function [panorama, rgbAnnotation] = renderPanorama(images, imgSize, cameras, mode, ref_idx, opts)
    % RENDERPANORAMA Render a panorama in the chosen projection with optional blending.
    %   [panorama, rgbAnnotation] = renderPanorama(images, cameras, mode, ref_idx, opts)
    %   composes the set of input images onto a common panorama surface using the
    %   provided camera intrinsics/extrinsics. Supports multiple projection modes and
    %   blending strategies, with optional GPU acceleration and tiling for memory.
    %
    %   Inputs
    %   - images  : 1xN or Nx1 cell array of RGB images (uint8 or single).
    %   - imgSize : 1xN or Nx1 array of [height, width, channels] for each image.
    %   - cameras : 1xN or Nx1 struct array with fields R (3x3 world->cam) and K (3x3).
    %   - mode    : 'cylindrical' | 'spherical' | 'equirectangular' | 'planar' |
    %               'perspective' | 'stereographic'.
    %   - ref_idx : Reference camera index (positive integer). Used to set planar frame
    %               and default panorama focal length.
    %   - opts    : Options struct (fields documented inline below) controlling focal
    %               length, resolution, tiling, GPU, blending, and annotations.
    %
    %   Outputs
    %   - panorama      : Rendered panorama image as uint8 [H x W x 3].
    %   - rgbAnnotation : Optional annotated panorama (only when enabled in opts),
    %                     else empty [].
    %
    %   Notes
    %   - For 'spherical' and 'equirectangular', the mapping is identical here.
    %   - 'stereographic' produces a fisheye-like projection centered on the reference view.
    %   - Use tiling and GPU options for large panoramas to control memory usage.
    %
    %   See also gainCompensation, multiBandBlending, gpuArray, gather

    arguments
        images cell
        imgSize (:, 3) {mustBeNumeric, mustBeFinite, mustBePositive}
        cameras struct
        mode {mustBeTextScalar}
        ref_idx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        opts (1, 1) struct = struct()
    end

    if nargin < 5, opts = struct(); end
    if ~isfield(opts, 'f_pan'), opts.f_pan = cameras(ref_idx).K(1, 1); end % Seed focal length
    if ~isfield(opts, 'res_scale'), opts.res_scale = 1.0; end % Resolution scale factor
    if ~isfield(opts, 'angle_power'), opts.angle_power = 1; end % View-angle weight power
    if ~isfield(opts, 'crop_border'), opts.crop_border = true; end % Crop black border
    if ~isfield(opts, 'margin'), opts.margin = 0.01; end % Margin fraction on bounds
    if ~isfield(opts, 'use_gpu'), opts.use_gpu = true; end % Use GPU if available
    if ~isfield(opts, 'parfor'), opts.parfor = true; end % Use parfor for tiling
    if ~isfield(opts, 'tile'), opts.tile = []; end % [tileH tileW] or []
    if ~isfield(opts, 'max_megapix'), opts.max_megapix = 50; end % cap total pixels (e.g., 50 MP)
    if ~isfield(opts, 'robust_pct'), opts.robust_pct = [1 99]; end % percentile clip for planar bounds
    if ~isfield(opts, 'uv_abs_cap'), opts.uv_abs_cap = 8.0; end % |u|,|v| max in plane units (≈ FOV clamp)
    if ~isfield(opts, 'pix_pad'), opts.pix_pad = 24; end % extra border in **pixels** on panorama plane
    if ~isfield(opts, 'auto_ref'), opts.auto_ref = true; end % auto-pick best planar ref_idx
    if ~isfield(opts, 'canvas_color'), opts.canvas_color = 'black'; end % 'black' | 'white'
    if ~isfield(opts, 'gain_compensation'), opts.gain_compensation = true; end % Gain compensation on/off
    if ~isfield(opts, 'sigma_N'), opts.sigma_N = 10.0; end % Noise stddev for gain comp.
    if ~isfield(opts, 'sigma_g'), opts.sigma_g = 0.1; end % Gain stddev for gain comp.
    if ~isfield(opts, 'blending'), opts.blending = 'multiband'; end % 'none' | 'linear' | 'multiband'
    if ~isfield(opts, 'overlap_stride'), opts.overlap_stride = 4; end % stride on panorama grid for overlap sampling
    if ~isfield(opts, 'pyr_levels'), opts.pyr_levels = 3; end % for multiband
    if ~isfield(opts, 'compose_none_policy'), opts.compose_none_policy = 'last'; end % pixel overwrite policy:
    % 'last' | 'first' | 'maxangle'
    % "last" (default): later images overwrite earlier ones (matches your old code order-dependent paste).
    % "first": first valid source wins; later images don't overwrite filled pixels.
    % "maxangle": pick the single camera with the largest view-angle weight (still no blending; weight only decides which source wins).
    if ~isfield(opts, 'showPanoramaImgsNums'), opts.showPanoramaImgsNums = false; end % show image nums on pano
    if ~isfield(opts, 'showCropBoundingBox'), opts.showCropBoundingBox = false; end % show crop box on pano
    if ~isfield(opts, 'blend_device'), opts.blend_device = 'auto'; end % 'gpu' | 'cpu' | 'auto'
    if ~isfield(opts, 'tile_min'), opts.tile_min = [512 768]; end % lower bound for tiling
    if ~isfield(opts, 'gpu_mem_frac'), opts.gpu_mem_frac = 0.5; end % keep peak <55 % free mem

    N = numel(images);

    % ---- Auto-pick best planar ref_idx by minimizing canvas area ----
    if (strcmpi(mode, 'planar') || strcmpi(mode, 'perspective')) && opts.auto_ref
        bestArea = inf; best_idx = ref_idx;

        for ii = 1:N
            Rref_i = cameras(ii).R;
            [u_min, u_max, v_min, v_max] = planar_bounds( ...
                cameras, imgSize, Rref_i, opts.robust_pct, opts.uv_abs_cap);

            % apply same growth as in your planar case
            du = u_max - u_min; dv = v_max - v_min;
            u_min2 = u_min - opts.margin * du - opts.pix_pad / opts.f_pan;
            u_max2 = u_max + opts.margin * du + opts.pix_pad / opts.f_pan;
            v_min2 = v_min - opts.margin * dv - opts.pix_pad / opts.f_pan;
            v_max2 = v_max + opts.margin * dv + opts.pix_pad / opts.f_pan;

            W_i = max(1, ceil(opts.f_pan * (u_max2 - u_min2) * opts.res_scale));
            H_i = max(1, ceil(opts.f_pan * (v_max2 - v_min2) * opts.res_scale));
            area_i = double(W_i) * double(H_i);

            if area_i < bestArea
                bestArea = area_i; best_idx = ii;
            end

        end

        ref_idx = best_idx; % use the best reference going forward
    end

    % -------- 1) Bounds on chosen surface (using R' = c2w) --------
    switch lower(mode)
        case 'cylindrical'
            [th_min, th_max, h_min, h_max] = cylindrical_bounds(cameras, imgSize);
            dt = th_max - th_min; dh = h_max - h_min;
            th_min = th_min - opts.margin * dt; th_max = th_max + opts.margin * dt;
            h_min = h_min - opts.margin * dh; h_max = h_max + opts.margin * dh;

            W = max(1, ceil(opts.f_pan * (th_max - th_min) * opts.res_scale));
            H = max(1, ceil(opts.f_pan * (h_max - h_min) * opts.res_scale));

            th0 = th_min; h0 = h_min;

        case 'spherical'
            [th_min, th_max, ph_min, ph_max] = spherical_bounds(cameras, imgSize);
            dt = th_max - th_min; dp = ph_max - ph_min;
            th_min = th_min - opts.margin * dt; th_max = th_max + opts.margin * dt;
            ph_min = ph_min - opts.margin * dp; ph_max = ph_max + opts.margin * dp;

            W = max(1, ceil(opts.f_pan * (th_max - th_min) * opts.res_scale));
            H = max(1, ceil(opts.f_pan * (ph_max - ph_min) * opts.res_scale));

            th0 = th_min; ph0 = ph_min;

        case {'planar', 'perspective'}
            % Tight bounds on the reference image plane (z_ref = +1)
            Rref = cameras(ref_idx).R; % world->ref
            [u_min, u_max, v_min, v_max] = planar_bounds( ...
                cameras, imgSize, Rref, opts.robust_pct, opts.uv_abs_cap);

            du = u_max - u_min; dv = v_max - v_min;
            u_min = u_min - opts.margin * du; u_max = u_max + opts.margin * du;
            v_min = v_min - opts.margin * dv; v_max = v_max + opts.margin * dv;

            % --- add small padding in pixel units on the panorama plane ---
            u_min = u_min - opts.pix_pad / opts.f_pan;
            u_max = u_max + opts.pix_pad / opts.f_pan;
            v_min = v_min - opts.pix_pad / opts.f_pan;
            v_max = v_max + opts.pix_pad / opts.f_pan;

            W = max(1, ceil(opts.f_pan * (u_max - u_min) * opts.res_scale));
            H = max(1, ceil(opts.f_pan * (v_max - v_min) * opts.res_scale));

            % --- global pixel cap (auto downscale if needed) ---
            max_px = round(opts.max_megapix * 1e6);
            HW_est = double(H) * double(W);

            if HW_est > max_px
                s = sqrt(max_px / HW_est); % uniform scale on both axes
                opts.res_scale = opts.res_scale * s;
                W = max(1, ceil(opts.f_pan * (u_max - u_min) * opts.res_scale));
                H = max(1, ceil(opts.f_pan * (v_max - v_min) * opts.res_scale));
            end

            u0 = u_min; v0 = v_min; % lower-left in plane coords

        case {'equirectangular'} % alias of spherical
            [th_min, th_max, ph_min, ph_max] = spherical_bounds(cameras, imgSize);
            dt = th_max - th_min; dp = ph_max - ph_min;
            th_min = th_min - opts.margin * dt; th_max = th_max + opts.margin * dt;
            ph_min = ph_min - opts.margin * dp; ph_max = ph_max + opts.margin * dp;

            W = max(1, ceil(opts.f_pan * (th_max - th_min) * opts.res_scale));
            H = max(1, ceil(opts.f_pan * (ph_max - ph_min) * opts.res_scale));

            th0 = th_min; ph0 = ph_min;

        case 'stereographic'
            % Bounds on the stereographic plane (relative to the reference camera)
            Rref = cameras(ref_idx).R; % world->ref
            [a_min, a_max, b_min, b_max] = stereographic_bounds( ...
                cameras, imgSize, Rref, opts.robust_pct, opts.uv_abs_cap);

            da = a_max - a_min; db = b_max - b_min;
            a_min = a_min - opts.margin * da; a_max = a_max + opts.margin * da;
            b_min = b_min - opts.margin * db; b_max = b_max + opts.margin * db;

            % pixel padding in plane units
            a_min = a_min - opts.pix_pad / opts.f_pan;
            a_max = a_max + opts.pix_pad / opts.f_pan;
            b_min = b_min - opts.pix_pad / opts.f_pan;
            b_max = b_max + opts.pix_pad / opts.f_pan;

            W = max(1, ceil(opts.f_pan * (a_max - a_min) * opts.res_scale));
            H = max(1, ceil(opts.f_pan * (b_max - b_min) * opts.res_scale));

            % global pixel cap (same as planar)
            max_px = round(opts.max_megapix * 1e6);
            HW_est = double(H) * double(W);

            if HW_est > max_px
                s = sqrt(max_px / HW_est);
                opts.res_scale = opts.res_scale * s;
                W = max(1, ceil(opts.f_pan * (a_max - a_min) * opts.res_scale));
                H = max(1, ceil(opts.f_pan * (b_max - b_min) * opts.res_scale));
            end

            u0 = a_min; v0 = b_min; % stereographic plane origin (same naming as planar)

        otherwise
            error('mode must be cylindrical, spherical, or planar/perspective');
    end

    % Choose device
    onGPU = opts.use_gpu && (gpuDeviceCount > 0);

    % Check for panorama size and if the projection is strange
    panorama = [];
    rgbAnnotation = [];

    % --- Estimate bytes we need ---
    % main float panorama (single) + logical covered + a working tile buffer margin
    bytes_pano = double(H) * double(W) * 3 * 4; % single RGB
    bytes_cover = double(H) * double(W) * 1; % logical ~1 byte
    bytes_margin = 128e6; % small slack (tune as you like)
    bytes_needed = bytes_pano + bytes_cover + bytes_margin;

    % --- Allocate only after we know it fits ---
    try
        panorama = zeros(H, W, 3, 'single'); % ~12 bytes per pixel
        covered = false(H, W); % ~1 byte per pixel

        if onGPU
            panorama = gpuArray(panorama);
            covered = gpuArray(covered);
        end

    catch ME
        warning('renderPanorama:Alloc', ...
            'Skipping panorama (allocation failed): %s', ME.message);
        panorama = [];
        return
    end

    if ~canFit(bytes_needed, onGPU)
        warning('renderPanorama:TooLarge', ...
            ['Skipping panorama: %dx%d requires ~%.2f GB; not enough %s memory.'], ...
            H, W, bytes_needed / 1e9, tern(onGPU, 'GPU', 'CPU'));
        return
    end

    % ---------- Simple auto-tiler (mode-aware, streaming-safe) ----------
    if isempty(opts.tile)
        N = numel(images);
        % available memory
        if opts.use_gpu && gpuDeviceCount > 0
            g = gpuDevice;
            avail = opts.gpu_mem_frac * double(g.AvailableMemory); % bytes
        else
            % very rough CPU fallback: assume 6 GB usable (adjust if you like)
            avail = 6e9;
        end

        % bytes per pixel per image (single precision)
        % 'none' ≈ RGB only; 'linear' ≈ RGB + weight; 'multiband' ≈ add 1.33 pyr factor
        switch lower(opts.blending)
            case 'none', bpppi = 4 * (3); pyr = 1.0; % ~12 B/px/img
            case 'linear', bpppi = 4 * (3 + 1); pyr = 1.0; % ~16 B/px/img
            otherwise % 'multiband'
                bpppi = 4 * (3 + 1); pyr = 1.33; % ~21.3 B/px/img
        end

        % Use a safety factor (40% memory) and assume worst-case contributors = N
        bytes_per_px = N * bpppi * pyr; % bytes per pano-pixel for this tile
        px_budget = max(1, floor(0.4 * avail / bytes_per_px));
        side = floor(sqrt(px_budget));

        % clamp to sane bounds and image size
        lo = 512; hi = 2048; % good general-purpose limits
        side = max(lo, min([side, H, W, hi]));
        opts.tile = [side side];
    end

    % Warp weights for all images (for gain compensation and blending)
    srcW = warpWeights(images, N);

    if opts.gain_compensation

        switch lower(mode)
            case {'planar', 'perspective', 'stereographic'}
                tic;
                gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
                    H, W, u0, v0, [], [], [], srcW);
                fprintf('Gain compensation planar/stereographic panorama rendering time : %f seconds\n', toc);

            case 'cylindrical'
                tic;
                gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
                    H, W, [], [], th0, h0, [], srcW);
                fprintf('Gain compensation cylindrical panorama rendering time : %f seconds\n', toc);

            case {'spherical', 'equirectangular'}
                tic;
                gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
                    H, W, [], [], th0, [], ph0, srcW);
                fprintf('Gain compensation spherical/equirectangular panorama rendering time : %f seconds\n', toc);

            otherwise
                error('Unsupported mode for gain compensation.');
        end

    else
        gains = ones(N, 3, 'single');
    end

    % Get signm and tile sizes
    if ~isfield(opts, 'pyr_sigma'), opts.pyr_sigma = 1.0; end
    tileH = opts.tile(1); tileW = opts.tile(2);

    % === ACCUMULATION & BLENDING (STREAMING, TILE-FIRST) =======================
    tic;
    % ---------------------- TILE LOOP (streaming) ------------------------------
    % You can parallelize **over tiles** (not over images) if you want:
    % parfor r = 1:tileH:H

    for r = 1:tileH:H
        rr = r:min(r + tileH - 1, H);

        for c = 1:tileW:W
            cc = c:min(c + tileW - 1, W);

            % --- build tile WORLD directions DWt ---
            [xp_t, yp_t] = meshgrid(single(cc - 1), single(rr - 1));

            switch lower(mode)
                case 'cylindrical'
                    theta = single(th0) + xp_t / single(opts.f_pan);
                    hline = single(h0) + yp_t / single(opts.f_pan);
                    dwx = -sin(theta); dwy = -hline; dwz = cos(theta);

                case {'spherical', 'equirectangular'}
                    theta = single(th0) + xp_t / single(opts.f_pan);
                    phi = single(ph0) + yp_t / single(opts.f_pan);
                    cphi = cos(phi); sphi = sin(phi);
                    dwx = -cphi .* sin(theta); dwy = -sphi; dwz = cphi .* cos(theta);

                case {'planar', 'perspective'}
                    u = single(u0) + xp_t / single(opts.f_pan);
                    v = single(v0) + yp_t / single(opts.f_pan);
                    dz_ref = ones(size(u), 'like', u);
                    Rref = single(cameras(ref_idx).R);
                    dwx = Rref(1, 1) .* u + Rref(2, 1) .* v + Rref(3, 1) .* dz_ref;
                    dwy = Rref(1, 2) .* u + Rref(2, 2) .* v + Rref(3, 2) .* dz_ref;
                    dwz = Rref(1, 3) .* u + Rref(2, 3) .* v + Rref(3, 3) .* dz_ref;

                case 'stereographic'
                    a = single(u0) + xp_t / single(opts.f_pan);
                    b = single(v0) + yp_t / single(opts.f_pan);
                    r2 = a .* a + b .* b; denom = (1 + r2);
                    dxr = 2 * a ./ denom; dyr = 2 * b ./ denom; dzr = (1 - r2) ./ denom;
                    Rref = single(cameras(ref_idx).R);
                    dwx = Rref(1, 1) .* dxr + Rref(2, 1) .* dyr + Rref(3, 1) .* dzr;
                    dwy = Rref(1, 2) .* dxr + Rref(2, 2) .* dyr + Rref(3, 2) .* dzr;
                    dwz = Rref(1, 3) .* dxr + Rref(2, 3) .* dyr + Rref(3, 3) .* dzr;
                otherwise
                    error('mode');
            end

            DWt = [dwx(:), dwy(:), dwz(:)];
            nrm = sqrt(sum(DWt .^ 2, 2));
            nrm = max(nrm, 1e-8); % guard
            DWt = DWt ./ nrm; % unit world rays

            if onGPU, DWt = gpuArray(DWt); end

            % --- fuse this tile immediately ---
            [F_tile, Wsum_tile] = fuse_tile(rr, cc, DWt, images, cameras, onGPU, srcW, gains, opts);

            % --- write back & coverage ---
            if ~onGPU && isa(F_tile, 'gpuArray'), F_tile = gather(F_tile); end
            panorama(rr, cc, :) = F_tile;

            if ~onGPU && isa(Wsum_tile, 'gpuArray'), Wsum_tile = gather(Wsum_tile); end
            covered(rr, cc) = covered(rr, cc) | (Wsum_tile > 0);
        end

    end

    mask_void = ~covered; % [H x W] logical

    fprintf('Streaming %s blending panorama rendering time : %f seconds\n', opts.blending, toc);
    % === END ACCUMULATION & BLENDING (STREAMING) ===============================

    % Paint the empty pixels to the requested canvas color
    if strcmpi(opts.canvas_color, 'white')
        % expand mask to 3 channels and set to 1.0 (before uint8 conversion)
        panorama(repmat(mask_void, [1 1 3])) = 1;
    else
        % black canvas is already zero; nothing needed
        % (optional) ensure zeros where void:
        panorama(repmat(mask_void, [1 1 3])) = 0;
    end

    % Gather & convert
    if onGPU, panorama = gather(panorama); end
    panorama = uint8(max(0, min(255, round(255 * panorama))));

    % -------- 4) Optional crop --------
    crop_rect = [1 size(panorama, 1) 1 size(panorama, 2)]; % default (no shift)

    if opts.crop_border
        [panorama, crop_rect, ~] = crop_nonzero_bbox(panorama, opts.canvas_color);
    end

    r1 = crop_rect(1); c1 = crop_rect(3); % top and left of the crop
    dx = c1 - 1; dy = r1 - 1; % shift to apply to x/y

    % --- Draw debugging annotations -----------------------------------------
    rgbAnnotation = [];

    if opts.showPanoramaImgsNums && opts.showCropBoundingBox

        switch lower(mode)
            case {'planar', 'perspective', 'stereographic'}
                [xBoxes, yBoxes, centers] = all_warped_boxes( ...
                    images, cameras, mode, ref_idx, opts, u0, v0, [], [], []);
            case 'cylindrical'
                [xBoxes, yBoxes, centers] = all_warped_boxes( ...
                    images, cameras, mode, ref_idx, opts, [], [], th0, h0, []);
            case {'spherical', 'equirectangular'}
                [xBoxes, yBoxes, centers] = all_warped_boxes( ...
                    images, cameras, mode, ref_idx, opts, [], [], th0, [], ph0);
        end

        % Shift by crop origin (panorama was cropped after coords were computed)
        xBoxes = cellfun(@(x) x - dx, xBoxes, 'uni', 0);
        yBoxes = cellfun(@(y) y - dy, yBoxes, 'uni', 0);
        centers = centers - [dx dy];

        % Validate, pack, and draw
        isValid = cellfun(@(x, y) numel(x) == numel(y) && numel(x) >= 3 && ...
            all(isfinite(x)) && all(isfinite(y)), ...
            xBoxes, yBoxes);

        xBoxesV = xBoxes(isValid);
        yBoxesV = yBoxes(isValid);
        centersV = centers(isValid, :);

        polyCells = cellfun(@(x, y) reshape([x(:) y(:)]', 1, []), ...
            xBoxesV, yBoxesV, 'uni', 0);

        rgbAnnotation = insertShape(panorama, 'Polygon', polyCells, 'LineWidth', 2);
        labels = arrayfun(@num2str, 1:numel(xBoxesV), 'uni', 0);
        rgbAnnotation = insertText(rgbAnnotation, centersV, labels, ...
            'FontSize', 24, 'BoxColor', 'red', 'TextColor', 'white');
    end

end

% Per-blending-mode tile worker -------------------------------------------
function [F_tile, Wsum_tile] = fuse_tile(rr, cc, DWt, images, cameras, onGPU, srcW, gains, opts)
    % FUSE_TILE Fuse a panorama tile using the selected blending strategy.
    %   [F_tile, Wsum_tile] = fuse_tile(rr, cc, DWt, images, cameras, onGPU, srcW, gains, opts)
    %   computes the blended RGB tile and a coverage indicator for the region
    %   specified by rr (rows) and cc (cols) using the world direction vectors DWt.
    %
    %   F_tile is [ht x wt x 3] single; Wsum_tile indicates coverage.

    arguments
        rr (:, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        cc (:, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        DWt (:, 3) {mustBeNumeric}
        images cell
        cameras struct
        onGPU (1, 1) logical
        srcW cell
        gains (:, 3) {mustBeNumeric}
        opts (1, 1) struct
    end

    N = numel(images);
    ht = numel(rr); wt = numel(cc);

    switch lower(opts.blending)
        case 'none'
            % For 'none', we implement policies streaming:
            %  - 'last'    : later images overwrite earlier ones
            %  - 'first'   : first valid wins
            %  - 'maxangle': select by largest angle weight
            out_tile = zeros(ht, numel(cc), 3, 'single');
            fill_tile = false(ht, numel(cc));

            if strcmpi(opts.compose_none_policy, 'maxangle')
                bestW = zeros(ht, numel(cc), 'single');
            end

            for iImg = 1:N
                [Si, Mi, Wang] = sample_one(images{iImg}, cameras(iImg), ...
                    DWt, ht, wt, onGPU, ...
                    opts.angle_power, srcW{iImg}, gains(iImg, :));
                Simg = reshape(Si, [ht wt 3]);
                Mimg = reshape(Mi, [ht wt]);

                assert(isequal(size(Simg), [ht wt 3]));
                assert(isequal(size(Mimg), [ht wt]));

                switch lower(opts.compose_none_policy)
                    case 'last'
                        mask2d = Mimg;
                        m3 = repmat(mask2d, [1 1 3]); % [ht x wt x 3]
                        out_tile(m3) = Simg(m3); % write all 3 channels
                        fill_tile = fill_tile | mask2d;

                    case 'first'
                        mask2d = Mimg & ~fill_tile;
                        m3 = repmat(mask2d, [1 1 3]);
                        out_tile(m3) = Simg(m3);
                        fill_tile(mask2d) = true;

                    case 'maxangle'
                        Wimg = reshape(Wang, [ht wt]);
                        assert(isequal(size(Wimg), [ht wt]));
                        upd2d = Mimg & (Wimg > bestW);
                        upd3 = repmat(upd2d, [1 1 3]);
                        out_tile(upd3) = Simg(upd3);
                        bestW(upd2d) = Wimg(upd2d);
                        fill_tile(upd2d) = true;
                    otherwise
                        error('compose_none_policy must be ''last'',''first'' or ''maxangle''.');
                end

            end

            F_tile = out_tile;
            Wsum_tile = single(fill_tile); % only used for coverage

        case 'linear'
            ht = numel(rr); wt = numel(cc);
            ht = numel(rr); wt = numel(cc);

            if onGPU
                accum_tile = gpuArray.zeros(ht, wt, 3, 'single');
                wsum_tile = gpuArray.zeros(ht, wt, 1, 'single');
            else
                accum_tile = zeros(ht, wt, 3, 'single');
                wsum_tile = zeros(ht, wt, 1, 'single');
            end

            any_valid = false(ht, wt);
            bestW = zeros(ht, wt, 'single');
            bestRGB = zeros(ht, wt, 3, 'single');

            for iImg = 1:N
                [Si, Mi, Wang, Wf] = sample_one(images{iImg}, cameras(iImg), ...
                    DWt, ht, wt, onGPU, ...
                    opts.angle_power, srcW{iImg}, gains(iImg, :));
                if ~any(Mi), continue; end

                % safe feather (clamp to [eps,1])
                Wf(~isfinite(Wf)) = 0;
                Wf = max(Wf, single(1e-4));

                w_vec = Wang .* Wf; w_vec(~Mi) = 0;

                Simg = reshape(Si, [ht wt 3]);
                Wimg = reshape(w_vec, [ht wt]);
                Mimg = reshape(Mi, [ht wt]);

                accum_tile = accum_tile + bsxfun(@times, Simg, Wimg);
                wsum_tile = wsum_tile + Wimg;

                any_valid = any_valid | Mimg; % << NEW

                better = Mimg & (Wimg > bestW);

                if any(better, 'all')
                    bestW(better) = Wimg(better);
                    b3 = repmat(better, [1 1 3]);
                    bestRGB(b3) = Simg(b3);
                end

            end

            F_tile = zeros(ht, wt, 3, 'single');
            z = (wsum_tile > 1e-12);

            if any(z, 'all')
                wsafe = wsum_tile; wsafe(~z) = 1;
                F_tile = bsxfun(@rdivide, accum_tile, wsafe);
            end

            need_fb = ~z & any_valid; % << per-pixel fallback

            if any(need_fb, 'all')
                nf3 = repmat(need_fb, [1 1 3]);
                F_tile(nf3) = bestRGB(nf3);
            end

            F_tile = gather(F_tile);
            Wsum_tile = gather(wsum_tile);

        case 'multiband'
            ht = numel(rr); wt = numel(cc);
            Ci = {}; Wc = {};
            any_valid = false(ht, wt);

            for iImg = 1:N
                [Si, Mi, Wang, Wf] = sample_one(images{iImg}, cameras(iImg), ...
                    DWt, ht, wt, onGPU, ...
                    opts.angle_power, srcW{iImg}, gains(iImg, :));
                if ~any(Mi), continue; end

                % same safe weight as linear
                w_vec = Wang .* Wf; w_vec(~Mi) = 0;

                Simg = reshape(Si, [ht wt 3]);
                Wimg = reshape(w_vec, [ht wt]);

                Ci{end + 1} = Simg; %#ok<AGROW>
                Wc{end + 1} = Wimg; %#ok<AGROW>
                any_valid = any_valid | (Wimg > 0);
            end

            if isempty(Ci)
                F_tile = zeros(ht, wt, 3, 'single');
                Wsum_tile = zeros(ht, wt, 'single');
                return;
            end

            % Normalize masks to sum~1 (avoid division by 0 outside contributors)
            sumW = zeros(ht, wt, 'like', Wc{1});
            for k = 1:numel(Wc), sumW = sumW + Wc{k}; end
            sumWnz = sumW > 1e-8;
            invSum = zeros(ht, wt, 'like', sumW);
            invSum(sumWnz) = 1 ./ sumW(sumWnz);

            for k = 1:numel(Wc)
                Wc{k} = Wc{k} .* invSum; % now \sum_k Wc{k} ≈ 1 where there are contributors
            end

            % Device choice (as you had)
            useGPUBlend = onGPU;
            if strcmpi(opts.blend_device, 'cpu'), useGPUBlend = false; end

            if strcmpi(opts.blend_device, 'auto') && onGPU
                g = gpuDevice; Ktile = numel(Ci);
                need_bytes = 4 * 1.33 * Ktile * ht * wt * 4;
                if need_bytes > 0.45 * g.AvailableMemory, useGPUBlend = false; end
            end

            if ~useGPUBlend

                for k = 1:numel(Ci)
                    if isa(Ci{k}, 'gpuArray'), Ci{k} = gather(Ci{k}); end
                    if isa(Wc{k}, 'gpuArray'), Wc{k} = gather(Wc{k}); end
                end

            end

            F_tile = multiBandBlending(Ci, Wc, opts.pyr_levels, useGPUBlend, opts.pyr_sigma);

            % coverage: any place that had at least one contributor is "covered"
            Wsum_tile = single(any_valid);

            % Wsum_tile = zeros(ht, wt, 'like', Wc{1});
            for k = 1:numel(Wc), Wsum_tile = Wsum_tile + Wc{k}; end

        otherwise
            error('Unknown opts.blending.');
    end

end

% Per-image per-tile sampler (streaming) -----------------------------------
function [S_vec, M_vec, Wang_vec, Wf_vec] = sample_one( ...
        I, cam, DWt, ht, wt, onGPU, angle_pow, srcW_i, gain_i)
    % SAMPLE_ONE Sample one image onto a panorama tile with weights and mask.
    %   [S_vec, M_vec, Wang_vec, Wf_vec] = sample_one(I, cam, DWt, ht, wt, onGPU, angle_pow, srcW_i, gain_i)
    %   projects world directions DWt into image I using camera cam, then samples
    %   colors and weights. Returns flattened vectors for colors S_vec, validity
    %   mask M_vec, angle-based weights Wang_vec, and feather weights Wf_vec.

    arguments
        I (:, :, :) {mustBeNumeric}
        cam (1, 1) struct
        DWt (:, 3) {mustBeNumeric}
        ht (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        wt (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        onGPU (1, 1) logical
        angle_pow (1, 1) {mustBeNumeric, mustBeFinite}
        srcW_i (:, :) {mustBeNumeric}
        gain_i (1, 3) {mustBeNumeric}
    end

    if ~isfield(cam, 'R') || ~isfield(cam, 'K')
        error('sample_one:CamFields', 'cam must contain fields R and K.');
    end

    % Convert & gain
    if isa(I, 'uint8'), Ic = single(I) / 255; else, Ic = single(I); end
    Ic = bsxfun(@times, Ic, reshape(gain_i, 1, 1, 3));
    if onGPU, Ic = gpuArray(Ic); end

    % Project WORLD dirs to this camera
    R = single(cam.R); K = single(cam.K);
    fx = K(1, 1); fy = K(2, 2); cx = K(1, 3); cy = K(2, 3);

    dirc = DWt * R.'; % [T x 3] in *camera* frame
    cx_w = dirc(:, 1); cy_w = dirc(:, 2); cz_w = dirc(:, 3);

    epsZ = single(1e-6);
    front = cz_w > epsZ;
    cz = max(cz_w, epsZ);

    u = fx * (cx_w ./ cz) + cx;
    v = fy * (cy_w ./ cz) + cy;

    % View-angle weight (cosine falloff)
    % For w2c R, camera +Z in WORLD coords is fw_world = R' * [0;0;1] = (R(3,:)).'
    % ---- in sample_one() BEFORE computing Wang_vec ----
    fw = R(3, :).'; % camera forward in WORLD coords
    Wang_vec = max(0, DWt * fw) .^ single(angle_pow); % [T x 1]
    Wang_vec = Wang_vec .* single(front); % zero out back-facing

    % Samples
    S_vec = fast_sample_block(Ic, u, v, ht, wt, onGPU); % [T x 3]
    Wf = fast_sample_block(srcW_i, u, v, ht, wt, onGPU); % [T x 1]
    Wf_vec = Wf(:, 1); % feather only

    % Validity mask (independent of weights)
    M_vec = all(isfinite(S_vec), 2) & Wang_vec > 0; % discard back-facing

    % ---- ADD THIS BLOCK (exactly like the OOM path's behavior) ----
    bad = ~M_vec; % invalid or out-of-bounds

    if any(bad)
        S_vec(bad, :) = 0; % kill NaNs in color
        Wf_vec(bad) = 0; % kill NaNs in feather
        Wang_vec(bad) = 0; % ensure zero weight everywhere bad
    end

end

function [xBoxes, yBoxes, centers] = all_warped_boxes(images, cameras, mode, ref_idx, opts, u0, v0, th0, h0, ph0)
    % ALL_WARPED_BOXES Compute projected image boxes and centers on the panorama.
    %   [xBoxes, yBoxes, centers] = all_warped_boxes(images, cameras, mode, ref_idx, opts, ...)
    %   returns polygon coordinates and centers for each input image in panorama
    %   coordinates for annotation.

    arguments
        images cell
        cameras struct
        mode {mustBeTextScalar}
        ref_idx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        opts (1, 1) struct
        u0
        v0
        th0
        h0
        ph0
    end

    N = numel(images);
    xBoxes = cell(N, 1); yBoxes = cell(N, 1); centers = zeros(N, 2);

    for i = 1:N
        [xPoly, yPoly, ctr] = warped_box_in_pano(size(images{i}), cameras(i), mode, ...
            ref_idx, cameras, opts, u0, v0, th0, h0, ph0, opts.f_pan);
        xBoxes{i} = xPoly; yBoxes{i} = yPoly; centers(i, :) = ctr;
    end

end

function [xPoly, yPoly, centroid] = warped_box_in_pano(imageSize, cam, mode, ...
        ref_idx, cameras, opts, u0, v0, th0, h0, ph0, f_pan)
    % WARPED_BOX_IN_PANO Project the image boundary into panorama coordinates.
    %   [xPoly, yPoly, centroid] = warped_box_in_pano(imageSize, cam, mode, ref_idx, cameras, opts, ...)
    %   computes the 4-corner polygon and its centroid on the panorama surface.

    arguments
        imageSize (:, :) {mustBeNumeric}
        cam (1, 1) struct
        mode {mustBeTextScalar}
        ref_idx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        cameras struct
        opts (1, 1) struct
        u0
        v0
        th0
        h0
        ph0
        f_pan (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    Hc = imageSize(1); Wc = imageSize(2);
    corners = [1, 1; 1, Wc; Hc, Wc; Hc, 1]; % [row, col]
    xy1 = [corners(:, 2)'; corners(:, 1)'; ones(1, 4)]; % [u;v;1] (col = x, row = y)

    ray_c = cam.K \ xy1;
    ray_w = cam.R' * ray_c; % to world

    % Map world ray to panorama coords (u,v) or (theta,h) or (theta,phi)
    switch lower(mode)
        case 'planar'
            Rref = cameras(ref_idx).R;
            ray_r = Rref * ray_w; zr = ray_r(3, :);
            ur = ray_r(1, :) ./ zr; vr = ray_r(2, :) ./ zr; % plane coords
            x = (ur - u0) * f_pan; y = (vr - v0) * f_pan;

        case 'cylindrical'
            xw = ray_w(1, :); yw = ray_w(2, :); zw = ray_w(3, :);
            theta = -atan2(xw, zw); h =- yw ./ hypot(xw, zw);
            x = (theta - th0) * f_pan; y = (h - h0) * f_pan;

        case 'spherical'
            xw = ray_w(1, :); yw = ray_w(2, :); zw = ray_w(3, :);
            theta = -atan2(xw, zw); phi = atan2(-yw, hypot(xw, zw));
            x = (theta - th0) * f_pan; y = (phi - ph0) * f_pan;

        case 'equirectangular' % alias of 'spherical'
            xw = ray_w(1, :); yw = ray_w(2, :); zw = ray_w(3, :);
            theta = -atan2(xw, zw); phi = atan2(-yw, hypot(xw, zw));
            x = (theta - th0) * f_pan; y = (phi - ph0) * f_pan;

        case 'stereographic'
            % Project to the reference camera frame first
            Rref = cameras(ref_idx).R;
            ray_r = Rref * ray_w; % to ref camera frame
            % normalize direction
            nr = sqrt(sum(ray_r .^ 2, 1));
            xr = ray_r(1, :) ./ nr; yr = ray_r(2, :) ./ nr; zr = ray_r(3, :) ./ nr;

            % Stereographic forward map (plane tangent at +Z; project from -Z):
            % a = x / (1 + z),  b = y / (1 + z)
            denom = 1 + zr;
            % guard degenerate (z ~ -1)
            denom = max(denom, 1e-6);
            a = xr ./ denom;
            b = yr ./ denom;

            x = (a - u0) * f_pan;
            y = (b - v0) * f_pan;
        otherwise , error('mode');
    end

    xPoly = [x, x(1)]; yPoly = [y, y(1)];
    centroid = [mean(x), mean(y)];
end

function srcW = warpWeights(images, N)
    % WARPWEIGHTS Build simple per-image feathering weights.
    %   srcW = warpWeights(images, N) returns a cell array of [h x w] single
    %   weight maps with a gentle center emphasis, one per input image.

    arguments
        images cell
        N (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    srcW = cell(N, 1);

    for i = 1:N
        [h, w, ~] = size(images{i});
        wx = ones(1, w, 'single');
        wx(1:ceil(w / 2)) = linspace(0, 1, ceil(w / 2));
        wx(floor(w / 2) + 1:w) = linspace(1, 0, w - floor(w / 2));
        wy = ones(h, 1, 'single');
        wy(1:ceil(h / 2)) = linspace(0, 1, ceil(h / 2));
        wy(floor(h / 2) + 1:h) = linspace(1, 0, h - floor(h / 2));
        srcW{i} = wy * wx; % [h x w], single
    end

end

function tf = canFit(bytes_needed, onGPU)
    % CANFIT Quick check for available CPU/GPU memory.
    % tf = canFit(bytes_needed, onGPU)
    % - bytes_needed: required bytes.
    % - onGPU: true=GPU, false=CPU (default false).
    % Returns true if allocation is likely to fit (best-effort).
    arguments
        bytes_needed (1, 1) {mustBeNumeric, mustBeFinite, mustBeNonnegative}
        onGPU (1, 1) logical = false
    end

    if onGPU

        try
            g = gpuDevice;
            % leave a safety headroom (50%)
            tf = bytes_needed < 0.5 * double(g.AvailableMemory);
        catch
            tf = false; % no GPU available / error
        end

    else
        % CPU side: use MaxPossibleArrayBytes as a proxy; keep a headroom
        try
            m = memory; % Windows/desktop MATLAB only
            tf = bytes_needed < 0.5 * double(m.MaxPossibleArrayBytes);
        catch
            % Fallback if 'memory' unsupported (e.g., Linux w/o swap info):
            % be conservative (require < 2 GB).
            tf = bytes_needed < 2e9;
        end

    end

end

function s = tern(cond, a, b)
    % TERN Ternary-like conditional selection.
    % s = tern(cond, a, b)
    % - cond: logical scalar or array. If true selects from a; otherwise from b.
    % - a: value(s) returned where cond is true. Must be size-compatible with cond/b.
    % - b: value(s) returned where cond is false. Must be size-compatible with cond/a.
    % Returns s: selected value(s); size and type follow standard MATLAB assignment rules.
    % Notes:
    % - If cond is a scalar, returns a when cond is true, otherwise b.
    % - If cond is an array, selection is element-wise; scalars in a or b are expanded as needed.
    % Examples:
    % - s = tern(true, 1, 2)              % returns 1
    % - s = tern([true false], 10, [1 2]) % returns [10 2]
    arguments
        cond {mustBeNumericOrLogical}
        a
        b
    end

    if cond
        s = a;
    else
        s = b;
    end

end

% ============================================================
function S = fast_sample_block(Ic, u, v, H, W, onGPU)
    % FAST_SAMPLE_BLOCK Vectorized bilinear sampling using interp2 with NaN extrapolation.
    %   S = fast_sample_block(Ic, u, v, H, W, onGPU) samples the C-channel image Ic at
    %   query points (u,v) and returns S as a [numel(u) x C] array. Out-of-bounds yield NaN.

    arguments
        Ic (:, :, :) {mustBeNumeric}
        u (:, 1) {mustBeNumeric}
        v (:, 1) {mustBeNumeric}
        H (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        W (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        onGPU (1, 1) logical
    end

    % Touch H/W to avoid unused-arg warnings given signature constraints
    H = H; %#ok<NASGU>
    W = W; %#ok<NASGU>

    [h, w, C] = size(Ic); %#ok<ASGLU> % h,w unused except for grid creation

    % Build grids for interp2 (1..w columns, 1..h rows)
    X = single(1:w);
    Y = single(1:h);

    if onGPU
        % Ensure grids live on GPU when sampling on GPU
        X = gpuArray(X);
        Y = gpuArray(Y);
    end

    % Sanitize coords; any non-finite -> will yield NaN from interp2
    bad = ~isfinite(u) | ~isfinite(v);

    if any(bad, 'all')
        % No need to force -1; extrapval=NaN handles oob and bad points
        % but keep them finite to avoid interp2 errors
        u(bad) = 1; % arbitrary in-bounds
        v(bad) = 1;
    end

    % Allocate result like u (CPU or GPU), with C channels
    S = zeros(numel(u), C, 'like', u);

    % Interpolate each channel with NaN extrapolation
    if C == 1
        S(:, 1) = interp2(X, Y, Ic, u, v, 'linear', NaN);
    else

        for ch = 1:C
            S(:, ch) = interp2(X, Y, Ic(:, :, ch), u, v, 'linear', NaN);
        end

    end

end

% ============================================================
function [pano_cropped, rect, didCrop] = crop_nonzero_bbox(panorama, canvas_color)
    % CROP_NONZERO_BBOX Crop panorama to non-canvas content with small padding.
    %   [pano_cropped, rect, didCrop] = crop_nonzero_bbox(panorama, canvas_color)
    %   returns the cropped panorama, the crop rectangle [r1 r2 c1 c2], and a
    %   boolean didCrop.

    arguments
        panorama (:, :, :) {mustBeNumeric}
        canvas_color {mustBeTextScalar}
    end

    % rect = [r1 r2 c1 c2] (1-based, inclusive), didCrop = true if cropped

    G = rgb2gray(panorama);

    if strcmpi(canvas_color, 'white')
        fg = (G < 255); % foreground is anything not pure white
    else
        fg = (G > 0); % foreground is anything not pure black
    end

    [r, c] = find(fg);

    if ~isempty(r)
        pad = 6; H = size(panorama, 1); W = size(panorama, 2);
        r1 = max(1, min(r) - pad); r2 = min(H, max(r) + pad);
        c1 = max(1, min(c) - pad); c2 = min(W, max(c) + pad);
        pano_cropped = panorama(r1:r2, c1:c2, :);
        rect = [r1 r2 c1 c2];
        didCrop = true;
    else
        pano_cropped = panorama;
        rect = [1 size(panorama, 1) 1 size(panorama, 2)];
        didCrop = false;
    end

end

% ============================================================
function [th_min, th_max, h_min, h_max] = cylindrical_bounds(cams, imgSize)
    % CYLINDRICAL_BOUNDS Compute cylindrical bounds (theta,h) over all cameras.
    %   [th_min, th_max, h_min, h_max] = cylindrical_bounds(cams, imgSize)
    %   samples each camera frame and accumulates global min/max in cylindrical coords.

    arguments
        cams struct
        imgSize (:, 3) {mustBeNumeric}
    end

    th_min = inf; th_max = -inf; h_min = inf; h_max = -inf;
    nx = 48; ny = 32;

    parfor i = 1:numel(cams)
        H = imgSize(i, 1); W = imgSize(i, 2);
        xs = linspace(1, W, nx); ys = linspace(1, H, ny);
        [U, V] = meshgrid(xs, ys); u = U(:)'; v = V(:)';
        xy1 = [u; v; ones(1, numel(u), 'like', u)];
        ray_c = cams(i).K \ xy1;
        ray_w = cams(i).R' * ray_c; % to world
        x = ray_w(1, :); y = ray_w(2, :); z = ray_w(3, :);
        theta = -atan2(x, z);
        h =- y ./ hypot(x, z);
        th_min = min(th_min, min(theta)); th_max = max(th_max, max(theta));
        h_min = min(h_min, min(h)); h_max = max(h_max, max(h));
    end

end

function [th_min, th_max, ph_min, ph_max] = spherical_bounds(cams, imgSize)
    % SPHERICAL_BOUNDS Compute spherical bounds (theta,phi) over all cameras.
    %   [th_min, th_max, ph_min, ph_max] = spherical_bounds(cams, imgSize)
    %   samples each camera frame and accumulates global min/max in spherical coords.

    arguments
        cams struct
        imgSize (:, 3) {mustBeNumeric}
    end

    th_min = inf; th_max = -inf; ph_min = inf; ph_max = -inf;
    nx = 48; ny = 32;

    parfor i = 1:numel(cams)
        H = imgSize(i, 1); W = imgSize(i, 2);
        xs = linspace(1, W, nx); ys = linspace(1, H, ny);
        [U, V] = meshgrid(xs, ys); u = U(:)'; v = V(:)';
        xy1 = [u; v; ones(1, numel(u), 'like', u)];
        ray_c = cams(i).K \ xy1;
        ray_w = cams(i).R' * ray_c;
        x = ray_w(1, :); y = ray_w(2, :); z = ray_w(3, :);
        theta = -atan2(x, z);
        phi = atan2(-y, hypot(x, z));
        th_min = min(th_min, min(theta)); th_max = max(th_max, max(theta));
        ph_min = min(ph_min, min(phi)); ph_max = max(ph_max, max(phi));
    end

end

function [u_min, u_max, v_min, v_max] = planar_bounds( ...
        cams, imgSize, Rref, robust_pct, uv_abs_cap)
    % PLANAR_BOUNDS Robust bounds on the reference image plane z_ref = +1.
    %   [u_min, u_max, v_min, v_max] = planar_bounds(cams, imgSize, Rref, robust_pct, uv_abs_cap)
    %   computes planar coordinate bounds with percentile clipping and hard caps.

    arguments
        cams struct
        imgSize (:, 3) {mustBeNumeric}
        Rref (3, 3) {mustBeNumeric, mustBeFinite}
        robust_pct (1, 2) {mustBeNumeric}
        uv_abs_cap (1, 1) {mustBeNumeric}
    end

    u_min = inf; u_max = -inf;
    v_min = inf; v_max = -inf;

    nx = 48; ny = 32; % coarse interior
    ne = 512; % dense borders
    z_eps = 1e-4; % slightly bigger than before for stability

    parfor i = 1:numel(cams)
        H = imgSize(i, 1); W = imgSize(i, 2);

        % interior samples
        xs = linspace(1, W, nx); ys = linspace(1, H, ny);
        [U, V] = meshgrid(xs, ys); u = U(:)'; v = V(:)';

        % border samples
        xb = linspace(1, W, ne); yb = linspace(1, H, ne);
        u_b = [xb, xb, ones(1, ne), W * ones(1, ne)];
        v_b = [ones(1, ne), H * ones(1, ne), yb, yb];

        u = [u, u_b]; v = [v, v_b];

        xy1 = [u; v; ones(1, numel(u), 'like', u)];
        ray_c = cams(i).K \ xy1;
        ray_w = cams(i).R' * ray_c; % to world
        ray_r = Rref * ray_w; % to ref camera frame

        zr = ray_r(3, :);
        m = zr > z_eps; % strictly front-facing and away from grazing

        if ~any(m), continue; end

        ur = ray_r(1, m) ./ zr(m);
        vr = ray_r(2, m) ./ zr(m);

        % Hard cap to avoid extreme blow-ups
        if isfinite(uv_abs_cap) && uv_abs_cap > 0
            ur = max(-uv_abs_cap, min(uv_abs_cap, ur));
            vr = max(-uv_abs_cap, min(uv_abs_cap, vr));
        end

        % Per-camera percentile clip (robust bounds)
        lo = robust_pct(1); hi = robust_pct(2);
        u_lo = prctile(ur, lo); u_hi = prctile(ur, hi);
        v_lo = prctile(vr, lo); v_hi = prctile(vr, hi);

        % Update global bounds
        u_min = min(u_min, u_lo); u_max = max(u_max, u_hi);
        v_min = min(v_min, v_lo); v_max = max(v_max, v_hi);
    end

    % Safety: if something went wrong, fall back to a sane box
    if ~isfinite(u_min) || ~isfinite(u_max) || u_min >= u_max
        u_min = -1; u_max = 1;
    end

    if ~isfinite(v_min) || ~isfinite(v_max) || v_min >= v_max
        v_min = -1; v_max = 1;
    end

end

function [a_min, a_max, b_min, b_max] = stereographic_bounds( ...
        cams, imgSize, Rref, robust_pct, abs_cap)
    % STEREOGRAPHIC_BOUNDS Robust bounds on the stereographic plane.
    %   [a_min, a_max, b_min, b_max] = stereographic_bounds(cams, imgSize, Rref, robust_pct, abs_cap)
    %   computes bounds after stereographic mapping with percentile clipping and caps.

    arguments
        cams struct
        imgSize (:, 3) {mustBeNumeric}
        Rref (3, 3) {mustBeNumeric, mustBeFinite}
        robust_pct (1, 2) {mustBeNumeric}
        abs_cap (1, 1) {mustBeNumeric}
    end

    a_min = inf; a_max = -inf;
    b_min = inf; b_max = -inf;

    nx = 48; ny = 32; % interior grid
    ne = 512; % dense borders

    parfor i = 1:numel(cams)
        H = imgSize(i, 1); W = imgSize(i, 2);

        % interior samples
        xs = linspace(1, W, nx); ys = linspace(1, H, ny);
        [U, V] = meshgrid(xs, ys); u = U(:)'; v = V(:)';

        % border samples
        xb = linspace(1, W, ne); yb = linspace(1, H, ne);
        u_b = [xb, xb, ones(1, ne), W * ones(1, ne)];
        v_b = [ones(1, ne), H * ones(1, ne), yb, yb];

        u = [u, u_b]; v = [v, v_b];

        xy1 = [u; v; ones(1, numel(u), 'like', u)];
        ray_c = cams(i).K \ xy1; % camera frame (not unit)
        ray_w = cams(i).R' * ray_c; % to world
        ray_r = Rref * ray_w; % to ref frame

        % normalize to unit directions
        nr = sqrt(sum(ray_r .^ 2, 1));
        xr = ray_r(1, :) ./ nr; yr = ray_r(2, :) ./ nr; zr = ray_r(3, :) ./ nr;

        % stereographic forward map (plane tangent at +Z; project from -Z)
        denom = 1 + zr;
        % avoid explosion near zr ~ -1 (those go to infinity anyway)
        valid = denom > 1e-6;
        if ~any(valid), continue; end

        a = xr(valid) ./ denom(valid);
        b = yr(valid) ./ denom(valid);

        % optional hard cap (like planar uv_abs_cap)
        if isfinite(abs_cap) && abs_cap > 0
            a = max(-abs_cap, min(abs_cap, a));
            b = max(-abs_cap, min(abs_cap, b));
        end

        % per-camera percentile clip then update global
        lo = robust_pct(1); hi = robust_pct(2);
        a_lo = prctile(a, lo); a_hi = prctile(a, hi);
        b_lo = prctile(b, lo); b_hi = prctile(b, hi);

        a_min = min(a_min, a_lo); a_max = max(a_max, a_hi);
        b_min = min(b_min, b_lo); b_max = max(b_max, b_hi);
    end

    % safety fallback
    if ~isfinite(a_min) || ~isfinite(a_max) || a_min >= a_max
        a_min = -1; a_max = 1;
    end

    if ~isfinite(b_min) || ~isfinite(b_max) || b_min >= b_max
        b_min = -1; b_max = 1;
    end

end
