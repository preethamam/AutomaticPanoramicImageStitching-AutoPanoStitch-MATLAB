function [panorama, rgbAnnotation] = renderPanorama(images, cameras, mode, ref_idx, opts)
% mode: 'cylindrical' | 'spherical' | 'planar' (aka 'perspective')
% cameras(i).R : world->camera (w2c)
% cameras(i).K : intrinsics
% ref_idx only used to set default panorama focal and (for planar) the plane frame
%
% SPEED TIPS:
%   - Set opts.use_gpu = true (requires Parallel Toolbox & GPU) for big boosts.
%   - Set opts.parfor = true to parallelize across cameras.
%   - Use tiling via opts.tile = [tileH tileW] for very large panoramas or GPUs.
%
% DEPENDS ON:
%   - cyl_bounds_dense_fast, sph_bounds_dense_fast, plan_bounds_dense_fast (below)
%   - fast_sample_block (interp2-based sampler)
%   - crop_nonzero_bbox (optional crop)

if nargin<5, opts=struct(); end
if ~isfield(opts,'f_pan'),       opts.f_pan       = cameras(ref_idx).K(1,1); end
if ~isfield(opts,'res_scale'),   opts.res_scale   = 1.0;                     end
if ~isfield(opts,'angle_power'), opts.angle_power = 1;                       end
if ~isfield(opts,'crop_border'), opts.crop_border = true;                    end
if ~isfield(opts,'margin'),      opts.margin      = 0.05;                  end
if ~isfield(opts,'use_gpu'),     opts.use_gpu     = true;                    end
if ~isfield(opts,'parfor'),      opts.parfor      = true;                    end
if ~isfield(opts,'tile'),        opts.tile        = [];                      end   % [tileH tileW] or []
if ~isfield(opts,'max_megapix'),  opts.max_megapix  = 50;     end % cap total pixels (e.g., 50 MP)
if ~isfield(opts,'robust_pct'),   opts.robust_pct   = [1 99]; end % percentile clip for planar bounds
if ~isfield(opts,'uv_abs_cap'),   opts.uv_abs_cap   = 8.0;    end % |u|,|v| max in plane units (≈ FOV clamp)
if ~isfield(opts,'pix_pad'),      opts.pix_pad      = 24;     end % extra border in **pixels** on panorama plane
if ~isfield(opts,'auto_ref'),    opts.auto_ref    = true;   end
if ~isfield(opts,'canvas_color'), opts.canvas_color = 'black'; end  % 'black' | 'white'
if ~isfield(opts,'gain_compensation'), opts.gain_compensation = true; end
if ~isfield(opts,'sigma_N'),             opts.sigma_N             = 10.0;end
if ~isfield(opts,'sigma_g'),             opts.sigma_g             = 0.1; end
if ~isfield(opts,'blending'),          opts.blending = 'multiband'; end % 'none'|'linear'|'multiband'
if ~isfield(opts,'overlap_stride'),    opts.overlap_stride = 4; end   % stride on panorama grid for overlap sampling
if ~isfield(opts,'pyr_levels'),        opts.pyr_levels = 3; end       % for multiband
if ~isfield(opts,'compose_none_policy'), opts.compose_none_policy = 'last'; end
% 'last' | 'first' | 'maxangle'
% "last" (default): later images overwrite earlier ones (matches your old code order-dependent paste).
% "first": first valid source wins; later images don't overwrite filled pixels.
% "maxangle": pick the single camera with the largest view-angle weight (still no blending; weight only decides which source wins).
if ~isfield(opts,'showPanoramaImgsNums'), opts.showPanoramaImgsNums = false; end
if ~isfield(opts,'showCropBoundingBox'), opts.showCropBoundingBox = false; end
if ~isfield(opts,'blend_device'), opts.blend_device = 'auto'; end  % 'gpu'|'cpu'|'auto'
if ~isfield(opts,'tile_min'),     opts.tile_min     = [512 768];  end % lower bound for tiling
if ~isfield(opts,'gpu_mem_frac'), opts.gpu_mem_frac = 0.55;       end % keep peak <55% free mem


N = numel(images);
imgSize = zeros(N,2);
for i=1:N, imgSize(i,:) = [size(images{i},1), size(images{i},2)]; end

% ---- Auto-pick best planar ref_idx by minimizing canvas area ----
if (strcmpi(mode,'planar') || strcmpi(mode,'perspective')) && opts.auto_ref
    bestArea = inf; best_idx = ref_idx;
    for ii = 1:N
        Rref_i = cameras(ii).R;
        [u_min,u_max,v_min,v_max] = planar_bounds( ...
            cameras, imgSize, Rref_i, opts.robust_pct, opts.uv_abs_cap);

        % apply same growth as in your planar case
        du = u_max - u_min; dv = v_max - v_min;
        u_min2 = u_min - opts.margin*du - opts.pix_pad/opts.f_pan;
        u_max2 = u_max + opts.margin*du + opts.pix_pad/opts.f_pan;
        v_min2 = v_min - opts.margin*dv - opts.pix_pad/opts.f_pan;
        v_max2 = v_max + opts.margin*dv + opts.pix_pad/opts.f_pan;

        W_i = max(1, ceil(opts.f_pan * (u_max2 - u_min2) * opts.res_scale));
        H_i = max(1, ceil(opts.f_pan * (v_max2 - v_min2) * opts.res_scale));
        area_i = double(W_i) * double(H_i);

        if area_i < bestArea
            bestArea = area_i; best_idx = ii;
        end
    end
    ref_idx = best_idx;   % use the best reference going forward
end


% -------- 1) Bounds on chosen surface (using R' = c2w) --------
switch lower(mode)
    case 'cylindrical'
        [th_min, th_max, h_min, h_max] = cylindrical_bounds(cameras, imgSize);
        dt = th_max - th_min; dh = h_max - h_min;
        th_min = th_min - opts.margin*dt; th_max = th_max + opts.margin*dt;
        h_min  = h_min  - opts.margin*dh; h_max  = h_max  + opts.margin*dh;
    
        W = max(1, ceil(opts.f_pan * (th_max - th_min) * opts.res_scale));
        H = max(1, ceil(opts.f_pan * (h_max  - h_min ) * opts.res_scale));
    
        th0 = th_min; h0 = h_min;
    
    case 'spherical'
        [th_min, th_max, ph_min, ph_max] = spherical_bounds(cameras, imgSize);
        dt = th_max - th_min; dp = ph_max - ph_min;
        th_min = th_min - opts.margin*dt; th_max = th_max + opts.margin*dt;
        ph_min = ph_min - opts.margin*dp; ph_max = ph_max + opts.margin*dp;
    
        W = max(1, ceil(opts.f_pan * (th_max - th_min) * opts.res_scale));
        H = max(1, ceil(opts.f_pan * (ph_max - ph_min) * opts.res_scale));
    
        th0 = th_min; ph0 = ph_min;

    case {'planar','perspective'}
        % Tight bounds on the reference image plane (z_ref = +1)
        Rref = cameras(ref_idx).R;  % world->ref
        [u_min, u_max, v_min, v_max] = planar_bounds( ...
        cameras, imgSize, Rref, opts.robust_pct, opts.uv_abs_cap);
    
        du = u_max - u_min; dv = v_max - v_min;
        u_min = u_min - opts.margin*du; u_max = u_max + opts.margin*du;
        v_min = v_min - opts.margin*dv; v_max = v_max + opts.margin*dv;
        
        % --- add small padding in pixel units on the panorama plane ---
        u_min = u_min - opts.pix_pad / opts.f_pan;
        u_max = u_max + opts.pix_pad / opts.f_pan;
        v_min = v_min - opts.pix_pad / opts.f_pan;
        v_max = v_max + opts.pix_pad / opts.f_pan;
        
        W = max(1, ceil(opts.f_pan * (u_max - u_min) * opts.res_scale));
        H = max(1, ceil(opts.f_pan * (v_max - v_min) * opts.res_scale));
        
        % --- global pixel cap (auto downscale if needed) ---
        max_px = round(opts.max_megapix*1e6);
        HW_est = double(H)*double(W);
        if HW_est > max_px
            s = sqrt(max_px / HW_est);                 % uniform scale on both axes
            opts.res_scale = opts.res_scale * s;
            W = max(1, ceil(opts.f_pan * (u_max - u_min) * opts.res_scale));
            H = max(1, ceil(opts.f_pan * (v_max - v_min) * opts.res_scale));
        end
    
        u0 = u_min; v0 = v_min;  % lower-left in plane coords


    otherwise
        error('mode must be cylindrical, spherical, or planar/perspective');
end

% ---- Auto tiler based on GPU memory + K contributors ----
if isempty(opts.tile)
    % approximate worst-case bytes per tile on GPU during multiband:
    % per-image: ~ 4 bytes * (h*w*(C + 1))  (color + weight at base level)
    % plus transient for pyr building; multiply by ~ (1 + 1/4 + 1/16 + ...) ≈ 1.33
    Kmax = numel(images); Cc = 3; scale = 1.33; 
    bytes_per_px_per_img = 4*(Cc+1)*scale; % single precision
    peak_budget = inf;
    if opts.use_gpu && (gpuDeviceCount>0)
        g = gpuDevice;
        peak_budget = opts.gpu_mem_frac * g.AvailableMemory;  % bytes we can safely use
    end
    % choose tile so that Kmax * bytes_per_px_per_img * (tileH*tileW) <= peak_budget/2  (headroom)
    if isfinite(peak_budget)
        px_budget = floor(0.5 * double(peak_budget) / (Kmax * bytes_per_px_per_img));
        side = max( min( floor(sqrt(px_budget)), min(H,W) ),  min(opts.tile_min) );
        tileH = max(opts.tile_min(1), min(H, side));
        tileW = max(opts.tile_min(2), min(W, side));
        opts.tile = [tileH tileW];
    else
        % CPU or unknown GPU mem: still tile a bit for safety
        opts.tile = max(opts.tile_min, [min(H,2048) min(W,2048)]);
    end
end


% -------- 2) Precompute WORLD direction grid --------
[xp, yp] = meshgrid(single(0:W-1), single(0:H-1));
switch lower(mode)
    case 'cylindrical'
        theta = single(th0) + xp/single(opts.f_pan);
        h     = single(h0)  + yp/single(opts.f_pan);
        dwx = -sin(theta);
        dwy = -h;
        dwz =  cos(theta);
    
    case 'spherical'
        theta = single(th0) + xp/single(opts.f_pan);
        phi   = single(ph0) + yp/single(opts.f_pan);
        cphi  = cos(phi); sphi = sin(phi);
        dwx = -cphi.*sin(theta);
        dwy = -sphi;
        dwz =  cphi.*cos(theta);

    case {'planar','perspective'}
        % Plane coords on the reference image plane (z_ref = +1)
        u = single(u0) + xp/single(opts.f_pan);
        v = single(v0) + yp/single(opts.f_pan);
    
        % Directions in REF camera frame
        dx_ref = u;  dy_ref = v;  dz_ref = ones(size(u), 'like', u);
    
        % Convert to WORLD directions: dir_w = Rref' * dir_ref   (NOT Rref)
        Rref = single(cameras(ref_idx).R);  % world->ref
        % use columns of Rref (i.e., multiply by Rref')
        dwx =  Rref(1,1).*dx_ref + Rref(2,1).*dy_ref + Rref(3,1).*dz_ref;
        dwy =  Rref(1,2).*dx_ref + Rref(2,2).*dy_ref + Rref(3,2).*dz_ref;
        dwz =  Rref(1,3).*dx_ref + Rref(2,3).*dy_ref + Rref(3,3).*dz_ref;
    
    otherwise
        error('mode must be cylindrical, spherical, or planar/perspective');

end

% (B) optional global gains from sparse overlaps
srcW = warpWeights(images, N);

if opts.gain_compensation
    switch lower(mode)
        case {'planar','perspective'}
            gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
                                          H, W, u0, v0, [], [], [], srcW);
        case 'cylindrical'
            gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
                                          H, W, [], [], th0, h0, [], srcW);
        case 'spherical'
            gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
                                          H, W, [], [], th0, [], ph0, srcW);
        otherwise
            error('Unsupported mode for gain compensation.');
    end
else
    gains = ones(N,3,'single');
end


% Flatten once (vector math)
HW = H*W;
DW = [dwx(:), dwy(:), dwz(:)];  % [HW x 3], single
clear dwx dwy dwz

% Choose device
onGPU = opts.use_gpu && (gpuDeviceCount>0);
if onGPU
    DW = gpuArray(DW);
end

if ~strcmpi(opts.blending,'multiband')
    % Single global accumulators (linear / none)
    accum = zeros(HW,3,'single');    % global color sum
    w_sum = zeros(HW,1,'single');    % global weight sum
    if onGPU, accum = gpuArray(accum); w_sum = gpuArray(w_sum); end
else
    % For multiband, store per-image per-pixel samples to fuse later
    colorCells  = cell(N,1);   % each: [HW x 3] single (CPU or GPU)
    weightCells = cell(N,1);   % each: [HW x 1] single
end

% === ACCUMULATION & BLENDING =================================================
switch lower(opts.blending)
case 'none'
    % ---- plain mosaic, zero blending: collect per-camera samples ----
    S_list  = cell(N,1);   % [HW x 3] single (CPU/GPU)
    M_list  = cell(N,1);   % [HW x 1] logical
    Wsel_ls = cell(N,1);   % [HW x 1] single (view-angle score)

    if opts.parfor
        parfor i=1:N
            [Si, meta] = process_one_camera( ...
                images{i}, cameras(i), DW, H, W, onGPU, ...
                opts.angle_power, srcW{i}, gains(i,:), 'none', 1);
            S_list{i}  = Si;
            M_list{i}  = meta.mask;
            Wsel_ls{i} = meta.sel;
        end
    else
        for i=1:N
            [Si, meta] = process_one_camera( ...
                images{i}, cameras(i), DW, H, W, onGPU, ...
                opts.angle_power, srcW{i}, gains(i,:), 'none', 1);
            S_list{i}  = Si;
            M_list{i}  = meta.mask;
            Wsel_ls{i} = meta.sel;
        end
    end

    % Compose according to policy: 'last' | 'first' | 'maxangle'
    [pano_hw3, fill_mask] = compose_none(S_list, M_list, Wsel_ls, N, HW, opts.compose_none_policy, onGPU);
    panorama = reshape(pano_hw3, [H, W, 3]);
    mask_void = ~reshape(fill_mask, [H, W]);        % logical [H x W]


case 'linear'
    % ---- weighted average (your previous linear path) ----
    % prepare global accumulators
    accum = zeros(HW,3,'single'); w_sum = zeros(HW,1,'single');
    if onGPU, accum = gpuArray(accum); w_sum = gpuArray(w_sum); end

    if opts.parfor
        accumParts = cell(N,1); wsumParts = cell(N,1);
        parfor i=1:N
            [acc_i, w_i] = process_one_camera( ...
                images{i}, cameras(i), DW, H, W, onGPU, ...
                opts.angle_power, srcW{i}, gains(i,:), 'linear', opts.pyr_levels);
            accumParts{i} = acc_i; wsumParts{i} = w_i;
        end
        for i=1:N
            accum = accum + accumParts{i};
            w_sum = w_sum + wsumParts{i};
        end
    else
        for i=1:N
            [acc_i, w_i] = process_one_camera( ...
                images{i}, cameras(i), DW, H, W, onGPU, ...
                opts.angle_power, srcW{i}, gains(i,:), 'linear', opts.pyr_levels);
            accum = accum + acc_i;
            w_sum = w_sum + w_i;
        end
    end

    % finalize linear
    mask_void = (w_sum < 1e-12);
    w_sum(mask_void) = 1;
    panorama = reshape(accum ./ w_sum, [H, W, 3]);
    mask_void = reshape(mask_void, [H, W]);  % make it [H x W] for painting

case 'multiband'
    % ---- store per-image color/weight; fuse tile-wise later ----
    colorCells  = cell(N,1);   % [HW x 3]
    weightCells = cell(N,1);   % [HW x 1]

    if opts.parfor
        parfor i=1:N
            [Ci, Wi] = process_one_camera( ...
                images{i}, cameras(i), DW, H, W, onGPU, ...
                opts.angle_power, srcW{i}, gains(i,:), 'multiband', opts.pyr_levels);
            colorCells{i}  = Ci;    % raw color
            weightCells{i} = Wi;    % per-pixel weight
        end
    else
        for i=1:N
            [Ci, Wi] = process_one_camera( ...
                images{i}, cameras(i), DW, H, W, onGPU, ...
                opts.angle_power, srcW{i}, gains(i,:), 'multiband', opts.pyr_levels);
            colorCells{i}  = Ci;
            weightCells{i} = Wi;
        end
    end

    % tile-wise fuse with your GPU-aware multiBandBlending
    if isempty(opts.tile), tileH = H; tileW = W;
    else, tileH = opts.tile(1); tileW = opts.tile(2);
    end
    if ~isfield(opts,'pyr_sigma'),  opts.pyr_sigma  = 1.0; end
    if onGPU, covered = gpuArray.false(H, W); else, covered = false(H, W); end


    panorama = zeros(H, W, 3, 'single'); if onGPU, panorama = gpuArray(panorama); end
    for r = 1:tileH:H
        rr = r:min(r+tileH-1, H);
        for c = 1:tileW:W
            cc = c:min(c+tileW-1, W);

            % Build Ci/Wi per tile for all images that contribute
            Ci_tile = {}; Wi_tile = {};
            for i = 1:N
                Wi_full = reshape(weightCells{i}, [H, W]);
                Wi_sub = Wi_full(rr, cc);
                if ~any(Wi_sub(:)), continue; end
                Ci_full = reshape(colorCells{i}, [H, W, 3]);
                Ci_tile{end+1} = Ci_full(rr, cc, :);   %#ok<AGROW>
                Wi_tile{end+1} = Wi_sub;               %#ok<AGROW>
            end
            if isempty(Ci_tile)
                continue;
            end
            
            % Update coverage: any pixel with >0 total weight is covered
            Wsum_tile = zeros(numel(rr), numel(cc), 'like', Wi_tile{1});
            for k = 1:numel(Wi_tile), Wsum_tile = Wsum_tile + Wi_tile{k}; end
            covered(rr, cc) = covered(rr, cc) | (Wsum_tile > 0);
            
            % Decide blending device for this tile
            useGPUBlend = onGPU;
            if strcmpi(opts.blend_device,'cpu'), useGPUBlend = false; end
            if strcmpi(opts.blend_device,'auto')
                useGPUBlend = onGPU;
                if onGPU
                    g = gpuDevice;
                    % Heuristic: if (tile px * K contributors) is large relative to free mem, switch to CPU
                    Ktile = numel(Wi_tile);
                    tile_px = numel(rr) * numel(cc);
                    need_bytes = 4 * 1.33 * Ktile * tile_px * 4;  % ~ 4 channels worth of temps
                    if need_bytes > 0.45 * g.AvailableMemory
                        useGPUBlend = false;
                    end
                end
            end

            % Optionally bring Ci/Wi to CPU for blend (sampling can stay on GPU)
            if ~useGPUBlend
                for kk=1:numel(Ci_tile)
                    if isa(Ci_tile{kk},'gpuArray'), Ci_tile{kk} = gather(Ci_tile{kk}); end
                    if isa(Wi_tile{kk},'gpuArray'), Wi_tile{kk} = gather(Wi_tile{kk}); end
                end
            end


            % --- Harmonize per-image tile inputs (defensive) ---
            for kk = 1:numel(Ci_tile)
                [hC,wC,~] = size(Ci_tile{kk});
                [hW,wW]   = size(Wi_tile{kk});
                if hC~=hW || wC~=wW
                    Wi_tile{kk} = imresize(Wi_tile{kk}, [hC wC], 'bilinear');
                end
            end

            % Fuse this tile
            F_tile = multiBandBlending(Ci_tile, Wi_tile, opts.pyr_levels, onGPU, opts.pyr_sigma);
            panorama(rr, cc, :) = F_tile;

        end
    end

    mask_void = ~covered;     % [H x W] logical

otherwise
    error('Unknown opts.blending.');
end
% === END ACCUMULATION & BLENDING =========================================

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
panorama = uint8( max(0, min(255, round(255*panorama))) );

% -------- 4) Optional crop --------
crop_rect = [1 size(panorama,1) 1 size(panorama,2)];  % default (no shift)
if opts.crop_border
    [panorama, crop_rect, ~] = crop_nonzero_bbox(panorama, opts.canvas_color);
end
r1 = crop_rect(1);  c1 = crop_rect(3);      % top and left of the crop
dx = c1 - 1;        dy = r1 - 1;            % shift to apply to x/y

% --- Draw debugging annotations -----------------------------------------
rgbAnnotation = [];
if opts.showPanoramaImgsNums && opts.showCropBoundingBox
    switch lower(mode)
        case {'planar','perspective'}
            [xBoxes, yBoxes, centers] = all_warped_boxes(...
                images, cameras, mode, ref_idx, opts, u0,v0,[],[],[]);
        case 'cylindrical'
            [xBoxes, yBoxes, centers] = all_warped_boxes(...
                images, cameras, mode, ref_idx, opts, [],[],th0,h0,[]);
        case 'spherical'
            [xBoxes, yBoxes, centers] = all_warped_boxes(...
                images, cameras, mode, ref_idx, opts, [],[],th0,[],ph0);
    end

    % Shift by crop origin (panorama was cropped after coords were computed)
    xBoxes = cellfun(@(x) x - dx, xBoxes, 'uni', 0);
    yBoxes = cellfun(@(y) y - dy, yBoxes, 'uni', 0);
    centers = centers - [dx dy];

    % Validate, pack, and draw
    isValid = cellfun(@(x,y) numel(x)==numel(y) && numel(x)>=3 && ...
                               all(isfinite(x)) && all(isfinite(y)), ...
                      xBoxes, yBoxes);

    xBoxesV  = xBoxes(isValid);
    yBoxesV  = yBoxes(isValid);
    centersV = centers(isValid,:);

    polyCells = cellfun(@(x,y) reshape([x(:) y(:)]',1,[]), ...
                        xBoxesV, yBoxesV, 'uni', 0);

    rgbAnnotation = insertShape(panorama,'Polygon',polyCells,'LineWidth',2);
    labels = arrayfun(@num2str, 1:numel(xBoxesV), 'uni', 0);
    rgbAnnotation = insertText(rgbAnnotation, centersV, labels, ...
                               'FontSize',24,'BoxColor','red','TextColor','white');
end
end

function [out, fill] = compose_none(S_list, M_list, Wsel_ls, N, HW, policy, onGPU)
if onGPU, out=gpuArray.zeros(HW,3,'single'); fill=gpuArray.false(HW,1);
else,      out=zeros(HW,3,'single');         fill=false(HW,1);
end

switch lower(policy)
case 'last'
    for i=1:N
        M = M_list{i};
        if ~any(M), continue; end
        S = S_list{i};
        out(M,:) = S(M,:);
        fill = fill | M;
    end

case 'first'
    for i=1:N
        M = M_list{i} & ~fill;
        if ~any(M), continue; end
        S = S_list{i};
        out(M,:) = S(M,:);
        fill(M)  = true;
    end

case 'maxangle'
    if onGPU, bestW=gpuArray.zeros(HW,1,'single'); else, bestW=zeros(HW,1,'single'); end
    for i=1:N
        M = M_list{i}; if ~any(M), continue; end
        W = Wsel_ls{i}; S = S_list{i};
        upd = M & (W > bestW);
        if any(upd)
            out(upd,:) = S(upd,:);
            bestW(upd) = W(upd);
            fill(upd)  = true;
        end
    end

otherwise
    error('compose_none_policy must be ''last'',''first'' or ''maxangle''.');
end
end


function [xBoxes, yBoxes, centers] = all_warped_boxes(images, cameras, mode, ref_idx, opts, u0,v0,th0,h0,ph0)
    N = numel(images);
    xBoxes = cell(N,1); yBoxes = cell(N,1); centers = zeros(N,2);
    for i=1:N
      [xPoly, yPoly, ctr] = warped_box_in_pano(size(images{i}), cameras(i), mode, ...
                             ref_idx, cameras, opts, u0,v0,th0,h0,ph0, opts.f_pan);
      xBoxes{i} = xPoly; yBoxes{i} = yPoly; centers(i,:) = ctr;
    end
end

function [xPoly, yPoly, centroid] = warped_box_in_pano(imageSize, cam, mode, ...
    ref_idx, cameras, opts, u0,v0, th0,h0,ph0, f_pan)

    Hc = imageSize(1); Wc = imageSize(2);
    corners = [1,1; 1,Wc; Hc,Wc; Hc,1]; % [row, col]
    xy1 = [corners(:,2)'; corners(:,1)'; ones(1,4)]; % [u;v;1] (col = x, row = y)
    
    ray_c = cam.K \ xy1;
    ray_w = cam.R' * ray_c;  % to world
    
    % Map world ray to panorama coords (u,v) or (theta,h) or (theta,phi)
    switch lower(mode)
     case 'planar'
      Rref = cameras(ref_idx).R;
      ray_r = Rref * ray_w;  zr = ray_r(3,:);
      ur = ray_r(1,:)./zr;   vr = ray_r(2,:)./zr; % plane coords
      x = (ur - u0) * f_pan; y = (vr - v0) * f_pan;
    
     case 'cylindrical'
      xw=ray_w(1,:); yw=ray_w(2,:); zw=ray_w(3,:);
      theta = -atan2(xw,zw); h = - yw ./ hypot(xw,zw);
      x = (theta - th0) * f_pan; y = (h - h0) * f_pan;
    
     case 'spherical'
      xw=ray_w(1,:); yw=ray_w(2,:); zw=ray_w(3,:);
      theta = -atan2(xw,zw); phi = atan2(-yw, hypot(xw,zw));
      x = (theta - th0) * f_pan; y = (phi - ph0) * f_pan;
    
     otherwise, error('mode');
    end
    
    xPoly = [x, x(1)]; yPoly = [y, y(1)];
    centroid = [mean(x), mean(y)];
end

function srcW = warpWeights(images, N)
    srcW = cell(N,1);
    for i=1:N
        [h,w,~] = size(images{i});
        wx = ones(1,w,'single');
        wx(1:ceil(w/2)) = linspace(0,1,ceil(w/2));
        wx(floor(w/2)+1:w) = linspace(1,0,w-floor(w/2));
        wy = ones(h,1,'single');
        wy(1:ceil(h/2)) = linspace(0,1,ceil(h/2));
        wy(floor(h/2)+1:h) = linspace(1,0,h-floor(h/2));
        srcW{i} = wy * wx;                     % [h x w], single
    end
end

% ============================================================
function [A, B] = process_one_camera(I, cam, DW, H, W, onGPU, angle_power, srcW_i, gain_i, blend_mode, pyr_levels)
% For 'none', returns:
%   A = S          [HW x 3] raw samples (no gain? we still apply gain for consistency)
%   B = M or Wsel  [HW x 1] selection score:
%        - for 'none'/'last'/'first' -> logical mask M (valid samples)
%        - for 'none'/'maxangle'     -> selection score w_angle (float)
% For 'linear'/'multiband' (unchanged): A=accum color or raw color, B=weight

% Convert & gain
if isa(I,'uint8'), Ic = single(I)/255; else, Ic = single(I); end
Ic = bsxfun(@times, Ic, reshape(gain_i,1,1,3));
if onGPU, Ic = gpuArray(Ic); end

% Project
R = single(cam.R); K = single(cam.K);
fx=K(1,1); fy=K(2,2); cx=K(1,3); cy=K(2,3);
dirc = DW * R.';  cx_w=dirc(:,1); cy_w=dirc(:,2); cz_w=dirc(:,3);
front = cz_w > 1e-6;
u = single(fx * (cx_w ./ cz_w) + cx);
v = single(fy * (cy_w ./ cz_w) + cy);

% Angle weight for selection/blending
fw = R(:,3);
w_angle = max(0, DW * fw).^single(angle_power);
w_angle = w_angle .* single(front);

% Sample color
S = fast_sample_block(Ic, u, v, H, W, onGPU);     % [HW x 3]
M = all(isfinite(S),2) & (w_angle > 0);

switch lower(blend_mode)
    case 'none'
        % Return raw color and *selection* (mask or score; caller decides)
        A = S;                    % [HW x 3]
        B = struct('mask', M, 'sel', w_angle);  % struct to support policies

    case 'linear'
        % Linear
        Wf = fast_sample_block(srcW_i, u, v, H, W, onGPU); Wf = Wf(:,1);
        S(~M,:) = 0; Wf(~M) = 0; w_angle(~M) = 0;
        Wfinal = w_angle .* Wf;
        A = S .* Wfinal;          % accum color
        B = Wfinal;               % weight

    case 'multiband'
        Wf = fast_sample_block(srcW_i, u, v, H, W, onGPU); Wf = Wf(:,1);
        S(~M,:) = 0; Wf(~M) = 0; w_angle(~M) = 0;
        A = S;                    % raw color
        B = w_angle .* Wf;        % per-pixel weight
    otherwise
        error('Unknown blend mode.');
end
end

% ============================================================
function S = fast_sample_block(Ic, u, v, H, W, onGPU)
% Vectorized bilinear sampling using interp2 with NaN extrapolation.
% Ic : [h x w x C] (single, CPU or gpuArray). C can be 1 or 3 (or any).
% u,v: [HW x 1] single (same device as Ic if onGPU==true)
% S  : [HW x C] (same device as u/v). NaNs for out-of-bounds.

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
    u(bad) = 1;  % arbitrary in-bounds
    v(bad) = 1;
end

% Allocate result like u (CPU or GPU), with C channels
S = zeros(numel(u), C, 'like', u);

% Interpolate each channel with NaN extrapolation
if C==1
    S(:,1) = interp2(X, Y, Ic, u, v, 'linear', NaN);
else
    for ch=1:C
        S(:,ch) = interp2(X, Y, Ic(:,:,ch), u, v, 'linear', NaN);
    end
end

end


% ============================================================
function [pano_cropped, rect, didCrop] = crop_nonzero_bbox(panorama, canvas_color)
% rect = [r1 r2 c1 c2] (1-based, inclusive), didCrop = true if cropped

G = rgb2gray(panorama);
if strcmpi(canvas_color,'white')
    fg = (G < 255);   % foreground is anything not pure white
else
    fg = (G > 0);     % foreground is anything not pure black
end

[r,c] = find(fg);
if ~isempty(r)
    pad=6; H=size(panorama,1); W=size(panorama,2);
    r1=max(1,min(r)-pad); r2=min(H,max(r)+pad);
    c1=max(1,min(c)-pad); c2=min(W,max(c)+pad);
    pano_cropped = panorama(r1:r2, c1:c2, :);
    rect   = [r1 r2 c1 c2];
    didCrop = true;
else
    pano_cropped = panorama;
    rect   = [1 size(panorama,1) 1 size(panorama,2)];
    didCrop = false;
end
end


% ============================================================
function [th_min, th_max, h_min, h_max] = cylindrical_bounds(cams, imgSize)
th_min= inf; th_max=-inf; h_min= inf; h_max=-inf;
nx=48; ny=32;
parfor i=1:numel(cams)
    H = imgSize(i,1); W = imgSize(i,2);
    xs = linspace(1,W,nx); ys = linspace(1,H,ny);
    [U,V] = meshgrid(xs,ys); u = U(:)'; v = V(:)';
    xy1   = [u; v; ones(1,numel(u),'like',u)];
    ray_c = cams(i).K \ xy1;
    ray_w = cams(i).R' * ray_c;     % to world
    x=ray_w(1,:); y=ray_w(2,:); z=ray_w(3,:);
    theta = -atan2(x,z);
    h     = - y ./ hypot(x,z);
    th_min = min(th_min, min(theta)); th_max = max(th_max, max(theta));
    h_min  = min(h_min,  min(h));    h_max  = max(h_max,  max(h));
end
end

function [th_min, th_max, ph_min, ph_max] = spherical_bounds(cams, imgSize)
th_min= inf; th_max=-inf; ph_min= inf; ph_max=-inf;
nx=48; ny=32;
parfor i=1:numel(cams)
    H = imgSize(i,1); W = imgSize(i,2);
    xs = linspace(1,W,nx); ys = linspace(1,H,ny);
    [U,V] = meshgrid(xs,ys); u = U(:)'; v = V(:)';
    xy1   = [u; v; ones(1,numel(u),'like',u)];
    ray_c = cams(i).K \ xy1;
    ray_w = cams(i).R' * ray_c;
    x=ray_w(1,:); y=ray_w(2,:); z=ray_w(3,:);
    theta = -atan2(x,z);
    phi   = atan2(-y, hypot(x,z));
    th_min = min(th_min, min(theta)); th_max = max(th_max, max(theta));
    ph_min = min(ph_min, min(phi));   ph_max = max(ph_max,  max(phi));
end
end

function [u_min, u_max, v_min, v_max] = planar_bounds( ...
        cams, imgSize, Rref, robust_pct, uv_abs_cap)
% Robust bounds on the reference image plane z_ref = +1.
% - Discards rays with tiny z (grazing).
% - Per-camera percentile clipping (e.g., [1 99]).
% - Hard caps |u|,|v| to uv_abs_cap to guard pathological cases.
%
% Inputs:
%   robust_pct = [lo hi], e.g., [1 99]
%   uv_abs_cap = scalar, e.g., 8.0  (≈ ~130° half-FOV in plane units)
%
% Notes:
%   u = x/z, v = y/z in ref camera coords (z>0 only).

u_min =  inf; u_max = -inf;
v_min =  inf; v_max = -inf;

nx = 48; ny = 32;     % coarse interior
ne = 512;             % dense borders
z_eps = 1e-4;         % slightly bigger than before for stability

parfor i = 1:numel(cams)
    H = imgSize(i,1); W = imgSize(i,2);

    % interior samples
    xs = linspace(1,W,nx); ys = linspace(1,H,ny);
    [U,V] = meshgrid(xs,ys); u = U(:)'; v = V(:)';

    % border samples
    xb = linspace(1,W,ne); yb = linspace(1,H,ne);
    u_b = [xb, xb,           ones(1,ne),   W*ones(1,ne)];
    v_b = [ones(1,ne), H*ones(1,ne), yb,   yb         ];

    u = [u, u_b];  v = [v, v_b];

    xy1   = [u; v; ones(1,numel(u),'like',u)];
    ray_c = cams(i).K \ xy1;
    ray_w = cams(i).R' * ray_c;       % to world
    ray_r = Rref * ray_w;             % to ref camera frame

    zr = ray_r(3,:);
    m  = zr > z_eps;                  % strictly front-facing and away from grazing

    if ~any(m), continue; end

    ur = ray_r(1,m) ./ zr(m);
    vr = ray_r(2,m) ./ zr(m);

    % Hard cap to avoid extreme blow-ups
    if isfinite(uv_abs_cap) && uv_abs_cap > 0
        ur = max(-uv_abs_cap, min(uv_abs_cap, ur));
        vr = max(-uv_abs_cap, min(uv_abs_cap, vr));
    end

    % Per-camera percentile clip (robust bounds)
    lo = robust_pct(1); hi = robust_pct(2);
    u_lo = prctile(ur, lo);  u_hi = prctile(ur, hi);
    v_lo = prctile(vr, lo);  v_hi = prctile(vr, hi);

    % Update global bounds
    u_min = min(u_min, u_lo); u_max = max(u_max, u_hi);
    v_min = min(v_min, v_lo); v_max = max(v_max, v_hi);
end

% Safety: if something went wrong, fall back to a sane box
if ~isfinite(u_min) || ~isfinite(u_max) || u_min>=u_max
    u_min = -1; u_max = 1;
end
if ~isfinite(v_min) || ~isfinite(v_max) || v_min>=v_max
    v_min = -1; v_max = 1;
end
end
