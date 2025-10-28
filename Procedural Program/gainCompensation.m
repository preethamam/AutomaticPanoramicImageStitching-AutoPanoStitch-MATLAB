function gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
        H, W, u0, v0, th0, h0, ph0, srcW)
    % GAINCOMPENSATION Estimate per-image RGB gains using Brown–Lowe (2007) Eq. (29).
    %   gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
    %           H, W, u0, v0, th0, h0, ph0, srcW) computes brightness gain factors
    %   for each image so that overlapping regions have consistent exposure. The
    %   method tiles a subsampled panorama grid, accumulates per-overlap statistics,
    %   and solves a small linear system per channel for the gains. Optional GPU
    %   support keeps intermediate arrays on the device.
    %
    %   Inputs
    %   - images    : 1xN or Nx1 cell of RGB images (numeric arrays).
    %   - cameras   : 1xN or Nx1 struct array with fields R (3x3), K (3x3).
    %   - mode      : projection mode: 'planar' | 'cylindrical' | 'spherical' |
    %                 'equirectangular' | 'stereographic'.
    %   - ref_idx   : reference camera index (positive integer).
    %   - opts      : struct of options. Fields used include:
    %                 overlap_stride, min_overlap_samples, sigma_N, sigma_g,
    %                 lambda_diag, parfor_tiles, anchor_ref, tile, use_gpu, f_pan.
    %   - H, W      : panorama height and width (subsampled grid derives from stride).
    %   - u0, v0    : planar/stereographic center offsets.
    %   - th0, h0   : cylindrical base angles/height offsets.
    %   - ph0       : spherical/equirectangular latitude offset.
    %   - srcW      : 1xN or Nx1 cell of single-channel weight/coverage maps.
    %
    %   Output
    %   - gains     : N-by-3 single matrix of per-image RGB gains, clamped to [0.25, 4].
    %
    %   Notes
    %   - We accumulate only per-edge statistics required by Eq.(29): counts N_ij and
    %     channel sums over overlaps; means are recovered as sums / N_ij.
    %   - Tiling is done over a SUBSAMPLED grid to reduce memory; we never build full DWs.
    %   - GPU path keeps arrays on device and reduces there; final A/b assembly happens on CPU.
    %
    %   See also svd, gpuArray, gather, interp2

    arguments
        images cell
        cameras struct
        mode {mustBeTextScalar}
        ref_idx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        opts (1, 1) struct
        H (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        W (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        u0
        v0
        th0
        h0
        ph0
        srcW cell
    end

    % Basic consistency and camera validation
    N = numel(images);

    if ~(isvector(images) && isvector(srcW) && numel(srcW) == N)
        error('gainCompensation:SrcWSize', 'srcW must be a vector cell array with the same length as images.');
    end

    if ~(isstruct(cameras) && numel(cameras) == N)
        error('gainCompensation:CamerasSize', 'cameras must be a struct array with one entry per image.');
    end

    for i = 1:N

        if ~isfield(cameras(i), 'R') || ~isfield(cameras(i), 'K')
            error('gainCompensation:CamFields', 'cameras(%d) must contain fields R and K.', i);
        end

        Ri = cameras(i).R; Ki = cameras(i).K;

        if ~(isnumeric(Ri) && isequal(size(Ri), [3, 3]) && all(isfinite(Ri(:))))
            error('gainCompensation:RShape', 'cameras(%d).R must be a 3x3 finite numeric matrix.', i);
        end

        if ~(isnumeric(Ki) && isequal(size(Ki), [3, 3]) && all(isfinite(Ki(:))))
            error('gainCompensation:KShape', 'cameras(%d).K must be a 3x3 finite numeric matrix.', i);
        end

    end

    N = numel(images);
    gains = ones(N, 3, 'single');

    % -------- Defaults --------
    if ~isfield(opts, 'overlap_stride'), opts.overlap_stride = 5; end
    if ~isfield(opts, 'min_overlap_samples'), opts.min_overlap_samples = 50; end
    if ~isfield(opts, 'sigma_N'), opts.sigma_N = 10.0; end
    if ~isfield(opts, 'sigma_g'), opts.sigma_g = 0.1; end
    if ~isfield(opts, 'lambda_diag'), opts.lambda_diag = 1e-8; end
    if ~isfield(opts, 'parfor_tiles'), opts.parfor_tiles = false; end
    if ~isfield(opts, 'anchor_ref'), opts.anchor_ref = false; end
    if ~isfield(opts, 'tile'), opts.tile = [512 512]; end

    % Auto GPU?
    if ~isfield(opts, 'use_gpu')
        opts.use_gpu = (gpuDeviceCount > 0);
    end

    stride = max(1, opts.overlap_stride);
    minOv = max(1, opts.min_overlap_samples);
    sN2 = (opts.sigma_N) ^ 2;
    sg2 = (opts.sigma_g) ^ 2;

    % -------- Subsampled grid size --------
    xp_all = single(1:stride:W);
    yp_all = single(1:stride:H);
    Hs = numel(yp_all);
    Ws = numel(xp_all);

    tileH = min(opts.tile(1), Hs);
    tileW = min(opts.tile(2), Ws);

    % -------- Accumulators (host/CPU) --------
    % For each (i,j), i<j:
    Nij = zeros(N, N, 'double'); % counts
    sumCi_ij = zeros(N, N, 3, 'double'); % sum of Ci per channel
    sumCj_ij = zeros(N, N, 3, 'double'); % sum of Cj per channel

    % -------- Tile enumeration over subsampled grid --------
    x_tiles = 1:tileW:Ws;
    y_tiles = 1:tileH:Hs;

    % To optionally run tiles in parallel:
    tileJobs = [];

    for ty = y_tiles

        for tx = x_tiles
            tileJobs(end + 1, :) = [ty, tx]; %#ok<AGROW>
        end

    end

    % -------- Run tiles (optionally parfor) --------
    if opts.parfor_tiles
        % PARFOR over tiles (good when GPU not used or when GPU is busy elsewhere)
        parfor t = 1:size(tileJobs, 1)
            [Nij_l, sCi_l, sCj_l] = do_one_tile(tileJobs(t, 1), tileJobs(t, 2), opts, mode, ref_idx, cameras, ...
                u0, v0, th0, h0, ph0, srcW, images, tileH, tileW, ...
                Hs, Ws, xp_all, yp_all, N);
            % Accumulate into master (parfor requires reduction variables)
            Nij = Nij + Nij_l;
            sumCi_ij = sumCi_ij + sCi_l;
            sumCj_ij = sumCj_ij + sCj_l;
        end

    else

        for t = 1:size(tileJobs, 1)
            [Nij_l, sCi_l, sCj_l] = do_one_tile(tileJobs(t, 1), tileJobs(t, 2), opts, mode, ref_idx, cameras, ...
                u0, v0, th0, h0, ph0, srcW, images, tileH, tileW, ...
                Hs, Ws, xp_all, yp_all, N);
            Nij = Nij + Nij_l;
            sumCi_ij = sumCi_ij + sCi_l;
            sumCj_ij = sumCj_ij + sCj_l;
        end

    end

    % -------- Build edge list that meets min overlap --------
    edges = [];

    for i = 1:N - 1

        for j = i + 1:N

            if Nij(i, j) >= minOv
                edges(end + 1, :) = [i j]; %#ok<AGROW>
            end

        end

    end

    if isempty(edges)
        return; % gains = ones
    end

    % -------- Build A g = b per channel from Eq. (29) --------
    A = zeros(N, N, 3, 'double');
    b = zeros(N, 1, 'double');

    for e = 1:size(edges, 1)
        i = edges(e, 1); j = edges(e, 2);
        Kij = Nij(i, j);
        if Kij < minOv, continue; end

        % Means
        Ibar_ij = reshape(sumCi_ij(i, j, :), [1 3]) / Kij; % mean Ci on (i,j)
        Ibar_ji = reshape(sumCj_ij(i, j, :), [1 3]) / Kij; % mean Cj on (i,j)

        wN = double(Kij) / sN2; % data weight
        wG = double(Kij) / sg2; % prior weight (paper multiplies by Nij)

        for ch = 1:3
            aii = wN * (Ibar_ij(ch) * Ibar_ij(ch)) + wG;
            ajj = wN * (Ibar_ji(ch) * Ibar_ji(ch)) + wG;
            aij =- wN * (Ibar_ij(ch) * Ibar_ji(ch));

            A(i, i, ch) = A(i, i, ch) + aii;
            A(j, j, ch) = A(j, j, ch) + ajj;
            A(i, j, ch) = A(i, j, ch) + aij;
            A(j, i, ch) = A(j, i, ch) + aij;
        end

        % RHS (prior) adds equally to b(i), b(j)
        b(i) = b(i) + wG;
        b(j) = b(j) + wG;
    end

    % Optional hard anchor to pin the reference gain to 1
    if opts.anchor_ref
        pin = max(1, min(N, ref_idx));

        for ch = 1:3
            A(:, :, ch) = A(:, :, ch) + opts.lambda_diag * eye(N);
            A(pin, :, ch) = 0; A(:, pin, ch) = 0; A(pin, pin, ch) = 1e6;
        end

        b(pin) = 1e6; % drives g_ref -> 1
    else

        for ch = 1:3
            A(:, :, ch) = A(:, :, ch) + opts.lambda_diag * eye(N);
        end

    end

    % -------- Solve per channel --------
    for ch = 1:3
        x = A(:, :, ch) \ b;
        gains(:, ch) = single(max(0.25, min(4.0, x)));
    end

end % gainCompensation

% -------- Per-tile worker (nested for clarity) --------
function [Nij_loc, sumCi_loc, sumCj_loc] = do_one_tile(ty, tx, opts, mode, ref_idx, cameras, ...
        u0, v0, th0, h0, ph0, weights, imgs, tileH, tileW, ...
        Hs, Ws, xp_all, yp_all, N)
    % DO_ONE_TILE Accumulate overlap statistics for a single subsampled tile.
    %   [Nij_loc, sumCi_loc, sumCj_loc] = do_one_tile(ty, tx, opts, mode, ref_idx, cameras,
    %       u0, v0, th0, h0, ph0, weights, imgs, tileH, tileW, Hs, Ws, xp_all, yp_all, N)
    %   computes, for the tile starting at (ty, tx) on the subsampled grid, the
    %   per-pair overlap counts and per-channel color sums needed for gain solving.
    %
    %   Outputs are double and sized to N-by-N (and N-by-N-by-3 for color sums).

    arguments
        ty (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        tx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        opts (1, 1) struct
        mode {mustBeTextScalar}
        ref_idx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        cameras struct
        u0
        v0
        th0
        h0
        ph0
        weights cell
        imgs cell
        tileH (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        tileW (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        Hs (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        Ws (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        xp_all (:, 1) {mustBeNumeric, mustBeFinite}
        yp_all (:, 1) {mustBeNumeric, mustBeFinite}
        N (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    % ranges in subsampled-index space
    y2 = min(ty + tileH - 1, Hs);
    x2 = min(tx + tileW - 1, Ws);
    yy = yp_all(ty:y2); xx = xp_all(tx:x2);
    [xp_s, yp_s] = meshgrid(xx, yy); % small tile grid (subsampled)

    % Build world directions for this tile (on chosen device)
    [DWx, DWy, DWz] = pano_dirs_for_grid_tile(xp_s, yp_s, mode, ref_idx, cameras, ...
        opts.f_pan, u0, v0, th0, h0, ph0, opts.use_gpu);

    DWs = cat(3, DWx, DWy, DWz);

    % Pre-alloc per-image projections/masks for this tile (device arrays)
    uL = cell(N, 1); vL = cell(N, 1); cov = cell(N, 1);

    % Project into each image and compute coverage from srcW
    for i = 1:N
        [u_i, v_i, front_i] = project_to_image(DWs, cameras(i), opts.use_gpu);
        % Sample srcW{i} (single-channel) at (u_i,v_i). Returns [HW x 1]
        Wi = sample_linear(weights{i}, u_i, v_i, size(xp_s), opts.use_gpu);

        Mi = isfinite(Wi) & (Wi > 0) & front_i & isfinite(u_i) & isfinite(v_i);
        uL{i} = u_i; vL{i} = v_i; cov{i} = Mi;
    end

    % Accumulators local to the tile (host-friendly at end)
    Nij_loc = zeros(N, N, 'double');
    sumCi_loc = zeros(N, N, 3, 'double');
    sumCj_loc = zeros(N, N, 3, 'double');

    % For each pair (i,j), accumulate only the sums and counts
    for i = 1:N - 1
        Mi = cov{i};
        if ~any(Mi, 'all'), continue; end
        ui = uL{i}; vi = vL{i};

        % Sample the (small) subset per-channel only where needed by pairs
        % We'll sample lazily inside each pair to reuse masks.

        for j = i + 1:N
            Mj = cov{j};
            if ~any(Mj, 'all'), continue; end
            Mij = Mi & Mj;
            if ~any(Mij, 'all'), continue; end

            % Flatten mask indices for this tile
            idx = find(Mij(:));
            if isempty(idx), continue; end

            % Sample colors for i,j at these idx only (device)
            ui_s = ui(idx); vi_s = vi(idx);
            uj_s = uL{j}(idx); vj_s = vL{j}(idx);

            Ci = sample_linear_rgb(single(imgs{i}), ui_s, vi_s, opts.use_gpu); % [K x 3]
            Cj = sample_linear_rgb(single(imgs{j}), uj_s, vj_s, opts.use_gpu);

            % Filter any non-finites (rare; NaNs from borders)
            good = all(isfinite(Ci), 2) & all(isfinite(Cj), 2);
            if ~any(good), continue; end

            Ci = Ci(good, :); Cj = Cj(good, :);
            k = size(Ci, 1);

            % Reduce on device, then gather
            sCi = double(gather(sum(Ci, 1)));
            sCj = double(gather(sum(Cj, 1)));

            Nij_loc(i, j) = Nij_loc(i, j) + k;
            sumCi_loc(i, j, :) = sumCi_loc(i, j, :) + reshape(sCi, [1, 1, 3]);
            sumCj_loc(i, j, :) = sumCj_loc(i, j, :) + reshape(sCj, [1, 1, 3]);
        end

    end

end

function [dwx, dwy, dwz] = pano_dirs_for_grid_tile(xp_s, yp_s, mode, ...
        ref_idx, cameras, f_pan, u0, v0, ...
        th0, h0, ph0, useGPU)
    % PANO_DIRS_FOR_GRID_TILE Compute world direction vectors for a pano grid tile.
    %   [dwx, dwy, dwz] = pano_dirs_for_grid_tile(xp_s, yp_s, mode, ref_idx, cameras,
    %       f_pan, u0, v0, th0, h0, ph0, useGPU) returns per-pixel world direction
    %   vectors for the given projection mode, relative to a reference camera.
    %
    %   Returns three arrays of the same size as xp_s/yp_s containing the x/y/z
    %   components of the direction vectors.

    arguments
        xp_s (:, :) {mustBeNumeric, mustBeFinite}
        yp_s (:, :) {mustBeNumeric, mustBeFinite}
        mode {mustBeTextScalar}
        ref_idx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        cameras struct
        f_pan (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        u0 
        v0 
        th0
        h0 
        ph0
        useGPU (1, 1) logical
    end

    if useGPU
        xp_s = gpuArray(xp_s); yp_s = gpuArray(yp_s);
    end

    switch lower(mode)
        case 'planar'
            u = single(u0) + xp_s / single(f_pan);
            v = single(v0) + yp_s / single(f_pan);
            dx_ref = u; dy_ref = v; dz_ref = ones(size(u), 'like', u);
            Rref = single(cameras(ref_idx).R);
            Rt = Rref.';
            dwx = Rt(1, 1) * dx_ref + Rt(1, 2) * dy_ref + Rt(1, 3) * dz_ref;
            dwy = Rt(2, 1) * dx_ref + Rt(2, 2) * dy_ref + Rt(2, 3) * dz_ref;
            dwz = Rt(3, 1) * dx_ref + Rt(3, 2) * dy_ref + Rt(3, 3) * dz_ref;
        case 'cylindrical'
            theta = single(th0) + xp_s / single(f_pan);
            h = single(h0) + yp_s / single(f_pan);
            dwx = -sin(theta); dwy = -h; dwz = cos(theta);
        case {'spherical', 'equirectangular'}
            theta = single(th0) + xp_s / single(f_pan);
            phi = single(ph0) + yp_s / single(f_pan);
            cphi = cos(phi); sphi = sin(phi);
            dwx = -cphi .* sin(theta); dwy = -sphi; dwz = cphi .* cos(theta);
        case 'stereographic'
            a = single(u0) + xp_s / single(f_pan);
            b = single(v0) + yp_s / single(f_pan);
            r2 = a .* a + b .* b; denom = 1 + r2;
            dx_ref = 2 * a ./ denom;
            dy_ref = 2 * b ./ denom;
            dz_ref = (1 - r2) ./ denom;
            Rref = single(cameras(ref_idx).R);
            Rt = Rref.';
            dwx = Rt(1, 1) * dx_ref + Rt(1, 2) * dy_ref + Rt(1, 3) * dz_ref;
            dwy = Rt(2, 1) * dx_ref + Rt(2, 2) * dy_ref + Rt(2, 3) * dz_ref;
            dwz = Rt(3, 1) * dx_ref + Rt(3, 2) * dy_ref + Rt(3, 3) * dz_ref;
        otherwise
            error('Unknown mode "%s"', mode);
    end

end

function [u, v, front] = project_to_image(DWs, cam, useGPU)
    % PROJECT_TO_IMAGE Project world direction vectors into a camera image.
    %   [u, v, front] = project_to_image(DWs, cam, useGPU) projects the world
    %   directions DWs (HxWx3 or [M x N x 3]) into pixel coordinates using the
    %   camera intrinsics K and orientation R. Returns pixel arrays u, v and a
    %   logical mask 'front' for directions with positive z in camera space.

    arguments
        DWs (:, :, :) {mustBeNumeric}
        cam (1, 1) struct
        useGPU (1, 1) logical
    end

    if size(DWs, 3) ~= 3
        error('project_to_image:DWsShape', 'DWs must be an array with size(DWs,3) == 3.');
    end

    if ~isfield(cam, 'R') || ~isfield(cam, 'K')
        error('project_to_image:CamFields', 'cam struct must contain fields R and K.');
    end

    DW = reshape(DWs, [], 3); % [M x 3]

    % Compact version returning only what we need (u,v,front)
    R = single(cam.R);
    K = single(cam.K);

    dir_c = DW * R.'; % world -> camera
    cx_w = dir_c(:, 1);
    cy_w = dir_c(:, 2);
    cz_w = dir_c(:, 3);

    front = cz_w > 1e-6;
    u = K(1, 1) * (cx_w ./ cz_w) + K(1, 3);
    v = K(2, 2) * (cy_w ./ cz_w) + K(2, 3);

    if useGPU
        % nothing to do; already on GPU
    else
        % keep as single on CPU
        u = single(u); v = single(v); front = logical(front);
    end

end

function S = sample_linear(I, u, v, tileHW, useGPU)
    % SAMPLE_LINEAR Bilinear sampler for single-channel images with NaN extrapolation.
    %   S = sample_linear(I, u, v, tileHW, useGPU) samples the single-channel
    %   image/map I at floating-point pixel coordinates (u,v) using bilinear
    %   interpolation. Out-of-range values are set to NaN. The result is a
    %   column vector matching the number of query points.

    arguments
        I (:, :) {mustBeNumeric}
        u (:, 1) {mustBeNumeric}
        v (:, 1) {mustBeNumeric}
        tileHW (1, 2) {mustBeNumeric}
        useGPU (1, 1) logical
    end

    % Touch tileHW to avoid unused-arg warning when validating signature
    tileHW = tileHW; %#ok<NASGU>

    % Single-channel sampler (srcW)
    [hh, ww, ~] = size(I); %#ok<ASGLU>
    X = single(1:ww); Y = single(1:hh);

    if useGPU
        I = gpuArray(single(I));
        X = gpuArray(X); Y = gpuArray(Y);
    end

    % interp2 tolerates out-of-range with extrapval = NaN
    S = interp2(X, Y, single(I), u, v, 'linear', NaN);
    S = reshape(S, [], 1); % [HW x 1] or [K x 1]
end

function S = sample_linear_rgb(I, u, v, useGPU)
    % SAMPLE_LINEAR_RGB Bilinear sampler for RGB images with NaN extrapolation.
    %   S = sample_linear_rgb(I, u, v, useGPU) samples the RGB image I at
    %   floating-point pixel coordinates (u,v) and returns a K-by-3 array where
    %   each row contains the sampled RGB values. Out-of-range values are NaN.

    arguments
        I (:, :, :) {mustBeNumeric}
        u (:, 1) {mustBeNumeric}
        v (:, 1) {mustBeNumeric}
        useGPU (1, 1) logical
    end

    % RGB sampler returning [K x 3]
    [hh, ww, ~] = size(I); %#ok<ASGLU>
    X = single(1:ww); Y = single(1:hh);

    if useGPU
        I = gpuArray(single(I));
        X = gpuArray(X); Y = gpuArray(Y);
    else
        I = single(I);
    end

    K = numel(u);
    S = zeros(K, 3, 'like', u);
    S(:, 1) = interp2(X, Y, I(:, :, 1), u, v, 'linear', NaN);
    S(:, 2) = interp2(X, Y, I(:, :, 2), u, v, 'linear', NaN);
    S(:, 3) = interp2(X, Y, I(:, :, 3), u, v, 'linear', NaN);
end
