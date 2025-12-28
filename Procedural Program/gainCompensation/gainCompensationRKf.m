function gains = gainCompensationRKf(images, cameras, mode, refIdx, opts, ...
        H, W, u0, v0, th0, h0, ph0, srcW)
    % GAINCOMPENSATIONRKF Estimate per-image RGB gains using Brown–Lowe (2007) Eq. (29).
    %   gains = gainCompensationRKf(images, cameras, mode, refIdx, opts, ...
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
    %   - refIdx   : reference camera index (positive integer).
    %   - opts      : struct of options. Fields used include:
    %                 overlapStride, minOverlapSamples, sigmaN, sigmag,
    %                 lambdaDiag, parforTiles, anchorRef, tile, useGPU, fPan.
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
    %   - We accumulate only per-edge statistics required by Eq.(29): counts Nij and
    %     channel sums over overlaps; means are recovered as sums / Nij.
    %   - Tiling is done over a SUBSAMPLED grid to reduce memory; we never build full DWs.
    %   - GPU path keeps arrays on device and reduces there; final A/b assembly happens on CPU.
    %
    %   See also svd, gpuArray, gather, interp2

    arguments
        images cell
        cameras struct
        mode {mustBeTextScalar}
        refIdx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
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
        error('gainCompensationRKf:SrcWSize', 'srcW must be a vector cell array with the same length as images.');
    end

    if ~(isstruct(cameras) && numel(cameras) == N)
        error('gainCompensationRKf:CamerasSize', 'cameras must be a struct array with one entry per image.');
    end

    for i = 1:N

        if ~isfield(cameras(i), 'R') || ~isfield(cameras(i), 'K')
            error('gainCompensationRKf:CamFields', 'cameras(%d) must contain fields R and K.', i);
        end

        Ri = cameras(i).R; Ki = cameras(i).K;

        if ~(isnumeric(Ri) && isequal(size(Ri), [3, 3]) && all(isfinite(Ri(:))))
            error('gainCompensationRKf:RShape', 'cameras(%d).R must be a 3x3 finite numeric matrix.', i);
        end

        if ~(isnumeric(Ki) && isequal(size(Ki), [3, 3]) && all(isfinite(Ki(:))))
            error('gainCompensationRKf:KShape', 'cameras(%d).K must be a 3x3 finite numeric matrix.', i);
        end

    end

    N = numel(images);
    gains = ones(N, 3, 'single');

    % -------- Defaults --------
    if ~isfield(opts, 'overlapStride'), opts.overlapStride = 5; end
    if ~isfield(opts, 'minOverlapSamples'), opts.minOverlapSamples = 50; end
    if ~isfield(opts, 'sigmaN'), opts.sigmaN = 10.0; end
    if ~isfield(opts, 'sigmag'), opts.sigmag = 0.1; end
    if ~isfield(opts, 'lambdaDiag'), opts.lambdaDiag = 1e-8; end
    if ~isfield(opts, 'parforTiles'), opts.parforTiles = false; end
    if ~isfield(opts, 'anchorRef'), opts.anchorRef = false; end
    if ~isfield(opts, 'tile'), opts.tile = [512 512]; end

    % Auto GPU?
    if ~isfield(opts, 'useGPU')
        opts.useGPU = (gpuDeviceCount > 0);
    end

    stride = max(1, opts.overlapStride);
    minOv = max(1, opts.minOverlapSamples);
    sN2 = (opts.sigmaN) ^ 2;
    sg2 = (opts.sigmag) ^ 2;

    % -------- Subsampled grid size --------
    xpAll = single(1:stride:W);
    ypAll = single(1:stride:H);
    Hs = numel(ypAll);
    Ws = numel(xpAll);

    tileH = min(opts.tile(1), Hs);
    tileW = min(opts.tile(2), Ws);

    % -------- Accumulators (host/CPU) --------
    % For each (i,j), i<j:
    Nij = zeros(N, N, 'double'); % counts
    sumCiij = zeros(N, N, 3, 'double'); % sum of Ci per channel
    sumCjij = zeros(N, N, 3, 'double'); % sum of Cj per channel

    % -------- Tile enumeration over subsampled grid --------
    xTiles = 1:tileW:Ws;
    yTiles = 1:tileH:Hs;

    % To optionally run tiles in parallel:
    tileJobs = [];

    for ty = yTiles

        for tx = xTiles
            tileJobs(end + 1, :) = [ty, tx]; %#ok<AGROW>
        end

    end

    % -------- Run tiles (optionally parfor) --------
    if opts.parforTiles
        % PARFOR over tiles (good when GPU not used or when GPU is busy elsewhere)
        parfor t = 1:size(tileJobs, 1)
            [NijL, sCiL, sCjL] = processOneTile(tileJobs(t, 1), tileJobs(t, 2), opts, mode, refIdx, cameras, ...
                u0, v0, th0, h0, ph0, srcW, images, tileH, tileW, ...
                Hs, Ws, xpAll, ypAll, N);
            % Accumulate into master (parfor requires reduction variables)
            Nij = Nij + NijL;
            sumCiij = sumCiij + sCiL;
            sumCjij = sumCjij + sCjL;
        end

    else

        for t = 1:size(tileJobs, 1)
            [NijL, sCiL, sCjL] = processOneTile(tileJobs(t, 1), tileJobs(t, 2), opts, mode, refIdx, cameras, ...
                u0, v0, th0, h0, ph0, srcW, images, tileH, tileW, ...
                Hs, Ws, xpAll, ypAll, N);
            Nij = Nij + NijL;
            sumCiij = sumCiij + sCiL;
            sumCjij = sumCjij + sCjL;
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
        Ibarij = reshape(sumCiij(i, j, :), [1 3]) / Kij; % mean Ci on (i,j)
        Ibarji = reshape(sumCjij(i, j, :), [1 3]) / Kij; % mean Cj on (i,j)

        wN = double(Kij) / sN2; % data weight
        wG = double(Kij) / sg2; % prior weight (paper multiplies by Nij)

        for ch = 1:3
            aii = wN * (Ibarij(ch) * Ibarij(ch)) + wG;
            ajj = wN * (Ibarji(ch) * Ibarji(ch)) + wG;
            aij =- wN * (Ibarij(ch) * Ibarji(ch));

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
    if opts.anchorRef
        pin = max(1, min(N, refIdx));

        for ch = 1:3
            A(:, :, ch) = A(:, :, ch) + opts.lambdaDiag * eye(N);
            A(pin, :, ch) = 0; A(:, pin, ch) = 0; A(pin, pin, ch) = 1e6;
        end

        b(pin) = 1e6; % drives g_ref -> 1
    else

        for ch = 1:3
            A(:, :, ch) = A(:, :, ch) + opts.lambdaDiag * eye(N);
        end

    end

    % -------- Solve per channel --------
    for ch = 1:3
        x = A(:, :, ch) \ b;
        gains(:, ch) = single(max(0.25, min(4.0, x)));
    end

end % gainCompensation

% -------- Per-tile worker (nested for clarity) --------
function [Nijloc, sumCiloc, sumCjloc] = processOneTile(ty, tx, opts, mode, refIdx, cameras, ...
        u0, v0, th0, h0, ph0, weights, imgs, tileH, tileW, ...
        Hs, Ws, xpAll, ypAll, N)
    % PROCESSONETILE Accumulate overlap statistics for a single subsampled tile.
    %   [Nijloc, sumCiloc, sumCjloc] = processOneTile(ty, tx, opts, mode, refIdx, cameras,
    %       u0, v0, th0, h0, ph0, weights, imgs, tileH, tileW, Hs, Ws, xpAll, ypAll, N)
    %   computes, for the tile starting at (ty, tx) on the subsampled grid, the
    %   per-pair overlap counts and per-channel color sums needed for gain solving.
    %
    %   Inputs:
    %   - ty:       starting y-index (subsampled grid) for this tile
    %   - tx:       starting x-index (subsampled grid) for this tile
    %   - opts:     options struct (fields used: fPan, useGPU, ...)
    %   - mode:     projection mode string
    %   - refIdx:   reference camera index
    %   - cameras:  camera struct array
    %   - u0,v0:    planar/stereographic center offsets
    %   - th0,h0:   cylindrical parameters (theta base, height offset)
    %   - ph0:      spherical/equirectangular latitude offset
    %   - weights:  cell array of single-channel coverage/weight maps (srcW)
    %   - imgs:     cell array of RGB images
    %   - tileH,tileW: tile size in subsampled pixels
    %   - Hs,Ws:    subsampled panorama grid height and width
    %   - xpAll,ypAll: subsampled grid coordinate vectors
    %   - N:        number of images
    %
    %   Outputs:
    %   - Nijloc:   N-by-N double matrix of per-pair overlap counts
    %   - sumCiloc: N-by-N-by-3 double matrix of summed Ci values over overlaps
    %   - sumCjloc: N-by-N-by-3 double matrix of summed Cj values over overlaps

    arguments
        ty (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        tx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        opts (1, 1) struct
        mode {mustBeTextScalar}
        refIdx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
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
        xpAll (:, 1) {mustBeNumeric, mustBeFinite}
        ypAll (:, 1) {mustBeNumeric, mustBeFinite}
        N (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    % ranges in subsampled-index space
    y2 = min(ty + tileH - 1, Hs);
    x2 = min(tx + tileW - 1, Ws);
    yy = ypAll(ty:y2); xx = xpAll(tx:x2);
    [xpS, ypS] = meshgrid(xx, yy); % small tile grid (subsampled)

    % Build world directions for this tile (on chosen device)
    [DWx, DWy, DWz] = panoDirsGridTile(xpS, ypS, mode, refIdx, cameras, ...
        opts.fPan, u0, v0, th0, h0, ph0, opts.useGPU);

    DWs = cat(3, DWx, DWy, DWz);

    % Pre-alloc per-image projections/masks for this tile (device arrays)
    uL = cell(N, 1); vL = cell(N, 1); cov = cell(N, 1);

    % Project into each image and compute coverage from srcW
    for i = 1:N
        [ui, vi, fronti] = projectToImage(DWs, cameras(i), opts.useGPU);
        % Sample srcW{i} (single-channel) at (ui,vi). Returns [HW x 1]
        Wi = sampleLinear(weights{i}, ui, vi, size(xpS), opts.useGPU);

        Mi = isfinite(Wi) & (Wi > 0) & fronti & isfinite(ui) & isfinite(vi);
        uL{i} = ui; vL{i} = vi; cov{i} = Mi;
    end

    % Accumulators local to the tile (host-friendly at end)
    Nijloc = zeros(N, N, 'double');
    sumCiloc = zeros(N, N, 3, 'double');
    sumCjloc = zeros(N, N, 3, 'double');

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
            uis = ui(idx); vis = vi(idx);
            ujs = uL{j}(idx); vjs = vL{j}(idx);

            Ci = sampleLinearRGB(single(imgs{i}), uis, vis, opts.useGPU); % [K x 3]
            Cj = sampleLinearRGB(single(imgs{j}), ujs, vjs, opts.useGPU);

            % Filter any non-finites (rare; NaNs from borders)
            good = all(isfinite(Ci), 2) & all(isfinite(Cj), 2);
            if ~any(good), continue; end

            Ci = Ci(good, :); Cj = Cj(good, :);
            k = size(Ci, 1);

            % Reduce on device, then gather
            sCi = double(gather(sum(Ci, 1)));
            sCj = double(gather(sum(Cj, 1)));

            Nijloc(i, j) = Nijloc(i, j) + k;
            sumCiloc(i, j, :) = sumCiloc(i, j, :) + reshape(sCi, [1, 1, 3]);
            sumCjloc(i, j, :) = sumCjloc(i, j, :) + reshape(sCj, [1, 1, 3]);
        end

    end

end

function [dwx, dwy, dwz] = panoDirsGridTile(xps, yps, mode, ...
        refIdx, cameras, fPan, u0, v0, ...
        th0, h0, ph0, useGPU)
    % PANODIRSGRIDTILE Compute world direction vectors for a pano grid tile.
    %   [dwx, dwy, dwz] = panoDirsGridTile(xps, yps, mode, refIdx, cameras,
    %       fPan, u0, v0, th0, h0, ph0, useGPU) returns per-pixel world direction
    %   vectors for the given projection mode, relative to a reference camera.
    %
    %   Returns three arrays of the same size as xps/yps containing the x/y/z
    %   components of the direction vectors.
    %
    %   Inputs:
    %   - xps, yps: matrices of subsampled panorama grid coordinates for the tile
    %   - mode:     projection mode string
    %   - refIdx:   reference camera index
    %   - cameras:  camera struct array
    %   - fPan:     focal/scale parameter for panorama projection
    %   - u0,v0:    planar/stereographic center offsets
    %   - th0,h0:   cylindrical parameters (theta base, height offset)
    %   - ph0:      spherical/equirectangular latitude offset
    %   - useGPU:   logical flag to perform computations on GPU
    %
    %   Outputs:
    %   - dwx,dwy,dwz: arrays matching xps/yps with world direction components (x,y,z)

    arguments
        xps (:, :) {mustBeNumeric, mustBeFinite}
        yps (:, :) {mustBeNumeric, mustBeFinite}
        mode {mustBeTextScalar}
        refIdx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        cameras struct
        fPan (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        u0
        v0
        th0
        h0
        ph0
        useGPU (1, 1) logical
    end

    if useGPU
        xps = gpuArray(xps); yps = gpuArray(yps);
    end

    switch lower(mode)
        case {'planar', 'perspective'}
            u = single(u0) + xps / single(fPan);
            v = single(v0) + yps / single(fPan);
            dxRef = u; dyRef = v; dzRef = ones(size(u), 'like', u);
            Rref = single(cameras(refIdx).R);
            Rt = Rref.';
            dwx = Rt(1, 1) * dxRef + Rt(1, 2) * dyRef + Rt(1, 3) * dzRef;
            dwy = Rt(2, 1) * dxRef + Rt(2, 2) * dyRef + Rt(2, 3) * dzRef;
            dwz = Rt(3, 1) * dxRef + Rt(3, 2) * dyRef + Rt(3, 3) * dzRef;
        case 'cylindrical'
            theta = single(th0) + xps / single(fPan);
            h = single(h0) + yps / single(fPan);
            dwx = sin(theta); dwy = h; dwz = cos(theta);
        case {'spherical', 'equirectangular'}
            theta = single(th0) + xps / single(fPan);
            phi = single(ph0) + yps / single(fPan);
            cphi = cos(phi); sphi = sin(phi);
            dwx = cphi .* sin(theta); dwy = sphi; dwz = cphi .* cos(theta);
        case 'stereographic'
            a = single(u0) + xps / single(fPan);
            b = single(v0) + yps / single(fPan);
            r2 = a .* a + b .* b; denom = 1 + r2;
            dxRef = 2 * a ./ denom;
            dyRef = 2 * b ./ denom;
            dzRef = (1 - r2) ./ denom;
            Rref = single(cameras(refIdx).R);
            Rt = Rref.';
            dwx = Rt(1, 1) * dxRef + Rt(1, 2) * dyRef + Rt(1, 3) * dzRef;
            dwy = Rt(2, 1) * dxRef + Rt(2, 2) * dyRef + Rt(2, 3) * dzRef;
            dwz = Rt(3, 1) * dxRef + Rt(3, 2) * dyRef + Rt(3, 3) * dzRef;
        otherwise
            error('Unknown mode "%s"', mode);
    end

end

function [u, v, front] = projectToImage(DWs, cam, useGPU)
    % PROJECTTOIMAGE Project world direction vectors into a camera image.
    %   [u, v, front] = projectToImage(DWs, cam, useGPU) projects the world
    %   directions DWs (HxWx3 or [M x N x 3]) into pixel coordinates using the
    %   camera intrinsics K and orientation R. Returns pixel arrays u, v and a
    %   logical mask 'front' for directions with positive z in camera space.
    %
    %   Inputs:
    %   - DWs:  array of world direction vectors (HxWx3 or MxNx3)
    %   - cam:  camera struct with fields R (3x3) and K (3x3)
    %   - useGPU: logical flag indicating GPU arrays expected
    %
    %   Outputs:
    %   - u,v:  column vectors of projected pixel coordinates (u = x, v = y)
    %   - front: logical mask indicating directions in front of the camera (z>0)

    arguments
        DWs (:, :, :) {mustBeNumeric}
        cam (1, 1) struct
        useGPU (1, 1) logical
    end

    if size(DWs, 3) ~= 3
        error('projectToImage:DWsShape', 'DWs must be an array with size(DWs,3) == 3.');
    end

    if ~isfield(cam, 'R') || ~isfield(cam, 'K')
        error('projectToImage:CamFields', 'cam struct must contain fields R and K.');
    end

    DW = reshape(DWs, [], 3); % [M x 3]

    % Compact version returning only what we need (u,v,front)
    R = single(cam.R);
    K = single(cam.K);

    dirC = DW * R.'; % world -> camera
    cxW = dirC(:, 1);
    cyW = dirC(:, 2);
    czW = dirC(:, 3);

    front = czW > 1e-6;
    u = K(1, 1) * (cxW ./ czW) + K(1, 3);
    v = K(2, 2) * (cyW ./ czW) + K(2, 3);

    if useGPU
        % nothing to do; already on GPU
    else
        % keep as single on CPU
        u = single(u); v = single(v); front = logical(front);
    end

end

function S = sampleLinear(I, u, v, tileHW, useGPU)
    % SAMPLELINEAR Bilinear sampler for single-channel images with NaN extrapolation.
    %   S = sampleLinear(I, u, v, tileHW, useGPU) samples the single-channel
    %   image/map I at floating-point pixel coordinates (u,v) using bilinear
    %   interpolation. Out-of-range values are set to NaN. The result is a
    %   column vector matching the number of query points.
    %
    %   Inputs:
    %   - I:      single-channel image or map (HxW)
    %   - u,v:    column vectors of floating-point sample x,y coordinates
    %   - tileHW: size of the tile as [height, width] (unused except for signature)
    %   - useGPU: logical flag indicating whether to use GPU arrays
    %
    %   Outputs:
    %   - S:      column vector of sampled values (NaN for out-of-range)

    arguments
        I (:, :) {mustBeNumeric}
        u (:, 1) {mustBeNumeric}
        v (:, 1) {mustBeNumeric}
        tileHW (1, 2) {mustBeNumeric}
        useGPU (1, 1) logical
    end

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

function S = sampleLinearRGB(I, u, v, useGPU)
    % SAMPLELINEARRGB Bilinear sampler for RGB images with NaN extrapolation.
    %   S = sampleLinearRGB(I, u, v, useGPU) samples the RGB image I at
    %   floating-point pixel coordinates (u,v) and returns a K-by-3 array where
    %   each row contains the sampled RGB values. Out-of-range values are NaN.
    %
    %   Inputs:
    %   - I:      RGB image (HxWx3)
    %   - u,v:    column vectors of floating-point sample x,y coordinates
    %   - useGPU: logical flag indicating whether to use GPU arrays
    %
    %   Outputs:
    %   - S:      K-by-3 array of sampled RGB values (NaN for out-of-range)

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
