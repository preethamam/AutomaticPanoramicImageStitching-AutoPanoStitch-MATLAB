function gains = gainCompensationH(Iw, Ww, opts)
    %GAINCOMPENSATIONH Robust gain compensation from warped images.
    %   gains = gainCompensationWarped(Iw, Ww, opts)
    %   Iw/Ww: 1xN cell, each image HxWx3, weight HxW (same canvas).
    %   Returns N x 3 gains in [0.25, 4].
    %
    %   This version is CPU+GPU compatible and heavily vectorized:
    %   - If Iw{1} / Ww{1} are gpuArray, the overlap accumulation runs on GPU.
    %   - The small N x N linear systems are solved on CPU.
    %
    %   Inputs:
    %   - Iw: 1xN or Nx1 cell array of RGB images (H x W x 3) already warped to
    %         a common canvas (can be gpuArray of singles).
    %   - Ww: 1xN or Nx1 cell array of single-channel weight/coverage maps (H x W).
    %   - opts: options struct. Fields used include: overlapDownsample,
    %           minOverlapSamples, sigmaN, sigmag, lambdaDiag, anchorRef,
    %           refIdx, tileSize, gpuMemFrac.
    %
    %   Outputs:
    %   - gains: N-by-3 single matrix of per-image RGB gains clamped to [0.25, 4].

    N = numel(Iw);
    gains = ones(N, 3, 'single');
    if N <= 1, return; end

    % ---- defaults ----
    if ~isfield(opts, 'overlapDownsample'), opts.overlapDownsample = 4; end
    if ~isfield(opts, 'minOverlapSamples'), opts.minOverlapSamples = 100; end
    if ~isfield(opts, 'sigmaN'), opts.sigmaN = 10.0; end
    if ~isfield(opts, 'sigmag'), opts.sigmag = 10.0; end
    if ~isfield(opts, 'lambdaDiag'), opts.lambdaDiag = 1e-8; end
    if ~isfield(opts, 'anchorRef'), opts.anchorRef = false; end
    if ~isfield(opts, 'refIdx'), opts.refIdx = 1; end
    if ~isfield(opts, 'tileSize'), opts.tileSize = []; end % Empty = auto
    if ~isfield(opts, 'gpuMemFrac'), opts.gpuMemFrac = 0.4; end % Safety factor

    % ---- GPU setup ----
    try
        useGPU = parallel.gpu.GPUDevice.isAvailable && gpuDeviceCount > 0;
    catch
        useGPU = false;
    end

    % ---- DOWNSAMPLE ONCE (on CPU to save memory) ----
    ds = opts.overlapDownsample;
    IwDs = cell(N, 1);
    WwDs = cell(N, 1);

    for k = 1:N
        IwDs{k} = Iw{k}(1:ds:end, 1:ds:end, :);
        WwDs{k} = Ww{k}(1:ds:end, 1:ds:end);
    end

    % Get downsampled canvas size
    [Hds, Wds, ~] = size(IwDs{1});

    % Auto-tiler to select the optimal tile size based on GPU memory
    opts = autoTiler(opts, useGPU, N, Hds, Wds);

    % Clear original full-res data
    clear Iw Ww;

    % ---- Preallocate GLOBAL overlap statistics ----
    Nij = zeros(N, N, 'double');
    sumCi = zeros(N, N, 3, 'double');
    sumCj = zeros(N, N, 3, 'double');

    % ---- Define tiles on downsampled canvas ----
    tileH = opts.tileSize(1);
    tileW = opts.tileSize(2);

    rowStarts = 1:tileH:Hds;
    colStarts = 1:tileW:Wds;
    numTilesY = length(rowStarts);
    numTilesX = length(colStarts);

    % ---- Process each tile ----
    for ty = 1:numTilesY

        for tx = 1:numTilesX
            % Tile bounds
            r1 = rowStarts(ty);
            r2 = min(r1 + tileH - 1, Hds);
            c1 = colStarts(tx);
            c2 = min(c1 + tileW - 1, Wds);

            % Extract tiles for all images
            IwTile = cell(N, 1);
            WwTile = cell(N, 1);

            for k = 1:N
                IwTile{k} = IwDs{k}(r1:r2, c1:c2, :);
                WwTile{k} = WwDs{k}(r1:r2, c1:c2);

                % Move to GPU
                if useGPU
                    IwTile{k} = gpuArray(single(IwTile{k}));
                    WwTile{k} = gpuArray(single(WwTile{k}));
                end

            end

            % Concatenate (this is now small)
            Iw4 = cat(4, IwTile{:});
            Ww3 = cat(3, WwTile{:});

            [Htile, Wtile, ~, ~] = size(Iw4);
            P = Htile * Wtile;

            IPix = reshape(Iw4, [P, 3, N]);
            WPix = reshape(Ww3, [P, N]);

            % Valid mask
            valid = (WPix > 0) & all(isfinite(IPix), 2);

            % ---- Accumulate statistics ----
            for i = 1:N - 1
                jRange = (i + 1):N;
                nj = length(jRange);

                vi = valid(:, i);
                vj = valid(:, jRange);
                overlap = vi & vj;

                counts = sum(overlap, 1, 'double');
                Nij(i, jRange) = Nij(i, jRange) + gather(counts);

                Ci = IPix(:, :, i);

                for ch = 1:3
                    cIch = Ci(:, ch);
                    sumCi(i, jRange, ch) = sumCi(i, jRange, ch) + ...
                        gather(sum(cIch .* double(overlap), 1, 'double'));

                    cJch = squeeze(IPix(:, ch, jRange));

                    if nj == 1
                        cJch = reshape(cJch, [P, 1]);
                    end

                    sumCj(i, jRange, ch) = sumCj(i, jRange, ch) + ...
                        gather(sum(cJch .* double(overlap), 1, 'double'));
                end

            end

        end

    end

    % ---- Rest is identical ----
    mask = triu(Nij >= opts.minOverlapSamples, 1);
    [iIdx, jIdx] = find(mask);
    edges = [iIdx, jIdx];

    if isempty(edges)
        return;
    end

    A = zeros(N, N, 3, 'double');
    b = zeros(N, 1, 'double');

    sN2 = opts.sigmaN ^ 2;
    sg2 = opts.sigmag ^ 2;

    for e = 1:size(edges, 1)
        i = edges(e, 1);
        j = edges(e, 2);

        Kij = Nij(i, j);
        Ibari = reshape(sumCi(i, j, :), [1 3]) / Kij;
        Ibarj = reshape(sumCj(i, j, :), [1 3]) / Kij;

        wN = double(Kij) / sN2;

        for ch = 1:3
            aii = wN * (Ibari(ch) ^ 2);
            ajj = wN * (Ibarj(ch) ^ 2);
            aij = -wN * (Ibari(ch) * Ibarj(ch));

            A(i, i, ch) = A(i, i, ch) + aii;
            A(j, j, ch) = A(j, j, ch) + ajj;
            A(i, j, ch) = A(i, j, ch) + aij;
            A(j, i, ch) = A(j, i, ch) + aij;
        end

    end

    participating = unique([edges(:, 1); edges(:, 2)]);
    wG = 1.0 / sg2;

    for i = participating'

        for ch = 1:3
            A(i, i, ch) = A(i, i, ch) + wG;
        end

        b(i) = wG;
    end

    if opts.anchorRef
        pin = max(1, min(N, opts.refIdx));

        for ch = 1:3
            A(:, :, ch) = A(:, :, ch) + opts.lambdaDiag * eye(N);
            A(pin, :, ch) = 0;
            A(:, pin, ch) = 0;
            A(pin, pin, ch) = 1e6;
        end

        b(pin) = 1e6;
    else

        for ch = 1:3
            A(:, :, ch) = A(:, :, ch) + opts.lambdaDiag * eye(N);
        end

    end

    for ch = 1:3
        x = A(:, :, ch) \ b;
        gains(:, ch) = single(max(0.25, min(4.0, x)));
    end

end

function opts = autoTiler(opts, useGPU, N, Hds, Wds)
    %AUTOTILER Choose a tile size based on available memory for tiled processing.
    %   opts = autoTiler(opts, useGPU, N, Hds, Wds) inspects available memory
    %   (GPU or CPU fallback) and computes a reasonable square `opts.tileSize`
    %   value when it is empty. The decision aims to fit temporary per-tile
    %   arrays for N images into memory with a conservative safety margin.
    %
    %   Inputs:
    %   - opts: options struct which may contain `tileSize` (empty to auto-select)
    %   - useGPU: logical flag indicating GPU use
    %   - N: number of images
    %   - Hds, Wds: downsampled canvas height and width
    %
    %   Outputs:
    %   - opts: the same options struct with `opts.tileSize` set to [tileH tileW]
    %           if it was previously empty.

    if isempty(opts.tileSize)
        % Get available memory
        if useGPU
            g = gpuDevice;
            avail = opts.gpuMemFrac * double(g.AvailableMemory); % bytes
        else
            % CPU fallback: conservative estimate
            avail = 4e9; % 4 GB
        end

        % Memory model for gain compensation tile processing:
        % For each tile of size [tileH, tileW] with N images:
        %
        % GPU memory needed (worst case, all arrays on GPU simultaneously):
        % 1. IwTile{k}: N × tileH × tileW × 3 × 4 bytes (RGB singles)
        % 2. WwTile{k}: N × tileH × tileW × 4 bytes (weight singles)
        % 3. Iw4 (concatenated): tileH × tileW × 3 × N × 4 bytes
        % 4. Ww3 (concatenated): tileH × tileW × N × 4 bytes
        % 5. Ipix (reshaped): P × 3 × N × 4 bytes (P = tileH*tileW)
        % 6. Wpix (reshaped): P × N × 4 bytes
        % 7. valid mask: P × N × 1 byte (logical)
        % 8. overlap mask: P × N × 1 byte
        % 9. Temporary arrays during computation
        %
        % Conservative estimate (assuming overlap, some reuse):
        % Per pixel: N × [(3+1)×2 + 3 + 1 + 0.5 + 0.5] × 4 bytes
        %          = N × 12.5 × 4 ≈ 50N bytes per pixel

        bytesPerPixel = 50 * N * 4; % Conservative factor

        % Add safety margin for GPU overhead, temp arrays
        safetyFactor = 1.5;
        bytesPerPixel = bytesPerPixel * safetyFactor;

        % Calculate pixel budget
        pixBudget = max(1, floor(avail / bytesPerPixel));

        % Square tile assumption
        side = floor(sqrt(pixBudget));

        % Clamp to reasonable bounds
        minTile = 128; % Don't go too small (overhead dominates)
        maxTile = 2048; % Don't go too large (defeats purpose)
        side = max(minTile, min(side, maxTile));

        % Don't exceed canvas dimensions
        side = min([side, Hds, Wds]);

        opts.tileSize = [side side];
    end

end
