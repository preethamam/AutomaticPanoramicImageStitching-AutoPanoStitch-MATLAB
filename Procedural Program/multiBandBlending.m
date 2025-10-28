function F = multiBandBlending(Ci, Wi, levels, onGPU, sigma)
    if nargin < 5 || isempty(sigma), sigma = 1.0; end
    K = numel(Ci); assert(K == numel(Wi) && K >= 1);

    % Ensure types/devices
    for k = 1:K
        if ~isa(Ci{k}, 'single'), Ci{k} = single(Ci{k}); end
        if ~isa(Wi{k}, 'single'), Wi{k} = single(Wi{k}); end
    end

    inputsOnGPU = isa(Ci{1}, 'gpuArray') || isa(Wi{1}, 'gpuArray');

    if onGPU && ~inputsOnGPU
        for k = 1:K, Ci{k} = gpuArray(Ci{k}); Wi{k} = gpuArray(Wi{k}); end
    elseif ~onGPU && inputsOnGPU
        for k = 1:K, Ci{k} = gather(Ci{k}); Wi{k} = gather(Wi{k}); end
    end

    % Ensure each Ci{k} and Wi{k} have identical HxW at entry
    [h0, w0, ~] = size(Ci{1});

    for k = 1:K
        [hc, wc, ~] = size(Ci{k});
        [hw, ww] = size(Wi{k});
        assert(hc == h0 && wc == w0, 'Ci{%d} size differs from Ci{1}.', k);
        assert(hw == h0 && ww == w0, 'Wi{%d} size differs from Ci{1}.', k);
    end

    % Pre-normalize weights at full-res (Σ_k W = 1 where covered)
    Wsum = zeros(size(Wi{1}), 'like', Wi{1});

    for k = 1:K
        Wi{k} = max(0, Wi{k});
        Wsum = Wsum + Wi{k};
    end

    epsW = 1e-8; mask_nz = (Wsum > epsW);

    for k = 1:K
        Wn = zeros(size(Wi{k}), 'like', Wi{k});
        Wn(mask_nz) = Wi{k}(mask_nz) ./ Wsum(mask_nz);
        Wi{k} = Wn;
    end

    % 3-channel internal
    [h, w, C0] = size(Ci{1});

    if C0 == 1
        for k = 1:K, Ci{k} = repmat(Ci{k}, [1 1 3]); end
    end

    % --- derive per-level sizes (base -> coarsest) ---
    maxLevels = floor(log2(min(h0, w0)));
    levels = max(1, min(levels, maxLevels));

    % precompute sizes for each level
    levH = zeros(levels, 1, 'uint32');
    levW = zeros(levels, 1, 'uint32');
    levH(1) = h0; levW(1) = w0;

    for l = 2:levels
        levH(l) = max(1, floor(double(levH(l - 1)) / 2));
        levW(l) = max(1, floor(double(levW(l - 1)) / 2));
    end

    % allocate numerator pyramid with per-level sizes
    NumP = cell(levels, 1);

    for l = 1:levels
        NumP{l} = zeros(double(levH(l)), double(levW(l)), 3, 'like', Ci{1});
    end

    % ---- STREAMED ACCUMULATION (level-by-level, per image) ----
    for k = 1:K
        Gc = Ci{k}; % color (HxWx3)
        Gw = Wi{k}; % weight (HxW)

        % descend to levels-1, accumulating Laplacian*weight at each level size
        for l = 1:levels - 1
            hL = size(Gc, 1); wL = size(Gc, 2);
            % target size for next level
            newH = double(levH(l + 1));
            newW = double(levW(l + 1));

            % blur, then downsample BOTH to exactly [newH newW]
            Gc_blur = imgaussfilt(Gc, sigma, 'Padding', 'replicate');
            Dc = imresize(Gc_blur, [newH newW], 'bilinear');

            Gw_blur = imgaussfilt(Gw, sigma, 'Padding', 'replicate');
            Dw = imresize(Gw_blur, [newH newW], 'bilinear');

            % upsample back to current level size (explicit!) and form Laplacian
            Uc = imresize(Dc, [hL wL], 'bilinear');
            Lc = Gc - Uc; % (hL x wL x 3)

            % accumulate at current level size
            % NumP{l} is guaranteed allocated as [hL x wL x 3]
            NumP{l} = NumP{l} + bsxfun(@times, Lc, Gw); % Gw broadcasts to 3 ch

            % step down
            Gc = Dc; % now size = [newH newW 3]
            Gw = Dw; % now size = [newH newW]
        end

        % coarsest level (Gaussian*weight) — enforce exact match just in case
        if size(Gw, 1) ~= levH(levels) || size(Gw, 2) ~= levW(levels)
            Gw = imresize(Gw, [double(levH(levels)) double(levW(levels))], 'bilinear');
        end

        if size(Gc, 1) ~= levH(levels) || size(Gc, 2) ~= levW(levels)
            Gc = imresize(Gc, [double(levH(levels)) double(levW(levels))], 'bilinear');
        end

        NumP{levels} = NumP{levels} + bsxfun(@times, Gc, Gw);
    end

    % ---- collapse numerator pyramid back to full res ----
    F3 = NumP{levels};

    for l = levels - 1:-1:1
        F3 = imresize(F3, [double(levH(l)) double(levW(l))], 'bilinear') + NumP{l};
    end

    % restore channels
    if C0 == 1, F = F3(:, :, 1); else, F = F3; end
    F = min(1, max(0, F));

end
