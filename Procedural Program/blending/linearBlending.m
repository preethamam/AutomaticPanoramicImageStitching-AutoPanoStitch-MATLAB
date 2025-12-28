function imageBlended = linearBlending(warpedImages, warpedWeights)
    % LINEARBLENDING Blend multiple warped images by linear weight averaging.
    %
    %   imageBlended = linearBlending(warpedImages, warpedWeights)
    %
    % Description:
    %   Performs per-pixel linear blending of N warped images using the
    %   corresponding per-pixel weights. Supports CPU arrays and GPU arrays
    %   (gpuArray). Accumulation is performed in single precision; the final
    %   result is cast back to the class and device of the first input image.
    %
    % Inputs:
    %   warpedImages  - 1xN cell array of images. Each image must be HxWxC
    %                   (C = 1 or 3). Allowed classes: integer or floating
    %                   point (e.g. uint8, single, double). Images may be
    %                   located on the GPU as a `gpuArray`.
    %   warpedWeights - 1xN cell array of weight maps. Each weight map must
    %                   be HxW (or HxWx1) and is expected to be in [0,1].
    %                   Can be on CPU or GPU and may be single/double.
    %
    % Output:
    %   imageBlended  - HxWxC blended image. Returned on the same device
    %                   (CPU/GPU) and cast to the same numeric class as the
    %                   first entry of `warpedImages`.
    %
    % Notes:
    %   - For integer input types the result is rounded and clamped to the
    %     valid integer range for that type.
    %   - When the weights sum to zero at a pixel, a small epsilon prevents
    %     division-by-zero and the blended value will be zero at that pixel.
    %   - Implicit expansion is used to apply HxW weights across channels.
    %
    % Example:
    %   I1 = im2single(imread('img1.jpg'));
    %   I2 = im2single(imread('img2.jpg'));
    %   W1 = createWeightMap(I1); W2 = createWeightMap(I2);
    %   out = linearBlending({I1, I2}, {W1, W2});
    %
    % See also: gpuArray

    arguments
        warpedImages cell
        warpedWeights cell
    end

    N = numel(warpedImages);

    if N == 0
        imageBlended = [];
        return;
    end

    baseImg = warpedImages{1};
    inClass = class(baseImg); % e.g.,'uint8', 'single', ...
        isGPU = isa(baseImg, 'gpuArray');

    % Ensure base image is at least 3D (H x W x C)
    if ndims(baseImg) == 2
        baseImg = reshape(baseImg, size(baseImg, 1), size(baseImg, 2), 1);
    end

    [H, W, C] = size(baseImg);

    % Accumulate in single precision on same device (CPU/GPU)
    accNum = zeros(H, W, C, 'like', single(baseImg)); % numerator
    accDen = zeros(H, W, 'like', single(baseImg)); % denominator (2D, shared across channels)

    for k = 1:N
        Ik = warpedImages{k};
        Wk = warpedWeights{k};

        % Move to GPU if base is GPU
        if isGPU
            if ~isa(Ik, 'gpuArray'), Ik = gpuArray(Ik); end
            if ~isa(Wk, 'gpuArray'), Wk = gpuArray(Wk); end
        end

        % Force shapes: HxWxC for image, HxW for weights
        if ndims(Ik) == 2
            Ik = reshape(Ik, size(Ik, 1), size(Ik, 2), 1);
        end

        Ik = single(Ik);
        Wk = single(Wk);

        if ndims(Wk) == 3
            Wk = Wk(:, :, 1); % reduce to HxW if someone passed HxWx1
        end

        % Accumulate numerator and denominator
        % Implicit expansion: Wk [H x W] -> [H x W x C]
        accNum = accNum + Ik .* Wk;
        accDen = accDen + Wk;
    end

    % Avoid division by zero: outside support accNum is zero anyway
    tiny = eps('single');
    accDenSafe = max(accDen, tiny); % [H x W]

    % Broadcast denominator over channels
    blended = accNum ./ accDenSafe; % [H x W x C], single

    % ---- Cast back to original class (preserve device) ----
    switch inClass
        case {'uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32'}
            % Clamp to valid integer range
            minv = double(intmin(inClass));
            maxv = double(intmax(inClass));
            blended = max(minv, min(maxv, double(blended)));
            imageBlended = cast(round(blended), inClass);

        otherwise
            % float input -> float output
            imageBlended = cast(blended, inClass);
    end

end
