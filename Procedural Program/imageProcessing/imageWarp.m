function warped = imageWarp(image, tform, outputView, options)
    % IMAGEWARP Highly vectorized image warping using a homography.
    %
    % Inputs
    %   image       - HxWxC numeric array. Can be on CPU or a `gpuArray`.
    %   tform       - 3x3 numeric homography matrix (projective transform).
    %   outputView  - `imref2d` object defining the output image limits and
    %                 pixel extents.
    %   options     - (optional) struct with fields:
    %                   * `method`     : 'nearest' | 'bilinear' | 'bicubic'
    %                   * `fillValue`  : value used for pixels mapped outside
    %                                   the source image (default: 0)
    %
    % Outputs
    %   warped      - outHeight x outWidth x C image of the same class as
    %                 the input `image`, containing the warped result.

    % Argument validation
    arguments
        image {mustBeNumeric}
        tform double
        outputView 
        options struct = struct()
    end

    % Default options
    if ~isfield(options, 'method'), options.method = 'bilinear'; end
    if ~isfield(options, 'fillValue'), options.fillValue = 0; end

    % Detect GPU usage from input image
    % ---- choose CPU/GPU here ----
    try
        useGPU = parallel.gpu.GPUDevice.isAvailable || isa(image, 'gpuArray');
    catch
        useGPU = false;
    end

    % ----- Resolve output size and world grid -----
    outHeight = outputView.ImageSize(1);
    outWidth = outputView.ImageSize(2);

    % Build grid in WORLD coordinates (upper-left pixel centers)
    x0 = outputView.XWorldLimits(1);
    y0 = outputView.YWorldLimits(1);
    sx = outputView.PixelExtentInWorldX;
    sy = outputView.PixelExtentInWorldY;

    % Build on CPU then move to GPU if needed (keeps code simple & fast)
    [Xcpu, Ycpu] = meshgrid(double(x0 + (0:outWidth - 1) * sx), ...
        double(y0 + (0:outHeight - 1) * sy));

    if useGPU
        X = gpuArray(Xcpu);
        Y = gpuArray(Ycpu);
    else
        X = Xcpu;
        Y = Ycpu;
    end

    % ----- Get homography in column-vector convention and invert it -----
    H = double(tform); % assume already column-form 3x3

    if H(3, 3) ~= 0
        H = H ./ H(3, 3);
    end

    % ----- Back-map output samples to source -----
    total_pixels = outHeight * outWidth;
    XY1 = [X(:)'; ...
               Y(:)'; ...
               ones(1, total_pixels, 'like', X)];

    % Temporarily silence near-singular warnings (CPU path only)
    id1 = 'MATLAB:nearlySingularMatrix';
    id2 = 'MATLAB:singularMatrix';
    id3 = 'MATLAB:illConditionedMatrix';

    if ~useGPU
        s1 = warning('query', id1); warning('off', id1);
        s2 = warning('query', id2); warning('off', id2);
        s3 = warning('query', id3); warning('off', id3);
        cleanupObj = onCleanup(@() restoreWarnings(s1, s2, s3, id1, id2, id3)); %#ok<NASGU>
    end

    % Stable solve (prefer \ over inv, add micro-regularization if ill-conditioned)
    rc = rcond(H);

    if ~isfinite(rc) || rc < 1e-14
        eps_reg = 1e-12 * norm(H, 2);
        H(1:2, 1:2) = H(1:2, 1:2) + eps_reg * eye(2, 'like', H);
    end

    % Cast H to live on same device as X/Y
    H = cast(H, 'like', X);
    src = H \ XY1;

    % Guard the projective scale to avoid NaN/Inf without changing sign
    w = src(3, :);
    w = sign(w) .* max(abs(w), 1e-12);
    srcX = reshape(src(1, :) ./ w, outHeight, outWidth);
    srcY = reshape(src(2, :) ./ w, outHeight, outWidth);

    % ----- Initialize output -----
    [inHeight, inWidth, numChannels] = size(image);
    fill_val = cast(options.fillValue, 'like', image);
    warped = repmat(fill_val, [outHeight, outWidth, numChannels]);

    switch lower(options.method)
        case 'nearest'
            % Round coordinates and create mask
            x = round(srcX);
            y = round(srcY);
            valid = x >= 1 & x <= inWidth & y >= 1 & y <= inHeight;

            % Convert to linear indices
            indices = sub2ind([inHeight, inWidth], y(valid), x(valid));
            validLinear = valid;

            % Process all channels at once using reshaping
            imageReshaped = reshape(image, [], numChannels);
            warpedReshaped = reshape(warped, [], numChannels);
            warpedReshaped(validLinear, :) = imageReshaped(indices, :);
            warped = reshape(warpedReshaped, outHeight, outWidth, numChannels);

        case 'bilinear'
            % Floor coordinates and compute weights
            x1 = floor(srcX);
            y1 = floor(srcY);
            x2 = x1 + 1;
            y2 = y1 + 1;

            wx = srcX - x1;
            wy = srcY - y1;

            % Find valid coordinates
            valid = x1 >= 1 & x2 <= inWidth & y1 >= 1 & y2 <= inHeight;
            validLinear = find(valid);

            if ~isempty(validLinear)
                % Get corner indices for valid pixels
                i11 = sub2ind([inHeight, inWidth], y1(valid), x1(valid));
                i12 = sub2ind([inHeight, inWidth], y2(valid), x1(valid));
                i21 = sub2ind([inHeight, inWidth], y1(valid), x2(valid));
                i22 = sub2ind([inHeight, inWidth], y2(valid), x2(valid));

                % Extract weights for valid pixels
                wxv = wx(valid);
                wyv = wy(valid);

                % Compute weights
                w11 = (1 - wxv) .* (1 - wyv);
                w12 = (1 - wxv) .* wyv;
                w21 = wxv .* (1 - wyv);
                w22 = wxv .* wyv;

                % Process all channels simultaneously using matrix operations
                imageReshaped = reshape(cast(image, 'double'), [], numChannels);
                warpedReshaped = reshape(warped, [], numChannels);

                % Vectorized interpolation for all channels
                interpVals = w11 .* imageReshaped(i11, :) + ...
                    w12 .* imageReshaped(i12, :) + ...
                    w21 .* imageReshaped(i21, :) + ...
                    w22 .* imageReshaped(i22, :);

                warpedReshaped(validLinear, :) = cast(interpVals, 'like', image);
                warped = reshape(warpedReshaped, outHeight, outWidth, numChannels);
            end

        case 'bicubic'
            % Floor coordinates
            x = floor(srcX);
            y = floor(srcY);
            dx = srcX - x;
            dy = srcY - y;

            % Find valid coordinates (need one extra pixel on each side for bicubic)
            valid = x >= 2 & x <= inWidth - 2 & y >= 2 & y <= inHeight - 2;
            validLinear = find(valid);

            if ~isempty(validLinear)
                % Extract valid coordinates
                xValid = x(valid);
                yValid = y(valid);
                dxValid = dx(valid);
                dyValid = dy(valid);

                % Pre-compute weights for x and y directions
                numPts = numel(validLinear);
                xWeights = zeros(numPts, 4, 'like', dxValid);
                yWeights = zeros(numPts, 4, 'like', dyValid);

                % Compute x and y weights vectorized
                for ii = -1:2
                    xWeights(:, ii + 2) = bicubicKernel(ii - dxValid);
                    yWeights(:, ii + 2) = bicubicKernel(ii - dyValid);
                end

                % Create indices matrix for all 16 sample points
                indices = zeros(numPts, 16, 'like', validLinear);

                idx = 1;

                for dyOff = -1:2

                    for dxOff = -1:2
                        indices(:, idx) = sub2ind([inHeight, inWidth], ...
                            yValid + dyOff, xValid + dxOff);
                        idx = idx + 1;
                    end

                end

                % Process all channels simultaneously
                if isinteger(image)
                    imageClass = class(image);
                    maxVal = double(intmax(imageClass));
                    imageReshaped = reshape(double(image), [], numChannels); % double precision
                else
                    imageReshaped = reshape(double(image), [], numChannels);
                    maxVal = 1.0;
                end

                warpedReshaped = reshape(warped, [], numChannels);

                % Get all sample points for all channels
                samples = imageReshaped(indices, :);
                samples = reshape(samples, [], 4, 4, numChannels);

                % Apply bicubic interpolation using matrix operations
                interpVals = zeros(numPts, numChannels, 'double');

                for c = 1:numChannels
                    temp = squeeze(samples(:, :, :, c)); % [num_pts x 4 x 4]
                    tempReshaped = reshape(temp, numPts, 4, 4);

                    % First interpolate in x direction
                    xInterp = zeros(numPts, 4, 'double');

                    for ii = 1:4

                        for jj = 1:4
                            xInterp(:, jj) = xInterp(:, jj) + ...
                                tempReshaped(:, ii, jj) .* double(xWeights(:, ii));
                        end

                    end

                    % Then interpolate in y direction
                    interpVals(:, c) = sum(xInterp .* double(yWeights), 2);
                end

                % Handle output conversion
                if isinteger(image)
                    interpVals = max(0, min(maxVal, round(interpVals)));
                    warpedReshaped(validLinear, :) = cast(interpVals, imageClass);
                else
                    interpVals = max(0, min(maxVal, interpVals));
                    warpedReshaped(validLinear, :) = cast(interpVals, 'like', image);
                end

                % Warped image
                warped = reshape(warpedReshaped, outHeight, outWidth, numChannels);
            end

    end

    % Add this at the very end, before "end":
    if useGPU
        warped = gather(warped);
    end

end

% ----- Nested bicubic kernel (GPU-safe) -----
function w = bicubicKernel(x)
    % BICUBICKERNEL Vectorized bicubic interpolation kernel.
    %   w = bicubicKernel(x) computes bicubic kernel weights for input x,
    %   supporting vectorized evaluation for interpolation weights.
    %
    % Inputs
    %   x - numeric array of relative coordinates (can be vector/matrix)
    %
    % Outputs
    %   w - numeric array of same size as x containing kernel weights

    arguments
        x {mustBeNumeric}
    end

    absx = abs(x);
    w = zeros(size(x), 'like', x);

    mask1 = absx <= 1;
    mask2 = absx <= 2 & ~mask1;

    absx2 = absx .^ 2;
    absx3 = absx .^ 3;

    w(mask1) = 1.5 * absx3(mask1) - 2.5 * absx2(mask1) + 1;
    w(mask2) = -0.5 * absx3(mask2) + 2.5 * absx2(mask2) - 4 * absx(mask2) + 2;
end

% ----- Local helper to restore warning states (CPU only) -----
function restoreWarnings(s1, s2, s3, id1, id2, id3)
    % RESTOREWARNINGS Restore previously saved warning states for three IDs.
    %   restoreWarnings(s1,s2,s3,id1,id2,id3) restores the warning state
    %   recorded in the structs s1,s2,s3 for the warning IDs id1,id2,id3.
    %
    % Inputs
    %   s1,s2,s3 - structs returned by warning('query', id) containing a .state field
    %   id1,id2,id3 - character vectors specifying the warning identifiers
    %
    % Outputs
    %   None - this function restores warning states and does not return values

    arguments
        s1 struct
        s2 struct
        s3 struct
        id1 char
        id2 char
        id3 char
    end

    warning(s1.state, id1);
    warning(s2.state, id2);
    warning(s3.state, id3);
end
