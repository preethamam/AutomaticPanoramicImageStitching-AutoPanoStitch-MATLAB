function croppedImage = panoramaCropper(input, stitchedImage)
    % PANORAMACROPPER Crop a stitched panorama to its content bounds.
    %   croppedImage = panoramaCropper(input, stitchedImage) computes a tight
    %   crop rectangle around the valid content of the stitched RGB image
    %   stitchedImage by thresholding with respect to the canvas color and
    %   returns the cropped image.
    %
    %   Inputs
    %   - input: struct with fields:
    %       canvasColor        - "black" or "white" background for empty canvas.
    %       blackRange          - Scalar in [0,255] threshold when canvasColor = "black".
    %       whiteRange          - Scalar in [0,255] threshold when canvasColor = "white".
    %       showCropBoundingBox - logical flag to visualize crop rectangle.
    %       displayPanoramas    - logical flag that controls interactive display.
    %   - stitchedImage: M-by-N-by-3 numeric RGB image to crop.
    %
    %   Output
    %   - croppedImage: Cropped RGB image. If a valid crop cannot be computed
    %     (e.g., due to holes in the background mask), the input image is
    %     returned unchanged and a warning is issued.
    %
    %   Notes
    %   - If showCropBoundingBox and displayPanoramas are both true, the crop
    %     rectangle is drawn and exported to 'pano_bbox.jpg'.
    %   - Assumes an 8-bit intensity range; thresholds are interpreted in [0,255].
    %
    %   See also rgb2gray, imbinarize, imfill, imcomplement, rectangle, exportgraphics

    arguments
        input (1, 1) struct
        stitchedImage (:, :, 3) {mustBeNumeric, mustBeNonempty}
    end

    % Validate required fields and values in input struct
    reqFields = ["canvasColor", "blackRange", "whiteRange", "showCropBoundingBox", "displayPanoramas"];
    missing = reqFields(~isfield(input, reqFields));

    if ~isempty(missing)
        error('panoramaCropper:MissingField', ...
            'Missing required input fields: %s', strjoin(missing, ', '));
    end

    canvasColor = lower(string(input.canvasColor));

    if ~(canvasColor == "black" || canvasColor == "white")
        error('panoramaCropper:InvalidCanvasColor', ...
        'input.canvasColor must be "black" or "white".');
    end

    if ~(isscalar(input.blackRange) && isnumeric(input.blackRange) && isfinite(input.blackRange) && input.blackRange >= 0 && input.blackRange <= 255)
        error('panoramaCropper:InvalidBlackRange', 'input.blackRange must be a numeric scalar in [0,255].');
    end

    if ~(isscalar(input.whiteRange) && isnumeric(input.whiteRange) && isfinite(input.whiteRange) && input.whiteRange >= 0 && input.whiteRange <= 255)
        error('panoramaCropper:InvalidWhiteRange', 'input.whiteRange must be a numeric scalar in [0,255].');
    end

    if ~(islogical(input.showCropBoundingBox) && isscalar(input.showCropBoundingBox))
        error('panoramaCropper:InvalidFlag', 'input.showCropBoundingBox must be a logical scalar.');
    end

    if ~(islogical(input.displayPanoramas) && isscalar(input.displayPanoramas))
        error('panoramaCropper:InvalidFlag', 'input.displayPanoramas must be a logical scalar.');
    end

    % Initialize the variables
    w = size(stitchedImage, 2);
    h = size(stitchedImage, 1);

    % Convert the stitched image to grayscale and threshold it
    % such that all pixels greater than zero are set to 255
    % (foreground) while all others remain 0 (background)
    gray = rgb2gray(stitchedImage);

    if canvasColor == "black"
        BW = imbinarize(gray, double(input.blackRange) / 255);
    else
        BW = imbinarize(gray, double(input.whiteRange) / 255);
        BW = imcomplement(BW);
    end

    % Find all external contours in the threshold image then find
    % the *largest* contour which will be the contour/outline of
    % the stitched image
    BW2 = imfill(BW, 'holes');

    % Canvas outer indices
    canvasOuterIndices = BW2 == 0;

    % Normalize the image to -1 and others
    stitched = double(stitchedImage);
    stitched(repmat(canvasOuterIndices, 1, 1, 3)) = -255;
    stitched = stitched / 255.0;

    % Get the crop indices
    maxarea = 0;
    height = zeros(1, w);
    left = zeros(1, w);
    right = zeros(1, w);

    ll = 0;
    rr = 0;
    hh = 0;
    nl = 0;

    for line = 1:h

        for k = 1:w
            p = stitched(line, k, :);
            m = max(max(p(1), p(2)), p(3));

            if m < 0
                height(k) = 0;
            else
                height(k) = height(k) + 1; % find Color::NO
            end

        end

        for k = 1:w
            left(k) = k;

            while (left(k) > 1 && (height(k) <= height(left(k) - 1)))
                left(k) = left(left(k) - 1);
            end

        end

        for k = w - 1:-1:1
            right(k) = k;

            while (right(k) < w - 1 && (height(k) <= height(right(k) + 1)))
                right(k) = right(right(k) + 1);
            end

        end

        for k = 1:w
            val = (right(k) - left(k) + 1) * height(k);

            if (maxarea < val)
                maxarea = val;
                ll = left(k);
                rr = right(k);
                hh = height(k);
                nl = line;
            end

        end

    end

    % Crop indexes
    cropH = hh + 1;
    cropW = rr - ll + 1;
    offsetx = ll;
    offsety = nl - hh + 1;

    % Cropped image
    try
        croppedImage = stitchedImage(offsety:offsety + cropH, offsetx:offsetx + cropW, :);
    catch
        warning('Cannot crop the image. Image has background holes.');
        croppedImage = stitchedImage;
    end

    % Show tight bounding box
    if (input.showCropBoundingBox && input.displayPanoramas)
        figure('Name', 'Crop rectangle panorama image');
        imshow(stitchedImage);
        hold on
        rectangle('Position', [offsetx offsety cropW cropH], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--')
        hold off

        ax = gcf;
        exportgraphics(ax, 'pano_bbox.jpg')
    end

end
