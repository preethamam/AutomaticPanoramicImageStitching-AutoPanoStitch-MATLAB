function imageFilesResized = resizeImagesToLimits(imageFiles, heightLimit, widthLimit, mode)
    %RESIZEIMAGESTOLIMITS  Aspect-ratio preserving multi-mode image resizing.
    %
    %   imageFilesResized = resizeImagesToLimits(imageFiles, heightLimit, widthLimit, mode)
    %
    %   - Only runs if any input image exceeds [heightLimit,widthLimit].
    %   - Otherwise, returns input unmodified.
    %
    %   Modes:
    %     'fit'      : Stage-1 fit within [Hlim,Wlim]; if sizes differ, Stage-2
    %                  upscales smaller ones to the largest Stage-1 size (no padding).
    %     'pad'      : Fit within, then pad to exactly [Hlim,Wlim].
    %     'fillcrop' : Cover [Hlim,Wlim], then center-crop to exactly [Hlim,Wlim].
    %
    %   Uses parfor internally for speed.

    % -------- Input validation --------
    if nargin < 4 || isempty(mode), mode = 'fit'; end
    mode = validatestring(mode, {'fit', 'pad', 'fillcrop'}, mfilename, 'mode', 4);

    if ~iscell(imageFiles) || isempty(imageFiles)
        error('imageFiles must be a nonempty cell array of images.');
    end

    if ~isscalar(heightLimit) || ~isscalar(widthLimit) || heightLimit <= 0 || widthLimit <= 0
        error('heightLimit and widthLimit must be positive scalars.');
    end

    % -------- Early exit if nothing exceeds limits --------
    initialSizes = cellfun(@size, imageFiles, 'UniformOutput', false);
    tooLarge = cellfun(@(s) (s(1) > heightLimit) || (s(2) > widthLimit), initialSizes);
    
    % Check if image is small and same size
    if ~any(tooLarge)
        % Only exit if all sizes are identical
        sizeMat = cell2mat(cellfun(@(s) s(1:2), initialSizes, 'UniformOutput', false));
        if size(unique(sizeMat, 'rows'), 1) == 1
            imageFilesResized = imageFiles; % all same size, nothing to do
            return;
        end
        % Different sizes within limits - fall through to normalize
    end

    % -------- Stage 1: fit/pad/fillcrop --------
    n = numel(imageFiles);
    stage1 = cell(size(imageFiles));
    sizes1 = zeros(n, 2); % [H,W]

    parfor k = 1:n
        J = [];
        I = imageFiles{k};
        if isempty(I), stage1{k} = I; sizes1(k, :) = [0 0]; continue; end
        [h, w, ~] = size(I);

        switch mode
            case {'fit', 'pad'}
                s = min(heightLimit / h, widthLimit / w); % isotropic fit
                if ~isfinite(s) || s <= 0, s = 1; end
                if s < 1
                    J = imresize(I, s, 'bicubic');
                else
                    J = I;  % keep original if within limits
                end

                if strcmp(mode, 'pad')
                    J = padToBox(J, heightLimit, widthLimit);
                end

            case 'fillcrop'
                s = max(heightLimit / h, widthLimit / w); % cover then crop
                if ~isfinite(s) || s <= 0, s = 1; end
                Jbig = imresize(I, s, 'bicubic');
                J = centerCrop(Jbig, heightLimit, widthLimit);
        end

        stage1{k} = J;
        sizes1(k, :) = [size(J, 1), size(J, 2)];
    end

    % -------- Stage 2 logic --------
    if strcmp(mode, 'pad') || strcmp(mode, 'fillcrop')
        % All identical already
        imageFilesResized = stage1;
        return;
    end

    % mode == 'fit'
    uniqueSizes = unique(sizes1, 'rows');

    if size(uniqueSizes, 1) == 1
        imageFilesResized = stage1; % already identical
        return;
    end

    % Determine largest Stage-1 size
    Hmax = max(sizes1(:, 1));
    Wmax = max(sizes1(:, 2));

    % Upscale smaller ones (anisotropic to match largest Stage-1 size)
    imageFilesResized = cell(size(imageFiles));

    parfor k = 1:n
        J = stage1{k};
        if isempty(J), imageFilesResized{k} = J; continue; end
        imageFilesResized{k} = imresize(J, [Hmax, Wmax], 'bicubic');
    end

end

% -------- Helpers --------
function Jpad = padToBox(J, Ht, Wt)
    % PADTOBOX  Pad image to target box using edge replication.
    %
    %   Jpad = padToBox(J, Ht, Wt)
    %
    % Inputs:
    %   J  - HxW x C image (numeric)
    %   Ht - target height (positive integer)
    %   Wt - target width (positive integer)
    %
    % Outputs:
    %   Jpad - Ht x Wt x C image padded using replicate padding

    arguments
        J (:, :, :) {mustBeNumeric}
        Ht (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        Wt (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    [h, w, c] = size(J);

    if h > Ht || w > Wt
        J = centerCrop(J, min(Ht, h), min(Wt, w));
        [h, w, ~] = size(J);
    end

    padTop = floor((Ht - h) / 2);
    padBottom = Ht - h - padTop;
    padLeft = floor((Wt - w) / 2);
    padRight = Wt - w - padLeft;

    if c == 1
        Jpad = padarray(J, [padTop padLeft], 'replicate', 'pre');
        Jpad = padarray(Jpad, [padBottom padRight], 'replicate', 'post');
    else
        Jpad = padarray(J, [padTop padLeft 0], 'replicate', 'pre');
        Jpad = padarray(Jpad, [padBottom padRight 0], 'replicate', 'post');
    end

end

function Jcrop = centerCrop(J, Ht, Wt)
    % CENTERCROP  Center-crop an image to the requested box size.
    %
    %   Jcrop = centerCrop(J, Ht, Wt)
    %
    % Inputs:
    %     J  - HxW x C image
    %     Ht - desired height (positive integer)
    %     Wt - desired width (positive integer)
    %
    % Outputs:
    %     Jcrop - Ht x Wt x C image, cropped from center

    arguments
        J (:, :, :) {mustBeNumeric}
        Ht (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        Wt (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    [h, w, ~] = size(J);
    Ht = min(Ht, h); Wt = min(Wt, w);
    y0 = floor((h - Ht) / 2) + 1;
    x0 = floor((w - Wt) / 2) + 1;
    Jcrop = J(y0:y0 + Ht - 1, x0:x0 + Wt - 1, :);
end
