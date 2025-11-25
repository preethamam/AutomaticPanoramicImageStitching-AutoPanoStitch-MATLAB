function [keypoints, allDescriptors, imagesOriginal, imagesProcessed, imageSizes, ...
              imageNames, numImgs] = loadImages(input, imgSetVector, myImg)
%LOADIMAGES Read an image set, extract features, and return metadata.
%
% Syntax
%   [keypoints, allDescriptors, images, imageSizes, imageNames, numImgs] = ...
%       loadImages(input, imgSetVector, myImg)
%
% Description
%   Loads all images from the folder specified by imgSetVector(myImg).folder/name,
%   optionally resizes them, converts grayscale to 3-channel, and extracts
%   keypoints and descriptors using the detector specified in input.detector.
%   Returns stacked images, per-image keypoints/descriptors, image sizes,
%   image file names, and the number of images.
%
% Inputs
%   input        - Struct of parameters. Expected fields include:
%                  • resizeImage (logical)
%                  • detector (char/string): 'HARRIS'|'SIFT'|'vl_SIFT'|'FAST'|'SURF'|'BRISK'|'ORB'|'KAZE'
%                  • Additional detector-specific fields (e.g., SIFT params)
%   imgSetVector - Struct array with fields 'folder' and 'name' defining dataset roots.
%   myImg        - Positive integer index selecting the dataset entry.
%
% Outputs
%   keypoints      - 1-by-N cell; each cell is 2-by-K array of keypoint coordinates (columns).
%   allDescriptors - 1-by-N cell; descriptor arrays per image, format per detector.
%   imagesOriginal  - 1-by-N cell; each cell is the original loaded image.
%   imagesProcessed - 1-by-N cell; each cell is a processed RGB image (HxWx3).
%   imageSizes     - N-by-3 numeric array: [rows, cols, channels] per image.
%   imageNames     - 1-by-N cell array of image file names (with extensions).
%   numImgs        - Scalar count of images in the dataset folder.
%
% Notes
%   - If a loaded image is grayscale, it is replicated to 3 channels.
%   - Resizing logic maintains 480x640 bounds depending on original size.
%   - Keypoints are returned as 2-by-K (x;y) to match downstream expectations.
%
% See also: imageDatastore, extractFeatures, detectSIFTFeatures, vl_sift

arguments
    input (1, 1) struct
    imgSetVector (1, :) struct
    myImg (1, 1) double {mustBeInteger, mustBePositive}
end

% Runtime validations
if myImg > numel(imgSetVector)
    error('loadImages:IndexOutOfRange', 'myImg (%d) exceeds imgSetVector length (%d).', myImg, numel(imgSetVector));
end

if ~isfield(imgSetVector, 'folder') || ~isfield(imgSetVector, 'name')
    error('loadImages:InvalidImgSetVector', 'imgSetVector must contain fields ''folder'' and ''name''.');
end

% Read images
imgFolder = fullfile(imgSetVector(myImg).folder, imgSetVector(myImg).name);
imds = imageDatastore(imgFolder);
imds.ReadFcn = @imreadAutoRotate;
imagesOriginal = readall(imds);

% Check if grayscale
imagesOriginal = cellfun(@convertToRGB, imagesOriginal, 'UniformOutput', false);
imageFiles = imagesOriginal;

% Resize images respecting the aspect ratio
if input.resizeImage
    imageFiles = resizeImagesToLimits(imagesOriginal, input.heightLimit, input.widthLimit, 'fit');
end

[~, imageNames, ext] = fileparts(imds.Files);
imageNames = strcat(imageNames, ext);

% Number of images in the folder
numImgs = length(imageFiles);

% Initialize the cell arrays
keypoints = cell(1, numImgs);
allDescriptors = cell(1, numImgs);
imagesProcessed = cell(1, numImgs);

% Feature extraction
parfor i = 1:numImgs

    % Sequential mages
    image = imageFiles{i};

    % Get size of the image
    imageSizes(i, :) = size(image);

    % Stack images
    imagesProcessed{i} = image;

    % Get features and valid points
    [descriptors, points] = getFeaturePoints(input, image);

    % Concatenate the descriptors and points
    keypoints{i} = points;
    allDescriptors{i} = descriptors;
end

end

function out = convertToRGB(img)
    % CONVERTTORG Convert single-channel image to 3-channel RGB.
    %   out = convertToRGB(img) ensures the output is HxWx3 by replicating a
    %   grayscale input across three channels. If `img` already has three
    %   channels, it is returned unchanged.
    %
    % Inputs
    %   img  - HxWx1 or HxWx3 numeric image array.
    %
    % Outputs
    %   out  - HxWx3 numeric image array.

    arguments
        img (:, :, :) {mustBeNumeric}
    end

    if size(img, 3) == 1
        out = repmat(img, [1, 1, 3]);
    else
        out = img;
    end

end

function img = imreadAutoRotate(filename)
    % IMREADAUTOROTATE Read an image file and correct its orientation using EXIF data.
    %
    %   IMG = IMREADAUTOROTATE(FILENAME) reads the image at the path FILENAME,
    %   inspects available EXIF orientation metadata (if present), and returns
    %   IMG rotated and/or flipped so that it appears in the standard upright
    %   orientation. The function preserves image class (e.g., uint8, uint16)
    %   and alpha channel when possible.
    %
    %   Syntax
    %   ------
    %   img = imreadAutoRotate(filename)
    %
    %   Inputs
    %   ------
    %   filename  - string or character vector specifying the path to the image
    %               file. Can be an absolute or relative path.
    %
    %   Outputs
    %   -------
    %   img       - image array (MxN for grayscale, MxNx3 for RGB, MxNx4 when an
    %               alpha channel is present) with orientation corrected.
    %
    %   Behavior / Notes
    %   ----------------
    %   - The function uses imread to load the image and imfinfo (or equivalent)
    %     to query EXIF metadata. If no orientation tag is found, the image is
    %     returned unchanged.
    %   - EXIF orientation values and typical corrective actions:
    %       1: no change
    %       2: flip left-right (mirror horizontal)
    %       3: rotate 180 degrees
    %       4: flip up-down (mirror vertical)
    %       5: transpose then rotate (mirror + rotate)
    %       6: rotate 90 degrees clockwise
    %       7: transverse then rotate (mirror + rotate)
    %       8: rotate 270 degrees clockwise
    %     The implementation maps these values to combinations of rot90,
    %     flipud, and fliplr as appropriate.
    %   - Alpha channel: if the image includes an alpha channel, the same
    %     geometric transforms are applied to the alpha channel to preserve
    %     transparency alignment.
    %   - Class preservation: the output IMG retains the numeric class returned
    %     by imread (e.g., uint8, uint16, single, double).
    %   - Errors: if the file cannot be read or the path is invalid, the
    %     function causes an error consistent with imread/imfinfo behavior.
    %
    %   Examples
    %   --------
    %   img = imreadAutoRotate('photo.jpg');      % read and auto-rotate a JPEG
    %   img = imreadAutoRotate('C:\img\scan.tif');% read a TIFF and correct orient
    %
    %   See also
    %   --------
    %   imread, imfinfo, rot90, fliplr, flipud

    % Image reading with EXIF orientation handling
    img = imread(filename);

    try
        % Suppress all warnings during EXIF parsing
        warnState = warning('off', 'all');
        cleanupObj = onCleanup(@() warning(warnState));

        info = imfinfo(filename);

        if isfield(info, 'Orientation')

            switch info.Orientation
                case 1
                    % Normal - no rotation needed
                case 2
                    img = flip(img, 2); % Horizontal flip
                case 3
                    img = rot90(img, 2); % 180°
                case 4
                    img = flip(img, 1); % Vertical flip
                case 5
                    img = rot90(flip(img, 2), -1);
                case 6
                    img = rot90(img, -1); % 90° CW (common for portrait)
                case 7
                    img = rot90(flip(img, 2), 1);
                case 8
                    img = rot90(img, 1); % 90° CCW
            end

        end

    catch
        % No EXIF or couldn't read - keep as-is
    end

end
