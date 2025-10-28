function [keypoints, allDescriptors, images, imageSizes, imageNames, numImgs] = loadImages(input, imgSetVector, myImg)
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
    %   images         - 1-by-N cell; each cell is an RGB image (HxWx3).
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
    imageFiles = readall(imds);
    [~, imageNames, ext] = fileparts(imds.Files);
    imageNames = strcat(imageNames, ext);

    % Number of images in the folder
    numImgs = length(imageFiles);

    % Initialize the cell arrays
    keypoints = cell(1, numImgs);
    allDescriptors = cell(1, numImgs);
    images = cell(1, numImgs);

    % Feature extraction
    parfor i = 1:numImgs

        % Sequential mages
        image = imageFiles{i};

        % Get size of the image
        [imRows, imCols, imChannel] = size(image);

        % Image resize
        if input.resizeImage

            if imRows > 480 && imCols > 640
                image = imresize(image, [480, 640]);
            elseif imRows > 480 && imCols < 640
                image = imresize(image, [480, imCols]);
            elseif imRows < 480 && imCols > 640
                image = imresize(image, [imRows, 640]);
            end

        end

        % Replicate the third channel
        if imChannel == 1
            image = repmat(image, 1, 1, 3);
        end

        % Get size of the image
        [imRows, imCols, imChannel] = size(image);
        imageSizes(i, :) = [imRows, imCols, imChannel];

        % Stack images
        images{i} = image;

        % Get features and valid points
        [descriptors, points] = getFeaturePoints(input, image);

        % Concatenate the descriptors and points
        keypoints{i} = points';
        allDescriptors{i} = descriptors;
    end

end

% Get the feature points
function [features, validPts] = getFeaturePoints(input, ImageOriginal)
    %GETFEATUREPOINTS Detect keypoints and extract features for an image.
    %
    % Syntax
    %   [features, validPts] = getFeaturePoints(input, ImageOriginal)
    %
    % Description
    %   Converts the input image to grayscale if needed and detects features
    %   according to input.detector. Supports MATLAB detectors and 'vl_SIFT'.
    %   Returns the descriptor matrix and a N-by-2 array of keypoint locations.
    %
    % Inputs
    %   input         - Struct with at least field 'detector' specifying the method.
    %                   Additional fields may be required per detector (e.g., SIFT params).
    %   ImageOriginal - Numeric image, grayscale or RGB.
    %
    % Outputs
    %   features - Descriptor matrix as returned by the selected extractor.
    %   validPts - N-by-2 array of [x y] keypoint coordinates (double).

    arguments
        input (1, 1) struct
        ImageOriginal {mustBeNumeric}
    end

    if size(ImageOriginal, 3) > 1
        grayImage = rgb2gray(ImageOriginal);
    else
        grayImage = ImageOriginal;
    end

    switch input.detector
        case 'HARRIS'
            points = detectHarrisFeatures(grayImage);

        case 'SIFT'
            points = detectSIFTFeatures(grayImage, 'NumLayersInOctave', input.NumLayersInOctave, ...
                ContrastThreshold = input.ContrastThreshold, ...
                EdgeThreshold = input.EdgeThreshold, ...
                Sigma = input.Sigma);
        case 'vl_SIFT'
            [locations, features] = vl_sift(single(grayImage), 'Octaves', 8);
            features = features';

            if ~isempty(locations)
                validPts = locations(1:2, :)';
            else
                validPts = [];
            end

        case 'FAST'
            points = detectFASTFeatures(grayImage);

        case 'SURF'
            points = detectSURFFeatures(grayImage, 'NumOctaves', 8);

        case 'BRISK'
            points = detectBRISKFeatures(grayImage);

        case 'ORB'
            points = detectORBFeatures(grayImage);

        case 'KAZE'
            points = detectKAZEFeatures(grayImage);

        otherwise
            error('Need a valid input!')
    end

    % Get features and valid points for other than vl_SIFT
    if ~(strcmp(input.detector, 'vl_SIFT'))
        [features, validPts] = extractFeatures(grayImage, points);
        validPts = double(validPts.Location);
    end

end