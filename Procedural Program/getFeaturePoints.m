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
