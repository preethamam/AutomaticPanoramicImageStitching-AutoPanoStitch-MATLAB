function [keypoints, allDescriptors, images, imageSizes, imageNames, numImgs] = loadImages(input, imgSetVector, myImg)
% function imageFiles = loadImages(imgSetVector, myImg)
    
    %%***********************************************************************%
    %*                   Automatic panorama stitching                       *%
    %*                          Images loader                               *%
    %*                                                                      *%
    %* Code author: Preetham Manjunatha                                     *%
    %* Github link: https://github.com/preethamam                           *%
    %* Date: 05/14/2024                                                     *%
    %************************************************************************%

    % Read images    
    imgFolder = fullfile(imgSetVector(myImg).folder, imgSetVector(myImg).name);
    imds = imageDatastore(imgFolder);
    imageFiles = readall(imds);    
    [~,imageNames,ext] = fileparts(imds.Files);
    imageNames = strcat(imageNames,ext);

    % Number of images in the folder
    numImgs = length(imageFiles);

    % Initialize the cell arrays
    keypoints = cell(1,numImgs);
    allDescriptors = cell(1,numImgs);
    images = cell(1,numImgs);
    
    % Feature extraction
    parfor i = 1:numImgs
        
        % Sequential mages
        image = imageFiles{i};
        
        % Get size of the image
        [imRows, imCols, imChannel] = size(image);
        
        % Image resize
        if input.resizeImage
            if imRows > 480 && imCols > 640
                image = imresize(image,[480, 640]);
            elseif imRows > 480 && imCols < 640
                image = imresize(image,[480, imCols]);
            elseif imRows < 480 && imCols > 640
                image = imresize(image,[imRows, 640]);
            end
        end       

        % Replicate the third channel
        if imChannel == 1
            image = repmat(image, 1, 1, 3);
        end

        % Get size of the image
        [imRows, imCols, imChannel] = size(image);
        imageSizes(i,:) = [imRows, imCols, imChannel];
        
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
    if size(ImageOriginal,3) > 1
        grayImage = rgb2gray(ImageOriginal);
    else
        grayImage = ImageOriginal;
    end
        
    switch input.detector
        case 'HARRIS'
            points = detectHarrisFeatures(grayImage);
    
        case 'SIFT'        
            points = detectSIFTFeatures(grayImage,'NumLayersInOctave',input.NumLayersInOctave, ...
                                        ContrastThreshold=input.ContrastThreshold, ...
                                        EdgeThreshold=input.EdgeThreshold, ...
                                        Sigma=input.Sigma);
        case 'vl_SIFT'
            [locations, features]  = vl_sift(single(grayImage), 'Octaves', 8);
            features = features';
            if ~isempty(locations)                
                validPts = locations(1:2,:)';
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