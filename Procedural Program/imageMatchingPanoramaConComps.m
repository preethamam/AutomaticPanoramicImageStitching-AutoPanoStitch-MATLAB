function [allMatchesRefined, numMatches, initialTforms, ...
              imagesProcessed, imageSizes, keypoints, numMatchesG, concomps, ccBinSizes] = imageMatchingPanoramaConComps(input, numImgs, keypoints, ...
    featureMatches, imagesOriginal, imagesProcessed, imageSizes)
% IMAGEMATCHINGPANORAMACONCOMPS Image matching with panorama connected-components handling.
%   This wrapper performs initial image matching, finds connected components
%   in the match graph, optionally resizes images per component, re-extracts
%   features, and runs a second-pass matching to comply with panorama clusters.
%
% Inputs
%   input           - options struct (passes through to imageMatching)
%   numImgs         - number of images
%   keypoints       - 1xN cell of keypoints (may be empty; re-computed if resizing)
%   featureMatches  - N-by-N cell of putative feature matches
%   imagesOriginal  - 1xN cell of original images
%   imagesProcessed - 1xN cell of preprocessed images (may be modified)
%   imageSizes      - N-by-? array with image sizes
%
% Outputs
%   allMatchesRefined - N-by-N cell of verified inlier matches
%   numMatches        - N-by-N matrix of inlier counts
%   initialTforms     - N-by-N cell array of pairwise transforms
%   imagesProcessed   - possibly updated processed images
%   imageSizes        - possibly updated image sizes
%   keypoints         - possibly updated keypoints cell
%   numMatchesG       - graph object of matches (upper triangular)
%   concomps          - component indices per image
%   ccBinSizes        - sizes of connected components

arguments
    input struct
    numImgs (1, 1) double {mustBeInteger, mustBePositive}
    keypoints cell
    featureMatches cell
    imagesOriginal cell
    imagesProcessed cell
    imageSizes (:, :) {mustBeNumeric}
end

% First pass image matching
[allMatchesRefined, numMatches, initialTforms] = imageMatching(input, numImgs, keypoints, featureMatches, imagesProcessed);

% Find connected components of image matches
numMatchesG = graph(numMatches, 'upper');
[concomps, ccBinSizes] = conncomp(numMatchesG);
conCompsUnqNum = unique(concomps);

% Resize images based on the panorama connected components local sizes
if input.resizeImage == 1 && input.resizeImagePanoramaCluster == 1 && numel(ccBinSizes) > 1

    % Resize the images as per panorama clusters matched connected
    % components
    for i = 1:numel(conCompsUnqNum)
        idxs = find(concomps == conCompsUnqNum(i));
        imagesProcessed(idxs) = resizeImagesToLimits(cell(imagesOriginal(idxs)), ...
            input.heightLimit, input.widthLimit, 'fit');
    end

    % Panorama connected components compliant image sizes
    imageSizes = cellfun(@size, imagesProcessed, 'UniformOutput', false);
    imageSizes = vertcat(imageSizes{:});

    % Initialize the cell arrays
    keypoints = cell(1, numImgs);
    allDescriptors = cell(1, numImgs);

    % Feature extraction
    parfor i = 1:numImgs

        % Sequential mages
        image = imagesProcessed{i};

        % Get features and valid points
        [descriptors, points] = getFeaturePoints(input, image);

        % Concatenate the descriptors and points
        keypoints{i} = points;
        allDescriptors{i} = descriptors;
    end

    % Get the matched features for panorama connected components
    % compliant images
    featurMatchesPanConComp = featureMatching(input, allDescriptors, numImgs);

    % Second pass matching for panorama connected components compliant
    [allMatchesRefined, numMatches, initialTforms] = imageMatching(input, numImgs, ...
        keypoints, featurMatchesPanConComp, imagesProcessed);
end

end
