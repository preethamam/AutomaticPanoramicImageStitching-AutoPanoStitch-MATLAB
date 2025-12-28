%//%*************************************************************************%
%//%*                 Automatic Panorama Image Stitcher                     *%
%//%*           Stitches multiple images using feature points               *%
%//%*                                                                       *%
%//%*                    Name: Dr. Preetham Manjunatha                      *%
%//%*               GitHub: https://github.com/preethamam	                *%
%//%*                   Repo Name: AutoPanoStitch                           *%
%//%*                    Rewritten Date: 11/24/2025                         *%
%%***************************************************************************%
%* Citation 1: Automatic Panoramic Image Stitching using Invariant Features.*%
%* M. Brown and D. Lowe. International Journal of Computer Vision. 74(1),   *%
%* pages 59-73, 2007                                                        *%
%*                                                                          *%
%* Citation 2: Recognising Panoramas. M. Brown and D. G. Lowe.              *%
%* International Conference on Computer Vision (ICCV2003). pages 1218-1225, *%
%* Nice, France, 2003.                                                      *%
%****************************************************************************%

%% Start
%--------------------------------------------------------------------------
clear; close all; clc;
clcwaitbarz = findall(0, 'type', 'figure', 'tag', 'TMWWaitbar');
delete(clcwaitbarz);
warning('on', 'all');

%% Get inputs
%--------------------------------------------------------------------------
% Inputs file
%--------------------------------------------------------------------------
inputs;

% Add folders
addpath(genpath(pwd));

%% Parallel workers start
%--------------------------------------------------------------------------
% Parallel pools
%--------------------------------------------------------------------------
if (isempty(gcp('nocreate')))

    if strcmp(input.poolType, 'numcores')
        parpool(input.numCores);
    else
        parpool('Threads')
    end

end

Start = tic;

%% Get image filenames and store image names
%--------------------------------------------------------------------------
% Image sets
%--------------------------------------------------------------------------
imgFolders = dir(fullfile(folderPath, folderName));
imgFolders = imgFolders([imgFolders(:).isdir]);
imgFolders = imgFolders(~ismember({imgFolders(:).name}, {'.', '..'}));

% Dataset name and folders length
datasetName = {imgFolders.name}';
foldersLen = length(datasetName);

% Log command window/terminal outputs
if input.writeCommandWindowOutput
    try
        diary off     % safe even if diary is not on
    catch
        warning('Cannot turn off the diary.')
    end

    logPath = fullfile(pwd, 'log.txt'); 
    if exist(logPath,'file')
        delete(logPath);   % remove old log
    end
    diary(logPath)     % start recording
    diary on
end

%% Panorama stitcher
%--------------------------------------------------------------------------
% Stitches panoramas
%--------------------------------------------------------------------------
for myImg = 1:foldersLen
    stitchStart = tic;
    fprintf('Image number: %i | Current folder: %s\n', myImg, imgFolders(myImg).name);

    %% Load images
    loadimagestic = tic;
    [keypoints, allDescriptors, imagesOriginal, imagesProcessed, imageSizes, ...
        imageNames, numImg] = loadImages(input, imgFolders, myImg);
    fprintf('Loaded %d images (+ features): %f seconds\n', numel(imagesProcessed), toc(loadimagestic));

    %% Get feature matrices and keypoints
    featureMatchtic = tic;
    if input.matchFeaturesPairwise == true
        featureMatches = featureMatchingPairwise(input, allDescriptors, numImg);
    else
        featureMatches = featureMatchingGlobal(input, allDescriptors, numImg);
    end
    fprintf('Matched features: %f seconds\n', toc(featureMatchtic));

    %% Find matches
    imageMatchtic = tic;    
    [allMatchesRefined, numMatches, initialTforms, imagesProcessed, imageSizes, keypoints, ...
        numMatchesG, concomps, ccBinSizes] ...
        = imageMatchingPanoramaConComps(input, numImg, keypoints, featureMatches, ...
          imagesOriginal, imagesProcessed, imageSizes);
    fprintf('Matched images: %f seconds\n', toc(imageMatchtic));       

    %% Recognize panoramas and perform bundle adjustment
    recPanosnBAtic = tic;
    [bundlerTforms, finalrefIdxs, panoIndices, concomps, panaromaCCs, connCompsNumber] = recognizePanoramas(input, ...
                numMatches, allMatchesRefined, keypoints, imageSizes, ...
                initialTforms, numMatchesG, concomps, ccBinSizes);
    fprintf('Recognized panoramas and performed bundle adjustment: %f seconds\n', toc(recPanosnBAtic));    

    %% Automatic panorama straightening
    straightentic = tic;       
    finalPanoramaTforms = straightening(input, bundlerTforms);   
    fprintf('Automatic panorama straightening: %f seconds\n', toc(straightentic));

    %% Render and display panoramas
    rendertic = tic;
    reportGPUMemory(sprintf('Before dataset %d', myImg));
    panoStoreRGBAnno = displayPanorama(input, finalPanoramaTforms, finalrefIdxs, imagesProcessed, imageSizes, panoIndices);
    reportGPUMemory(sprintf('After dataset %d', myImg));
    fprintf('Rendering and display time: %f seconds\n', toc(rendertic));

    %% Crop and save panoramas
    cropsavetic = tic;
    panoStoreRGBCropAnno = cropNsavePanorama(input, panoStoreRGBAnno, myImg, datasetName);
    fprintf('Crop and save time: %f seconds\n', toc(cropsavetic));

    %% Print complete runtime to stitch images
    fprintf('----------------------------------------------------------------------------------------\n');
    fprintf('Total runtime (stitching): %f seconds\n', toc(stitchStart));
    fprintf('----------------------------------------------------------------------------------------\n\n', toc); %#ok<CTPCT>
end

%% End parameters
%--------------------------------------------------------------------------
% Final cleanup
if gpuDeviceCount > 0
    reset(gpuDevice);
end

clcwaitbarz = findall(0, 'type', 'figure', 'tag', 'TMWWaitbar');
delete(clcwaitbarz);
statusFclose = fclose('all');
diary off     % safe even if diary is not on
if (statusFclose == 0)
    disp('All files are closed.')
end

Runtime = toc(Start);
fprintf('Total runtime : %f seconds\n', Runtime);
currtime = datetime('now');
display(currtime)