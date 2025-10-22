%//%*************************************************************************%
%//%*                 Automatic Panorama Image Stitcher                     *%
%//%*           Stitches multiple images using feature points               *%
%//%*                                                                       *%
%//%*                    Name: Dr. Preetham Manjunatha                      *%
%//%*               GitHub: https://github.com/preethamam	                *%
%//%*                   Repo Name: AutoPanoStitch                           *%
%//%*                    Written Date: 04/01/2022                           *%
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
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
warning('off','all');

%% Get inputs
%--------------------------------------------------------------------------
% Inputs file
%--------------------------------------------------------------------------
inputs;

%% Parallel workers start
%--------------------------------------------------------------------------
% Parallel pools
%--------------------------------------------------------------------------
if(isempty(gcp('nocreate')))
    if strcmp(input.poolType,'numcores')
        parpool(input.numCores);
    else
        parpool('Threads')
    end
end

Start = tic;
warning('off','all')

%% Get image filenames and store image names
%--------------------------------------------------------------------------
% Image sets
%--------------------------------------------------------------------------
imgFolders = dir(fullfile(folderPath,folderName));
imgFolders = imgFolders([imgFolders(:).isdir]);
imgFolders = imgFolders(~ismember({imgFolders(:).name},{'.','..'}));

% Dataset name and folders length
datasetName  = {imgFolders.name}';
foldersLen = length(datasetName);

%% Panorama stitcher
%--------------------------------------------------------------------------
% Stitches panoramas
%--------------------------------------------------------------------------
for myImg = 25 %20 %1:foldersLen
    stitchStart = tic;
    fprintf('Image number: %i | Current folder: %s\n', myImg, imgFolders(myImg).name);
    
    %% Load images
    loadimagestic = tic;
    [keypoints, allDescriptors, images, imageSizes, imageNames, numImg] = loadImages(input, imgFolders, myImg);
    fprintf('Loading images (+ features): %f seconds\n', toc(loadimagestic));

    %% Get feature matrices and keypoints    
    featureMatchtic = tic;
    matches = featureMatching(input, allDescriptors, numImg);
    fprintf('Matching features : %f seconds\n', toc(featureMatchtic));
    
    %% Find matches        
    imageMatchtic = tic;
    [allMatches, numMatches, initialTforms] = imageMatching(input, numImg, keypoints, matches, images);
    fprintf('Matching images: %f seconds\n', toc(imageMatchtic));               

    %% Bundle adjustment
    bALMtic = tic;
    [bundlerTforms, concomps, imageNeighbors, finalrefIdxs] = bundleAdjustmentLM(input, matches, keypoints, imageSizes, initialTforms, numMatches);
    fprintf('Final alignment (Bundle adjustment): %f seconds\n', toc(bALMtic));  
    
    %% Automatic panorama straightening
    straightentic = tic;
    finalPanoramaTforms = straightening({bundlerTforms});    
    fprintf('Automatic panorama straightening: %f seconds\n', toc(straightentic));

    %% Render panoramas
    rendertic = tic;    
    [allPanoramas, croppedPanoramas] = displayPanorama(input, finalPanoramaTforms, finalrefIdxs, images, ...
                                                       myImg, datasetName);
    fprintf('Rendering panorama : %f seconds\n', toc(rendertic));
    fprintf('Total runtime (stitching) : %f seconds\n', toc(stitchStart));
    fprintf('--------------------------------\n\n', toc); %#ok<CTPCT>        
end

%% End parameters
%--------------------------------------------------------------------------
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
statusFclose = fclose('all');
if(statusFclose == 0)
    disp('All files are closed.')
end
Runtime = toc(Start);
fprintf('Total runtime : %f seconds\n', Runtime);
currtime = datetime('now');
display(currtime)
