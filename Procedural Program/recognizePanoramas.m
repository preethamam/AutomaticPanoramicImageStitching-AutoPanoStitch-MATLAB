function [bundlerTformsAll, finalrefIdxsAll, panoIndices, concomps, panaromaCCs, connCompsNumber] = recognizePanoramas(input, ...
                numMatches, matchesAll, keypointsAll, imageSizesAll, initialTformsAll)
        
        
    % Find connected components of image matches
    numMatchesG = graph(numMatches, 'upper');
    [concomps, ccBinSizes] = conncomp(numMatchesG);
    panaromaCCsAll = find(ccBinSizes >= 1);
    panaromaCCs = find(ccBinSizes > 1);
    connCompsNumber = numel(panaromaCCsAll);
    
    % Initialize
    bundlerTformsAll = cell(numel(panaromaCCs),1);
    finalrefIdxsAll = zeros(numel(panaromaCCs),1);
    panoIndices = cell(numel(panaromaCCs),1);

    for i = 1:connCompsNumber
        idxs = find(concomps == i);
        if isscalar(idxs)
            % Skip bundle adjustment for single-image panorama
            fprintf('Skipping bundle adjustment as only one image found.\n');
            continue;
        end
        
        % Store panorama image indices
        panoIndices{i} = idxs;

        % Get sub-matches keypoints, image sizes, matcehd and initial
        % transformation matrices
        keypoints = keypointsAll(idxs);
        imageSizes = imageSizesAll(idxs,:);
        matches = matchesAll(idxs, idxs);
        initialTforms = initialTformsAll(idxs, idxs);

        % Perform bundle adjustment
        BAtic = tic;
        [bundlerTforms, finalrefIdxs] = bundleAdjustmentLM(input, matches, keypoints, ...
            imageSizes, initialTforms, ...
            'MaxLMIters', input.maxIterLM, 'Lambda0', input.lambda, ...
            'SigmaHuber', input.sigmaHuber, 'Verbose', input.verboseLM);

        fprintf('Final alignment (Bundle adjustment): %f seconds\n', toc(BAtic));
        
        % Store panoramas R, K and f and reference indexes
        bundlerTformsAll{i} = bundlerTforms;
        finalrefIdxsAll(i) = finalrefIdxs;
    end
end