function [allMatches, numMatches, tforms] = imageMatching(input, n, keypoints, matchesAll, images)

    % Initialize
    allMatches = cell(n);
    numMatches = zeros(n);
    matSize = size(allMatches);

    % Use symmetry and run for upper triangular matrix
    linearIdxs = reshape(1:numel(allMatches), size(allMatches));
    IuptriIdx = nonzeros(triu(linearIdxs, 1));
    IlowtriIdx = nonzeros(triu(linearIdxs', 1));

    % Initialize
    allMatches_temp = cell(1, length(IuptriIdx));
    numMatches_temp = zeros(1, length(IuptriIdx));

    % Initialize transformation matrix
    tforms = cell(n, n);
    tforms_ij = cell(1, length(IuptriIdx));
    tforms_ji = cell(1, length(IuptriIdx));

    % Minimum number of features/keypoints
    if strcmp(input.transformationType, 'translation')
        nfmin = 1;
    elseif strcmp(input.transformationType, 'rigid') || strcmp(input.transformationType, 'similarity')
        nfmin = 2;
    elseif strcmp(input.transformationType, 'affine')
        nfmin = 3;
    else
        nfmin = 4;
    end

    % Match images
    parfor i = 1:length(IuptriIdx)

        % IND2SUB converts from a "linear" index into individual
        % subscripts
        [ii, jj] = ind2sub(matSize, IuptriIdx(i));

        % Keypoints matches
        matches = matchesAll{ii, jj};

        % Number of features
        nf = size(matches, 2);

        % Image matching
        % Filter matches using RANSAC (model maps keypt i to keypt j)
        if nf >= nfmin
            [inliers, model] = refineMatch(input, keypoints{ii}, keypoints{jj}, matches, ...
                images{ii}, images{jj});

            if input.useMATLABImageMatching == 1
                model = model.A;
            end

            % Number of inliers
            ni = length(inliers);

            % Verify image matches using probabilistic model
            if strcmp(input.transformationType, 'rigid') || ...
                    strcmp(input.transformationType, 'similarity') || ...
                    strcmp(input.transformationType, 'translation')

                if ni > 5 + 0.025 * nf %2 % accept as correct image match
                    allMatches_temp{i} = matches(:, inliers);
                    numMatches_temp(i) = ni;
                    tforms_ij{i} = model;
                    tforms_ji{i} = inv(model);
                end

            elseif strcmp(input.transformationType, 'affine')

                if ni > 5 + 0.15 * nf %3 % accept as correct image match
                    allMatches_temp{i} = matches(:, inliers);
                    numMatches_temp(i) = ni;
                    tforms_ij{i} = model;
                    tforms_ji{i} = inv(model);
                end

            else

                if ni > 8 + 0.3 * nf
                    allMatches_temp{i} = matches(:, inliers);
                    numMatches_temp(i) = ni;
                    tforms_ij{i} = model;
                    tforms_ji{i} = inv(model);
                end

            end

        end

    end

    % Populate all matches symmetric matrix
    allMatches(IuptriIdx) = allMatches_temp;

    % Populate number of matches symmetric matrix
    numMatches(IuptriIdx) = numMatches_temp;

    % Populate transformation matrix
    tforms(IuptriIdx) = tforms_ij;
    tforms(IlowtriIdx) = tforms_ji;
end

%--------------------------------------------------------------------------------------------------------
% Auxillary functions
%--------------------------------------------------------------------------------------------------------
% [inliers, model] = refineMatch(P1, P2, matches)
%
% Returns the set of inliers and corresponding homography that maps matched
% points from P1 to P2.
function [inliers, model] = refineMatch(input, P1, P2, matches, image1, image2)
    % Matched points
    matchedPts_1 = P1(:, matches(1, :))';
    matchedPts_2 = P2(:, matches(2, :))';

    % Estimate the transformation using the RANSAC or MLESAC
    if input.useMATLABImageMatching == 1
        % Estimate the transformation matrix
        [model, inliers, status] = estgeotform2d(matchedPts_2, matchedPts_1, input.transformationType, ...
            'Confidence', input.inliersConfidence, 'MaxNumTrials', input.maxIter, ...
            'MaxDistance', input.maxDistance);
    else

        switch input.imageMatchingMethod
            case 'ransac'
                [model, inliers] = estimateTransformationRANSAC(matchedPts_2, matchedPts_1, ...
                    input.transformationType, input);
            case 'mlesac'
                [model, inliers] = estimateTransformationMLESAC(matchedPts_2, matchedPts_1, ...
                    input.transformationType, input);
            otherwise
                error('Valid image matching method is required.')
        end

    end

    % Find inliers
    inliers = find(inliers);

    % Plot of the feature matches
    if (input.showKeypointsPlot == 1)
        figure(1);
        subplot(1, 3, 1)
        montage({image1, image2});
        title('Original Images')

        subplot(1, 3, 2)
        showMatchedFeatures(image1, image2, matchedPts_1, matchedPts_2, 'montage')
        title('Putative Matches')

        subplot(1, 3, 3)
        showMatchedFeatures(image1, image2, matchedPts_1(inliers, :), matchedPts_2(inliers, :), 'montage')
        title('Inlier Matches')
    end

end
