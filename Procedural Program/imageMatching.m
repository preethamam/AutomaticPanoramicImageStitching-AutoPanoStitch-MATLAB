function [allMatches, numMatches, tforms] = imageMatching(input, n, keypoints, matchesAll, imagesProcessed)
    %IMAGEMATCHING Verify pairwise image matches and estimate pair transforms.
    %
    % Syntax
    %   [allMatches, numMatches, tforms] = imageMatching(input, n, keypoints, matchesAll, images)
    %
    % Description
    %   Consumes putative keypoint matches between image pairs and filters them
    %   using a robust geometric model (RANSAC or MLESAC) to produce verified
    %   inlier correspondences and pairwise transforms. Only the upper-triangular
    %   (i<j) pairs are processed in parallel and used to populate outputs:
    %   - allMatches{i,j}: 2-by-K indices of inlier matches
    %   - numMatches(i,j): inlier counts
    %   - tforms{i,j}:     model mapping points in image i to image j
    %   The inverse transform is placed at tforms{j,i} for convenience.
    %
    % Inputs
    %   input     - Struct controlling model type and thresholds. Expected fields include:
    %               • transformationType: 'translation'|'rigid'|'similarity'|'affine'|'projective'
    %               • useMATLABImageMatching (logical): use estgeotform2d if true
    %               • imageMatchingMethod: 'ransac'|'mlesac' (when not using MATLAB)
    %               • inliersConfidence, maxIter, maxDistance (RANSAC/MLESAC params)
    %               • showKeypointsPlot (logical): visualize matches
    %   n         - Number of images.
    %   keypoints - 1-by-n cell array of 2-by-M keypoint coordinates per image.
    %   matchesAll- n-by-n cell array; matchesAll{i,j} is 2-by-K putative match indices.
    %   images    - 1-by-n cell array of images used for optional visualization.
    %
    % Outputs
    %   allMatches - n-by-n cell array; for i<j, 2-by-K inlier index pairs.
    %   numMatches - n-by-n numeric matrix; for i<j, inlier counts.
    %   tforms     - n-by-n cell array; tforms{i,j} is the i->j transform, and
    %                tforms{j,i} its inverse.
    %
    % Notes
    %   - Minimal required correspondences depend on the model: 1 (translation),
    %     2 (rigid/similarity), 3 (affine), 4 (projective).
    %   - Acceptance thresholds for inlier counts (ni) are model-dependent and scale
    %     with the number of putative matches (nf):
    %       • rigid/similarity/translation: ni > 5 + 0.025*nf
    %       • affine:                        ni > 5 + 0.15*nf
    %       • projective:                    ni > 8 + 0.3*nf
    %   - When input.useMATLABImageMatching is true, the transform returned by
    %     estgeotform2d is converted to a 3x3 matrix via its A property.
    %
    % See also: estgeotform2d, estimateTransformationRANSAC, estimateTransformationMLESAC,
    %           showMatchedFeatures, montage, parfor, triu, ind2sub

    arguments
        input (1, 1) struct
        n (1, 1) double {mustBeInteger, mustBePositive}
        keypoints cell
        matchesAll cell
        imagesProcessed cell
    end

    % Basic shape validations that depend on runtime values
    if ~isequal(size(matchesAll), [n, n])
        error('imageMatching:InvalidMatchesAllSize', 'matchesAll must be an n-by-n cell array.');
    end

    if numel(keypoints) ~= n
        error('imageMatching:InvalidKeypointsLength', 'keypoints must contain n elements (one per image).');
    end

    if numel(imagesProcessed) ~= n
        error('imageMatching:InvalidImagesLength', 'images must contain n elements (one per image).');
    end

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
        nf = size(matches, 1);

        % Image matching
        % Filter matches using RANSAC (model maps keypt i to keypt j)
        if nf >= nfmin
            [inliers, model] = refineMatch(input, keypoints{ii}, keypoints{jj}, matches, ...
                imagesProcessed{ii}, imagesProcessed{jj});

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
                    allMatches_temp{i} = matches(inliers, :);
                    numMatches_temp(i) = ni;
                    tforms_ij{i} = model;
                    tforms_ji{i} = inv(model);
                end

            elseif strcmp(input.transformationType, 'affine')

                if ni > 5 + 0.15 * nf %3 % accept as correct image match
                    allMatches_temp{i} = matches(inliers, :);
                    numMatches_temp(i) = ni;
                    tforms_ij{i} = model;
                    tforms_ji{i} = inv(model);
                end

            else

                if ni > 8 + 0.3 * nf
                    allMatches_temp{i} = matches(inliers, :);
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
    %REFINEMATCH Robustly fit geometric model and return inlier indices.
    %
    % Syntax
    %   [inliers, model] = refineMatch(input, P1, P2, matches, image1, image2)
    %
    % Description
    %   Given candidate matches between keypoints P1 (image i) and P2 (image j),
    %   estimate a geometric transform mapping P1->P2 using either MATLAB's
    %   estgeotform2d or custom RANSAC/MLESAC, then return the indices of inliers
    %   among the putative matches. Optionally visualizes original, putative, and
    %   inlier matches when input.showKeypointsPlot is enabled.
    %
    % Inputs
    %   input   - Parameter struct (see imageMatching). Key fields used here:
    %             • useMATLABImageMatching (logical)
    %             • transformationType, inliersConfidence, maxIter, maxDistance
    %             • imageMatchingMethod: 'ransac'|'mlesac' (if not using MATLAB)
    %             • showKeypointsPlot (logical)
    %   P1      - 2-by-M keypoints from image i.
    %   P2      - 2-by-M keypoints from image j.
    %   matches - 2-by-K putative index pairs referencing columns in P1 and P2.
    %   image1  - Image i (used only for visualization).
    %   image2  - Image j (used only for visualization).
    %
    % Outputs
    %   inliers - Linear indices into the columns of P1/P2 that are inlier matches.
    %   model   - 3-by-3 transform matrix mapping P1->P2.
    %
    % Notes
    %   - When using estgeotform2d, the returned object is converted to model.A.
    %   - For custom matching, the selected method is estimateTransformationRANSAC
    %     or estimateTransformationMLESAC depending on input.imageMatchingMethod.
    %   - Visualization uses montage and showMatchedFeatures.
    arguments
        input (1, 1) struct
        P1 (:, 2) double
        P2 (:, 2) double
        matches (:, 2) double {mustBeInteger, mustBePositive}
        image1
        image2
    end

    % Validate index ranges if matches are present
    if ~isempty(matches)

        if max(matches(:, 1)) > size(P1, 1) || max(matches(:, 2)) > size(P2, 1)
            error('refineMatch:MatchIndexOutOfBounds', 'Match indices exceed keypoint array sizes.');
        end

    end

    % Matched points
    matchedPts_1 = P1(matches(:, 1), :);
    matchedPts_2 = P2(matches(:, 2), :);

    % Estimate the transformation using the RANSAC or MLESAC
    if input.useMATLABImageMatching == 1
        % Estimate the transformation matrix
        [model, inliers, ~] = estgeotform2d(matchedPts_2, matchedPts_1, input.transformationType, ...
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
