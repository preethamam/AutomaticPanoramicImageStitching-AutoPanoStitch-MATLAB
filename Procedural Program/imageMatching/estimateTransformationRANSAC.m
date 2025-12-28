function [model, inliers, isFound] = estimateTransformationRANSAC(matchedPoints1, matchedPoints2, transformType, input)
    %ESTIMATETRANSFORMATIONRANSAC Robust 2-D transform estimation using RANSAC.
    %
    % Syntax
    %   [model, inliers, isFound] = estimateTransformationRANSAC(matchedPoints1, matchedPoints2, transformType)
    %   [model, inliers, isFound] = estimateTransformationRANSAC(matchedPoints1, matchedPoints2, transformType, input)
    %
    % Description
    %   Estimates a geometric transformation between matched 2-D point sets using
    %   a RANSAC loop with model-specific minimal samples. Supports transform types:
    %   'translation', 'rigid', 'similarity', 'affine', and 'projective'. Returns
    %   the 3x3 transform matrix, a logical inlier vector, and a success flag.
    %
    % Inputs
    %   matchedPoints1 - M-by-2 double array of [x y] points (source).
    %   matchedPoints2 - M-by-2 double array of [x y] points (destination).
    %   transformType  - Char/string: 'translation'|'rigid'|'similarity'|'affine'|'projective'.
    %   input          - Optional struct with fields:
    %                    • maxDistance (scalar pixels)         - Inlier threshold.
    %                    • inliersConfidence (0-100)           - Desired confidence (%).
    %                    • maxIter (integer)                   - Max RANSAC iterations.
    %                    • recomputeFromInliers (logical)      - Refit using all inliers (default true).
    %
    % Outputs
    %   model   - 3-by-3 homogeneous transform matrix.
    %   inliers - M-by-1 logical vector marking inlier correspondences.
    %   isFound - Logical flag indicating if a valid model was found.
    %
    % Notes
    %   - Minimal sample size depends on the model: 1 (translation), 2 (rigid/similarity),
    %     3 (affine), 4 (projective).
    %   - Internally adapts iteration budget based on observed inlier ratio.
    %   - Uses normalized DLT forms for affine/projective; median-based estimate for translation.

    arguments
        matchedPoints1 (:, 2) double
        matchedPoints2 (:, 2) double
        transformType {mustBeTextScalar}
        input struct = struct()
    end

    % Suppress warnings about matrix singularity
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');

    % Ensure warnings are restored when function exits
    cleanupObj = onCleanup(@() warning('on', 'all'));

    % Early consistency check
    if size(matchedPoints1, 1) ~= size(matchedPoints2, 1)
        error('estimateTransformationRANSAC:PointCountMismatch', 'matchedPoints1 and matchedPoints2 must have the same number of rows.');
    end

    % Set default parameters if not provided
    if nargin < 4
        maxDistance = 2.0;
        confidence = 99.9;
        maxTrials = 500;
        maxSkipTrials = maxTrials * 10;
        recomputeFromInliers = true;
    else
        maxDistance = input.maxDistance;
        confidence = input.inliersConfidence;
        maxTrials = input.maxIter;
        maxSkipTrials = maxTrials * 10;
        recomputeFromInliers = true;
    end

    % Get parameters for transform type
    params = getTransformParams(transformType);
    minPoints = params.minPoints;

    % Check if we have enough points
    if size(matchedPoints1, 1) < minPoints
        model = [];
        inliers = false(size(matchedPoints1, 1), 1);
        isFound = false;
        return;
    end

    % Precompute homogeneous coordinates
    numPoints = size(matchedPoints1, 1);
    pts1_homog = [matchedPoints1, ones(numPoints, 1)];
    pts2_homog = [matchedPoints2, ones(numPoints, 1)];

    % Initialize variables
    bestInliers = false(numPoints, 1);
    bestModel = [];
    bestError = inf;
    trial = 1;
    skipTrials = 0;

    % Main RANSAC loop
    while trial <= maxTrials && skipTrials < maxSkipTrials
        % Random selection without replacement
        sampleIdx = randperm(numPoints, minPoints);
        sample1 = matchedPoints1(sampleIdx, :);
        sample2 = matchedPoints2(sampleIdx, :);

        % Estimate transform from minimal set
        try
            H = estimateTransform(sample1, sample2, params);

            % Check if model is valid
            if ~checkModel(H)
                skipTrials = skipTrials + 1;
                continue;
            end

            % Evaluate model and find inliers
            [currentInliers, errors] = findInliers(H, pts1_homog, pts2_homog, maxDistance, params);
            numInliers = sum(currentInliers);

            % Update best model if better
            if numInliers >= minPoints
                meanError = mean(errors(currentInliers));

                if numInliers > sum(bestInliers) || ...
                        (numInliers == sum(bestInliers) && meanError < bestError)
                    bestInliers = currentInliers;
                    bestModel = H;
                    bestError = meanError;

                    % Update number of trials based on inlier ratio
                    inlierRatio = numInliers / numPoints;
                    % Avoid division by zero or log of zero
                    if inlierRatio > 0
                        maxTrials = min(maxTrials, ...
                            ceil(log(1 - confidence / 100) / log(1 - inlierRatio ^ minPoints)));
                    end

                end

            end

            trial = trial + 1;

        catch
            skipTrials = skipTrials + 1;
            continue;
        end

    end

    % Final step: recompute model using all inliers if found
    if sum(bestInliers) >= minPoints && recomputeFromInliers

        try
            inlierPts1 = matchedPoints1(bestInliers, :);
            inlierPts2 = matchedPoints2(bestInliers, :);
            model = estimateTransform(inlierPts1, inlierPts2, params);

            % Verify final model
            if checkModel(model)
                [inliers, ~] = findInliers(model, pts1_homog, pts2_homog, maxDistance, params);

                if sum(inliers) >= minPoints
                    isFound = true;
                else
                    model = bestModel;
                    inliers = bestInliers;
                    isFound = true;
                end

            else
                model = bestModel;
                inliers = bestInliers;
                isFound = sum(bestInliers) >= minPoints;
            end

        catch
            model = bestModel;
            inliers = bestInliers;
            isFound = sum(bestInliers) >= minPoints;
        end

    else
        model = bestModel;
        inliers = bestInliers;
        isFound = sum(bestInliers) >= minPoints;
    end

end

%--------------------------------------------------------------------------
% Transformation estimation functions
%--------------------------------------------------------------------------
function H = estimateHomography(pts1, pts2)
    %ESTIMATEHOMOGRAPHY Estimate 3x3 projective transform via normalized DLT.
    %
    % Syntax
    %   H = estimateHomography(pts1, pts2)
    %
    % Inputs
    %   pts1 - N-by-2 source points.
    %   pts2 - N-by-2 destination points.
    %
    % Output
    %   H    - 3-by-3 homography matrix.

    arguments
        pts1 (:, 2) double
        pts2 (:, 2) double
    end

    % Homography estimation using normalized DLT
    [pts1_norm, T1] = normalizePoints(pts1);
    [pts2_norm, T2] = normalizePoints(pts2);

    n = size(pts1_norm, 1);
    x = pts1_norm(:, 1); y = pts1_norm(:, 2);
    u = pts2_norm(:, 1); v = pts2_norm(:, 2);

    A = [
         -x, -y, -ones(n, 1), zeros(n, 3), x .* u, y .* u, u;
         zeros(n, 3), -x, -y, -ones(n, 1), x .* v, y .* v, v
         ];

    [~, ~, V] = svd(A);
    h = V(:, end);
    H_norm = reshape(h, 3, 3)';

    % Denormalize
    H = T2 \ (H_norm / H_norm(3, 3)) * T1;
end

function H = estimateAffine(pts1, pts2)
    %ESTIMATEAFFINE Estimate 2-D affine transform using normalized DLT.
    %
    % Syntax
    %   H = estimateAffine(pts1, pts2)
    %
    % Inputs
    %   pts1 - N-by-2 source points.
    %   pts2 - N-by-2 destination points.
    %
    % Output
    %   H    - 3-by-3 affine transform matrix.

    arguments
        pts1 (:, 2) double
        pts2 (:, 2) double
    end

    % Affine estimation using normalized DLT with improved constraints
    [pts1_norm, T1] = normalizePoints(pts1);
    [pts2_norm, T2] = normalizePoints(pts2);

    n = size(pts1_norm, 1);
    x = pts1_norm(:, 1); y = pts1_norm(:, 2);
    u = pts2_norm(:, 1); v = pts2_norm(:, 2);

    % Build the design matrix A with improved constraints
    A = [
         x, y, ones(n, 1), zeros(n, 3);
         zeros(n, 3), x, y, ones(n, 1)
         ];

    b = [u; v];

    % Solve using SVD for better numerical stability
    [U, S, V] = svd(A, 'econ'); % Use economy size SVD

    % Apply condition number threshold
    s = diag(S);
    cond_threshold = 1e-10;
    s(s < cond_threshold * s(1)) = 0;
    s_inv = zeros(size(s));
    s_inv(s > 0) = 1 ./ s(s > 0);

    % Solve the system with proper dimensions
    h = V * (diag(s_inv) * (U' * b));

    % Construct affine matrix
    H_norm = [
              h(1), h(2), h(3);
              h(4), h(5), h(6);
              0, 0, 1
              ];

    % Denormalize
    H = T2 \ H_norm * T1;

    % Ensure exact affine form
    H(3, 1:2) = 0;
    H(3, 3) = 1;
end

function H = estimateSimilarity(pts1, pts2)
    %ESTIMATESIMILARITY Estimate similarity transform (scale, rotation, translation).
    %
    % Syntax
    %   H = estimateSimilarity(pts1, pts2)
    %
    % Inputs
    %   pts1 - N-by-2 source points.
    %   pts2 - N-by-2 destination points.
    %
    % Output
    %   H    - 3-by-3 similarity transform matrix.

    arguments
        pts1 (:, 2) double
        pts2 (:, 2) double
    end

    % Similarity estimation with improved normalization and constraints
    [pts1_norm, T1] = normalizePoints(pts1);
    [pts2_norm, T2] = normalizePoints(pts2);

    % Center points
    centroid1 = mean(pts1_norm);
    centroid2 = mean(pts2_norm);

    pts1_centered = bsxfun(@minus, pts1_norm, centroid1);
    pts2_centered = bsxfun(@minus, pts2_norm, centroid2);

    % Compute optimal rotation
    [U, ~, V] = svd(pts2_centered' * pts1_centered);
    R = V * [1 0; 0 det(V * U')] * U';

    % Compute scale using multiple methods and take median
    scales = [];
    % Method 1: Frobenius norm ratio
    scales(1) = norm(pts2_centered, 'fro') / norm(pts1_centered, 'fro');
    % Method 2: Point-wise scale ratios
    pts1_norm_centered = sqrt(sum(pts1_centered .^ 2, 2));
    pts2_norm_centered = sqrt(sum(pts2_centered .^ 2, 2));
    valid_ratios = pts1_norm_centered > 1e-10;

    if any(valid_ratios)
        point_scales = pts2_norm_centered(valid_ratios) ./ pts1_norm_centered(valid_ratios);
        scales(2) = median(point_scales);
    end

    % Take median of scales
    s = median(scales);

    % Compute translation
    t = centroid2' - s * R * centroid1';

    % Construct similarity transform
    H_norm = [s * R, t; 0 0 1];
    H = T2 \ H_norm * T1;

    % Ensure exact similarity form
    H(3, 1:2) = 0;
    H(3, 3) = 1;
end

function H = estimateRigid(pts1, pts2)
    %ESTIMATERIGID Estimate rigid transform (rotation + translation).
    %
    % Syntax
    %   H = estimateRigid(pts1, pts2)
    %
    % Inputs
    %   pts1 - N-by-2 source points.
    %   pts2 - N-by-2 destination points.
    %
    % Output
    %   H    - 3-by-3 rigid transform matrix.

    arguments
        pts1 (:, 2) double
        pts2 (:, 2) double
    end

    % Rigid transform estimation with improved robustness
    [pts1_norm, T1] = normalizePoints(pts1);
    [pts2_norm, T2] = normalizePoints(pts2);

    % Center points
    centroid1 = mean(pts1_norm);
    centroid2 = mean(pts2_norm);

    pts1_centered = bsxfun(@minus, pts1_norm, centroid1);
    pts2_centered = bsxfun(@minus, pts2_norm, centroid2);

    % Compute optimal rotation with additional checks
    [U, S, V] = svd(pts2_centered' * pts1_centered);

    % Check condition number
    singular_values = diag(S);
    condition_number = singular_values(1) / max(singular_values(2), eps);

    if condition_number > 1e6
        % Poor conditioning, possibly degenerate configuration
        R = eye(2);
    else
        R = V * [1 0; 0 det(V * U')] * U';
    end

    % Ensure exact orthogonality
    [Ur, ~, Vr] = svd(R);
    R = Ur * Vr';

    % Compute translation
    t = centroid2' - R * centroid1';

    % Construct rigid transform
    H_norm = [R, t; 0 0 1];
    H = T2 \ H_norm * T1;

    % Ensure exact rigid form
    H(3, 1:2) = 0;
    H(3, 3) = 1;
end

function H = estimateTranslation(pts1, pts2)
    %ESTIMATETRANSLATION Estimate pure translation using median displacement.
    %
    % Syntax
    %   H = estimateTranslation(pts1, pts2)
    %
    % Inputs
    %   pts1 - N-by-2 source points.
    %   pts2 - N-by-2 destination points.
    %
    % Output
    %   H    - 3-by-3 translation transform matrix.

    arguments
        pts1 (:, 2) double
        pts2 (:, 2) double
    end

    % Translation estimation without normalization for large scale translations
    translations = pts2 - pts1;

    % Use median for robustness, separate for x and y
    tx = median(translations(:, 1));
    ty = median(translations(:, 2));

    % Build transformation matrix
    H = [1, 0, tx;
         0, 1, ty;
         0, 0, 1];
end

%--------------------------------------------------------------------------
% Inlier function
%--------------------------------------------------------------------------
function [inliers, errors] = findInliers(H, pts1_homog, pts2_homog, threshold, params)
    %FINDINLIERS Compute inliers and residuals for a candidate transform.
    %
    % Syntax
    %   [inliers, errors] = findInliers(H, pts1_homog, pts2_homog, threshold, params)
    %
    % Inputs
    %   H           - 3-by-3 transform matrix.
    %   pts1_homog  - N-by-3 homogeneous source points.
    %   pts2_homog  - N-by-3 homogeneous destination points.
    %   threshold   - Scalar inlier threshold (pixels; normalized for translation path).
    %   params      - Struct with fields 'type' and 'minPoints'.
    %
    % Outputs
    %   inliers     - N-by-1 logical vector of inliers.
    %   errors      - N-by-1 residual distances.

    arguments
        H (3, 3) double
        pts1_homog (:, 3) double
        pts2_homog (:, 3) double
        threshold (1, 1) double
        params (1, 1) struct
    end

    % Transform points
    transformed = (H * pts1_homog')';
    transformed = bsxfun(@rdivide, transformed, transformed(:, 3));

    switch params.type
        case 'projective'
            % Compute symmetric transfer error
            invHpts2 = (H \ pts2_homog')';
            invHpts2 = bsxfun(@rdivide, invHpts2, invHpts2(:, 3));

            d1 = sum((pts2_homog(:, 1:2) - transformed(:, 1:2)) .^ 2, 2);
            d2 = sum((pts1_homog(:, 1:2) - invHpts2(:, 1:2)) .^ 2, 2);
            errors = sqrt(d1 + d2);

        case {'affine', 'similarity', 'rigid'}
            errors = sqrt(sum((pts2_homog(:, 1:2) - transformed(:, 1:2)) .^ 2, 2));

        case 'translation'
            scale = max(abs([pts1_homog(:); pts2_homog(:)]));

            if scale > eps
                errors = sqrt(sum((pts2_homog(:, 1:2) - transformed(:, 1:2)) .^ 2, 2)) / scale;
                threshold = threshold / scale;
            else
                errors = sqrt(sum((pts2_homog(:, 1:2) - transformed(:, 1:2)) .^ 2, 2));
            end

    end

    % Handle numerical instabilities
    errors(~isfinite(errors)) = inf;
    errors(abs(transformed(:, 3)) < eps) = inf;

    % Determine inliers
    inliers = errors < threshold;

    % Check for degenerate configurations
    if sum(inliers) >= params.minPoints && ...
            (strcmp(params.type, 'projective') || strcmp(params.type, 'affine'))

        if isDegenerate(pts1_homog(inliers, 1:2), params.type)
            inliers(:) = false;
            errors(:) = inf;
        end

    end

end

function tf = checkModel(H)
    %CHECKMODEL Validate transformation matrix properties.
    %
    % Input
    %   H - 3-by-3 transform matrix.
    %
    % Output
    %   tf - Logical true if H is finite, well-conditioned, and non-singular.

    arguments
        H (3, 3) double
    end

    % Check if transformation matrix is valid
    tf = all(isfinite(H(:))) && ... % All elements are finite
        rcond(H) > eps && ... % Matrix is well-conditioned
        abs(det(H)) > eps; % Non-zero determinant
end

function isDegen = isDegenerate(points, transformType)
    %ISDEGENERATE Detect degenerate point configurations for a model.
    %
    % Syntax
    %   isDegen = isDegenerate(points, transformType)
    %
    % Inputs
    %   points         - N-by-2 points.
    %   transformType  - Char/string model: 'projective'|'affine'|...
    %
    % Output
    %   isDegen        - Logical true if configuration is degenerate for model.

    arguments
        points (:, 2) double
        transformType {mustBeTextScalar}
    end

    if size(points, 1) < 3
        isDegen = true;
        return;
    end

    % Center points
    centeredPoints = points - mean(points, 1);
    [~, S, ~] = svd(centeredPoints, 'econ');
    singularValues = diag(S);

    switch transformType
        case 'projective'
            isDegen = singularValues(2) / singularValues(1) < 1e-3;
        case 'affine'
            isDegen = singularValues(2) / singularValues(1) < 1e-3;
        otherwise
            isDegen = false;
    end

end

%--------------------------------------------------------------------------
% Normalization functions
%--------------------------------------------------------------------------
function [pts_norm, T] = normalizePoints(pts)
    %NORMALIZEPOINTS Normalize 2-D points for numerical stability.
    %
    % Syntax
    %   [pts_norm, T] = normalizePoints(pts)
    %
    % Input
    %   pts      - N-by-2 inhomogeneous points.
    %
    % Outputs
    %   pts_norm - N-by-2 normalized points (mean distance ~= 1).
    %   T        - 3-by-3 normalization matrix.

    arguments
        pts (:, 2) double
    end

    % Normalize points for improved numerical stability
    centroid = mean(pts);
    pts_centered = bsxfun(@minus, pts, centroid);

    % Use mean distance for scale instead of RMS distance
    scale = 1 / mean(sqrt(sum(pts_centered .^ 2, 2)));

    T = [scale, 0, -scale * centroid(1);
         0, scale, -scale * centroid(2);
         0, 0, 1];

    pts_homog = [pts, ones(size(pts, 1), 1)];
    pts_norm_homog = (T * pts_homog')';
    pts_norm = pts_norm_homog(:, 1:2);
end

%--------------------------------------------------------------------------
% Helper functions
%--------------------------------------------------------------------------
function params = getTransformParams(transformType)
    %GETTRANSFORMPARAMS Return minimal points and metadata for a model.
    %
    % Syntax
    %   params = getTransformParams(transformType)
    %
    % Input
    %   transformType - Char/string specifying the transform model.
    %
    % Output
    %   params - Struct with fields: minPoints, type, dof.

    arguments
        transformType {mustBeTextScalar}
    end

    % Returns parameters for different transform types
    switch lower(transformType)
        case 'projective'
            params.minPoints = 4;
            params.type = 'projective';
            params.dof = 8;
        case 'affine'
            params.minPoints = 3;
            params.type = 'affine';
            params.dof = 6;
        case 'similarity'
            params.minPoints = 2;
            params.type = 'similarity';
            params.dof = 4;
        case 'rigid'
            params.minPoints = 2;
            params.type = 'rigid';
            params.dof = 3;
        case 'translation'
            params.minPoints = 1;
            params.type = 'translation';
            params.dof = 2;
        otherwise
            error('Unknown transform type');
    end

end

function H = estimateTransform(pts1, pts2, params)
    %ESTIMATETRANSFORM Dispatch to model-specific estimator.
    %
    % Syntax
    %   H = estimateTransform(pts1, pts2, params)
    %
    % Inputs
    %   pts1   - N-by-2 source points.
    %   pts2   - N-by-2 destination points.
    %   params - Struct with field 'type'.
    %
    % Output
    %   H      - 3-by-3 transform matrix.

    arguments
        pts1 (:, 2) double
        pts2 (:, 2) double
        params (1, 1) struct
    end

    switch params.type
        case 'projective'
            H = estimateHomography(pts1, pts2);
        case 'affine'
            H = estimateAffine(pts1, pts2);
        case 'similarity'
            H = estimateSimilarity(pts1, pts2);
        case 'rigid'
            H = estimateRigid(pts1, pts2);
        case 'translation'
            H = estimateTranslation(pts1, pts2);
    end

end
