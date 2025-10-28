function [model, inliers, isFound] = estimateTransformationRANSAC(matchedPoints1, matchedPoints2, transformType, input)
    % Suppress warnings about matrix singularity
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');

    % Ensure warnings are restored when function exits
    cleanupObj = onCleanup(@() warning('on', 'all'));

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
    [Ur, Sr, Vr] = svd(R);
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
    % Check if transformation matrix is valid
    tf = all(isfinite(H(:))) && ... % All elements are finite
        rcond(H) > eps && ... % Matrix is well-conditioned
        abs(det(H)) > eps; % Non-zero determinant
end

function isDegen = isDegenerate(points, transformType)

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
