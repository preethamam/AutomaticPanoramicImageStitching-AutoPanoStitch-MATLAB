function [tform, inlierIdx, isFound] = estimateTransformationMLESAC(points1, points2, transformationType, input)

    % Set default parameters if not provided
    if nargin < 4
        params = struct();
        mlesacParams = struct();
        params = setDefaultParams(params);

        % Setup MLESAC parameters
        mlesacParams.maxDistance = params.maxDistance;
        mlesacParams.confidence = params.confidence;
        mlesacParams.maxNumTrials = params.maxIterations;
        mlesacParams.maxSkipTrials = params.maxSkipTrials;
        mlesacParams.recomputeModelFromInliers = params.recomputeFromInliers;
    else
        params = struct();
        mlesacParams = struct();
        params = setDefaultParams(params);

        % Setup MLESAC parameters
        mlesacParams.maxDistance = input.maxDistance;
        mlesacParams.confidence = input.inliersConfidence;
        mlesacParams.maxNumTrials = input.maxIter;
        mlesacParams.maxSkipTrials = params.maxSkipTrials;
        mlesacParams.recomputeModelFromInliers = params.recomputeFromInliers;
    end

    % Concatenate 2D points as 3D array
    points = cat(3, points1, points2);

    % Get the corresponding functions
    mlesacFuncs = getMLESACFunctions(transformationType);

    % Minimum number of features
    if strcmp(transformationType, 'translation')
        mlesacParams.sampleSize = 1;
    elseif strcmp(transformationType, 'rigid') || strcmp(transformationType, 'similarity')
        mlesacParams.sampleSize = 2;
    elseif strcmp(transformationType, 'affine')
        mlesacParams.sampleSize = 3;
    else
        mlesacParams.sampleSize = 4;
    end

    % MLESAC motion model transformation and inliers estimation
    [isFound, tform, inlierIdx] = mlesac( ...
        points, mlesacParams, mlesacFuncs);
end

%--------------------------------------------------------------------------
% MLESAC estimation
%--------------------------------------------------------------------------
% MLESAC iterative algorithm
function [isFound, bestModelParams, inliers, reachedMaxSkipTrials] = mlesac( ...
        allPoints, params, funcs, varargin)
    % MSAC M-estimator SAmple Consensus (MSAC) algorithm that is used for point
    % cloud model fitting. allPoints must be an M-by-N matrix, where each point
    % is a row vector.
    %
    % allPoints - M-by-2 or M-by-2-by-2 array of [x y] coordinates
    %
    % params    - struct containing the following fields:
    %               sampleSize
    %               maxDistance
    %               confidence
    %               maxNumTrials
    %
    % funcs     - struct containing the following function handles
    %               fitFunc
    %               evalFunc
    %               checkFunc

    % Copyright 2015-2022 The MathWorks, Inc.
    %
    % References:
    % ----------
    %   P. H. S. Torr and A. Zisserman, "MLESAC: A New Robust Estimator with
    %   Application to Estimating Image Geometry," Computer Vision and Image
    %   Understanding, 2000.

    confidence = params.confidence;
    sampleSize = params.sampleSize;
    maxDistance = params.maxDistance;

    threshold = cast(maxDistance, 'like', allPoints);
    numPts = size(allPoints, 1);
    idxTrial = 1;
    numTrials = int32(params.maxNumTrials);
    maxDis = cast(threshold * numPts, 'like', allPoints);
    bestDis = maxDis;

    if isfield(params, 'maxSkipTrials')
        maxSkipTrials = params.maxSkipTrials;
    else
        maxSkipTrials = params.maxNumTrials * 10;
    end

    skipTrials = 0;
    reachedMaxSkipTrials = false;

    bestInliers = false(numPts, 1);

    while idxTrial <= numTrials && skipTrials < maxSkipTrials
        % Random selection without replacement
        indices = randperm(numPts, sampleSize);

        % Compute a model from samples
        samplePoints = allPoints(indices, :, :);
        modelParams = funcs.fitFunc(samplePoints, varargin{:});

        % Validate the model
        isValidModel = funcs.checkFunc(modelParams, varargin{:});

        if isValidModel
            % Evaluate model with truncated loss
            [model, dis, accDis] = evaluateModel(funcs.evalFunc, modelParams, ...
                allPoints, threshold, varargin{:});

            % Update the best model found so far
            if accDis < bestDis
                bestDis = accDis;
                bestInliers = dis < threshold;
                bestModelParams = model;
                inlierNum = cast(sum(dis < threshold), 'like', allPoints);
                num = vision.internal.ransac.computeLoopNumber(sampleSize, ...
                    confidence, numPts, inlierNum);
                numTrials = min(numTrials, num);
            end

            idxTrial = idxTrial + 1;
        else
            skipTrials = skipTrials + 1;
        end

    end

    if isnumeric(bestModelParams)
        modelParamsToCheck = bestModelParams(:);
    else
        modelParamsToCheck = bestModelParams;
    end

    isFound = funcs.checkFunc(modelParamsToCheck, varargin{:}) && ...
        ~isempty(bestInliers) && sum(bestInliers(:)) >= sampleSize;

    if isFound

        if isfield(params, 'recomputeModelFromInliers') && ...
                params.recomputeModelFromInliers
            modelParams = funcs.fitFunc(allPoints(bestInliers, :, :), varargin{:});
            [bestModelParams, dis] = evaluateModel(funcs.evalFunc, modelParams, ...
                allPoints, threshold, varargin{:});

            if isnumeric(bestModelParams)
                modelParamsToCheck = bestModelParams(:);
            else
                modelParamsToCheck = bestModelParams;
            end

            isValidModel = funcs.checkFunc(modelParamsToCheck, varargin{:});
            inliers = (dis < threshold);

            if ~isValidModel || ~any(inliers)
                isFound = false;
                inliers = false(size(allPoints, 1), 1);
                return;
            end

        else
            inliers = bestInliers;
        end

    else
        inliers = false(size(allPoints, 1), 1);
    end

    reachedMaxSkipTrials = skipTrials >= maxSkipTrials;
end

%--------------------------------------------------------------------------
% MLESAC evaluation
function [modelOut, distances, sumDistances] = evaluateModel(evalFunc, modelIn, ...
        allPoints, threshold, varargin)
    dis = evalFunc(modelIn, allPoints, varargin{:});
    dis(dis > threshold) = threshold;
    accDis = sum(dis);

    if iscell(modelIn)
        [sumDistances, minIdx] = min(accDis);
        distances = dis(:, minIdx);
        modelOut = modelIn{minIdx(1)};
    else
        distances = dis;
        modelOut = modelIn;
        sumDistances = accDis;
    end

end

%--------------------------------------------------------------------------
% MLESAC function handlers
function mlesacFuncs = getMLESACFunctions(tformType)
    mlesacFuncs.checkFunc = @checkTForm;

    switch (tformType)
        case 'rigid'
            mlesacFuncs.fitFunc = @estimateRigid;
            mlesacFuncs.evalFunc = @evaluateTransform2d;
        case 'similarity'
            mlesacFuncs.fitFunc = @estimateSimilarity;
            mlesacFuncs.evalFunc = @evaluateTransform2d;
        case 'affine'
            mlesacFuncs.fitFunc = @estimateAffine;
            mlesacFuncs.evalFunc = @evaluateTransform2d;
        case 'projective'
            mlesacFuncs.fitFunc = @estimateHomography;
            mlesacFuncs.evalFunc = @evaluateTransform2d;
        otherwise % 't'
            mlesacFuncs.fitFunc = @estimateTranslation;
            mlesacFuncs.evalFunc = @evaluateTranslation2d;
    end

end

%--------------------------------------------------------------------------
% Transformation estimation functions
%--------------------------------------------------------------------------
% Homography transformation matrix
function T = estimateHomography(points)
    classToUse = class(points);

    [points1, points2, normMatrix1, normMatrix2] = ...
        normalizePoints(points, classToUse);

    % Get points
    numPts = size(points1, 1);
    p1x = points1(:, 1);
    p1y = points1(:, 2);
    p2x = points2(:, 1);
    p2y = points2(:, 2);

    % DLT contraints
    constraints = zeros(2 * numPts, 9, 'like', points);
    constraints(1:2:2 * numPts, :) = [zeros(numPts, 3), -points1, ...
                                        -ones(numPts, 1), p1x .* p2y, p1y .* p2y, p2y];
    constraints(2:2:2 * numPts, :) = [points1, ones(numPts, 1), ...
                                        zeros(numPts, 3), -p1x .* p2x, -p1y .* p2x, -p2x];

    % SVD
    [~, ~, V] = svd(constraints, 0);
    h = V(:, end);
    T = reshape(h, [3, 3])' / h(9);

    % Denormalize
    T = denormalizeTform(T, normMatrix1, normMatrix2);
end

%--------------------------------------------------------------------------
% Affine transformation matrix
function T = estimateAffine(points)
    classToUse = class(points);

    [points1, points2, normMatrix1, normMatrix2] = ...
        normalizePoints(points, classToUse);

    numPts = size(points1, 1);
    constraints = zeros(2 * numPts, 7, 'like', points);
    constraints(1:2:2 * numPts, :) = [zeros(numPts, 3), -points1, ...
                                        -ones(numPts, 1), points2(:, 2)];
    constraints(2:2:2 * numPts, :) = [points1, ones(numPts, 1), ...
                                        zeros(numPts, 3), -points2(:, 1)];

    [~, ~, V] = svd(constraints, 0);
    h = V(:, end);
    T = eye(3, 'like', points);
    T(1:2, :) = reshape(h(1:6), [3, 2])' / h(7);
    T = denormalizeTform(T, normMatrix1, normMatrix2);
end

%--------------------------------------------------------------------------
% Similarity transformation matrix
function T = estimateSimilarity(points)
    classToUse = class(points);

    [points1, points2, normMatrix1, normMatrix2] = ...
        normalizePoints(points, classToUse);

    numPts = size(points1, 1);
    constraints = zeros(2 * numPts, 5, 'like', points);
    constraints(1:2:2 * numPts, :) = [-points1(:, 2), points1(:, 1), ...
                                        zeros(numPts, 1), -ones(numPts, 1), points2(:, 2)];
    constraints(2:2:2 * numPts, :) = [points1, ones(numPts, 1), ...
                                        zeros(numPts, 1), -points2(:, 1)];

    [~, ~, V] = svd(constraints, 0);
    h = V(:, end);
    T = eye(3, 'like', points);
    T(1:2, :) = [h(1:3)'; [-h(2), h(1), h(4)]] / h(5);
    T = denormalizeTform(T, normMatrix1, normMatrix2);
end

%--------------------------------------------------------------------------
% Rigid transformation matrix
function T = estimateRigid(points)
    points1 = points(:, :, 1);
    points2 = points(:, :, 2);

    % Find data centroid and deviations from centroid
    centroid1 = mean(points1);
    centroid2 = mean(points2);

    normPoints1 = bsxfun(@minus, points1, centroid1);
    normPoints2 = bsxfun(@minus, points2, centroid2);

    % Covariance matrix
    C = normPoints1' * normPoints2;

    [U, ~, V] = svd(C);

    % Handle the reflection case
    R = V * diag([ones(1, size(points1, 2) - 1) sign(det(U * V'))]) * U';

    % Compute the translation
    t = centroid2' - R * centroid1';
    T = eye(3, 'like', points);
    T(1:2, :) = [R, t];
end

%--------------------------------------------------------------------------
% Translation transformation matrix
function T = estimateTranslation(points)
    % Function that computes translation in 2-D between two sets of matched points.
    points1 = points(:, :, 1);
    points2 = points(:, :, 2);
    T = eye(3, 'like', points);
    translation = mean(points2 - points1, 1);
    T(1:2, 3) = translation';
end

%--------------------------------------------------------------------------
% Evaluation functions
%--------------------------------------------------------------------------
function distance = evaluateTransform2d(tform, points)
    points1 = points(:, :, 1);
    points2 = points(:, :, 2);
    numPoints = size(points1, 1);
    pt1h = [points1, ones(numPoints, 1, 'like', points)];
    pt1h = (tform * pt1h')';
    w = pt1h(:, 3);
    pt = pt1h(:, 1:2) ./ [w, w];
    delta = pt - points2;
    distance = hypot(delta(:, 1), delta(:, 2));
    distance(abs(pt1h(:, 3)) < eps(class(points))) = inf;
end

function distance = evaluateTranslation2d(tform, points)
    % Function that evaluates translation in 2-D between two sets of matched
    % points.
    points1 = points(:, :, 1);
    points2 = points(:, :, 2);

    tpoints1 = points1 + tform(1:2, 3)';
    distance = sqrt((tpoints1(:, 1) - points2(:, 1)) .^ 2 + ...
        (tpoints1(:, 2) - points2(:, 2)) .^ 2);
end

%--------------------------------------------------------------------------
% Normalization functions
%--------------------------------------------------------------------------
function [samples1, samples2, normMatrix1, normMatrix2] = normalizePoints(points, ...
        classToUse)
    points1 = cast(points(:, :, 1), classToUse);
    points2 = cast(points(:, :, 2), classToUse);

    [samples1, normMatrix1] = normalizePointsHartleyZisserman(points1', 2, classToUse);
    [samples2, normMatrix2] = normalizePointsHartleyZisserman(points2', 2, classToUse);

    samples1 = samples1';
    samples2 = samples2';
end

function [normPoints, T, Tinv] = normalizePointsHartleyZisserman(p, numDims, outputClass)
    % strip off the homogeneous coordinate
    points = p(1:numDims, :);

    % compute centroid
    centroid = cast(mean(points, 2), outputClass);

    % translate points so that the centroid is at [0,0]
    translatedPoints = bsxfun(@minus, points, centroid);

    % compute the scale to make mean distance from centroid sqrt(2)
    meanDistanceFromCenter = cast(mean(sqrt(sum(translatedPoints .^ 2))), ...
        outputClass);

    if meanDistanceFromCenter > 0 % protect against division by 0
        scale = cast(sqrt(numDims), outputClass) / meanDistanceFromCenter;
    else
        scale = cast(1, outputClass);
    end

    % compute the matrix to scale and translate the points
    % the matrix is of the size numDims+1-by-numDims+1 of the form
    % [scale   0     ... -scale*center(1)]
    % [  0   scale   ... -scale*center(2)]
    %           ...
    % [  0     0     ...       1         ]
    T = diag(ones(1, numDims + 1) * scale);
    T(1:end - 1, end) = -scale * centroid;
    T(end) = 1;

    if size(p, 1) > numDims
        normPoints = T * p;
    else
        normPoints = translatedPoints * scale;
    end

    % the following must be true: mean(sqrt(sum(normPoints(1:2,:).^2, 1))) == sqrt(2)

    if nargout > 2
        Tinv = diag(ones(1, numDims + 1) / scale);
        Tinv(1:end - 1, end) = centroid;
        Tinv(end) = 1;
    end

end

function tform = denormalizeTform(tform, normMatrix1, normMatrix2)
    tform = (normMatrix2 \ tform) * normMatrix1;
    tform = tform ./ tform(end);
end

%--------------------------------------------------------------------------
% Helper functions
%--------------------------------------------------------------------------
function tf = checkTForm(tform)
    tf = all(isfinite(tform(:)));
end

function params = setDefaultParams(params)
    % Set default parameters if not provided
    if ~isfield(params, 'maxIterations')
        params.maxIterations = 1000;
    end

    if ~isfield(params, 'confidence')
        params.confidence = 0.999;
    end

    if ~isfield(params, 'maxDistance')
        params.maxDistance = 2.0;
    end

    if ~isfield(params, 'maxSkipTrials')
        params.maxSkipTrials = params.maxIterations * 10;
    end

    if ~isfield(params, 'recomputeFromInliers')
        params.recomputeFromInliers = true;
    end

end
