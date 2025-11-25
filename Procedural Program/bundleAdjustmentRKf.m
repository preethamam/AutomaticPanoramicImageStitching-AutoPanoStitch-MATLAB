function [cameras, seed] = bundleAdjustmentRKf(input, numMatches, matches, keypoints, ...
        imageSizes, initialTforms, varargin)
    % BUNDLEADJUSTMENTLM Bundle adjustment for panoramic image stitching
    %   [cameras, seed] = bundleAdjustmentLM(input, matches, keypoints, imageSizes, initialTforms, ...)
    %
    %   Implements Brown-Lowe incremental bundle adjustment strategy:
    %   - Cameras added in best-match order (not sequential)
    %   - Global optimization after each addition
    %   - Huber robust error function
    %   - Levenberg-Marquardt solver with priors
    %
    %   Inputs:
    %   - input         : struct with fields:
    %                     .focalEstimateMethod ('wConstraint', 'shumSzeliskiOneHPaper')
    %                     .transformationType (e.g., 'projective')
    %   - numMatches    : N×N array, numMatches(i,j) = number of matches between i and j
    %   - matches       : N×N cell array, matches{i,j} = 2×M [indices_in_i; indices_in_j]
    %   - keypoints     : 1×N cell array, keypoints{i} = K×2 [x,y] or 2×K
    %   - imageSizes    : N×2 or N×3 array [height, width] per image
    %   - initialTforms : N×N cell (projective2d or 3×3 H) or struct array with .i,.j,.H
    %
    %   Name-Value Parameters:
    %   - 'SigmaHuber'  : Huber threshold (default: 2.0 pixels)
    %   - 'MaxLMIters'  : Max LM iterations (default: 50)
    %   - 'Lambda0'     : Initial LM damping (default: 1e-2)
    %   - 'PriorSigmaF' : Focal prior std (default: 50 pixels)
    %   - 'Verbose'     : Verbosity level 0/1/2 (default: 1)
    %   - 'userSeed'    : Fixed seed image index (default: auto-select)
    %   - 'oneDirection' : if true, only use one-direction residuals in BA (default: false)
    %   - 'MaxMatches'   : cap max matches per pair (default: Inf)
    %   - 'SubsampleMode' : 'random'|'grid'|'polar' (default: 'random')
    %   - 'SubsampleGridBins' : [rows cols] for 'grid' subsampling (default: [4 4])
    %   - 'SubsamplePolarBins' : [nAngles nRadii] for 'polar' subsampling (default: [12 5])
    %   Outputs:
    %   - cameras : 1×N struct array with fields:
    %       .R           — 3×3 rotation (world-to-camera convention)
    %       .f           — scalar focal length (pixels)
    %       .K           — 3×3 intrinsics [f 0 cx; 0 f cy; 0 0 1]
    %       .initialized — logical flag
    %   - seed    : index of seed image
    %
    %   References:
    %   Brown, M. and Lowe, D.G., 2007. Automatic panoramic image stitching
    %   using invariant features. International Journal of Computer Vision,
    %   74(1), pp.59-73.

    % --- Input validation ---
    narginchk(6, inf);
    validateattributes(numMatches, {'double'}, {'2d', 'finite'});
    validateattributes(matches, {'cell'}, {}, mfilename, 'matches', 3);
    validateattributes(keypoints, {'cell'}, {}, mfilename, 'keypoints', 4);
    validateattributes(imageSizes, {'double'}, {'nonnan', 'finite'}, mfilename, 'imageSizes', 5);
    % initialTforms may be cell or struct; validate loosely if provided

    %% Parse optional parameters
    p = inputParser;
    addParameter(p, 'SigmaHuber', 2.0, @(x) x > 0);
    addParameter(p, 'MaxLMIters', 50, @(x) x > 0);
    addParameter(p, 'Lambda0', 1e-2, @(x) x > 0);
    addParameter(p, 'PriorSigmaF', 50, @(x) x > 0);
    addParameter(p, 'Verbose', 1, @(x) ismember(x, [0, 1, 2]));
    addParameter(p, 'userSeed', [], @(x) isempty(x) || (isscalar(x) && x > 0));
    addParameter(p, 'OneDirection', false);
    addParameter(p, 'MaxMatches', Inf);

    parse(p, varargin{:});
    opts = p.Results;
    if ~isfield(opts, 'SubsampleMode'), opts.SubsampleMode = 'random'; end % 'random' | 'grid' | 'polar'
    if ~isfield(opts, 'SubsampleGridBins'), opts.SubsampleGridBins = [4 4]; end % [rows cols]
    if ~isfield(opts, 'SubsamplePolarBins'), opts.SubsamplePolarBins = [12 5]; end % [nAngles nRadii]

    % NEW: Add focal smoothness control
    opts.FocalSmoothnessWeight = 'auto'; % or set manually, e.g., 100
    %   Weak coupling:   opts.FocalSmoothnessWeight = 500;
    %   Medium coupling: opts.FocalSmoothnessWeight = 2000;  (recommended)
    %   Strong coupling: opts.FocalSmoothnessWeight = 5000;
    opts.FocalMeanWeight = 50; % Optional: even tighter global constraint (50)

    N = numel(keypoints);

    % ------------------ Build pair list (compact) ------------------
    pairs = buildPairs(numMatches, matches, keypoints, initialTforms); % each pair has i,j, Ui (px), Uj (px); i<j

    % ------------------ Choose seed (image with max degree / matches) ------------------
    % Optionally override the automatic seed selection
    if ~isempty(opts.userSeed)
        seed = opts.userSeed;

        if ~(isscalar(seed) && isnumeric(seed) && isfinite(seed) && seed >= 1 && seed <= N)
            error('bundleAdjustmentLM:SeedIndex', 'userSeed must be a valid image index in [1..N].');
        end

    else
        % Automatic seed: image with maximum degree
        deg = zeros(1, N);

        for t = 1:numel(pairs)
            M = size(pairs(t).Ui, 1);
            deg(pairs(t).i) = deg(pairs(t).i) + M;
            deg(pairs(t).j) = deg(pairs(t).j) + M;
        end

        [~, seed] = max(deg);
    end

    % Print camera initialization
    if opts.Verbose
        fprintf('\n=== Camera Initialization ===\n');
        fprintf('Seed image: %d (best connected)\n', seed);
        fprintf('Using focal estimation method: %s\n', input.focalEstimateMethod);
    end

    % Robust camera initialization
    cameras = initializeCameraMatrices(input, pairs, matches, imageSizes, ...
        initialTforms, seed, N, numMatches);

    % Check if the panorama has translation
    if cameras(1).noRotation == 1 || input.forcePlanarScan
        % G0: 1xN cell of 3x3 absolute homographies projected onto Hseed=eye(3)
        H2refined = bundleAdjustmentH(input, pairs, N, seed, 'G0', {cameras(:).H2seed}, ...
            'MaxIters', opts.MaxLMIters, 'Huber', opts.SigmaHuber, ...
            'UseLSQ', false, 'Verbose', opts.Verbose, ... % LSQ is sometimes unstable due to numerical errors
            'ImageSizes', imageSizes, 'OneDirection', true, ...
            'MaxMatches', opts.MaxMatches, ...
            'SubsampleMode', 'random'); % 'random' | 'grid' | 'polar'
        [cameras.H2refined] = H2refined{:};
        return
    end

    % Prepare camera parameters
    cameras = prepareCameraCache(cameras, imageSizes);
    initialized = false(N, 1);
    initialized(seed) = true;
    cameras(seed).initialized = true;

    % Brown-Lowe Incremental Bundle Adjustment
    if opts.Verbose
        fprintf('\n=== Incremental Bundle Adjustment (Brown-Lowe) ===\n');
        fprintf('Step 1/%d: Initialized seed camera %d (f=%.1f px)\n', ...
            N, seed, cameras(seed).f);
    end

    % NEW: Track RMSE history
    rmseHistory = [];

    % Incremental addition in BEST-MATCH order
    for step = 2:N
        % Find uninitialized image with strongest connection to initialized set
        bestScore = 0;
        bestImage = -1;
        bestMatchTo = -1;

        for candidate = 1:N

            if initialized(candidate)
                continue
            end

            % Sum matches to ALL initialized images
            for initImg = 1:N

                if ~initialized(initImg)
                    continue
                end

                score = numMatches(candidate, initImg) + ...
                    numMatches(initImg, candidate);

                if score > bestScore
                    bestScore = score;
                    bestImage = candidate;
                    bestMatchTo = initImg;
                end

            end

        end

        if bestImage == -1

            if opts.Verbose
                fprintf('Warning: No more images with matches to add (added %d/%d)\n', ...
                    step - 1, N);
            end

            break
        end

        % Initialize new camera from best matching initialized camera
        % Build H (bestImage <- bestMatchTo), column form

        pairTemp.i = bestImage;
        pairTemp.j = bestMatchTo;
        % default to empty so downstream code is safe
        pairTemp.Ui = [];
        pairTemp.Uj = [];

        % First try: reuse from compact pairs (only has i<j entries)
        allijs = [cat(1, pairs.i), cat(1, pairs.j)];
        idxFromPairs = find(ismember(allijs, [min(bestImage, bestMatchTo), max(bestImage, bestMatchTo)], 'rows'), 1);

        if ~isempty(idxFromPairs)
            % Pull from pairs; need to ensure Ui/Uj correspond to (i=bestImage, j=bestMatchTo)
            p = pairs(idxFromPairs);

            if p.i == bestImage && p.j == bestMatchTo
                pairTemp.Ui = p.Ui;
                pairTemp.Uj = p.Uj;
            elseif p.i == bestMatchTo && p.j == bestImage
                % swap to keep j->i orientation consistent with pairTemp (i=bestImage, j=bestMatchTo)
                pairTemp.Ui = p.Uj; % points in bestImage
                pairTemp.Uj = p.Ui; % points in bestMatchTo
            end

        end

        % Second try: build from matches/keypoints cells
        if isempty(pairTemp.Ui) || isempty(pairTemp.Uj)
            M = matches{bestMatchTo, bestImage}; % columns: [idx_in_bestMatchTo; idx_in_bestImage]
            flipped = false;

            if isempty(M)
                M = matches{bestImage, bestMatchTo}; % columns: [idx_in_bestImage; idx_in_bestMatchTo]
                flipped = true;
            end

            if ~isempty(M)
                % Extract Kx2 or 2xK agnostic
                Ui = keypoints{bestImage};
                Uj = keypoints{bestMatchTo};
                if size(Ui, 1) == 2, Ui = Ui.'; end % -> Kx2
                if size(Uj, 1) == 2, Uj = Uj.'; end

                if ~flipped
                    % M = [idx_j; idx_i] w.r.t matches{j,i} convention → here j=bestMatchTo, i=bestImage
                    pairTemp.Uj = Uj(M(1, :).', :); % points in j (bestMatchTo), M rows are indices
                    pairTemp.Ui = Ui(M(2, :).', :); % points in i (bestImage)
                else
                    % M = [idx_i; idx_j] w.r.t matches{i,j}
                    pairTemp.Ui = Ui(M(1, :).', :);
                    pairTemp.Uj = Uj(M(2, :).', :);
                end

            end

        end

        % ---- Accept or ban this pair for this step ----
        if isempty(pairTemp.Ui) || size(pairTemp.Ui, 1) < 4 || isempty(pairTemp.Uj) || size(pairTemp.Uj, 1) < 4

            if opts.Verbose
                fprintf('  Skipping %d (no robust matches with %d)\n', bestImage, bestMatchTo);
            end

            % Ban just this (bestImage,bestMatchTo) option and try next best
            localScore(bestImage, bestMatchTo) = -inf;
            continue; % stay in while; pick the next best
        end

        % Get homography matrix
        Hij = initialTforms{pairTemp.i, pairTemp.j};

        % Best image rotation
        if ~isempty(Hij)
            Ki = buildIntrinsicMatrix(cameras(bestImage).f, imageSizes(bestImage, 1:2));
            Kj = buildIntrinsicMatrix(cameras(bestMatchTo).f, imageSizes(bestMatchTo, 1:2));
            Hji = Ki \ Hij * Kj;

            if all(isfinite(Hji(:)))
                Rij = projectToSO3(Hji); % approx R_i * R_j^T
                cameras(bestImage).R = Rij * cameras(bestMatchTo).R;
            else
                cameras(bestImage).R = cameras(bestMatchTo).R; % fallback
            end

        else
            cameras(bestImage).R = cameras(bestMatchTo).R; % fallback
        end

        % Best image K and f
        cameras(bestImage).f = cameras(bestMatchTo).f;
        cameras(bestImage).K = buildIntrinsicMatrix(cameras(bestImage).f, ...
            imageSizes(bestImage, 1:2));
        cameras(bestImage).initialized = true;
        initialized(bestImage) = true;

        if opts.Verbose
            fprintf('Step %d/%d: Added camera %d (best match to %d: %d pairs)\n', ...
                step, N, bestImage, bestMatchTo, bestScore);
        end

        % CRITICAL: Global bundle adjustment on ALL initialized cameras
        initializedList = find(initialized);
        optsIter = opts; % <— add

        if numel(initializedList) <= 3
            optsIter.FinalPass = true; % use σHuber = opts.SigmaHuber (e.g., 2 px)
        end

        if opts.Verbose >= 2
            fprintf('  Running global BA on %d cameras...\n', numel(initializedList));
        end

        [cameras, rmseCurrent] = runLevenbergMarquardt( ...
            cameras, initializedList, seed, ...
            matches, keypoints, imageSizes, optsIter);

        % Quality check
        if step > 3 && ~isempty(rmseHistory)
            medianRMSE = median(rmseHistory);

            if rmseCurrent > 2.5 * medianRMSE && medianRMSE > 0

                if opts.Verbose
                    fprintf('    ⚠️  WARNING: Elevated RMSE %.3f px (median: %.3f, ratio: %.1fx)\n', ...
                        rmseCurrent, medianRMSE, rmseCurrent / medianRMSE);
                    fprintf('    This may indicate a problematic camera addition.\n');
                end

            end

        end

        % Track RMSE history
        rmseHistory(end + 1) = rmseCurrent;
    end

    %% Final global bundle adjustment
    initializedList = find([cameras.initialized]);
    optsFinal = opts; % <— add
    optsFinal.FinalPass = true; % <— add

    if opts.Verbose
        fprintf('\n=== Final Bundle Adjustment ===\n');
        fprintf('Running final BA on all %d cameras...\n', numel(initializedList));
    end

    if numel(initializedList) > 1
        % Run multiple passes for long chains
        nPasses = min(2, ceil(numel(initializedList) / 10));

        for pass = 1:nPasses

            if nPasses > 1 && opts.Verbose
                fprintf('Pass %d/%d:\n', pass, nPasses);
            end

            [cameras, finalRMSE] = runLevenbergMarquardt( ...
                cameras, initializedList, seed, ...
                matches, keypoints, imageSizes, optsFinal);
        end

    end

    %% Report final parameters
    if opts.Verbose
        fprintf('\n=== Final Camera Parameters ===\n');
        focals = [cameras(initializedList).f];
        fprintf('Focal lengths: mean=%.1f px, std=%.1f px, range=[%.1f, %.1f]\n', ...
            mean(focals), std(focals), min(focals), max(focals));

        if opts.Verbose >= 2

            for i = initializedList'
                [yaw, pitch, roll] = extractEulerAngles(cameras(i).R);
                fprintf('  Camera %d: f=%.1f, yaw=%.1f°, pitch=%.1f°, roll=%.1f°\n', ...
                    i, cameras(i).f, yaw * 180 / pi, pitch * 180 / pi, roll * 180 / pi);
            end

        end

    end

end

function pairs = buildPairs(numMatches, matches, keypoints, initialTforms)
    % BUILDPAIRS Create compact pair list from match and keypoint cells.
    %   pairs = buildPairs(numMatches, matches, keypoints, initialTforms)
    %
    %   Inputs:
    %     - numMatches     : N×N numeric matrix of match counts per pair
    %     - matches        : N×N cell array of index-pair arrays
    %     - keypoints      : 1×N cell array of K×2 point coordinates
    %     - initialTforms  : N×N cell array of initial transforms (H)
    %
    %   Outputs:
    %     - pairs : 1×P struct array with fields:
    %         .i, .j : image indices (i<j)
    %         .Ui, .Uj : M×2 pixel coordinates of matched points
    %         .Hij : initial transform for the pair (if available)

    arguments
        numMatches (:, :) double
        matches (:, :) cell
        keypoints cell
        initialTforms cell
    end

    N = numel(keypoints);

    % VECTORIZED: Get upper triangle linear indices directly
    upperIdx = false(N, N);
    upperIdx(triu(true(N), 1)) = true;
    linearIdx = find(upperIdx);

    % VECTORIZED: Check all matches at once
    hasMatches = numMatches(linearIdx) ~= 0;
    validLinearIdx = linearIdx(hasMatches);
    nPairs = numel(validLinearIdx);

    if nPairs == 0
        pairs = struct('i', {}, 'j', {}, 'Ui', {}, 'Uj', {}, 'Hij', {});
        return;
    end

    % VECTORIZED: Convert linear indices to subscripts
    [validI, validJ] = ind2sub([N, N], validLinearIdx);

    % Pre-allocate output
    pairs(nPairs) = struct('i', [], 'j', [], 'Ui', [], 'Uj', [], 'Hij', []);

    % Extract keypoints (this loop is unavoidable due to ragged arrays)
    for k = 1:nPairs
        i = validI(k);
        j = validJ(k);
        Mij = matches{i, j};

        pairs(k).i = i;
        pairs(k).j = j;
        pairs(k).Ui = keypoints{i}(Mij(:, 1), :);
        pairs(k).Uj = keypoints{j}(Mij(:, 2), :);
        pairs(k).Hij = initialTforms{i, j};
    end

end

% Core optimization function
function [cameras, finalRMSE] = runLevenbergMarquardt(cameras, camList, seed, matches, ...
        keypoints, imageSizes, opts)
    % RUNLEVENBERGMARQUARDT Global BA using Levenberg–Marquardt.
    %   cameras = runLevenbergMarquardt(cameras, camList, seed, matches,
    %   keypoints, imageSizes, opts) optimizes rotations and focal lengths
    %   for the subset of cameras in camList given correspondences.
    %
    %   Inputs
    %   - cameras     : struct array with fields R,f,K,initialized
    %   - camList     : vector of camera indices to optimize
    %   - seed        : scalar index for seed camera (only f optimized)
    %   - matches     : N×N cell of index pairs
    %   - keypoints   : 1×N cell of K×2 or 2×K points
    %   - imageSizes  : N×2 [H W]
    %   - opts        : struct of solver options
    %   Output
    %   - cameras     : updated cameras struct array
    %   - finalRMSE   : final reprojection RMSE (pixels)

    % --- Arguments validation ---
    arguments
        cameras (1, :) struct
        camList (1, :) double {mustBeInteger, mustBePositive}
        seed (1, 1) double {mustBeInteger, mustBePositive}
        matches (:, :) cell
        keypoints cell
        imageSizes (:, 3) double {mustBeFinite}
        opts struct
    end

    % Robust schedule (as before)
    if isfield(opts, 'FinalPass') && opts.FinalPass
        sigmaHuber = opts.SigmaHuber;
    else
        sigmaHuber = 2.0;
    end

    lambda = opts.Lambda0;
    maxIters = opts.MaxLMIters;

    % Freeze σ_f the first time we enter LM
    if ~isfield(opts, 'sigmafFixed') || isempty(opts.sigmafFixed)
        f0 = median([cameras(camList).f]);
        opts.sigmafFixed = max(1, f0) / 10; % you can tighten to /20 for the first pass
    end

    % Add smoothness weight (key parameter!)
    if ~isfield(opts, 'FocalSmoothnessWeight') || ...
            (ischar(opts.FocalSmoothnessWeight) && strcmp(opts.FocalSmoothnessWeight, 'auto'))
        % Auto-compute weight based on chain length
        f0 = median([cameras(camList).f]);

        if numel(camList) <= 5
            opts.FocalSmoothnessWeight = (f0 / 20) ^ 2 * 0.5; % Weak coupling for short chains
        else
            opts.FocalSmoothnessWeight = (f0 / 50) ^ 2 * 2.0; % Strong coupling for long chains
        end

        if isfield(opts, 'Verbose') && opts.Verbose >= 2
            fprintf('      Auto-set FocalSmoothnessWeight = %.2e\n', opts.FocalSmoothnessWeight);
        end

    elseif opts.FocalSmoothnessWeight == 0
        % User explicitly disabled smoothness
        opts.FocalSmoothnessWeight = 0;
    end

    % NEW: cached solver state for AMD ordering / ichol precond
    H = []; g = []; E0 = 0; rmse0 = 0; % avoid "might be used before defined"
    solverState = struct; % define with a P field so isfield checks pass

    % caps
    thetaCap = deg2rad(5);

    % STAGED OPTIMIZATION
    for outer = 1:3
        % Stage 1: Very tight focal (fracDf → 0)
        % Stage 2-3: Gradually relax
        if outer == 1
            fracDf = 0.005; % 0.5 % cap initially
        elseif outer == 2
            fracDf = 0.01; % 1 % in middle
        else
            fracDf = 0.02; % 2 % for final refinement
        end

        if isfield(opts, 'Verbose') && opts.Verbose >= 2
            fprintf('      (outer %d) fracDf = %.3f, smoothness = %.2e\n', ...
                outer, fracDf, opts.FocalSmoothnessWeight);
        end

        [Phi, pmap] = buildDeltaVector(cameras, camList, seed);

        % Build prior with smoothness
        CpInv = buildBrownLowePrior(camList, seed, cameras, opts, pmap);

        % ... rest of LM loop unchanged ...
        [H, g, E0, rmse0] = accumulateNormalEqnsBlock( ...
            Phi, pmap, cameras, camList, seed, ...
            matches, keypoints, imageSizes, sigmaHuber, opts);

        if isfield(opts, 'Verbose') && opts.Verbose >= 2
            fprintf('      (relin) RMSE: %.3f px  nnz(H): %d\n', rmse0, nnz(H));
        end

        for iter = 1:maxIters
            A = H + CpInv + lambda * speye(size(H, 1));
            b = -g;

            [deltaRaw, solverState] = solveSpd(A, b, solverState);
            delta = capPerCameraStep(deltaRaw, pmap, cameras, camList, seed, thetaCap, fracDf);

            PhiTrial = Phi + delta;
            camTrial = applyIncrements(cameras, PhiTrial, pmap, camList, []);

            [~, ~, ETrial, rmseTrial] = accumulateNormalEqnsBlock( ...
                PhiTrial, pmap, cameras, camList, seed, ...
                matches, keypoints, imageSizes, sigmaHuber, opts);

            pred = 0.5 * (delta.' * (lambda * delta - g + CpInv * delta));
            if pred <= 0, rho = -Inf; else, rho = (E0 - ETrial) / pred; end

            if (ETrial < E0) && (rho > 0)
                cameras = camTrial;

                % Re-orthonormalize
                for kk = 1:numel(camList)
                    i = camList(kk);
                    R = cameras(i).R;
                    [U, ~, V] = svd(R);
                    cameras(i).R = U * diag([1, 1, sign(det(U * V'))]) * V';
                end

                % Lambda schedule
                if rho > 0.75, lambda = lambda / 2;
                elseif rho < 0.25, lambda = lambda * 2;
                end

                lambda = max(min(lambda, 1e6), 1e-10);

                % Re-linearize
                [Phi, pmap] = buildDeltaVector(cameras, camList, seed);
                CpInv = buildBrownLowePrior(camList, seed, cameras, opts, pmap);
                [H, g, E0, rmse0] = accumulateNormalEqnsBlock( ...
                    Phi, pmap, cameras, camList, seed, ...
                    matches, keypoints, imageSizes, sigmaHuber, opts);

                if isfield(opts, 'Verbose') && opts.Verbose >= 2
                    fprintf('      iter %d: RMSE %.3f px  (λ=%.2g)\n', iter, rmse0, lambda);
                end

                if abs(pred) < 1e-12 || abs(E0 - ETrial) < 1e-9
                    break;
                end

            else
                lambda = min(lambda * 4, 1e6);
                if lambda > 1e5, break; end
            end

        end

    end

    if isfield(opts, 'Verbose') && opts.Verbose >= 1
        fprintf('    Final RMSE: %.3f pixels\n', rmse0);
    end

    finalRMSE = rmse0; % NEW: return the final RMSE
end

function [H, g, E, rmse] = accumulateNormalEqnsBlock( ...
        Phi, pmap, baseCams, camList, seed, matches, keypoints, imageSizes, sigmaHuber, opts)
    % BUILDACCUMULATENORMALEQNSBLOCK Build H and g for a block of camera pairs.
    %   [H, g, E, rmse] = accumulateNormalEqnsBlock(Phi, pmap, baseCams, camList,
    %   seed, matches, keypoints, imageSizes, sigmaHuber, opts)
    %
    %   Inputs:
    %     - Phi        : parameter increment vector (column)
    %     - pmap       : mapping of parameters to cameras (from buildDeltaVector)
    %     - baseCams   : cameras used for Jacobian evaluation
    %     - camList    : indices of cameras included in this block
    %     - seed       : seed camera index
    %     - matches    : N×N cell of match index pairs
    %     - keypoints  : 1×N cell of point coordinates
    %     - imageSizes : N×3 numeric array [H W ...]
    %     - sigmaHuber : Huber threshold (scalar)
    %     - opts       : options struct
    %
    %   Outputs:
    %     - H    : sparse normal matrix (approx J'J)
    %     - g    : RHS vector (approx J'r)
    %     - E    : total energy (0.5 * sum weighted squared residuals)
    %     - rmse : root-mean-squared reprojection error (pixels)
    %
    %   Notes:
    %     Residuals are evaluated at the *incremented* cameras while the
    %     Jacobians are computed at the base cameras (Gauss–Newton linearization).

    % --- Arguments validation ---
    arguments
        Phi (:, 1) double {mustBeFinite}
        pmap (1, :) struct
        baseCams (1, :) struct
        camList (1, :) double {mustBeInteger, mustBePositive}
        seed (1, 1) double {mustBeInteger, mustBePositive}
        matches (:, :) cell
        keypoints cell
        imageSizes (:, 3) double {mustBeFinite}
        sigmaHuber (1, 1) double {mustBeFinite, mustBePositive}
        opts struct
    end

    % Cameras used for residual evaluation
    camLin = applyIncrements(baseCams, Phi, pmap, camList, []); %#ok<NASGU> (for clarity)

    % Column span per camera in Phi
    % seed has 1 dof (df), others 4 dof. Build a fast map.
    maxIdx = pmap(end).startIdx + (pmap(end).isSeed == 0) * 3;
    P = maxIdx; % number of parameters
    H = spalloc(P, P, 40 * numel(camList)); % coarse guess; will grow
    g = zeros(P, 1);

    % Pre-compute column indices per camera
    % colsMap{i} -> 1 or 4-element vector into [1..P]
    Nmax = max([camList(:).']);
    colsMap = cell(Nmax, 1);

    for t = 1:numel(pmap)
        i = pmap(t).camIdx;
        s = pmap(t).startIdx;

        if pmap(t).isSeed
            colsMap{i} = s; % [df]
        else
            colsMap{i} = s:(s + 3); % [dθx dθy dθz df]
        end

    end

    % Build compact list of (i,j) pairs within camList
    pairList = struct('i', {}, 'j', {}, 'Ui', {}, 'Uj', {});
    idx = 1;
    present = false(1, max(numel(keypoints), Nmax));
    present(camList) = true;

    for ii = 1:numel(camList)
        i = camList(ii);

        for jj = ii + 1:numel(camList)
            j = camList(jj);
            if isempty(matches{i, j}), continue; end
            mpairs = matches{i, j}; % M×2
            Ui = keypoints{i}(mpairs(:, 1), :); % M×2 (on image i)
            Uj = keypoints{j}(mpairs(:, 2), :); % M×2 (on image j)
            if isempty(Ui), continue; end

            % ---- Subsample over-connected edges (cap to opts.MaxMatches) ----
            if isfield(opts, 'MaxMatches') && isfinite(opts.MaxMatches)
                [Ui, Uj] = subsampleMatches( ...
                    Ui, Uj, baseCams(i), baseCams(j), imageSizes(i, :), imageSizes(j, :), opts);
                if isempty(Ui), continue; end
            end

            pairList(idx).i = i;
            pairList(idx).j = j;
            pairList(idx).Ui = Ui;
            pairList(idx).Uj = Uj;
            idx = idx + 1;
        end

    end

    if isempty(pairList)
        E = 0; rmse = 0; return;
    end

    % Per-pair accumulation (parallel if you like)
    out(numel(pairList)) = struct('bi', [], 'bj', [], 'Hii', [], 'Hjj', [], 'Hij', [], 'gi', [], 'gj', [], 'E', 0, 'R2sum', 0, 'Rcnt', 0); %#ok<NASGU>

    parfor p = 1:numel(pairList)
        i = pairList(p).i; j = pairList(p).j;
        Ui = pairList(p).Ui; Uj = pairList(p).Uj;

        % Residuals at incremented cams; Jacobian at base cams (Gauss-Newton)
        % We use the *current* baseCams for Jacobians and *Phi-updated* for residuals:
        % → good tradeoff of speed/stability
        [rij, Ji, Jj, Eij, r2sum, rcnt] = jacobianPair( ...
            Ui, Uj, baseCams(i), baseCams(j), camLin(i), camLin(j), ...
            imageSizes(i, :), imageSizes(j, :), sigmaHuber, colsMap{i}, colsMap{j}, ...
            opts);

        % Blocks (note: Ji,Jj already shaped as (2M × ci) and (2M × cj))
        Hii = Ji.' * Ji; Hjj = Jj.' * Jj; Hij = Ji.' * Jj;
        gi = Ji.' * rij; gj = Jj.' * rij;

        out(p).bi = colsMap{i};
        out(p).bj = colsMap{j};
        out(p).Hii = Hii; out(p).Hjj = Hjj; out(p).Hij = Hij;
        out(p).gi = gi; out(p).gj = gj;
        out(p).E = Eij; out(p).R2sum = r2sum; out(p).Rcnt = rcnt;
    end

    % --- Serial reduction (triplets -> single sparse assembly) ---
    P = size(H, 1);
    I = zeros(0, 1); J = zeros(0, 1); V = zeros(0, 1);
    gg = zeros(P, 1);
    E = 0; R2sum = 0; Rcnt = 0;

    for p = 1:numel(pairList)
        % Force column vectors so subsequent indexing returns columns
        bi = out(p).bi(:);
        bj = out(p).bj(:);

        % Hii
        [ii, jj, vv] = find(out(p).Hii);

        if ~isempty(vv)
            I = [I; bi(ii(:))];
            J = [J; bi(jj(:))];
            V = [V; vv(:)];
        end

        % Hjj
        [ii, jj, vv] = find(out(p).Hjj);

        if ~isempty(vv)
            I = [I; bj(ii(:))];
            J = [J; bj(jj(:))];
            V = [V; vv(:)];
        end

        % Hij and symmetric
        [ii, jj, vv] = find(out(p).Hij);

        if ~isempty(vv)
            I = [I; bi(ii(:))]; J = [J; bj(jj(:))]; V = [V; vv(:)];
            I = [I; bj(jj(:))]; J = [J; bi(ii(:))]; V = [V; vv(:)];
        end

        % RHS
        gg(bi) = gg(bi) + out(p).gi(:);
        gg(bj) = gg(bj) + out(p).gj(:);

        E = E + out(p).E;
        R2sum = R2sum + out(p).R2sum;
        Rcnt = Rcnt + out(p).Rcnt;
    end

    H = sparse(I, J, V, P, P);
    g = gg;

    rmse = sqrt(max(R2sum, 0) / max(Rcnt, 1));
end

function [rStacked, Ji, Jj, eSum, r2sum, rcnt] = jacobianPair( ...
        Ui, Uj, camIbase, camJbase, camIlin, camJlin, ...
        sizeI, sizeJ, sigmaHuber, colsI, colsJ, opts)
    % JACOBIANPAIR Build residuals and Jacobians for a matched pair (i,j).
    %   [rStacked, Ji, Jj, eSum, r2sum, rcnt] = jacobianPair(Ui, Uj, ...)
    %
    %   Inputs:
    %     - Ui, Uj   : M×2 matched pixel coordinates on image i and j
    %     - camIbase : camera struct for i at linearization point (Jacobians)
    %     - camJbase : camera struct for j at linearization point
    %     - camIlin  : camera struct for i used to evaluate residuals
    %     - camJlin  : camera struct for j used to evaluate residuals
    %     - sizeI, sizeJ : image sizes for i and j
    %     - sigmaHuber   : Huber threshold
    %     - colsI, colsJ : column index vectors for camera params in H/g
    %     - opts         : options struct (e.g. OneDirection)
    %
    %   Outputs:
    %     - rStacked : stacked residual vector (2M or 4M × 1)
    %     - Ji, Jj   : block Jacobians w.r.t camera-i and camera-j params
    %     - eSum     : scalar energy (0.5 * sum w*r^2)
    %     - r2sum    : sum of squared (weighted) residuals
    %     - rcnt     : number of scalar residuals (2M or 4M)

    % --- Arguments validation ---
    arguments
        Ui (:, 2) double {mustBeFinite}
        Uj (:, 2) double {mustBeFinite}
        camIbase (1, 1) struct
        camJbase (1, 1) struct
        camIlin (1, 1) struct
        camJlin (1, 1) struct
        sizeI (1, :) double {mustBeFinite}
        sizeJ (1, :) double {mustBeFinite}
        sigmaHuber (1, 1) double {mustBeFinite, mustBePositive}
        colsI (1, :) double {mustBeInteger, mustBePositive}
        colsJ (1, :) double {mustBeInteger, mustBePositive}
        opts struct = struct
    end

    if nargin < 12, opts = struct; end
    doBoth = ~(isfield(opts, 'OneDirection') && opts.OneDirection);

    M = size(Ui, 1);
    rowsPerMatch = 2 * (1 + doBoth); % 2 (one-dir) or 4 (both)
    R = rowsPerMatch * M;

    % Preallocate outputs
    rStacked = zeros(R, 1);
    Ji = zeros(R, numel(colsI));
    Jj = zeros(R, numel(colsJ));

    eSum = 0; r2sum = 0; rcnt = 0;

    % Tight loop with preallocated row pointer
    rp = 1;

    for k = 1:M
        ui = Ui(k, :).';
        uj = Uj(k, :).';

        % ---------- j -> i ----------
        % Jacobian at base cams, residual at lin cams
        [rijBase, JijI, JijJ] = computeSingleResidual( ...
            ui, uj, camIbase, camJbase, sizeI, sizeJ);
        [rijLin, ~, ~] = computeSingleResidual( ...
            ui, uj, camIlin, camJlin, sizeI, sizeJ);

        wij = huberWeight(norm(rijLin), sigmaHuber);
        swij = sqrt(wij);

        % Place j->i rows
        rStacked(rp:rp + 1) = swij * rijLin;
        Ji(rp:rp + 1, :) = swij * JijI(:, 1:numel(colsI));
        Jj(rp:rp + 1, :) = swij * JijJ(:, 1:numel(colsJ));

        eSum = eSum + 0.5 * (swij ^ 2) * (rijLin.' * rijLin);
        r2sum = r2sum + (swij ^ 2) * (rijLin.' * rijLin);
        rcnt = rcnt + 2;

        rp = rp + 2;

        % ---------- i -> j (optional) ----------
        if doBoth
            [rjiBase, JjiJ, JjiI] = computeSingleResidual( ...
                uj, ui, camJbase, camIbase, sizeJ, sizeI);
            [rjiLin, ~, ~] = computeSingleResidual( ...
                uj, ui, camJlin, camIlin, sizeJ, sizeI);

            wji = huberWeight(norm(rjiLin), sigmaHuber);
            swji = sqrt(wji);

            rStacked(rp:rp + 1) = swji * rjiLin;
            % Note role swap: J wrt camera-i params uses Jji_i; wrt camera-j uses Jji_j
            Ji(rp:rp + 1, :) = swji * JjiI(:, 1:numel(colsI));
            Jj(rp:rp + 1, :) = swji * JjiJ(:, 1:numel(colsJ));

            eSum = eSum + 0.5 * (swji ^ 2) * (rjiLin.' * rjiLin);
            r2sum = r2sum + (swji ^ 2) * (rjiLin.' * rjiLin);
            rcnt = rcnt + 2;

            rp = rp + 2;
        end

    end

end

function [x, state] = solveSpd(A, b, state)
    % SOLVESPD Cached solver for (near) SPD sparse systems.
    % state is an optional struct you can keep & pass back in/out.

    % Inputs:
    %   - A : symmetric positive-definite (or near SPD) matrix
    %   - b : RHS column vector
    %   - state : (optional) struct to hold cached factorization info
    % Outputs:
    %   - x : solution vector
    %   - state : updated state with solver cache metadata

    % --- Arguments validation ---
    arguments
        A double {mustBeFinite}
        b (:, 1) double {mustBeFinite}
        state struct = struct
    end

    persistent cache
    if nargin < 3 || isempty(state), state = struct; end
    key = [];

    if issparse(A)
        % Hash pattern (rows, cols, nnz) — cheap heuristic
        key = [size(A, 1), size(A, 2), nnz(A)];
    end

    useCache = issparse(A) && isfield(cache, 'key') && isequal(cache.key, key);

    if issparse(A)

        if ~useCache
            % Build & cache permutation and preconditioner
            p = symamd(A); Ap = A(p, p);
            cache.key = key; cache.p = p;

            % Try Cholesky once on the pattern
            [R, flag] = chol(Ap);

            if flag == 0
                cache.method = 'chol'; cache.R = R;
            else
                setup = struct('type', 'ict', 'droptol', 1e-3, 'diagcomp', 0.01);

                try
                    L = ichol(Ap, setup);
                    cache.method = 'pcg'; cache.L = L;
                catch
                    cache.method = 'slash'; % fallback
                end

            end

        else
            p = cache.p; Ap = A(p, p);
        end

        bp = b(p);

        switch cache.method
            case 'chol'
                R = chol(Ap); % numeric chol on same ordering is fast
                y = R \ (R' \ bp);
            case 'pcg'
                [y, flag] = pcg(Ap, bp, 1e-6, 200, cache.L, cache.L');
                if flag ~= 0, y = Ap \ bp; end
            otherwise
                y = Ap \ bp;
        end

        x = zeros(size(b));
        x(p) = y;
        if nargout > 2, state.solverCache = cache; end
    else
        x = A \ b;
    end

    % after solving:
    state.lastN = size(A, 1);
    state.lastNZ = nnz(A);
end

function deltaC = capPerCameraStep(delta, pmap, cameras, camList, seed, thetaCap, fracDf)
    % CAPPERCAMERASTEP Limit per-camera parameter update to stabilize LM.
    %   deltaC = capPerCameraStep(delta, pmap, cameras, camList, seed,
    %   thetaCap, fracDf) clamps rotation increments and fractional focal
    %   length updates per camera to stabilize Levenberg–Marquardt steps.
    %
    %   Inputs:
    %     - delta    : raw parameter increment vector
    %     - pmap     : parameter mapping (from buildDeltaVector)
    %     - cameras  : camera structs (to obtain current f)
    %     - camList  : list of camera indices in the vector
    %     - seed     : seed camera index
    %     - thetaCap : maximum rotation step (radians)
    %     - fracDf   : fractional cap for focal length change
    %
    %   Output:
    %     - deltaC : clamped parameter increment vector

    % --- Arguments validation ---
    arguments
        delta (:, 1) double {mustBeFinite}
        pmap (1, :) struct
        cameras (1, :) struct
        camList (1, :) double {mustBeInteger, mustBePositive}
        seed (1, 1) double {mustBeInteger, mustBePositive}
        thetaCap (1, 1) double {mustBePositive, mustBeFinite}
        fracDf (1, 1) double {mustBePositive, mustBeFinite}
    end

    % touch args possibly unused by MATLAB analyzer
    camList = camList; %#ok<NASGU>
    seed = seed; %#ok<NASGU>
    deltaC = delta;

    for k = 1:numel(pmap)
        s = pmap(k).startIdx;
        i = pmap(k).camIdx;

        if pmap(k).isSeed
            % clamp df
            f = cameras(i).f;
            df = deltaC(s);
            df = max(-fracDf * f, min(fracDf * f, df));
            deltaC(s) = df;
        else
            dth = deltaC(s:s + 2);
            a = norm(dth);

            if a > thetaCap
                dth = dth * (thetaCap / a);
                deltaC(s:s + 2) = dth;
            end

            f = cameras(i).f;
            df = deltaC(s + 3);
            df = max(-fracDf * f, min(fracDf * f, df));
            deltaC(s + 3) = df;
        end

    end

end

function [UiOut, UjOut] = subsampleMatches( ...
        Ui, Uj, camI, camJ, sizeI, sizeJ, opts)
    % SUBSAMPLEMATCHES Cap correspondences per edge with controlled sampling.
    %   [UiOut, UjOut] = subsampleMatches(Ui, Uj, camI, camJ, sizeI, sizeJ, opts)
    %
    %   Inputs:
    %     - Ui, Uj : M×2 pixel coordinates for matches on image i and j
    %     - camI, camJ : camera structs (may provide intrinsics)
    %     - sizeI, sizeJ : image sizes for i and j
    %     - opts : options struct with fields `MaxMatches`, `SubsampleMode`, etc.
    %
    %   Outputs:
    %     - UiOut, UjOut : subsampled correspondences (cap applied)

    % --- Arguments validation ---
    arguments
        Ui (:, 2) double {mustBeFinite}
        Uj (:, 2) double {mustBeFinite}
        camI (1, 1) struct
        camJ (1, 1) struct
        sizeI (1, :) double {mustBeFinite}
        sizeJ (1, :) double {mustBeFinite}
        opts struct
    end

    M = size(Ui, 1);
    cap = opts.MaxMatches;

    if M <= cap
        UiOut = Ui; UjOut = Uj; return;
    end

    switch lower(opts.SubsampleMode)
        case 'random'
            idx = randpermPerPair(M, cap, camI, camJ);
            UiOut = Ui(idx, :); UjOut = Uj(idx, :);

        case 'grid'
            % Stratify by a uniform grid on image i (can also average i/j)
            bins = opts.SubsampleGridBins; % [rows cols]
            idx = gridStratified(Ui, sizeI, cap, bins, camI, camJ);
            UiOut = Ui(idx, :); UjOut = Uj(idx, :);

        case 'polar'
            % Stratify by angle & radius around principal point on image i
            bins = opts.SubsamplePolarBins; % [nAngles nRadii]
            idx = polarStratified(Ui, sizeI, cap, bins, camI, camJ);
            UiOut = Ui(idx, :); UjOut = Uj(idx, :);

        otherwise
            % Fallback to random
            idx = randpermPerPair(M, cap, camI, camJ);
            UiOut = Ui(idx, :); UjOut = Uj(idx, :);
    end

end

function idx = randpermPerPair(M, K, camI, camJ)
    % deterministic per-pair permutation using a local stream (parfor-safe)
    % RANDPERMPERPAIR Deterministic per-pair random sampling indices.
    %   idx = randpermPerPair(M, K, camI, camJ) returns K indices in 1..M
    %   sampled deterministically from a seed derived from `camI` and `camJ`.
    %
    %   Inputs:
    %     - M : number of available items
    %     - K : number to sample (<= M)
    %     - camI, camJ : camera structs used to derive a deterministic seed
    %   Output:
    %     - idx : K×1 vector of indices (integers)

    % --- Arguments validation ---
    arguments
        M (1, 1) double {mustBeInteger, mustBeNonnegative}
        K (1, 1) double {mustBeInteger, mustBeNonnegative}
        camI (1, 1) struct
        camJ (1, 1) struct
    end

    % Build a cheap hash/seed from camera pointers (R address changes; use K,cx,cy)
    ci = double(round(1e3 * camI.K(1, 3) +2e3 * camI.K(2, 3)));
    cj = double(round(1e3 * camJ.K(1, 3) +2e3 * camJ.K(2, 3)));
    seed = mod(uint32(1664525) * uint32(ci) + uint32(1013904223) * uint32(cj), uint32(2 ^ 31 - 1));
    if seed == 0, seed = uint32(1); end

    try
        rs = RandStream('threefry', 'Seed', double(seed)); % fast & stateless
        idx = randperm(rs, M, K);
    catch
        % Fallback if 'threefry' not available
        rs = RandStream('mt19937ar', 'Seed', double(seed));
        idx = randperm(rs, M, K);
    end

end

function idx = gridStratified(Ui, sizeI, Kcap, bins, camI, camJ)
    % Ui: M×2 pixels on image i. bins=[rows cols].
    % GRIDSTRATIFIED Stratified sampling by uniform grid on image i.
    %   idx = gridStratified(Ui, sizeI, Kcap, bins, camI, camJ)
    %
    %   Inputs:
    %     - Ui : M×2 pixel coordinates on image i
    %     - sizeI : image size [H W]
    %     - Kcap : cap on number of samples to return
    %     - bins : [rows cols] grid binning
    %     - camI, camJ : camera structs (optional seeds)
    %   Output:
    %     - idx : indices of selected points from Ui (subset)

    % --- Arguments validation ---
    arguments
        Ui (:, 2) double {mustBeFinite}
        sizeI (1, :) double {mustBeFinite}
        Kcap (1, 1) double {mustBeInteger, mustBeNonnegative}
        bins (1, 2) double {mustBeInteger, mustBePositive}
        camI (1, 1) struct
        camJ (1, 1) struct
    end

    M = size(Ui, 1); %#ok<NASGU>
    rows = max(1, bins(1)); cols = max(1, bins(2));
    H = sizeI(1); W = sizeI(2);

    % Bin each point
    rbin = min(rows, max(1, ceil(Ui(:, 2) * rows / H)));
    cbin = min(cols, max(1, ceil(Ui(:, 1) * cols / W)));
    binId = (rbin - 1) * cols + cbin; % 1..rows*cols

    % Quota per bin (at least 1 if points exist)
    nBins = rows * cols;
    counts = accumarray(binId, 1, [nBins, 1], @sum, 0);
    nonEmpty = find(counts > 0);

    % Distribute cap approximately proportional to counts (with min 1)
    q = zeros(nBins, 1);

    if ~isempty(nonEmpty)
        prop = counts(nonEmpty) / sum(counts(nonEmpty));
        q(nonEmpty) = max(1, round(prop * Kcap));
        % Trim to exact Kcap
        over = sum(q) - Kcap;

        if over > 0
            % remove 1 from the largest bins until sums to Kcap
            [~, ord] = sort(q(nonEmpty), 'descend');
            t = nonEmpty(ord);
            k = 1;

            while over > 0 && k <= numel(t)

                if q(t(k)) > 1
                    q(t(k)) = q(t(k)) - 1; over = over - 1;
                end

                k = k + 1;
            end

        elseif over < 0
            % add to bins with most points
            [~, ord] = sort(counts(nonEmpty), 'descend');
            t = nonEmpty(ord);
            add = -over; k = 1;

            while add > 0 && k <= numel(t)
                q(t(k)) = q(t(k)) + 1; add = add - 1; k = k + 1;
            end

        end

    end

    % Sample within each bin deterministically
    idx = zeros(0, 1);

    for b = 1:nBins
        if q(b) == 0, continue; end
        members = find(binId == b);

        if numel(members) <= q(b)
            idx = [idx; members(:)];
        else
            % per-bin stream using bin id (deterministic)
            seed = uint32(2654435761) * uint32(b);

            try
                rs = RandStream('threefry', 'Seed', double(bitand(seed, 2 ^ 31 - 1)));
            catch
                rs = RandStream('mt19937ar', 'Seed', double(bitand(seed, 2 ^ 31 - 1)));
            end

            pick = members(randperm(rs, numel(members), q(b)));
            idx = [idx; pick(:)];
        end

    end

    % Safety: if due to rounding we got more than Kcap, trim deterministically
    if numel(idx) > Kcap
        idx = idx(1:Kcap);
    end

end

function idx = polarStratified(Ui, sizeI, Kcap, bins, camI, camJ)
    % bins = [nAngles nRadii]; center at principal point if K exists else image center.
    % POLARSTRATIFIED Stratified sampling by polar bins around principal point.
    %   idx = polarStratified(Ui, sizeI, Kcap, bins, camI, camJ)
    %
    %   Inputs:
    %     - Ui : M×2 pixel coordinates on image i
    %     - sizeI : image size [H W]
    %     - Kcap : cap on number of samples
    %     - bins : [nAngles nRadii]
    %     - camI, camJ : camera structs (may provide K principal point)
    %   Output:
    %     - idx : indices selected from Ui

    % --- Arguments validation ---
    arguments
        Ui (:, 2) double {mustBeFinite}
        sizeI (1, :) double {mustBeFinite}
        Kcap (1, 1) double {mustBeInteger, mustBeNonnegative}
        bins (1, 2) double {mustBeInteger, mustBePositive}
        camI (1, 1) struct
        camJ (1, 1) struct
    end

    M = size(Ui, 1); %#ok<NASGU>
    nA = max(1, bins(1)); nR = max(1, bins(2));

    % Center: use intrinsics if available, else image center
    if isfield(camI, 'K') && ~isempty(camI.K)
        cx = camI.K(1, 3); cy = camI.K(2, 3);
    else
        cx = sizeI(2) / 2; cy = sizeI(1) / 2;
    end

    d = Ui - [cx, cy]; % M×2
    ang = atan2(d(:, 2), d(:, 1)); % [-pi, pi]
    ang = mod(ang, 2 * pi); % [0, 2pi)
    rad = hypot(d(:, 1), d(:, 2)); % [0, ~max radius]
    % Normalize radius to [0,1] by max possible extent:
    rmax = hypot(max(cx, sizeI(2) - cx), max(cy, sizeI(1) - cy));
    rnorm = min(1, rad / max(rmax, eps));

    abin = min(nA, max(1, floor(ang / (2 * pi / nA)) + 1));
    rbin = min(nR, max(1, floor(rnorm * nR) + 1));
    binId = (abin - 1) * nR + rbin;
    nBins = nA * nR;

    counts = accumarray(binId, 1, [nBins, 1], @sum, 0);
    nonEmpty = find(counts > 0);

    % Quotas (like gridStratified)
    q = zeros(nBins, 1);

    if ~isempty(nonEmpty)
        prop = counts(nonEmpty) / sum(counts(nonEmpty));
        q(nonEmpty) = max(1, round(prop * Kcap));
        over = sum(q) - Kcap;

        if over > 0
            [~, ord] = sort(q(nonEmpty), 'descend');
            t = nonEmpty(ord);
            k = 1;

            while over > 0 && k <= numel(t)
                if q(t(k)) > 1, q(t(k)) = q(t(k)) - 1; over = over - 1; end
                k = k + 1;
            end

        elseif over < 0
            [~, ord] = sort(counts(nonEmpty), 'descend');
            t = nonEmpty(ord); add = -over; k = 1;

            while add > 0 && k <= numel(t)
                q(t(k)) = q(t(k)) + 1; add = add - 1; k = k + 1;
            end

        end

    end

    % Sample per bin deterministically
    idx = zeros(0, 1);

    for b = 1:nBins
        if q(b) == 0, continue; end
        members = find(binId == b);

        if numel(members) <= q(b)
            idx = [idx; members(:)];
        else
            seed = uint32(2166136261) * uint32(b); % FNV-ish

            try
                rs = RandStream('threefry', 'Seed', double(bitand(seed, 2 ^ 31 - 1)));
            catch
                rs = RandStream('mt19937ar', 'Seed', double(bitand(seed, 2 ^ 31 - 1)));
            end

            pick = members(randperm(rs, numel(members), q(b)));
            idx = [idx; pick(:)];
        end

    end

    if numel(idx) > Kcap
        idx = idx(1:Kcap);
    end

end

function [Phi, pmap] = buildDeltaVector(cameras, camList, seed)
    % BUILDDELTAVECTOR Create zeroed parameter vector and mapping for cameras.
    %   [Phi, pmap] = buildDeltaVector(cameras, camList, seed) returns an
    %   all-zero increment vector `Phi` and a parameter map `pmap` that
    %   describes for each camera in `camList` the starting index in `Phi`
    %   and whether that camera is the seed (seed only has a single df DOF).
    %
    %   Inputs:
    %     - cameras : struct array of camera parameters
    %     - camList : vector of camera indices to include
    %     - seed    : scalar index of the seed camera
    %
    %   Outputs:
    %     - Phi  : column vector of zeros sized to the total DOFs
    %     - pmap : struct array with fields `camIdx`, `startIdx`, `isSeed`

    % --- Arguments validation ---
    arguments
        cameras (1, :) struct
        camList (1, :) double {mustBeInteger, mustBePositive}
        seed (1, 1) double {mustBeInteger, mustBePositive}
    end

    % All-zero increments around current cameras
    pmap = struct('camIdx', {}, 'startIdx', {}, 'isSeed', false);
    Phi = [];
    idx = 1;

    for k = 1:numel(camList)
        i = camList(k);
        pmap(k).camIdx = i;
        pmap(k).startIdx = idx;

        if i == seed
            pmap(k).isSeed = true;
            Phi = [Phi; 0]; % df only
            idx = idx + 1;
        else
            pmap(k).isSeed = false;
            Phi = [Phi; 0; 0; 0; 0]; % [dθx dθy dθz df]
            idx = idx + 4;
        end

    end

end

function camsOut = applyIncrements(cameras, Phi, pmap, camList, seed)
    % APPLYINCREMENTS Apply parameter increments to camera structs.
    %   camsOut = applyIncrements(cameras, Phi, pmap, camList, seed)
    %
    %   Inputs:
    %     - cameras : 1×N struct array of camera parameters
    %     - Phi     : parameter increment column vector (from buildDeltaVector)
    %     - pmap    : parameter map describing start indices and seed flags
    %     - camList : list of camera indices included in pmap
    %     - seed    : (optional) seed index (not used in this function but kept for API)
    %
    %   Output:
    %     - camsOut : cameras struct array with updated `R`, `f`, and `K` fields

    % --- Arguments validation ---
    arguments
        cameras (1, :) struct
        Phi (:, 1) double {mustBeFinite}
        pmap (1, :) struct
        camList (1, :) double {mustBeInteger, mustBePositive}
        seed = []
    end

    % Touch possibly unused arg
    seed = seed; %#ok<NASGU>
    camsOut = cameras;

    for k = 1:numel(pmap)
        i = pmap(k).camIdx;
        s = pmap(k).startIdx;

        % Ensure cx,cy are cached (in case older structs slip in)
        if ~isfield(camsOut(i), 'cx') || isempty(camsOut(i).cx)
            % Fallback to current K if present, else image center is needed;
            % best practice: call prepareCameraCache(...) once before LM.
            if isfield(camsOut(i), 'K') && ~isempty(camsOut(i).K)
                camsOut(i).cx = camsOut(i).K(1, 3);
                camsOut(i).cy = camsOut(i).K(2, 3);
            else
                % last-resort defaults (won't be used if you pre-cache properly)
                camsOut(i).cx = 0; camsOut(i).cy = 0;
            end

        end

        cx = camsOut(i).cx; cy = camsOut(i).cy;

        if pmap(k).isSeed
            % Seed: df only
            df = Phi(s);
            oldf = camsOut(i).f;
            f = max(100, min(5000, oldf + df));

            if abs(f - oldf) > 1e-9
                camsOut(i).f = f;
                Ki = camsOut(i).K;
                if isempty(Ki), Ki = eye(3); end
                Ki(1, 1) = f; Ki(2, 2) = f;
                camsOut(i).K = Ki; % cx,cy unchanged
            end

        else
            % Non-seed: [dθx dθy dθz df]
            dth = Phi(s:s + 2);
            df = Phi(s + 3);

            % SO(3) increment (left-multiplicative)
            a = norm(dth);

            if a < 1e-12
                Rupd = eye(3) + skewSymmetric(dth);
            else
                v = dth / a; K = skewSymmetric(v);
                Rupd = eye(3) + sin(a) * K + (1 - cos(a)) * (K * K);
            end

            camsOut(i).R = Rupd * camsOut(i).R;

            % --- Micro-opt: only update K if f changed significantly ---
            oldf = camsOut(i).f;
            f = max(100, min(5000, oldf + df));

            if abs(f - oldf) > 1e-9
                camsOut(i).f = f;
                Ki = camsOut(i).K;
                if isempty(Ki), Ki = eye(3); end
                Ki(1, 1) = f; Ki(2, 2) = f;
                camsOut(i).K = Ki; % cx,cy unchanged
            end

        end

    end

end

function CpInv = buildBrownLowePrior(camList, seed, cameras, opts, pmap)
    % BUILDBROWnLOWEPRIOR Build prior precision matrix for Brown–Lowe regularizers.
    %   CpInv = buildBrownLowePrior(camList, seed, cameras, opts, pmap)
    %
    %   Inputs:
    %     - camList : vector of camera indices included in the optimization
    %     - seed    : seed camera index
    %     - cameras : struct array of camera parameters
    %     - opts    : options struct (may include FocalSmoothnessWeight, FocalMeanWeight)
    %     - pmap    : parameter map (from buildDeltaVector)
    %
    %   Output:
    %     - CpInv : sparse prior precision matrix to be added to H

    arguments
        camList (1, :) double {mustBeInteger, mustBePositive}
        seed (1, 1) double {mustBeInteger, mustBePositive}
        cameras (1, :) struct
        opts struct
        pmap (1, :) struct
    end

    fbar = mean([cameras(camList).f]);
    sigth = pi / 16;
    sigfGlobal = max(1, fbar / 20);

    P = pmap(end).startIdx + (pmap(end).isSeed == false) * 3;
    numCams = numel(camList);

    % === Extract info from pmap (vectorized) ===
    starts = [pmap.startIdx];
    isSeed = [pmap.isSeed];
    camIndices = [pmap.camIdx];

    focalCols = zeros(1, numCams);
    rotColsStart = zeros(1, numCams);
    hasRot = false(1, numCams);

    diagvals = zeros(P, 1);

    for k = 1:numCams
        s = starts(k);

        if isSeed(k)
            diagvals(s) = 1 / (sigfGlobal ^ 2);
            focalCols(k) = s;
        else
            diagvals(s:s + 2) = 1 / (sigth ^ 2);
            diagvals(s + 3) = 1 / (sigfGlobal ^ 2);
            focalCols(k) = s + 3;
            rotColsStart(k) = s;
            hasRot(k) = true;
        end

    end

    CpInv = spdiags(diagvals, 0, P, P);

    % === Build all smoothness constraints at once ===
    I = []; J = []; V = [];

    % === FOCAL SMOOTHNESS ===
    if isfield(opts, 'FocalSmoothnessWeight') && ...
            isnumeric(opts.FocalSmoothnessWeight) && ...
            opts.FocalSmoothnessWeight > 0

        lf = opts.FocalSmoothnessWeight;

        % Find all adjacent pairs (distance <= 2)
        pairList = [];

        for ki = 1:numCams - 1

            for kj = ki + 1:min(ki + 2, numCams)

                if abs(camIndices(ki) - camIndices(kj)) <= 2
                    pairList = [pairList; ki, kj];
                end

            end

        end

        if ~isempty(pairList)
            numPairs = size(pairList, 1);

            % Vectorized assembly for all pairs
            kiList = pairList(:, 1);
            kjList = pairList(:, 2);

            fiList = focalCols(kiList)';
            fjList = focalCols(kjList)';

            % Each pair contributes 4 entries: [1 -1; -1 1] * lambda
            I = [I; fiList; fjList; fiList; fjList];
            J = [J; fiList; fjList; fjList; fiList];
            V = [V; repmat(lf, numPairs, 1); repmat(lf, numPairs, 1);
                 repmat(-lf, numPairs, 1); repmat(-lf, numPairs, 1)];
        end

    end

    % === GLOBAL MEAN (optimized with outer product) ===
    if isfield(opts, 'FocalMeanWeight') && ...
            isnumeric(opts.FocalMeanWeight) && ...
            opts.FocalMeanWeight > 0

        lm = opts.FocalMeanWeight;
        fc = focalCols;
        nf = numCams;

        % Diagonal contribution
        I = [I; fc'];
        J = [J; fc'];
        V = [V; repmat(lm * (nf - 1) / nf, nf, 1)];

        % Off-diagonal: -lambda/n for all pairs
        % Use repelem/repmat for vectorized construction
        fcRow = repelem(fc, nf); % [f1 f1 ... f1 f2 f2 ... f2 ...]
        fcCol = repmat(fc, 1, nf); % [f1 f2 ... fn f1 f2 ... fn ...]

        % Remove diagonal entries
        offDiag = fcRow ~= fcCol;

        I = [I; fcRow(offDiag)'];
        J = [J; fcCol(offDiag)'];
        V = [V; repmat(-lm / nf, sum(offDiag), 1)];
    end

    % === Assemble ===
    if ~isempty(I)
        CpInv = CpInv + sparse(I, J, V, P, P);
    end

end

%---------------------------- Minimal to no for loops ------------------------------------
% Single residual and Jacobian
function [r, JObs, JSrc] = computeSingleResidual(uObs, uSrc, camObs, camSrc, ...
        imageSizeObs, imageSizeSrc)
    % COMPUTESINGLERESIDUAL Compute residual and Jacobians for one correspondence.
    %   [r, JObs, JSrc] = computeSingleResidual(uObs, uSrc, camObs, camSrc,
    %   imageSizeObs, imageSizeSrc)
    %
    %   Inputs:
    %     - uObs, uSrc : 2×1 or 1×2 image coordinates of the observed and source points
    %     - camObs, camSrc : camera structs for observation and source
    %     - imageSizeObs, imageSizeSrc : image sizes [H W]
    %
    %   Outputs:
    %     - r    : 2×1 residual vector (uObs - projected uSrc)
    %     - JObs : 2×4 Jacobian w.r.t observation camera params [dθx dθy dθz df]
    %     - JSrc : 2×4 Jacobian w.r.t source camera params

    % --- Arguments validation ---
    arguments
        uObs double {mustBeFinite}
        uSrc double {mustBeFinite}
        camObs (1, 1) struct
        camSrc (1, 1) struct
        imageSizeObs (1, :) double {mustBeFinite}
        imageSizeSrc (1, :) double {mustBeFinite}
    end

    uSrch = [uSrc; 1]; % Homogeneous coordinates

    % Project from src to obs: Brown-Lowe Eq. 15
    pH = camObs.K * camObs.R * camSrc.R' * (camSrc.K \ uSrch);

    % Dehomogenize
    if abs(pH(3)) < 1e-10
        pH(3) = 1e-10;
    end

    p = pH(1:2) / pH(3);

    % Residual: Brown-Lowe Eq. 14
    r = uObs - p;

    % Jacobians: Brown-Lowe Eq. 20-23
    JObs = computeJacobianWrtCamera(uSrch, camObs, camSrc, pH, 'obs');
    JSrc = computeJacobianWrtCamera(uSrch, camObs, camSrc, pH, 'src');
end

% Jacobian w.r.t. camera parameters
function J = computeJacobianWrtCamera(uSrch, camObs, camSrc, pH, type)
    % COMPUTEJACOBIANWRTCAMERA Jacobian of reprojection w.r.t. camera params.
    %   J = computeJacobianWrtCamera(uSrch, camObs, camSrc, pH, type)
    %
    %   Inputs:
    %     - uSrch : 3×1 homogeneous source coordinate [x; y; 1]
    %     - camObs, camSrc : camera structs (obs and src)
    %     - pH : 3×1 homogeneous projected point before dehomogenization
    %     - type : 'obs' or 'src' indicating which camera the Jacobian is for
    %
    %   Output:
    %     - J : 2×4 Jacobian matrix [dθx dθy dθz df]

    % --- Arguments validation ---
    arguments
        uSrch (3, 1) double {mustBeFinite}
        camObs (1, 1) struct
        camSrc (1, 1) struct
        pH (3, 1) double {mustBeFinite}
        type {mustBeTextScalar}
    end

    x = pH(1);
    y = pH(2);
    z = pH(3);

    if abs(z) < 1e-10
        z = 1e-10;
    end

    % Dehomogenization Jacobian: Brown-Lowe Eq. 21
    Jdehom = [1 / z, 0, -x / z ^ 2;
              0, 1 / z, -y / z ^ 2];

    % Residual Jacobian: dr/dp = -I
    Jchain = -Jdehom;

    if strcmp(type, 'obs')
        % Observation camera Jacobian

        % Rotation Jacobian: Brown-Lowe Eq. 22-23
        Jrot = zeros(2, 3);

        for m = 1:3
            eM = zeros(3, 1);
            eM(m) = 1;
            skewEm = skewSymmetric(eM);

            % ∂R/∂θ = R [eM]×
            dphDtheta = camObs.K * camObs.R * skewEm * camSrc.R' * ...
                (camSrc.K \ uSrch);

            Jrot(:, m) = Jchain * dphDtheta;
        end

        % Focal length Jacobian
        dKdf = [1, 0, 0; 0, 1, 0; 0, 0, 0];
        dphdf = dKdf * camObs.R * camSrc.R' * (camSrc.K \ uSrch);
        Jf = Jchain * dphdf;

        J = [Jrot, Jf]; % 2×4

    else
        % Source camera Jacobian
        Jrot = zeros(2, 3);

        for m = 1:3
            eM = zeros(3, 1);
            eM(m) = 1;
            skewEm = skewSymmetric(eM);

            % ∂(R^T)/∂θ = -R^T [eM]×
            dphDtheta = camObs.K * camObs.R * (-camSrc.R' * skewEm) * ...
                (camSrc.K \ uSrch);

            Jrot(:, m) = Jchain * dphDtheta;
        end

        % Focal length Jacobian (through K^{-1})
        f = camSrc.f;
        cx = camSrc.cx; % cached in prepareCameraCache
        cy = camSrc.cy;
        dKinvdf = [-1 / f ^ 2, 0, cx / f ^ 2;
                   0, -1 / f ^ 2, cy / f ^ 2;
                   0, 0, 0];
        % dKinvdf = [ -1/f^2,  0,       0;
        %       0,     -1/f^2,   0;
        %       0,      0,        0     ];
        dphdf = camObs.K * camObs.R * camSrc.R' * dKinvdf * uSrch;
        Jf = Jchain * dphdf;

        J = [Jrot, Jf]; % 2×4
    end

end

% Skew-symmetric matrix
function S = skewSymmetric(v)
    % SKEWSYMMETRIC Compute a 3x3 skew-symmetric matrix from a 3-vector.
    %   S = skewSymmetric(v) returns the matrix S such that S*x = v x x
    %   (cross-product as matrix multiplication).
    %
    %   Input:
    %     - v : 3×1 (or 1×3) numeric vector
    %   Output:
    %     - S : 3×3 skew-symmetric matrix

    % --- Arguments validation ---
    arguments
        v (3, 1) double {mustBeFinite}
    end

    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
         -v(2), v(1), 0];
end

% Huber weight function
function w = huberWeight(residualNorm, sigma)
    % HUBERWEIGHT Huber weighting for a residual norm.
    %   w = huberWeight(residualNorm, sigma) returns the scalar weight `w`
    %   used to re-weight residuals according to the Huber function.
    %
    %   Inputs:
    %     - residualNorm : scalar norm of the residual (>=0)
    %     - sigma        : positive threshold parameter
    %   Output:
    %     - w : scalar weight in (0, 1], where values <1 downweight outliers

    % --- Arguments validation ---
    arguments
        residualNorm (1, 1) double {mustBeFinite, mustBeNonnegative}
        sigma (1, 1) double {mustBeFinite, mustBePositive}
    end

    if residualNorm < sigma
        w = 1; % L2 for inliers
    else
        w = sigma / residualNorm; % L1 for outliers
    end

end

function cameras = prepareCameraCache(cameras, imageSizes)
    % PREPARECAMERACACHE Cache principal point per camera (cx,cy) once.
    %   cameras = prepareCameraCache(cameras, imageSizes) ensures each camera
    %   struct has fields `cx`, `cy` and a consistent `K` matrix based on
    %   its focal length.
    %
    %   Inputs:
    %     - cameras : 1×N struct array (may lack cx/cy/K)
    %     - imageSizes : N×3 numeric array with image heights and widths
    %   Output:
    %     - cameras : updated struct array with cached `cx`, `cy`, and `K`

    % --- Arguments validation ---
    arguments
        cameras (1, :) struct
        imageSizes (:, 3) double {mustBeFinite}
    end

    % If K already has cx,cy, keep them; otherwise, use image center.

    N = numel(cameras);

    for i = 1:N

        if ~isfield(cameras(i), 'cx') || isempty(cameras(i).cx)

            if isfield(cameras(i), 'K') && ~isempty(cameras(i).K)
                cx = cameras(i).K(1, 3); cy = cameras(i).K(2, 3);
            else
                % imageSizes(i,:) = [H W ...]
                cx = imageSizes(i, 2) / 2;
                cy = imageSizes(i, 1) / 2;
            end

            cameras(i).cx = cx;
            cameras(i).cy = cy;

            % normalize K to use these cached cx,cy
            if isfield(cameras(i), 'f') && ~isempty(cameras(i).f)
                f = cameras(i).f;
                cameras(i).K = [f, 0, cx; 0, f, cy; 0, 0, 1];
            end

        end

    end

end

% Build intrinsic matrix
function K = buildIntrinsicMatrix(f, imageSize)
    % BUILDINTRINSICMATRIX Create 3×3 intrinsics with principal point at center.
    %   K = buildIntrinsicMatrix(f, imageSize)
    %
    %   Inputs:
    %     - f : scalar focal length in pixels
    %     - imageSize : [H W] image size
    %   Output:
    %     - K : 3×3 intrinsic matrix [f 0 cx; 0 f cy; 0 0 1]

    % --- Arguments validation ---
    arguments
        f (1, 1) double {mustBeFinite, mustBePositive}
        imageSize (1, :) double {mustBeFinite}
    end

    cx = imageSize(2) / 2;
    cy = imageSize(1) / 2;
    K = [f, 0, cx;
         0, f, cy;
         0, 0, 1];
end

% Extract Euler angles from rotation matrix
function [yaw, pitch, roll] = extractEulerAngles(R)
    % EXTRACTEULERANGLES Return ZYX Euler angles from a rotation matrix (rad).
    %   [yaw, pitch, roll] = extractEulerAngles(R)
    %
    %   Input:
    %     - R : 3×3 rotation matrix
    %   Outputs:
    %     - yaw, pitch, roll : Euler angles in radians (ZYX convention)

    % --- Arguments validation ---
    arguments
        R (3, 3) double {mustBeFinite}
    end

    % ZYX Euler convention
    sy = sqrt(R(1, 1) ^ 2 + R(2, 1) ^ 2);

    if sy > 1e-6
        yaw = atan2(R(3, 2), R(3, 3));
        pitch = atan2(-R(3, 1), sy);
        roll = atan2(R(2, 1), R(1, 1));
    else
        yaw = atan2(-R(2, 3), R(2, 2));
        pitch = atan2(-R(3, 1), sy);
        roll = 0;
    end

end

function rotation = projectToSO3(M)
    % PROJECTTOSO3 Project a matrix to the closest proper rotation via SVD.
    %   rotation = projectToSO3(M)
    %
    %   Input:
    %     - M : 3×3 matrix (typically approximate rotation)
    %   Output:
    %     - rotation : closest proper rotation matrix (det(rotation)=+1)

    arguments
        M (3, 3) double {mustBeFinite}
    end

    % SVD projection to the closest proper rotation.
    [U, ~, V] = svd(M);
    rotation = U * diag([1, 1, sign(det(U * V'))]) * V';
end
