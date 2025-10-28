function [cameras, seed] = bundleAdjustmentLM(input, matches, keypoints, ...
        imageSizes, initialTforms, varargin)
    % BUNDLEADJUSTMENTLM Incremental LM bundle adjustment for panoramic stitching.
    %   [cameras, seed] = bundleAdjustmentLM(input, matches, keypoints, imageSizes, initialTforms, ...)
    %   Grows the panorama graph from a seed image and refines per-image rotations
    %   (R_i) and focal lengths (f_i) using Levenberg–Marquardt with Huber loss.
    %   The seed image’s rotation is fixed to identity to anchor the gauge.
    %
    %   Inputs
    %   - input         : struct with fields controlling initialization and LM
    %                     (e.g., transformationType, focalEstimateMethod, robust/damping flags).
    %   - matches       : N-by-N cell; matches{i,j} is M_ij-by-2 [idx_in_keypoints{i}, idx_in_keypoints{j}].
    %   - keypoints     : 1-by-N (or N-by-1) cell; keypoints{i} = K_i-by-2 [x,y] pixel coordinates.
    %   - imageSizes    : N-by-2 array [H W] per image.
    %   - initialTforms : [] | N-by-N cell (projective2d or 3x3 H) | struct array with fields .i,.j,.H (j->i).
    %
    %   Name-Value options (varargin)
    %   - 'SigmaHuber'  : Huber threshold (pixels). Default 2.0
    %   - 'MaxLMIters'  : LM iterations per growth step. Default 30
    %   - 'Lambda0'     : Initial LM damping. Default 1e-3
    %   - 'PriorSigmaF' : Focal prior stdev; empty uses mean/median heuristic
    %   - 'Verbose'     : 0/1. Default 1
    %   - 'userSeed'    : Optional fixed seed image index. Default: automatic (max degree)
    %
    %   Outputs
    %   - cameras: 1-by-N struct array with fields
    %       R           — 3x3 rotations (world->camera)
    %       f           — scalar focal length (pixels)
    %       K           — 3x3 intrinsics [f 0 cx; 0 f cy; 0 0 1]
    %       initialized — logical flag
    %   - seed   : index of the seed image used

    arguments
        input (1, 1) struct
        matches (:, :) cell
        keypoints cell
        imageSizes (:, 2) {mustBeNumeric, mustBeFinite}
        initialTforms
    end

    arguments (Repeating)
        varargin
    end

    % ------------------ Options ------------------
    p = inputParser;
    p.addParameter('SigmaHuber', 2.0);
    p.addParameter('MaxLMIters', 30);
    p.addParameter('Lambda0', 1e-3);
    p.addParameter('PriorSigmaF', []);
    p.addParameter('Verbose', 1);
    p.addParameter('userSeed', []);
    p.parse(varargin{:});
    opt = p.Results;

    % Debug input matches
    [n1, n2] = size(matches);

    N = numel(keypoints);
    opt.MaxGrow = N;

    % Verify matches cell array is square
    assert(n1 == n2, 'Matches cell array must be square (NxN)');
    num_images = n1;

    % Verify keypoints array size
    assert(size(keypoints, 2) >= num_images, 'Keypoints array must have entry for each image');

    % ------------------ Build pair list (compact) ------------------
    pairs = build_pairs_from_cells(matches, keypoints); % each pair has i,j, Ui (px), Uj (px); i<j

    % ------------------ Choose seed (image with max degree / matches) ------------------
    % Optionally override the automatic seed selection
    if ~isempty(opt.userSeed)
        seed = opt.userSeed;

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

    % ------------------ Initialize cameras via your function ------------------
    cameras = initializeCameraMatrices(input, pairs, imageSizes, initialTforms, seed, num_images);
    cameras(seed).initialized = true;

    if opt.Verbose
        fprintf('Seed image: %d | initial f≈%.2f px\n', seed, cameras(seed).f);
    end

    % Internal rotation parameterization (axis-angle per image)
    theta = zeros(N, 3);

    % Keep R matrices aligned with theta; seed is fixed to identity
    for i = 1:N
        if isempty(cameras(i).R), cameras(i).R = eye(3); end
    end

    cameras(seed).R = eye(3); theta(seed, :) = [0 0 0];

    % ------------------ Incremental growth ------------------

    growSteps = 1;

    while any(~[cameras.initialized]) && growSteps < opt.MaxGrow
        [nextImg, bestNbr] = pick_next_image([cameras.initialized], pairs, N);

        if isempty(nextImg)
            if opt.Verbose, fprintf('No more connected images to add.\n'); end
            break;
        end

        % Initialize new image from its best neighbor
        cameras(nextImg).R = cameras(bestNbr).R;
        cameras(nextImg).f = cameras(bestNbr).f;
        cx = imageSizes(nextImg, 2) / 2; cy = imageSizes(nextImg, 1) / 2;
        cameras(nextImg).K = [cameras(nextImg).f, 0, cx; 0, cameras(nextImg).f, cy; 0, 0, 1];
        cameras(nextImg).initialized = true;
        theta(nextImg, :) = theta(bestNbr, :);

        if opt.Verbose
            fprintf('>> Added image %d (best neighbor %d). Active = %d\n', ...
                nextImg, bestNbr, nnz([cameras.initialized]));
        end

        % LM over all active (seed rotation hard-fixed)
        active = find([cameras.initialized]);
        [theta, cameras] = run_lm(active, theta, cameras, pairs, imageSizes, seed, input, opt);

        growSteps = growSteps + 1;
    end

    % ------------------ After all growth + LM steps ------------------
    rms_stats = check_w2c_sanity(pairs, cameras, imageSizes);
    fprintf('Final w2c RMS mean = %.3f px\n', rms_stats.mean);

    compare_conventions(pairs, cameras, imageSizes);

end

function stats = check_w2c_sanity(pairs, cameras, imageSizes, num_pairs)
    % CHECK_W2C_SANITY Probe reprojection RMS under w2c convention on random pairs.
    %   stats = check_w2c_sanity(pairs, cameras, imageSizes)
    %   stats = check_w2c_sanity(pairs, cameras, imageSizes, num_pairs)
    %   Returns a struct with fields mean, median, max, n based on a random
    %   subset of pairwise reprojections using world-to-camera rotations.
    %
    %   Inputs
    %   - pairs       : struct array with fields .i,.j,.Ui,.Uj (pixel coords)
    %   - cameras     : 1xN struct array with fields .R (3x3), .f (scalar)
    %   - imageSizes  : N-by-2 [H W]
    %   - num_pairs   : optional number of random pairs to evaluate
    %
    %   Output
    %   - stats: struct with fields mean, median, max, n

    arguments
        pairs (1, :) struct
        cameras (1, :) struct
        imageSizes (:, 2) {mustBeNumeric, mustBeFinite}
        num_pairs (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite} = min(20, numel(pairs))
    end

    if nargin < 4, num_pairs = min(20, numel(pairs)); end
    idx = randperm(numel(pairs), num_pairs);

    rms_w2c = nan(num_pairs, 1);
    z_eps = 1e-6;

    for k = 1:num_pairs
        t = idx(k);
        i = pairs(t).i; j = pairs(t).j;
        Ui = pairs(t).Ui; Uj = pairs(t).Uj;

        Ri = cameras(i).R; Rj = cameras(j).R; % w2c
        [Ki, ~] = make_K_and_inv(cameras(i).f, imageSizes(i, :));
        [~, KjInv] = make_K_and_inv(cameras(j).f, imageSizes(j, :));

        % j -> i (w2c)
        yj = [Uj, ones(size(Uj, 1), 1)] * KjInv.'; % rays in cam j
        XYZ = ((yj * Rj) * Ri.') * Ki.'; % pre-divide coords in cam i pixel space
        z = XYZ(:, 3);
        good = z > z_eps; % only front-facing / non-grazing
        if ~any(good), continue; end

        uv = XYZ(good, 1:2) ./ z(good);
        err = Ui(good, :) - uv;
        rms_w2c(k) = sqrt(mean(sum(err .^ 2, 2)));
    end

    % robust summarize
    rms_w2c = rms_w2c(isfinite(rms_w2c));
    stats.mean = mean(rms_w2c);
    stats.median = median(rms_w2c);
    stats.max = max(rms_w2c);
    stats.n = numel(rms_w2c);

    fprintf('w2c RMS (z>0 only): mean=%.3f  median=%.3f  max=%.3f  (n=%d)\n', ...
        stats.mean, stats.median, stats.max, stats.n);
end

function compare_conventions(pairs, cameras, imageSizes, num_pairs)
    % COMPARE_CONVENTIONS Compare RMS if rotations are w2c vs c2w by probing pairs.
    %   compare_conventions(pairs, cameras, imageSizes)
    %   compare_conventions(pairs, cameras, imageSizes, num_pairs)
    %   Prints mean RMS under both assumptions to help diagnose convention.
    %
    %   Inputs
    %   - pairs       : struct array with fields .i,.j,.Ui,.Uj
    %   - cameras     : 1xN struct array with .R (3x3), .f (scalar)
    %   - imageSizes  : N-by-2 [H W]
    %   - num_pairs   : optional number of random pairs

    arguments
        pairs (1, :) struct
        cameras (1, :) struct
        imageSizes (:, 2) {mustBeNumeric, mustBeFinite}
        num_pairs (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite} = min(20, numel(pairs))
    end

    if nargin < 4, num_pairs = min(20, numel(pairs)); end
    idx = randperm(numel(pairs), num_pairs);

    rms_w2c = zeros(num_pairs, 1);
    rms_c2w = zeros(num_pairs, 1);

    for k = 1:num_pairs
        t = idx(k);
        i = pairs(t).i; j = pairs(t).j;
        Ui = pairs(t).Ui; Uj = pairs(t).Uj;

        Ri = cameras(i).R; Rj = cameras(j).R;
        [Ki, ~] = make_K_and_inv(cameras(i).f, imageSizes(i, :));
        [~, KjInv] = make_K_and_inv(cameras(j).f, imageSizes(j, :));

        % ---- w2c chain (your BA/renderer convention) ----
        yj = [Uj, ones(size(Uj, 1), 1)] * KjInv.'; % rays in cam j
        uvW = ((yj * Rj) * Ri.') * Ki.'; % j->world->i->pix
        uvW = uvW(:, 1:2) ./ uvW(:, 3);
        rms_w2c(k) = sqrt(mean(sum((Ui - uvW) .^ 2, 2)));

        % ---- c2w chain (if R is actually c2w) ----
        uvC = ((yj * Rj.') * Ri) * Ki.'; % j->world->i->pix (c2w)
        uvC = uvC(:, 1:2) ./ uvC(:, 3);
        rms_c2w(k) = sqrt(mean(sum((Ui - uvC) .^ 2, 2)));
    end

    fprintf('Probe: w2c mean RMS=%.3f px | c2w mean RMS=%.3f px (n=%d)\n', ...
        mean(rms_w2c), mean(rms_c2w), num_pairs);
end

% ======================================================================
% ======================  BUILD PAIR LIST FROM CELLS  ==================
% ======================================================================
function pairs = build_pairs_from_cells(matches, keypoints)
    % BUILD_PAIRS_FROM_CELLS Convert matches/keypoints cells to a compact pair list.
    %   pairs = build_pairs_from_cells(matches, keypoints)
    %   For each i<j with non-empty matches, collects Ui, Uj pixel coordinates.
    %
    %   Inputs
    %   - matches   : N-by-N cell; matches{i,j} is Mx2 indices into keypoints{i/j}
    %   - keypoints : 1xN or Nx1 cell; keypoints{i} is 2xK_i (x;y) or K_i-by-2
    %
    %   Output
    %   - pairs     : struct with fields .i,.j,.Ui,.Uj (each Mx2)

    arguments
        matches (:, :) cell
        keypoints cell
    end

    N = numel(keypoints);
    plist = [];

    for i = 1:N

        for j = i + 1:N
            Mij = matches{i, j}';
            if isempty(Mij), continue; end
            Ui = keypoints{i}(:, Mij(:, 1))'; % M×2 pixels
            Uj = keypoints{j}(:, Mij(:, 2))';
            if isempty(Ui), continue; end
            pr.i = i; pr.j = j; pr.Ui = Ui; pr.Uj = Uj;
            plist = [plist, pr]; %#ok<AGROW>
        end

    end

    pairs = plist;
end

% ======================================================================
% =========================  PICK NEXT IMAGE  ==========================
% ======================================================================
function [nxt, bestNbr] = pick_next_image(initialized, pairs, N)
    % PICK_NEXT_IMAGE Choose the next uninitialized image with highest match score.
    %   [nxt, bestNbr] = pick_next_image(initialized, pairs, N)
    %   Scores each not-yet-initialized image by total matches to initialized ones
    %   and returns the best image and its best neighbor.
    %
    %   Inputs
    %   - initialized : 1xN logical mask of active images
    %   - pairs       : struct array with fields .i,.j,.Ui
    %   - N           : total number of images
    %
    %   Outputs
    %   - nxt      : selected image index or [] if none
    %   - bestNbr  : the neighbor that scored it best (or first active)

    arguments
        initialized (1, :) logical
        pairs (1, :) struct
        N (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
    end

    scores = zeros(1, N);
    nbr = zeros(1, N);
    act = find(initialized);

    for t = 1:numel(pairs)
        i = pairs(t).i; j = pairs(t).j; M = size(pairs(t).Ui, 1);

        if initialized(i) && ~initialized(j)
            scores(j) = scores(j) + M; nbr(j) = i;
        elseif initialized(j) && ~initialized(i)
            scores(i) = scores(i) + M; nbr(i) = j;
        end

    end

    scores(initialized) = -inf;
    [~, nxt] = max(scores);
    if isinf(scores(nxt)), nxt = []; bestNbr = []; return; end
    bestNbr = nbr(nxt); if bestNbr == 0, bestNbr = act(1); end
end

% ======================================================================
% =============================  LM CORE  ==============================
% ======================================================================
function [thetaA, cameras] = run_lm(active, theta0, cameras, pairs, imageSizes, seed, input, opt)
    % RUN_LM Levenberg–Marquardt over active cameras with robust Huber loss.
    %   [thetaA, cameras] = run_lm(active, theta0, cameras, pairs, imageSizes, seed, input, opt)
    %   Optimizes 3D rotations (axis-angle) and focal(s) for active images.
    %
    %   Inputs
    %   - active      : vector of image indices to optimize
    %   - theta0      : N-by-3 initial axis-angle parameters
    %   - cameras     : 1xN struct (.R, .f, .K, .initialized)
    %   - pairs       : struct array (.i,.j,.Ui,.Uj)
    %   - imageSizes  : N-by-2 [H W]
    %   - seed        : image index with fixed rotation (and optionally focal)
    %   - input,opt   : option structs controlling priors and damping
    %
    %   Outputs
    %   - thetaA      : updated axis-angles (same size as theta0)
    %   - cameras     : updated camera structs

    arguments
        active (1, :) double {mustBeInteger, mustBePositive, mustBeFinite}
        theta0 (:, 3) double {mustBeFinite}
        cameras (1, :) struct
        pairs (1, :) struct
        imageSizes (:, 2) double {mustBeFinite}
        seed (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
        input (1, 1) struct
        opt (1, 1) struct
    end

    thetaA = theta0;

    lambda = opt.Lambda0;
    sigHub = opt.SigmaHuber;

    % f_ref = median([cameras.f]);   % or cameras(seed).f most stable (ties everyone to the seed's f)
    % BEFORE the LM iterations in run_lm (once):
    % before LM loop (once per growth step):
    act0 = active;
    f_ref_step = median([cameras(act0).f]); % fixed reference

    % --- Fixed Brown–Lowe scale for f (used for LM damping each iteration)
    if isempty(opt.PriorSigmaF)
        sigma_f_ref = max(1e-6, f_ref_step / 10); % stable reference for this growth step
    else
        sigma_f_ref = opt.PriorSigmaF;
    end

    kappa = 0.18; % allows ~±25 % moves
    sigma_f_prior = max(1e-6, kappa * f_ref_step);
    w_fprior = 1 / (sigma_f_prior ^ 2);

    for it = 1:opt.MaxLMIters
        [H0, g, Eold, nres] = accumulate_normal_equations(thetaA, cameras, pairs, active, imageSizes, sigHub);

        % --- apples-to-apples metrics like OpenPano ---
        [R_abs_mean, R_abs_max] = avg_abs_residual(thetaA, cameras, pairs, active, imageSizes);

        if opt.Verbose
            fprintf('   avg|r|=%.3f px  max|r|=%.3f px\n', R_abs_mean, R_abs_max);
        end

        % --- choose a per-iteration focal center (adaptive soft prior center)
        % (damping still uses the fixed sigma_f_ref set before the loop)
        if input.UseFocalSoftPrior

            if input.UseIterFocalCenter
                f_ref_center = median([cameras(active).f]);
            else
                f_ref_center = f_ref_step; % your fixed value from before the loop
            end

            for k = 1:numel(active)
                f_idx = (k - 1) * 4 + 4;
                fi = cameras(active(k)).f;
                H0(f_idx, f_idx) = H0(f_idx, f_idx) + w_fprior;
                g(f_idx) = g(f_idx) + w_fprior * (fi - f_ref_center);
            end

        end

        if input.UseFocalSprings
            % build a quick active-set mask
            actMask = false(1, numel(cameras)); actMask(active) = true;

            % pick a spring strength relative to the prior
            alpha = 0.35; % very weak vs prior
            % estimate a robust scale of inlier counts to normalize weights
            mvec = [];

            for t = 1:numel(pairs)

                if actMask(pairs(t).i) && actMask(pairs(t).j)
                    mvec(end + 1) = size(pairs(t).Ui, 1); %#ok<AGROW> % adapt to your field name
                end

            end

            m0 = max(1, median(mvec));

            for t = 1:numel(pairs)
                i = pairs(t).i; j = pairs(t).j;
                if ~(actMask(i) && actMask(j)), continue; end

                ki = find(active == i, 1); kj = find(active == j, 1);
                ii = (ki - 1) * 4 + 4; jj = (kj - 1) * 4 + 4;

                mu = alpha * (size(pairs(t).Ui, 1) / m0) * w_fprior;

                % add mu*(f_i - f_j)^2
                H0(ii, ii) = H0(ii, ii) + mu;
                H0(jj, jj) = H0(jj, jj) + mu;
                H0(ii, jj) = H0(ii, jj) - mu;
                H0(jj, ii) = H0(jj, ii) - mu;

                fi = cameras(i).f; fj = cameras(j).f;
                g(ii) = g(ii) + mu * (fi - fj);
                g(jj) = g(jj) + mu * (fj - fi);
            end

        end

        % --- keep gauge (APPLY TO H0,g — the UNDAMPED system) ---
        seedPos = find(active == seed, 1);

        if ~isempty(seedPos)
            seedRows = (seedPos - 1) * 4 + (1:3); % fix seed rotation
            H0(seedRows, :) = 0; H0(:, seedRows) = 0;
            H0(sub2ind(size(H0), seedRows, seedRows)) = 1e12;
            g(seedRows) = 0;

            % also fix seed f to kill global f gauge
            if input.FixSeedFocal
                frow = (seedPos - 1) * 4 + 4;
                H0(frow, :) = 0; H0(:, frow) = 0; H0(frow, frow) = 1e12;
                g(frow) = 0;
            end

        end

        % --- Brown–Lowe scaled damping (fixed reference per growth step) ---
        if input.BrownLoweDamping
            sigma_theta = pi / 16;
            sigma_f = sigma_f_ref;

            Cinv = spalloc(size(H0, 1), size(H0, 2), size(H0, 1));

            for k = 1:numel(active)
                base = (k - 1) * 4;
                Cinv(base + 1, base + 1) = 1 / (sigma_theta ^ 2);
                Cinv(base + 2, base + 2) = 1 / (sigma_theta ^ 2);
                Cinv(base + 3, base + 3) = 1 / (sigma_theta ^ 2);
                Cinv(base + 4, base + 4) = 1 / (sigma_f ^ 2);
            end

            % --- form damped system ONLY for solving
            H0 = 0.5 * (H0 + H0.');
            H = H0 + lambda * Cinv;
        else
            % simplest paper-like damping:
            H = H0 + lambda * speye(size(H0));
        end

        b = -g;

        try
            L = chol(H, 'lower');
            d = L' \ (L \ b);
        catch
            d = H \ b;
        end

        % after solving for d but before apply_update:
        if input.ClipLogFStep
            delta = 0.08; % max |Δlog f| per iter (~±12 %)

            for k = 1:numel(active)
                f_idx = (k - 1) * 4 + 4;
                fi = cameras(active(k)).f;
                dfi = d(f_idx);
                s_step = dfi / max(1e-12, fi); % approx Δlog f
                s_step = max(-delta, min(delta, s_step));
                d(f_idx) = s_step * fi; % back to Δf
            end

        end

        if it == 1 && opt.Verbose
            jacobian_check(thetaA, cameras, pairs, active, imageSizes, sigHub, seed);
        end

        % Trial update
        [thetaTrial, camsTrial] = apply_update(thetaA, cameras, d, active, imageSizes, seed);

        % Evaluate true robust energy
        Enew = total_energy(thetaTrial, camsTrial, pairs, active, imageSizes, sigHub);

        % --- CORRECT predicted reduction (use UNDAMPED H0) ---
        if input.PredUsesUndampedH
            pred =- (g.' * d + 0.5 * (d.' * (H0 * d)));
        else
            pred =- (g.' * d + 0.5 * (d.' * (H * d))); % paper-consistent
        end

        if ~isfinite(pred) || pred <= 0
            rho = -Inf; % force reject
        else
            rho = (Eold - Enew) / pred;
        end

        % Accept/reject (require actual decrease)
        if (rho > 0) && (Enew < Eold)
            thetaA = thetaTrial;
            cameras = camsTrial;
            lambda = max(1e-9, lambda * max(1/3, 1 - (2 * rho - 1) ^ 3));

            if opt.Verbose && mod(it, 3) == 1
                fprintf('   it=%2d  E=%.3f  nres=%d  lambda=%.2e  accepted\n', it, Enew, nres, lambda);
                print_step_deltas(theta_before, thetaA, cams_before, cameras, active);
            end

            if norm(d) < 1e-8 || abs(Eold - Enew) < 1e-6, break; end
        else
            lambda = min(1e9, lambda * 10);
            % (do NOT update theta/cameras)
            if opt.Verbose && mod(it, 3) == 1
                fprintf('   it=%2d  E=%.3f  nres=%d  lambda=%.2e  REJECT\n', it, Eold, nres, lambda);
            end

        end

    end

end

function jacobian_check(theta, cameras, pairs, active, imageSizes, sigma, seed)
    % JACOBIAN_CHECK Finite-difference vs analytic gradient spot check.
    %   jacobian_check(theta, cameras, pairs, active, imageSizes, sigma, seed)

    arguments
        theta (:, 3) double {mustBeFinite}
        cameras (1, :) struct
        pairs (1, :) struct
        active (1, :) double {mustBeInteger, mustBePositive, mustBeFinite}
        imageSizes (:, 2) double {mustBeFinite}
        sigma (1, 1) double {mustBePositive, mustBeFinite}
        seed (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
    end

    rng(0); eps = 1e-6; % FD step size

    % Build analytic gradient on a tiny subset
    [~, g, ~, ~] = accumulate_normal_equations(theta, cameras, pairs, active, imageSizes, sigma);

    % Pick a random active image and parameter slot (θx,θy,θz,f)
    ai = active(randi(numel(active)));
    slot = randi(4);
    idx = (find(active == ai) - 1) * 4 + slot;

    % Finite difference gradient of total energy wrt that param
    d = zeros(size(g)); d(idx) = eps;

    % +eps
    [thetaP, camsP] = apply_update(theta, cameras, d, active, imageSizes, seed); % ai only used to keep seed fixed
    E1 = total_energy(thetaP, camsP, pairs, active, imageSizes, sigma);

    % -eps
    d(idx) = -eps;
    [thetaM, camsM] = apply_update(theta, cameras, d, active, imageSizes, seed);
    E2 = total_energy(thetaM, camsM, pairs, active, imageSizes, sigma);

    g_num = (E1 - E2) / (2 * eps);
    g_ana = g(idx); % note LM uses b=-g later

    fprintf('J-check img %d slot %d: analytic g=%.4e  numeric g=%.4e  diff=%.2e\n', ...
        ai, slot, g_ana, g_num, abs(g_ana - g_num));
end

function [m, mx] = avg_abs_residual(theta, cameras, pairs, active, imageSizes)
    % AVG_ABS_RESIDUAL Average and max absolute residual over active pairs.
    %   [m, mx] = avg_abs_residual(theta, cameras, pairs, active, imageSizes)

    arguments
        theta (:, 3) double {mustBeFinite}
        cameras (1, :) struct
        pairs (1, :) struct
        active (1, :) double {mustBeInteger, mustBePositive, mustBeFinite}
        imageSizes (:, 2) double {mustBeFinite}
    end

    activeMask = false(1, numel(cameras)); activeMask(active) = true;
    tot = 0; cnt = 0; mx = 0; z_eps = 1e-6;

    for t = 1:numel(pairs)
        i = pairs(t).i; j = pairs(t).j;
        if ~(activeMask(i) && activeMask(j)), continue; end
        Ui = pairs(t).Ui; Uj = pairs(t).Uj; M = size(Ui, 1);

        Ri = expm_hat(theta(i, :)); Rj = expm_hat(theta(j, :));
        [Ki, ~, ~, ~] = make_K_and_inv(cameras(i).f, imageSizes(i, :));
        [~, KjInv, ~, ~] = make_K_and_inv(cameras(j).f, imageSizes(j, :));

        % j->i
        yj = [Uj, ones(M, 1)] * KjInv.';
        XYZ = ((yj * Rj) * Ri.') * Ki.';
        z = XYZ(:, 3);
        good = z > z_eps;
        if ~any(good), continue; end
        pix = [XYZ(good, 1) ./ z(good), XYZ(good, 2) ./ z(good)];
        r = Ui(good, :) - pix;

        a = abs(r); tot = tot + sum(a(:)); cnt = cnt + numel(a);
        mx = max(mx, max(a(:)));
    end

    m = tot / max(1, cnt);
end

% ======================================================================
% ==================  Accumulate J^T(α)J and J^T(α)r  =================
% ======================================================================
function [H, g, E, nres] = accumulate_normal_equations(theta, cameras, pairs, active, imageSizes, sigmaHuber)
    % ACCUMULATE_NORMAL_EQUATIONS Build H=J'(α)J and g=J'(α)r with Huber weights.
    %   [H,g,E,nres] = accumulate_normal_equations(theta, cameras, pairs, active, imageSizes, sigmaHuber)
    %
    %   Inputs:
    %     theta      : N x 3 axis-angle rotation parameters for all images
    %     cameras    : 1 x N struct array with camera parameters (f, K, R, etc.)
    %     pairs      : struct array with fields .i, .j, .Ui, .Uj (matched pairs)
    %     active     : vector of indices of images to optimize
    %     imageSizes : N x 2 array of image sizes [H W]
    %     sigmaHuber : Huber loss scale parameter
    %
    %   Outputs:
    %     H     : sparse normal matrix (4m x 4m, m = numel(active))
    %     g     : gradient vector (4m x 1)
    %     E     : robust total energy (Huber loss)
    %     nres  : total number of residuals accumulated

    arguments
        theta (:, 3) double {mustBeFinite}
        cameras (1, :) struct
        pairs (1, :) struct
        active (1, :) double {mustBeInteger, mustBePositive, mustBeFinite}
        imageSizes (:, 2) double {mustBeFinite}
        sigmaHuber (1, 1) double {mustBePositive, mustBeFinite}
    end

    m = numel(active);
    blkSize = 4; % [theta(3), f(1)]

    % map image id -> block index [0..m-1]
    id2blk = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
    for k = 1:m, id2blk(int32(active(k))) = int32(k - 1); end

    % init block accumulators
    Hblocks = cell(m, m); gblocks = cell(m, 1);

    for a = 1:m
        gblocks{a} = zeros(blkSize, 1);
        for b = 1:m, Hblocks{a, b} = zeros(blkSize, blkSize); end
    end

    E = 0; nres = 0;

    % rotation generators
    E1 = [0 0 0; 0 0 -1; 0 1 0];
    E2 = [0 0 1; 0 0 0; -1 0 0];
    E3 = [0 -1 0; 1 0 0; 0 0 0];

    for t = 1:numel(pairs)
        i = pairs(t).i; j = pairs(t).j;
        if ~isKey(id2blk, int32(i)) || ~isKey(id2blk, int32(j)), continue; end
        bi = id2blk(int32(i)) + 1; bj = id2blk(int32(j)) + 1;

        Ui = pairs(t).Ui; Uj = pairs(t).Uj; M = size(Ui, 1);
        if M == 0, continue; end

        % Camera params
        Ri = expm_hat(theta(i, :)); Rj = expm_hat(theta(j, :));
        fi = cameras(i).f; fj = cameras(j).f;
        [Ki, KiInv, dKi_df, dKiInv_df] = make_K_and_inv(fi, imageSizes(i, :));
        [Kj, KjInv, dKj_df, dKjInv_df] = make_K_and_inv(fj, imageSizes(j, :));

        % --- j -> i (row-form, right-multiply) ---
        % Homogenize and back-project in j:
        yj = [Uj, ones(M, 1)] * KjInv.'; % Mx3  (rows)
        % Rotate world rays into camera i frame:
        sJ = yj * Rj; % Mx3  (since s = Rj^T * y  -> rows: y * Rj)
        % Project in camera i (pre-divide):
        XYZ = (sJ * Ri.') * Ki.'; % Mx3  (rows: s^T -> (s * Ri^T) * Ki^T)
        z = XYZ(:, 3);
        mask = z > 0;
        if ~any(mask), continue; end

        Ui_m = Ui(mask, :); % M1x2
        x = XYZ(mask, 1); y = XYZ(mask, 2); z = z(mask);
        M1 = numel(z);
        pix = [x ./ z, y ./ z];
        r = Ui_m - pix; % residuals

        [alpha, rho] = huber_alpha_and_energy(r, sigmaHuber);
        E = E + sum(rho); nres = nres + 2 * M1;

        % Perspective-divide Jacobian terms
        invz = 1 ./ z; invz2 = invz .^ 2;
        A11 = invz; A13 = -x .* invz2; A22 = invz; A23 = -y .* invz2;

        % 3D Jacobian pieces (all Mx3, computed by right-multiplying constants)
        % d/dθ_i: Ki*(Ri*E_k)*s   -> rows: s * (E_k^T * Ri^T * Ki^T)
        sJ_m = sJ(mask, :); yj_m = yj(mask, :);
        Ti1 = sJ_m * (E1.' * Ri.' * Ki.');
        Ti2 = sJ_m * (E2.' * Ri.' * Ki.');
        Ti3 = sJ_m * (E3.' * Ri.' * Ki.');

        % d/dθ_j: -Ki*Ri*(E_k*y)  -> rows: -y * (E_k^T * Ri^T * Ki^T)
        Tj1 =- yj_m * (E1.' * Ri.' * Ki.');
        Tj2 =- yj_m * (E2.' * Ri.' * Ki.');
        Tj3 =- yj_m * (E3.' * Ri.' * Ki.');

        % d/df_i: (dKi/df) * Ri * s        -> rows: s * (Ri^T * dKi_df^T)
        dfi = sJ_m * (Ri.' * dKi_df.');

        % d/df_j: Ki * Ri * Rj^T * (dKjInv/df) * [Uj;1]
        % rows: [Uj,1] * dKjInv_df^T * Rj * Ri^T * Ki^T
        dfj = ([Uj(mask, :), ones(M1, 1)] * dKjInv_df.') * (Rj * Ri.' * Ki.');

        [Hii, Hij, Hjj, gi, gj] = accumulate_blocks_huber(A11, A13, A22, A23, r, alpha, ...
            Ti1, Ti2, Ti3, Tj1, Tj2, Tj3, dfi, dfj);

        Hblocks{bi, bi} = Hblocks{bi, bi} + Hii;
        Hblocks{bj, bj} = Hblocks{bj, bj} + Hjj;
        Hblocks{bi, bj} = Hblocks{bi, bj} + Hij;
        Hblocks{bj, bi} = Hblocks{bj, bi} + Hij.';
        gblocks{bi} = gblocks{bi} + gi;
        gblocks{bj} = gblocks{bj} + gj;

        % --- i -> j ---
        % Back-project in i, rotate toward j, project with Kj
        yi = [Ui, ones(M, 1)] * KiInv.'; % Mx3
        sI = yi * Ri; % Mx3   (rows: y * Ri)
        XYZp = (sI * Rj.') * Kj.'; % Mx3
        z2 = XYZp(:, 3);

        % keep only forward rays
        mask2 = z2 > 0;
        if ~any(mask2), continue; end

        Uj_m = Uj(mask2, :); % M2x2
        x2 = XYZp(mask2, 1);
        y2 = XYZp(mask2, 2);
        z2 = z2(mask2);
        M2 = numel(z2);

        % residuals (i -> j)
        pix2 = [x2 ./ z2, y2 ./ z2];
        r2 = Uj_m - pix2;

        [alpha2, rho2] = huber_alpha_and_energy(r2, sigmaHuber);
        E = E + sum(rho2); nres = nres + 2 * M2;

        % perspective terms for masked rows
        invz2 = 1 ./ z2; invz2sq = invz2 .^ 2;
        B11 = invz2; % ∂(x/z)/∂x'
        B13 = -x2 .* invz2sq; % ∂(x/z)/∂z'
        B22 = invz2; % ∂(y/z)/∂y'
        B23 = -y2 .* invz2sq; % ∂(y/z)/∂z'

        % masked 3D rows
        sI_m = sI(mask2, :);

        % Jacobian pieces (match your row-form/right-multiply convention)
        % d/dθ_j:   Kj * (Rj * E_k) * s   => rows: s * (E_k^T * Rj^T * Kj^T)
        Sj1 = sI_m * (E1.' * Rj.' * Kj.'); % M2x3
        Sj2 = sI_m * (E2.' * Rj.' * Kj.');
        Sj3 = sI_m * (E3.' * Rj.' * Kj.');

        % d/dθ_i:  -Kj * Rj * (E_k * s)   => rows: -s * (E_k^T * Rj^T * Kj^T)
        Si1 =- sI_m * (E1.' * Rj.' * Kj.'); % M2x3
        Si2 =- sI_m * (E2.' * Rj.' * Kj.');
        Si3 =- sI_m * (E3.' * Rj.' * Kj.');

        % d/df_j: (dKj/df) * Rj * s       => rows: s * (Rj^T * dKj_df^T)
        dfj2 = sI_m * (Rj.' * dKj_df.'); % M2x3

        % d/df_i: Kj * Rj * Ri^T * (dKiInv/df) * [Ui;1]
        % rows: [Ui,1] * dKiInv_df^T * Ri * Rj^T * Kj^T
        dfi2 = ([Ui(mask2, :), ones(M2, 1)] * dKiInv_df.') * (Ri * Rj.' * Kj.');

        % Now accumulate with the same helper (note i/j order here)
        % accumulate_blocks_huber(B11,B13,B22,B23, r2, alpha2, ...
        %     Si1,Si2,Si3,  Sj1,Sj2,Sj3,  dfi2, dfj2)
        [Hii2, Hij2, Hjj2, gi2, gj2] = accumulate_blocks_huber(B11, B13, B22, B23, r2, alpha2, ...
            Si1, Si2, Si3, Sj1, Sj2, Sj3, dfi2, dfj2);

        Hblocks{bi, bi} = Hblocks{bi, bi} + Hii2;
        Hblocks{bj, bj} = Hblocks{bj, bj} + Hjj2;
        Hblocks{bi, bj} = Hblocks{bi, bj} + Hij2;
        Hblocks{bj, bi} = Hblocks{bj, bi} + Hij2.';
        gblocks{bi} = gblocks{bi} + gi2;
        gblocks{bj} = gblocks{bj} + gj2;
    end

    % Assemble sparse H, g
    Nblk = 4 * m;
    H = spalloc(Nblk, Nblk, 16 * m + 64 * m * (m - 1) / 2);
    g = zeros(Nblk, 1);

    for a = 1:m
        ia = (a - 1) * 4 + (1:4);
        g(ia) = gblocks{a};

        for b = 1:m
            ib = (b - 1) * 4 + (1:4);

            if any(Hblocks{a, b}(:))
                H(ia, ib) = H(ia, ib) + Hblocks{a, b};
            end

        end

    end

end

function [Hii, Hij, Hjj, gi, gj] = accumulate_blocks_huber(A11, A13, A22, A23, r, alpha, ...
        Ti1, Ti2, Ti3, Tj1, Tj2, Tj3, dfi, dfj)
    % ACCUMULATE_BLOCKS_HUBER  Accumulates Hessian blocks and gradient vectors using Huber loss.
    %
    %   [Hii, Hij, Hjj, gi, gj] = accumulate_blocks_huber(A11, A13, A22, A23, r, alpha, ...
    %       Ti1, Ti2, Ti3, Tj1, Tj2, Tj3, dfi, dfj)
    %
    %   This function computes the Hessian blocks (Hii, Hij, Hjj) and gradient vectors (gi, gj)
    %   for bundle adjustment using the Huber loss function. It is typically used in the context
    %   of Levenberg-Marquardt optimization for robust parameter estimation.
    %
    %   Inputs:
    %       A11, A13, A22, A23 - Submatrices of the Jacobian or Hessian related to parameters i and j.
    %       r       - Residual vector for the current observation.
    %       alpha   - Huber loss parameter controlling the threshold between quadratic and linear loss.
    %       Ti1, Ti2, Ti3 - Transformation matrices or vectors for parameter i.
    %       Tj1, Tj2, Tj3 - Transformation matrices or vectors for parameter j.
    %       dfi     - Derivative of the function with respect to parameter i.
    %       dfj     - Derivative of the function with respect to parameter j.
    %
    %   Outputs:
    %       Hii     - Accumulated Hessian block for parameter i.
    %       Hij     - Accumulated Hessian block between parameters i and j.
    %       Hjj     - Accumulated Hessian block for parameter j.
    %       gi      - Gradient vector for parameter i.
    %       gj      - Gradient vector for parameter j.
    %
    %   See also: bundleAdjustmentLM, huberLoss

    arguments
        A11 (:, 1) double {mustBeFinite}
        A13 (:, 1) double {mustBeFinite}
        A22 (:, 1) double {mustBeFinite}
        A23 (:, 1) double {mustBeFinite}
        r (:, 2) double {mustBeFinite}
        alpha (:, 1) double {mustBeFinite}
        Ti1 (:, 3) double {mustBeFinite}
        Ti2 (:, 3) double {mustBeFinite}
        Ti3 (:, 3) double {mustBeFinite}
        Tj1 (:, 3) double {mustBeFinite}
        Tj2 (:, 3) double {mustBeFinite}
        Tj3 (:, 3) double {mustBeFinite}
        dfi (:, 3) double {mustBeFinite}
        dfj (:, 3) double {mustBeFinite}
    end

    M = size(r, 1);
    Ji1 = two_by_three(A11, A13, A22, A23, Ti1); % M×2
    Ji2 = two_by_three(A11, A13, A22, A23, Ti2);
    Ji3 = two_by_three(A11, A13, A22, A23, Ti3);
    Jif = two_by_three(A11, A13, A22, A23, dfi);

    Jj1 = two_by_three(A11, A13, A22, A23, Tj1);
    Jj2 = two_by_three(A11, A13, A22, A23, Tj2);
    Jj3 = two_by_three(A11, A13, A22, A23, Tj3);
    Jjf = two_by_three(A11, A13, A22, A23, dfj);

    Ji = -cat(3, Ji1, Ji2, Ji3, Jif); % (M×2×4)
    Jj = -cat(3, Jj1, Jj2, Jj3, Jjf);

    Hii = zeros(4, 4); Hjj = zeros(4, 4); Hij = zeros(4, 4);
    gi = zeros(4, 1); gj = zeros(4, 1);

    for k = 1:M
        ak = alpha(k);
        rk = r(k, :).';
        Jik = reshape(Ji(k, :, :), [2, 4]);
        Jjk = reshape(Jj(k, :, :), [2, 4]);

        Hii = Hii + ak * (Jik.' * Jik);
        Hjj = Hjj + ak * (Jjk.' * Jjk);
        Hij = Hij + ak * (Jik.' * Jjk);

        gi = gi + ak * (Jik.' * rk);
        gj = gj + ak * (Jjk.' * rk);
    end

end

function J2 = two_by_three(A11, A13, A22, A23, T) % T: M×3
    % TWO_BY_THREE Apply perspective-division Jacobian to a 3-vector field.
    %   J2 = two_by_three(A11,A13,A22,A23,T)

    arguments
        A11 (:, 1) double {mustBeFinite}
        A13 (:, 1) double {mustBeFinite}
        A22 (:, 1) double {mustBeFinite}
        A23 (:, 1) double {mustBeFinite}
        T (:, 3) double {mustBeFinite}
    end

    Tx = T(:, 1); Ty = T(:, 2); Tz = T(:, 3);
    J2 = [A11 .* Tx + A13 .* Tz, A22 .* Ty + A23 .* Tz];
end

% ======================================================================
% ===========================  ENERGY  =================================
% ======================================================================
function E = total_energy(theta, cameras, pairs, active, imageSizes, sigma)
    % TOTAL_ENERGY Robust total Huber energy for symmetric reprojection.
    %   E = total_energy(theta, cameras, pairs, active, imageSizes, sigma)
    %
    %   Computes the robust (Huber) sum of squared reprojection residuals over
    %   all active image pairs. For each pair, residuals are evaluated in both
    %   directions (j->i and i->j) using the current rotations and focal
    %   lengths. Rays projecting behind a camera (negative depth) are ignored.
    %
    %   Inputs:
    %     theta      - N-by-3 axis–angle rotations for all images (w2c)
    %     cameras    - 1-by-N struct array with fields:
    %                  .f  scalar focal length in pixels
    %                  .K  3x3 intrinsics [f 0 cx; 0 f cy; 0 0 1]
    %     pairs      - struct array with fields:
    %                  .i,.j image indices; .Ui,.Uj are M-by-2 pixel coords
    %     active     - vector of image indices included in the energy
    %     imageSizes - N-by-2 [H W] image sizes in pixels
    %     sigma      - Huber scale (pixels); transition between L2 and L1
    %
    %   Output:
    %     E          - Scalar robust energy (sum over all residuals)
    %
    %   Notes:
    %   - Only forward-projected matches (positive depth) contribute.
    %   - Row-vector/right-multiply convention is used for projection.
    %   - Robustification performed by huber_energy_only per 2D residual.
    %
    %   See also: accumulate_normal_equations, huber_energy_only, expm_hat, make_K_and_inv

    arguments
        theta (:, 3) double {mustBeFinite}
        cameras (1, :) struct
        pairs (1, :) struct
        active (1, :) double {mustBeInteger, mustBePositive, mustBeFinite}
        imageSizes (:, 2) double {mustBeFinite}
        sigma (1, 1) double {mustBePositive, mustBeFinite}
    end

    activeSet = false(1, numel(cameras)); activeSet(active) = true;
    E = 0;

    for t = 1:numel(pairs)
        i = pairs(t).i; j = pairs(t).j;
        if ~(activeSet(i) && activeSet(j)), continue; end
        Ui = pairs(t).Ui; Uj = pairs(t).Uj; M = size(Ui, 1);

        Ri = expm_hat(theta(i, :)); Rj = expm_hat(theta(j, :));
        [Ki, KiInv] = make_K_and_inv(cameras(i).f, imageSizes(i, :));
        [Kj, KjInv] = make_K_and_inv(cameras(j).f, imageSizes(j, :));

        % --- j -> i ---
        yj = [Uj, ones(M, 1)] * KjInv.'; % Mx3
        sJ = yj * Rj; % Mx3
        XYZ = (sJ * Ri.') * Ki.'; z = XYZ(:, 3);
        mask = z > 0;

        if any(mask)
            pix = [XYZ(mask, 1) ./ z(mask), XYZ(mask, 2) ./ z(mask)];
            r = Ui(mask, :) - pix;
            E = E + sum(huber_energy_only(r, sigma));
        end

        % --- i -> j ---
        yi = [Ui, ones(M, 1)] * KiInv.'; % Mx3
        sI = yi * Ri; % Mx3
        XYZp = (sI * Rj.') * Kj.'; z2 = XYZp(:, 3);
        mask2 = z2 > 0;

        if any(mask2)
            pix2 = [XYZp(mask2, 1) ./ z2(mask2), XYZp(mask2, 2) ./ z2(mask2)];
            r2 = Uj(mask2, :) - pix2;
            E = E + sum(huber_energy_only(r2, sigma));
        end

    end

end

% ======================================================================
% ====================  UPDATES AND UTILITIES  =========================
% ======================================================================
function [thetaNew, camerasNew] = apply_update(theta, cameras, d, active, imageSizes, seed)
    % APPLY_UPDATE Apply parameter increment d to active images (θ and f).
    %   [thetaNew, camerasNew] = apply_update(theta, cameras, d, active, imageSizes, seed)

    arguments
        theta (:, 3) double {mustBeFinite}
        cameras (1, :) struct
        d (:, 1) double {mustBeFinite}
        active (1, :) double {mustBeInteger, mustBePositive, mustBeFinite}
        imageSizes (:, 2) double {mustBeFinite}
        seed (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
    end

    thetaNew = theta; camerasNew = cameras;

    for k = 1:numel(active)
        i = active(k);
        di = d((k - 1) * 4 + (1:4));
        % rotation: keep seed fixed
        if i ~= seed
            thetaNew(i, :) = theta(i, :) + di(1:3).';
            camerasNew(i).R = expm_hat(thetaNew(i, :));
        else
            thetaNew(i, :) = [0 0 0];
            camerasNew(i).R = eye(3);
        end

        % focal
        camerasNew(i).f = max(1e-6, cameras(i).f + di(4));
        cx = imageSizes(i, 2) / 2; cy = imageSizes(i, 1) / 2; f = camerasNew(i).f;
        camerasNew(i).K = [f, 0, cx; 0, f, cy; 0, 0, 1];
    end

end

function [K, Kinv, dK_df, dKinv_df] = make_K_and_inv(f, sizes)
    % MAKE_K_AND_INV Intrinsics K, its inverse, and their derivatives w.r.t f.
    %   [K,Kinv,dK_df,dKinv_df] = make_K_and_inv(f, sizes)

    arguments
        f (1, 1) double {mustBePositive, mustBeFinite}
        sizes (1, 2) double {mustBeFinite}
    end

    H = sizes(1); W = sizes(2);
    cx = W / 2; cy = H / 2;
    K = [f, 0, cx; 0, f, cy; 0, 0, 1];
    Kinv = [1 / f, 0, -cx / f; 0, 1 / f, -cy / f; 0, 0, 1];
    dK_df = [1, 0, 0; 0, 1, 0; 0, 0, 0];
    dKinv_df = [-1 / f ^ 2, 0, cx / f ^ 2; 0, -1 / f ^ 2, cy / f ^ 2; 0, 0, 0];
end

function [alpha, rho] = huber_alpha_and_energy(r, sigma)
    % HUBER_ALPHA_AND_ENERGY Huber weighting alpha and robust rho(r) per residual.
    %   [alpha,rho] = huber_alpha_and_energy(r, sigma)

    arguments
        r (:, 2) double {mustBeFinite}
        sigma (1, 1) double {mustBePositive, mustBeFinite}
    end

    nr = sqrt(sum(r .^ 2, 2));
    alpha = 2 * ones(size(nr));
    mask = nr >= sigma;
    alpha(mask) = 2 * sigma ./ max(nr(mask), 1e-12);
    rho = nr .^ 2;
    rho(mask) = 2 * sigma * nr(mask) - sigma ^ 2;
end

function rho = huber_energy_only(r, sigma)
    % HUBER_ENERGY_ONLY Robust Huber energy for 2D residuals.
    %   rho = huber_energy_only(r, sigma)

    arguments
        r (:, 2) double {mustBeFinite}
        sigma (1, 1) double {mustBePositive, mustBeFinite}
    end

    nr = sqrt(sum(r .^ 2, 2));
    rho = nr .^ 2;
    mask = nr >= sigma;
    rho(mask) = 2 * sigma * nr(mask) - sigma ^ 2;
end

function R = expm_hat(w)
    % EXPM_HAT Rodrigues' formula for SO(3) from axis-angle 3-vector.
    %   R = expm_hat(w)

    arguments
        w (1, 3) double {mustBeFinite}
    end

    th = norm(w);
    if th < 1e-12, R = eye(3); return; end
    k = w(:) / th;
    K = [0 -k(3) k(2);
         k(3) 0 -k(1);
         -k(2) k(1) 0];
    R = eye(3) + sin(th) * K + (1 - cos(th)) * (K * K);
end

function print_step_deltas(theta_old, theta_new, cams_old, cams_new, active)
    % PRINT_STEP_DELTAS Log median/max rotation and focal changes for active set.
    %   print_step_deltas(theta_old, theta_new, cams_old, cams_new, active)

    arguments
        theta_old (:, 3) double {mustBeFinite}
        theta_new (:, 3) double {mustBeFinite}
        cams_old (1, :) struct
        cams_new (1, :) struct
        active (1, :) double {mustBeInteger, mustBePositive, mustBeFinite}
    end

    rot_deg = zeros(numel(active), 1);
    df = zeros(numel(active), 1);

    for k = 1:numel(active)
        i = active(k);
        w = theta_new(i, :) - theta_old(i, :);
        rot_deg(k) = norm(w) * 180 / pi; % small-angle approx
        df(k) = cams_new(i).f - cams_old(i).f;
    end

    fprintf('   Δrot(deg) median=%.4g  max=%.4g | Δf px median=%.4g  max=%.4g\n', ...
        median(rot_deg), max(rot_deg), median(df), max(df));
end

%--------------------------------------------------------------------------
% Camera parameters estimation functions
%--------------------------------------------------------------------------
function cameras = initializeCameraMatrices(input, pairs, imageSizes, initialTforms, seed, num_images)
    % INITIALIZECAMERAMATRICES Build initial camera struct array (K,R,f,initialized).
    %   cameras = initializeCameraMatrices(input, pairs, imageSizes, initialTforms, seed, num_images)

    arguments
        input (1, 1) struct
        pairs (1, :) struct
        imageSizes (:, 2) double {mustBeFinite}
        initialTforms
        seed (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
        num_images (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
    end

    % You already have: pairs, imageSizes (N×2), num_images = N, and seed
    [K, R, f_used] = initializeKRf(input, pairs, imageSizes, num_images, seed, initialTforms);

    K = K(:); R = R(:);

    if isscalar(f_used)
        f_vec = repmat(f_used, num_images, 1);
    else
        f_vec = f_used(:);
    end

    cameras = struct( ...
        'f', num2cell(f_vec), ...
        'K', K(:), ...
        'R', R(:), ...
        'initialized', num2cell(false(num_images, 1)));

    cameras = cameras'; % now 1×N instead of N×1
end

%
function [K, R, f_used] = initializeKRf(input, pairs, imageSizes, num_images, seed, initialTforms)
    % INITIALIZEKRF Initialize intrinsics K, rotations R and focal(s) from H list.
    %   [K,R,f_used] = initializeKRf(input, pairs, imageSizes, num_images, seed, initialTforms)
    % initializeKRf
    % Robustly initialize intrinsics K, rotations R, and focal(s) for panorama BA.
    %
    % Inputs
    %   pairs          : struct array with fields:
    %                    .i, .j               (image indices, i<j recommended)
    %                    .Ui (M×2), .Uj (M×2) matched pixel coords (x,y)
    %   imageSizes     : N×2 [H W]
    %   num_images     : N
    %   seed           : chosen seed image index (gauge fix)
    %   initialTforms  : [] OR either:
    %                    - cell NxN with projective2d or 3×3 H mapping j->i
    %                    - struct array with fields .i, .j, .H (j->i)
    %
    % Outputs
    %   K      : 1×N cell, K{i} = [f 0 cx; 0 f cy; 0 0 1]
    %   R      : 1×N cell, absolute rotations (w2c) with R{seed}=I
    %   f_used : scalar if estimated globally; otherwise N×1 vector of per-image fallback focals

    arguments
        input (1, 1) struct
        pairs (1, :) struct
        imageSizes (:, 2) double {mustBeFinite}
        num_images (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
        seed (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
        initialTforms
    end

    N = num_images;
    K = cell(1, N);
    R = repmat({eye(3)}, 1, N);

    % ---------- (A) Gather homographies (j->i) for all pairs ----------
    Hlist = struct('i', {}, 'j', {}, 'H', {});

    for t = 1:numel(pairs)
        i = pairs(t).i; j = pairs(t).j;
        Hji = getHomog_j_to_i(input, pairs(t), imageSizes, initialTforms);

        if ~isempty(Hji) && all(isfinite(Hji(:))) && rank(Hji) == 3
            Hlist(end + 1).i = i; %#ok<AGROW>
            Hlist(end).j = j;
            Hlist(end).H = Hji;
        end

    end

    % Estimate focal lengths
    switch input.focalEstimateMethod
        case 'shumSzeliski'
            % ---------- (B) Estimate global focal via Shum–Szeliski over all edges ----------
            f_cands = [];

            for k = 1:numel(Hlist)
                i = Hlist(k).i; j = Hlist(k).j; H = Hlist(k).H;

                Hi = imageSizes(i, 1); Wi = imageSizes(i, 2);
                Hj = imageSizes(j, 1); Wj = imageSizes(j, 2);

                Hn = center_and_normalize_H(H, Wi, Hi, Wj, Hj);
                if isempty(Hn), continue; end

                f_ij = focal_from_H_shum_szeliski(Hn);

                if ~isempty(f_ij) && isfinite(f_ij) && f_ij > 0
                    f_cands(end + 1, 1) = f_ij; %#ok<AGROW>
                end

            end

            % Robust aggregation + plausibility clamp
            haveFocal = ~isempty(f_cands);

            if haveFocal
                base = median(max(imageSizes, [], 2));
                f_lo = 0.3 * base; % generous lower bound
                f_hi = 6.0 * base; % generous upper bound

                % MAD trim
                medf = median(f_cands);
                madf = mad(f_cands, 1);

                if madf == 0
                    keep = abs(f_cands - medf) <= 1e-6 * max(1, medf);
                else
                    keep = abs(f_cands - medf) <= 3 * madf;
                end

                f_cands = f_cands(keep);
                f_cands = f_cands(f_cands >= f_lo & f_cands <= f_hi);

                if isempty(f_cands)
                    haveFocal = false;
                else
                    f_used = median(f_cands);
                    fprintf('Estimated focal length (Shum–Szeliski, robust): %.4f pixels\n', f_used);
                end

            end

            if ~haveFocal
                % ---------- (C) Fallback focal(s): 0.8 * max(H,W) per image ----------
                f_fallback = 0.8 * max(imageSizes, [], 2); % N×1
                f_used = median(f_fallback);
                fprintf(['Cannot estimate focal lengths, %s motion model is used! ', ...
                         'Therefore, using the max(h,w) x 0.8 | f used: %.4f\n'], input.transformationType, f_used);
            end

        case 'wConstraint'
            % ---------- (B) Estimate global focal from homographies ----------
            ws = []; % collect positive candidates for w = 1/f^2

            for k = 1:numel(Hlist)
                i = Hlist(k).i;
                H = Hlist(k).H;

                % i, j are the images in H = H(j->i)
                Hi = imageSizes(i, 1); Wi = imageSizes(i, 2);
                Hj = imageSizes(j, 1); Wj = imageSizes(j, 2);

                Ci = [1 0 Wi / 2; 0 1 Hi / 2; 0 0 1]; % principal shift for i
                Cj = [1 0 Wj / 2; 0 1 Hj / 2; 0 0 1]; % principal shift for j

                Hc = Ci \ H * Cj; % **correct**: Ci^{-1} * H * Cj

                % normalize det to 1 (ideal rotation)
                d = det(Hc);
                if ~isfinite(d) || d == 0, continue; end
                s = sign(d) * nthroot(abs(d), 3); % keep orientation positive
                Hn = Hc / s;

                % constraints with omega=diag(1/f^2, 1/f^2, 1)
                h1 = Hn(:, 1); h2 = Hn(:, 2);

                denA = h1(1) * h2(1) + h1(2) * h2(2);

                if abs(denA) > eps
                    wA =- (h1(3) * h2(3)) / denA;
                    if isfinite(wA) && wA > 0, ws(end + 1) = wA; end %#ok<AGROW>
                end

                denB = (h1(1) ^ 2 + h1(2) ^ 2) - (h2(1) ^ 2 + h2(2) ^ 2);

                if abs(denB) > eps
                    wB = (h2(3) ^ 2 - h1(3) ^ 2) / denB;
                    if isfinite(wB) && wB > 0, ws(end + 1) = wB; end %#ok<AGROW>
                end

            end

            % ws collected from all pairs (w = 1/f^2), now robustify
            ws = ws(isfinite(ws) & ws > 0);

            if isempty(ws)
                haveFocal = false;
            else
                % MAD-based filtering on w
                medw = median(ws);
                madw = mad(ws, 1); % L1 MAD; robust scale

                if madw == 0
                    keep = abs(ws - medw) <= 1e-6 * max(1, medw); % degenerate but safe
                else
                    keep = abs(ws - medw) <= 3 * madw;
                end

                ws = ws(keep);
                haveFocal = ~isempty(ws);

                % Convert to f candidates and clamp to plausible range
                if haveFocal
                    f_cands = 1 ./ sqrt(ws);

                    % plausible focal band relative to image sizes
                    base = median(max(imageSizes, [], 2)); % typical "longer side"
                    f_lo = 0.3 * base; % generous low bound
                    f_hi = 6.0 * base; % generous high bound

                    f_cands = f_cands(isfinite(f_cands) & f_cands >= f_lo & f_cands <= f_hi);

                    if isempty(f_cands)
                        haveFocal = false;
                    else
                        f_used = median(f_cands); % robust pick
                        fprintf('Estimated focal length (robust): %.4f pixels\n', f_used);
                    end

                end

            end

            if ~haveFocal
                % ---------- (C) Fallback focal(s): 0.8 * max(H,W) per image ----------
                f_fallback = 0.8 * max(imageSizes, [], 2); % N×1
                f_used = median(f_fallback); % vector
                fprintf(['Cannot estimate focal lengths, %s motion model is used! ', ...
                         'Therefore, using the max(h,w) x 0.8 | f used: %.4f\n'], input.transformationType, f_used);
            end

        case 'shumSzeliskiOneH'
            % ----- Build Hlist (store i->j in column form) -----
            E = 0; Hlist = struct('i', {}, 'j', {}, 'H', {});

            for t = 1:numel(pairs)
                i = pairs(t).i; j = pairs(t).j;

                % You already have this:
                Hji = getHomog_j_to_i(input, pairs(t), imageSizes, initialTforms); % x_i ~ Hji * x_j

                if ~isempty(Hji)
                    Hij = inv(Hji); % convert to i->j: x_j ~ Hij * x_i
                    Hij = Hij ./ Hij(3, 3); % normalize scale
                    if det(Hij) < 0, Hij = -Hij; end % enforce det>0 (helps rotation projection)

                    E = E + 1;
                    Hlist(E).i = i;
                    Hlist(E).j = j;
                    Hlist(E).H = Hij; % *** store i->j ***
                end

            end

            % Estimate the focal lengths
            % Collect i->j homographies
            HAll = {Hlist(:).H};

            % Centering matrices (shift top-left origin to image center)
            C = @(w, h) [1 0 -w / 2; 0 1 -h / 2; 0 0 1];

            % Build centered homographies (still i->j)
            Hc = cell(1, numel(HAll));

            for e = 1:numel(HAll)
                i = Hlist(e).i; j = Hlist(e).j;
                Ci = C(imageSizes(i, 2), imageSizes(i, 1));
                Cj = C(imageSizes(j, 2), imageSizes(j, 1));
                % center both sides, keep column convention
                Hc{e} = Cj * (HAll{e} ./ HAll{e}(3, 3)) / Ci;
            end

            % Optional: also consider the opposite direction for robustness
            Hc_both = [Hc, cellfun(@inv, Hc, 'UniformOutput', false)];
            fvec = arrayfun(@(tform) focal_from_H_shum_szeliski_unnormalized(tform), Hc_both);

            % Keep valid, robust-aggregate
            fvec = fvec(isfinite(fvec) & fvec > 0);

            if ~isempty(fvec)
                f_used = median(fvec);
                fprintf('Estimated focal length Shum–Szeliski (single homography H): %.4f px\n', f_used);
            else
                % Fallback: 0.8*max(h,w) per image -> global median
                f_fallback = 0.8 * max(imageSizes, [], 2); % N×1
                f_used = median(f_fallback);
                fprintf(['Cannot estimate focal lengths, %s motion model is used! ', ...
                         'Therefore, using the max(h,w) x 0.8 | f used: %.4f\n'], ...
                    input.transformationType, f_used);
            end

        otherwise
            error('Require one focal estimate method.')
    end

    % ---------- (D) Build K for all images ----------
    for i = 1:N
        Hi = imageSizes(i, 1); Wi = imageSizes(i, 2);
        cx = Wi / 2; cy = Hi / 2;
        fi = f_used;
        K{i} = [fi, 0, cx;
                0, fi, cy;
                0, 0, 1];
    end

    % ---------- (E) Build relative rotations from H and propagate ----------
    % Prepare adjacency with edge IDs
    G = cell(N, 1);

    for e = 1:numel(Hlist)
        i = Hlist(e).i; j = Hlist(e).j;
        G{i} = [G{i}; j, e]; %#ok<AGROW>
        G{j} = [G{j}; i, e]; %#ok<AGROW>
    end

    % Seed gauge
    for i = 1:N, R{i} = eye(3); end
    visited = false(N, 1);
    visited(seed) = true;

    % BFS from seed
    q = seed;

    while ~isempty(q)
        u = q(1); q(1) = [];

        for r = 1:size(G{u}, 1)
            v = G{u}(r, 1);
            eid = G{u}(r, 2);

            if ~visited(v)
                i = Hlist(eid).i; j = Hlist(eid).j; H = Hlist(eid).H;

                if i == u && j == v
                    Rij = projectToSO3(K{u} \ H * K{v});
                    R{v} = R{u} * Rij;

                    if strcmp(input.focalEstimateMethod, 'shumSzeliskiOneH')
                        % K_j^{-1} H_ij K_i = R_j R_i^T
                        Rij = projectToSO3(K{j} \ H * K{i});
                        R{v} = Rij * R{u}; % R_j = (R_j R_i^T) * R_i
                    end

                elseif i == v && j == u
                    Rji = projectToSO3(K{v} \ H * K{u});
                    R{v} = R{u} * Rji'; % Rij = (Rji)'

                    if strcmp(input.focalEstimateMethod, 'shumSzeliskiOneH')
                        % Using H_ji
                        Rji = projectToSO3(K{u} \ H * K{v}); % K_u^{-1} H_ji K_v = R_u R_v^T
                        R{v} = (Rji') * R{u}; % R_v = (R_v R_u^T)^T * R_u
                    end

                else
                    continue
                end

                R{v} = projectToSO3(R{v});
                visited(v) = true;
                q(end + 1) = v; %#ok<AGROW>
            end

        end

    end

    % Ensure seed exactly identity (re-anchor)
    R{seed} = eye(3);
end

function f = focal_from_H_shum_szeliski_unnormalized(h)
    % FOCAL_FROM_H_SHUM_SZELISKI_UNNORMALIZED Estimate focal from one unnormalized H.
    %   f = focal_from_H_shum_szeliski_unnormalized(hCell)
    %   hCell should be a 1x1 cell containing a 3x3 homography matrix.

    arguments
        h cell
    end

    % Reference: Shum, HY., Szeliski, R. (2001). Construction of
    % Panoramic Image Mosaics with Global and Local Alignment.
    % In: Benosman, R., Kang, S.B. (eds) Panoramic Vision.
    % Monographs in Computer Science. Springer, New York,
    % NY. https://doi.org/10.1007/978-1-4757-3482-9_13

    % Get the transformation matrix from the cellarray
    h = h{1};

    % f1 focal length
    d1 = h(3, 1) * h(3, 2);
    d2 = (h(3, 2) - h(3, 1)) * (h(3, 2) + h(3, 1));
    v1 =- (h(1, 1) * h(1, 2) + h(2, 1) * h(2, 2)) / d1;
    v2 = (h(1, 1) ^ 2 + h(2, 1) ^ 2 - h(1, 2) ^ 2 - h(2, 2) ^ 2) / d2;

    % Swap v1 and v2
    if v1 < v2
        temp = v1;
        v1 = v2;
        v2 = temp;
    end

    % v1 and v2 > condition check
    if v1 > 0 && v2 > 0
        f1 = sqrt(v1 * (abs(d1) > abs(d2)) + v2 * (abs(d1) <= abs(d2)));
    elseif v1 > 0
        f1 = sqrt(v1);
    else
        f = 0;
        return;
    end

    % f0 focal length
    d1 = h(1, 1) * h(2, 1) + h(1, 2) * h(2, 2);
    d2 = h(1, 1) ^ 2 + h(1, 2) ^ 2 - h(2, 1) ^ 2 - h(2, 2) ^ 2;
    v1 = -h(1, 3) * h(2, 3) / d1;
    v2 = (h(2, 3) ^ 2 - h(1, 3) ^ 2) / d2;

    % Swap v1 and v2
    if v1 < v2
        temp = v1;
        v1 = v2;
        v2 = temp;
    end

    % v1 and v2 > condition check
    if v1 > 0 && v2 > 0
        f0 = sqrt(v1 * (abs(d1) > abs(d2)) + v2 * (abs(d1) <= abs(d2)));
    elseif v1 > 0
        f0 = sqrt(v1);
    else
        f = 0;
        return;
    end

    % Check for infinity
    if isinf(f1) || isinf(f0)
        f = 0;
        return;
    end

    % Geometric mean
    f = sqrt(f1 * f0);
end

function f = focal_from_H_shum_szeliski(Hn)
    % FOCAL_FROM_H_SHUM_SZELISKI Estimate focal from centered, det-normalized H.
    %   f = focal_from_H_shum_szeliski(Hn)

    % Reference: Shum, HY., Szeliski, R. (2001). Construction of
    % Panoramic Image Mosaics with Global and Local Alignment.
    % In: Benosman, R., Kang, S.B. (eds) Panoramic Vision.
    % Monographs in Computer Science. Springer, New York,
    % NY. https://doi.org/10.1007/978-1-4757-3482-9_13

    % Hn: centered & det-normalized column-form homography (j->i)
    % Returns a single positive focal candidate f (scalar) or [] if invalid.
    arguments
        Hn (3, 3) double {mustBeFinite}
    end

    % --- First focal (f1) from bottom row & left 2x2 block
    d1 = Hn(3, 1) * Hn(3, 2);
    d2 = (Hn(3, 2) - Hn(3, 1)) * (Hn(3, 2) + Hn(3, 1));

    v1 =- (Hn(1, 1) * Hn(1, 2) + Hn(2, 1) * Hn(2, 2));
    v2 = (Hn(1, 1) ^ 2 + Hn(2, 1) ^ 2 - Hn(1, 2) ^ 2 - Hn(2, 2) ^ 2);

    % --- Second focal (f0) from right column & left 2x2 block
    d1b = Hn(1, 1) * Hn(2, 1) + Hn(1, 2) * Hn(2, 2);
    d2b = Hn(1, 1) ^ 2 + Hn(1, 2) ^ 2 - Hn(2, 1) ^ 2 - Hn(2, 2) ^ 2;

    v1b = -Hn(1, 3) * Hn(2, 3);
    v2b = (Hn(2, 3) ^ 2 - Hn(1, 3) ^ 2);

    f1sq = candidate_v_to_f2(v1, v2, d1, d2);
    f0sq = candidate_v_to_f2(v1b, v2b, d1b, d2b);

    if isempty(f1sq) && isempty(f0sq), f = []; return; end
    if isempty(f1sq), f = sqrt(max(f0sq, 0)); return; end
    if isempty(f0sq), f = sqrt(max(f1sq, 0)); return; end

    % --- Correct geometric mean (Shum & Szeliski, Eq. (7)) ---
    f = sqrt(sqrt(f1sq) * sqrt(f0sq)); %  f = √(f1 * f0)

    if ~isfinite(f) || f <= 0, f = [];
    end

end

function f2 = candidate_v_to_f2(v1, v2, d1, d2)
    % CANDIDATE_V_TO_F2 Resolve focal^2 candidate from v1/v2 with denom guards.
    %   f2 = candidate_v_to_f2(v1, v2, d1, d2)

    arguments
        v1 (1, 1) double {mustBeFinite}
        v2 (1, 1) double {mustBeFinite}
        d1 (1, 1) double {mustBeFinite}
        d2 (1, 1) double {mustBeFinite}
    end

    % Implements the v1/v2 selection with denominator guards
    f2 = []; %#ok<NASGU>
    % guard tiny denominators
    tol = 1e-12;
    use1 = abs(d1) > tol; if use1, v1 = v1 / d1; else, v1 = -Inf; end
    use2 = abs(d2) > tol; if use2, v2 = v2 / d2; else, v2 = -Inf; end

    % put larger one in v1 as in your code
    if v1 < v2, tmp = v1; v1 = v2; v2 = tmp; end

    if v1 > 0 && v2 > 0
        % choose by larger denominator magnitude
        if abs(d1) > abs(d2), f2 = v1; else, f2 = v2; end
    elseif v1 > 0
        f2 = v1;
    else
        f2 = [];
    end

end

function Hn = center_and_normalize_H(H, Wi, Hi, Wj, Hj)
    % CENTER_AND_NORMALIZE_H Center H (j->i) and normalize determinant to 1.
    %   Hn = center_and_normalize_H(H, Wi, Hi, Wj, Hj)

    arguments
        H (3, 3) double {mustBeFinite}
        Wi (1, 1) double {mustBeFinite, mustBePositive}
        Hi (1, 1) double {mustBeFinite, mustBePositive}
        Wj (1, 1) double {mustBeFinite, mustBePositive}
        Hj (1, 1) double {mustBeFinite, mustBePositive}
    end

    % Column-form H (j->i). Returns centered (Ci^{-1} H Cj) and det-normalized.

    Ci = [1 0 Wi / 2; 0 1 Hi / 2; 0 0 1];
    Cj = [1 0 Wj / 2; 0 1 Hj / 2; 0 0 1];

    Hc = (Ci \ H) * Cj;

    d = det(Hc);
    if ~isfinite(d) || d == 0, Hn = []; return; end

    s = sign(d) * nthroot(abs(d), 3);
    Hn = Hc / s;
end

function Hji = getHomog_j_to_i(input, pair, imageSizes, initialTforms) %#ok<INUSD>
    % GETHOMOG_J_TO_I Obtain j->i homography (column form) from inputs or estimates.
    %   Hji = getHomog_j_to_i(input, pair, imageSizes, initialTforms)
    %   Returns a 3x3 homography mapping points in image j to image i in column form.
    %
    %   Inputs
    %   - input         : struct with options controlling matching/robustness
    %   - pair          : struct with fields .i,.j,.Ui,.Uj
    %   - imageSizes    : N-by-2 [H W]
    %   - initialTforms : [] | cell NxN | struct array with .i,.j,.H

    arguments
        input (1, 1) struct
        pair (1, 1) struct
        imageSizes (:, 2) double {mustBeFinite}
        initialTforms
    end

    % No-op reference to silence potential unused warnings in certain branches
    imageSizes = imageSizes; %#ok<NASGU>
    % Return a 3x3 column-form homography mapping j -> i: x_i ~ Hji * x_j

    i = pair.i; j = pair.j;
    Hji = [];

    if ~isempty(initialTforms)

        if iscell(initialTforms)
            T = initialTforms{i, j};

            if ~isempty(T)
                Hji = toColumn(T); % assume j->i slot
            else
                T = initialTforms{j, i}; % try opposite and invert

                if ~isempty(T)
                    Hij = toColumn(T); % i->j
                    if is_validH(Hij), Hji = inv(Hij); end
                end

            end

        else
            % struct array with .i,.j,.H
            idx = find([initialTforms.i] == i & [initialTforms.j] == j, 1);

            if ~isempty(idx)
                Hji = toColumn(initialTforms(idx).H);
            else
                idx = find([initialTforms.i] == j & [initialTforms.j] == i, 1);

                if ~isempty(idx)
                    Hij = toColumn(initialTforms(idx).H);
                    if is_validH(Hij), Hji = inv(Hij); end
                end

            end

        end

    end

    % Fallback: estimate from matches (return column-form j->i)
    if isempty(Hji) && size(pair.Ui, 1) >= 4

        try

            if input.useMATLABImageMatching == 1
                tform = estgeotform2d(pair.Uj, pair.Ui, input.transformationType, ...
                    'Confidence', input.inliersConfidence, ...
                    'MaxNumTrials', input.maxIter, ...
                    'MaxDistance', input.maxDistance);
                Hji = toColumn(tform); % *** transpose here ***

                Hji = Hji.A;
            else

                switch input.imageMatchingMethod
                    case 'ransac'
                        [Hraw, ~] = estimateTransformationRANSAC(pair.Uj, pair.Ui, ...
                            input.transformationType, input);
                    case 'mlesac'
                        [Hraw, ~] = estimateTransformationMLESAC(pair.Uj, pair.Ui, ...
                            input.transformationType, input);
                    otherwise
                        error('Valid image matching method is required.')
                end

                Hji = toColumn(Hraw);
            end

        catch
            Hji = [];
        end

    end

    % Final validation & orientation fix
    if ~is_validH(Hji)
        Hji = [];
        return
    end

    % Optional: force det>0 (helps the rotation projection)
    if det(Hji) < 0, Hji = -Hji; end

    function Hc = toColumn(T)
        % TOCOLUMN Converts the input transformation matrix T to column-form homography.
        %   Hc = TOCOLUMN(T) returns the column-form 3x3 homography matrix from T.
        %   If T is a projective2d object, returns its transpose.
        %   If T is a numeric matrix, returns it unchanged.
        %
        %   Input:
        %       T - projective2d object or 3x3 numeric matrix.
        %   Output:
        %       Hc - 3x3 column-form homography matrix.
        if isa(T, 'projective2d')
            % projective2d: row form [x y 1] * T  -> column form is T.'
            Hc = T.T.';
        else
            Hc = T;
        end

    end

    function ok = is_validH(H)
        % IS_VALIDH Check if a homography matrix H is valid (finite, non-empty, rank 3).
        %   ok = is_validH(H)
        %   Returns true if H is a non-empty, finite 3x3 matrix with full rank.

        ok = ~isempty(H) && all(isfinite(H(:))) && (rank(H) == 3);
    end

end

function R = projectToSO3(M)
    % PROJECTTOSO3 Project a matrix to the closest proper rotation via SVD.
    %   R = projectToSO3(M)

    arguments
        M (3, 3) double {mustBeFinite}
    end

    % SVD projection to the closest proper rotation.
    [U, ~, V] = svd(M);
    R = U * diag([1, 1, sign(det(U * V'))]) * V';
end
