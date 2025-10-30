function [cameras, seed] = bundleAdjustmentLM(input, numMatches, matches, keypoints, ...
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
    %                     .focalEstimateMethod ('wConstraint', 'shumSzeliski', 'shumSzeliskiOneH')
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
    %
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
    
    %% Parse optional parameters
    p = inputParser;
    addParameter(p, 'SigmaHuber', 2.0, @(x) x > 0);
    addParameter(p, 'MaxLMIters', 50, @(x) x > 0);
    addParameter(p, 'Lambda0', 1e-2, @(x) x > 0);
    addParameter(p, 'PriorSigmaF', 50, @(x) x > 0);
    addParameter(p, 'Verbose', 1, @(x) ismember(x, [0,1,2]));
    addParameter(p, 'userSeed', [], @(x) isempty(x) || (isscalar(x) && x > 0));
    
    parse(p, varargin{:});
    opts = p.Results;    
    if ~isfield(opts,'OneDirection'), opts.OneDirection = false; end
    if ~isfield(opts,'MaxMatches'),        opts.MaxMatches        = 300; end  % no cap by default
    if ~isfield(opts,'SubsampleMode'),     opts.SubsampleMode     = 'random'; end  % 'random'|'grid'|'polar'
    if ~isfield(opts,'SubsampleGridBins'), opts.SubsampleGridBins = [4 4];   end  % [rows cols]
    if ~isfield(opts,'SubsamplePolarBins'),opts.SubsamplePolarBins= [12 5];  end  % [nAngles nRadii]

    N = numel(keypoints);
    
    % ------------------ Build pair list (compact) ------------------
    pairs = build_pairs_from_cells(matches, keypoints); % each pair has i,j, Ui (px), Uj (px); i<j
    
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
    cameras = initializeCameraMatrices(input, pairs, imageSizes, initialTforms, seed, N); 
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
            for init_img = 1:N
                if ~initialized(init_img)
                    continue
                end
                
                score = numMatches(candidate, init_img) + ...
                        numMatches(init_img, candidate);
                
                if score > bestScore
                    bestScore = score;
                    bestImage = candidate;
                    bestMatchTo = init_img;
                end
            end
        end
        
        if bestImage == -1
            if opts.Verbose
                fprintf('Warning: No more images with matches to add (added %d/%d)\n', ...
                    step-1, N);
            end
            break
        end
        
        % Initialize new camera from best matching initialized camera
        % Build H (bestImage <- bestMatchTo), column form
        
        pair_tmp.i = bestImage; 
        pair_tmp.j = bestMatchTo;
        % default to empty so downstream code is safe
        pair_tmp.Ui = [];
        pair_tmp.Uj = [];
        
        % First try: reuse from compact pairs (only has i<j entries)
        allijs = [cat(1, pairs.i), cat(1, pairs.j)];
        idx_from_pairs = find(ismember(allijs, [min(bestImage,bestMatchTo), max(bestImage,bestMatchTo)], 'rows'), 1);
        
        if ~isempty(idx_from_pairs)
            % Pull from pairs; need to ensure Ui/Uj correspond to (i=bestImage, j=bestMatchTo)
            p = pairs(idx_from_pairs);
            if p.i == bestImage && p.j == bestMatchTo
                pair_tmp.Ui = p.Ui;
                pair_tmp.Uj = p.Uj;
            elseif p.i == bestMatchTo && p.j == bestImage
                % swap to keep j->i orientation consistent with pair_tmp (i=bestImage, j=bestMatchTo)
                pair_tmp.Ui = p.Uj;  % points in bestImage
                pair_tmp.Uj = p.Ui;  % points in bestMatchTo
            end
        end
        
        % Second try: build from matches/keypoints cells
        if isempty(pair_tmp.Ui) || isempty(pair_tmp.Uj)
            M = matches{bestMatchTo, bestImage};  % columns: [idx_in_bestMatchTo; idx_in_bestImage]
            flipped = false;
            if isempty(M)
                M = matches{bestImage, bestMatchTo};  % columns: [idx_in_bestImage; idx_in_bestMatchTo]
                flipped = true;
            end
            if ~isempty(M)
                % Extract Kx2 or 2xK agnostic
                Ui = keypoints{bestImage};
                Uj = keypoints{bestMatchTo};
                if size(Ui,1) == 2,  Ui = Ui.';  end   % -> Kx2
                if size(Uj,1) == 2,  Uj = Uj.';  end
        
                if ~flipped
                    % M = [idx_j; idx_i] w.r.t matches{j,i} convention → here j=bestMatchTo, i=bestImage
                    pair_tmp.Uj = Uj(M(1,:).', :);  % points in j (bestMatchTo), M rows are indices
                    pair_tmp.Ui = Ui(M(2,:).', :);  % points in i (bestImage)
                else
                    % M = [idx_i; idx_j] w.r.t matches{i,j}
                    pair_tmp.Ui = Ui(M(1,:).', :);
                    pair_tmp.Uj = Uj(M(2,:).', :);
                end
            end
        end

        % ---- Accept or ban this pair for this step ----
        if isempty(pair_tmp.Ui) || size(pair_tmp.Ui,1) < 4 || isempty(pair_tmp.Uj) || size(pair_tmp.Uj,1) < 4
            if opts.Verbose
                fprintf('  Skipping %d (no robust matches with %d)\n', bestImage, bestMatchTo);
            end
            % Ban just this (bestImage,bestMatchTo) option and try next best
            localScore(bestImage, bestMatchTo) = -inf;
            continue;   % stay in while; pick the next best
        end

        
        % Get homography matrix
        Hji = getHomog_j_to_i(input, pair_tmp, imageSizes, initialTforms);

        % Beast image rotation
        if ~isempty(Hji)
            Ki = buildIntrinsicMatrix(cameras(bestImage).f, imageSizes(bestImage,1:2));
            Kj = buildIntrinsicMatrix(cameras(bestMatchTo).f, imageSizes(bestMatchTo,1:2));
            Hij = Ki \ Hji * Kj;
            if all(isfinite(Hij(:)))
                Rij = projectToSO3(Hij);                     % approx R_i * R_j^T
                cameras(bestImage).R = Rij * cameras(bestMatchTo).R;
            else
                cameras(bestImage).R = cameras(bestMatchTo).R;   % fallback
            end
        else
            cameras(bestImage).R = cameras(bestMatchTo).R;       % fallback
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
        opts_iter = opts;                % <— add
        if numel(initializedList) <= 3
            opts_iter.FinalPass = true;   % use σHuber = opts.SigmaHuber (e.g., 2 px)
        end

        
        if opts.Verbose >= 2
            fprintf('  Running global BA on %d cameras...\n', numel(initializedList));
        end
        
        cameras = runLevenbergMarquardt( ...
            cameras, initializedList, seed, ...
            matches, keypoints, imageSizes(:,1:2), opts_iter);
    end
    
    %% Final global bundle adjustment
    initializedList = find([cameras.initialized]);
    opts_final = opts;               % <— add
    opts_final.FinalPass = true;     % <— add
    
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
           cameras = runLevenbergMarquardt( ...
                    cameras, initializedList, seed, ...
                    matches, keypoints, imageSizes(:,1:2), opts_iter);
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
                    i, cameras(i).f, yaw*180/pi, pitch*180/pi, roll*180/pi);
            end
        end
    end
end


function pairs = build_pairs_from_cells(matches, keypoints)
    arguments
        matches (:, :) cell
        keypoints cell
    end

    N = numel(keypoints);
    
    % VECTORIZED: Get upper triangle linear indices directly
    nMax = N * (N - 1) / 2;
    upperIdx = false(N, N);
    upperIdx(triu(true(N), 1)) = true;
    linearIdx = find(upperIdx);
    
    % VECTORIZED: Check all matches at once
    hasMatches = ~cellfun(@isempty, matches(linearIdx));
    validLinearIdx = linearIdx(hasMatches);
    nPairs = numel(validLinearIdx);
    
    if nPairs == 0
        pairs = struct('i', {}, 'j', {}, 'Ui', {}, 'Uj', {});
        return;
    end
    
    % VECTORIZED: Convert linear indices to subscripts
    [validI, validJ] = ind2sub([N, N], validLinearIdx);
    
    % Pre-allocate output
    pairs(nPairs) = struct('i', [], 'j', [], 'Ui', [], 'Uj', []);
    
    % Extract keypoints (this loop is unavoidable due to ragged arrays)
    for k = 1:nPairs
        i = validI(k);
        j = validJ(k);
        Mij = matches{i, j}';
        
        pairs(k).i = i;
        pairs(k).j = j;
        pairs(k).Ui = keypoints{i}(:, Mij(:, 1))';
        pairs(k).Uj = keypoints{j}(:, Mij(:, 2))';
    end
end



% Core optimization function
function cameras = runLevenbergMarquardt(cameras, camList, seed, matches, ...
    keypoints, imageSizes, opts)

    % Robust schedule (as before)
    if isfield(opts,'FinalPass') && opts.FinalPass
        sigmaHuber = opts.SigmaHuber;
    else
        sigmaHuber = 2.0;
    end

    lambda    = opts.Lambda0;
    maxIters  = opts.MaxLMIters;

    % NEW: cached solver state for AMD ordering / ichol precond
    H = []; g = []; E0 = 0; rmse0 = 0;  % avoid "might be used before defined"
    solver_state = struct;   % define with a P field so isfield checks pass    

    % caps (same as yours)
    theta_cap = deg2rad(5);
    frac_df   = 0.15;

    for outer = 1:3
        [Phi, pmap] = buildDeltaVector(cameras, camList, seed);

        % === NEW: build Brown–Lowe prior once per relinearization
        Cp_inv  = buildBrownLowePrior(camList, seed, cameras);

        % === NEW: accumulate block normal eqns (no explicit J)
        [H, g, E0, rmse0] = accumulateNormalEqns_block( ...
                Phi, pmap, cameras, camList, seed, ...
                matches, keypoints, imageSizes, sigmaHuber,opts);

        if isfield(opts,'Verbose') && opts.Verbose >= 2
            fprintf('      (relin) RMSE: %.3f px  nnz(H): %d\n', rmse0, nnz(H));
        end

        for iter = 1:maxIters
            % Solve (H + lambda*I + Cp_inv) * delta = -g
            A = H + Cp_inv + lambda*speye(size(H,1));
            b = -g;

            % NEW: pass and receive solver_state (reuses ordering/precond)
            [delta_raw, solver_state] = solve_spd(A, b, solver_state);

            % Step caps per camera (unchanged)
            delta = cap_per_camera_step(delta_raw, pmap, cameras, camList, seed, theta_cap, frac_df);

            % Trial cameras (apply to Phi then to cameras)
            Phi_trial = Phi + delta;
            cam_trial = applyIncrements(cameras, Phi_trial, pmap, camList, []);

            % Recompute energy at trial (no Jacobian needed)
            [~, ~, E_trial, rmse_trial] = accumulateNormalEqns_block( ...
                    Phi_trial, pmap, cameras, camList, seed, ...
                    matches, keypoints, imageSizes, sigmaHuber,opts);

            % Marquardt gain ratio using H,g,Cp_inv (no J anywhere)
            % pred = 0.5 * delta'*(lambda*delta - g + Cp_inv*delta)
            pred = 0.5 * (delta.'*(lambda*delta - g + Cp_inv*delta));
            if pred <= 0, rho = -Inf; else, rho = (E0 - E_trial) / pred; end

            if (E_trial < E0) && (rho > 0)
                % Accept
                cameras = cam_trial;

                % Re-orthonormalize (fast polar instead of full SVD — small speed win)
                for kk = 1:numel(camList)
                    i = camList(kk);  if i==seed, continue; end
                    R = cameras(i).R;
                    % One-step polar (Newton) ~ orthonormalize
                    X = R*(R');  % Rsym
                    R = R * ((3*eye(3) - X) / 2);   % approx (R * (RᵀR)^(-1/2))
                    % Final safeguard
                    [U,~,V] = svd(R);
                    cameras(i).R = U*diag([1,1,sign(det(U*V'))])*V';
                end

                % Lambda schedule (as before)
                if rho > 0.75, lambda = lambda/2;
                elseif rho < 0.25, lambda = lambda*2;
                end
                lambda = max(min(lambda, 1e6), 1e-10);

                % Re-linearize
                [Phi, pmap] = buildDeltaVector(cameras, camList, seed);
                Cp_inv      = buildBrownLowePrior(camList, seed, cameras);
                [H, g, E0, rmse0] = accumulateNormalEqns_block( ...
                        Phi, pmap, cameras, camList, seed, ...
                        matches, keypoints, imageSizes, sigmaHuber, opts);
                

                if isfield(opts,'Verbose') && opts.Verbose >= 2
                    fprintf('      iter %d: RMSE %.3f px  (λ=%.2g)\n', iter, rmse0, lambda);
                end

                % small convergence check
                if abs(pred) < 1e-12 || abs(E0 - E_trial) < 1e-9
                    break;
                end
            else
                % Reject
                lambda = min(lambda*4, 1e6);
                if lambda > 1e5, break; end
            end
        end
    end

    if isfield(opts,'Verbose') && opts.Verbose >= 1
        fprintf('    Final RMSE: %.3f pixels\n', rmse0);
    end
end


function [H, g, E, rmse] = accumulateNormalEqns_block( ...
    Phi, pmap, baseCams, camList, seed, matches, keypoints, imageSizes, sigmaHuber, opts)
% Build H = JᵀJ and g = Jᵀr without ever forming J.
% Residuals are evaluated at the *incremented* cameras, Jacobian at base (Gauss-Newton).
%
% Layout per camera: [dθx dθy dθz df] (seed still contributes only df)
% We keep the column layout returned by buildDeltaVector/pmap (so no API break).

    % Cameras used for residual evaluation
    cam_lin = applyIncrements(baseCams, Phi, pmap, camList, []);  %#ok<NASGU> (for clarity)

    % Column span per camera in Phi
    % seed has 1 dof (df), others 4 dof. Build a fast map.
    maxIdx = pmap(end).startIdx + (pmap(end).isSeed==0)*3;
    P = maxIdx;     % number of parameters
    H = spalloc(P, P, 40*numel(camList));   % coarse guess; will grow
    g = zeros(P,1);

      

    % Pre-compute column indices per camera
    % colsMap{i} -> 1 or 4-element vector into [1..P]
    Nmax = max([camList(:).']);
    colsMap = cell(Nmax,1);
    for t=1:numel(pmap)
        i = pmap(t).camIdx;
        s = pmap(t).startIdx;
        if pmap(t).isSeed
            colsMap{i} = s;          % [df]
        else
            colsMap{i} = s:(s+3);    % [dθx dθy dθz df]
        end
    end

    % Build compact list of (i,j) pairs within camList
    pairList = struct('i',{},'j',{},'Ui',{},'Uj',{});
    idx = 1;
    present = false(1, max(numel(keypoints), Nmax));
    present(camList) = true;

    for ii = 1:numel(camList)
        i = camList(ii);
        for jj = ii+1:numel(camList)
            j = camList(jj);
            if isempty(matches{i,j}), continue; end
            % Extract M×2 pixel coordinates (like your build_pairs_from_cells)
            mpairs = matches{i,j}.';              % M×2
            Ui = keypoints{i}(:,mpairs(:,1)).';   % M×2 (on image i)
            Uj = keypoints{j}(:,mpairs(:,2)).';   % M×2 (on image j)
            if isempty(Ui), continue; end
    
            % ---- Subsample over-connected edges (cap to opts.MaxMatches) ----
            if isfield(opts,'MaxMatches') && isfinite(opts.MaxMatches)
                [Ui, Uj] = subsample_matches( ...
                    Ui, Uj, baseCams(i), baseCams(j), imageSizes(i,:), imageSizes(j,:), opts);
                if isempty(Ui), continue; end
            end

            pairList(idx).i  = i;
            pairList(idx).j  = j;
            pairList(idx).Ui = Ui;
            pairList(idx).Uj = Uj;
            idx = idx + 1;
        end
    end

    if isempty(pairList)
        E = 0; rmse = 0; return;
    end
    
    M = size(Ui,1);
    % Per-pair accumulation (parallel if you like)
    out(numel(pairList)) = struct('bi',[],'bj',[],'Hii',[],'Hjj',[],'Hij',[],'gi',[],'gj',[],'E',0,'R2sum',0,'Rcnt',0); %#ok<NASGU>
    parfor p = 1:numel(pairList)
        i = pairList(p).i; j = pairList(p).j;
        Ui = pairList(p).Ui; Uj = pairList(p).Uj;

        % Residuals at incremented cams; Jacobian at base cams (Gauss-Newton)
        % We use the *current* baseCams for Jacobians and *Phi-updated* for residuals:
        % → good tradeoff of speed/stability (same as your previous design).
        [rij, Ji, Jj, E_ij, r2sum, rcnt] = jacobian_pair( ...
                Ui, Uj, baseCams(i), baseCams(j), cam_lin(i), cam_lin(j), ...
                imageSizes(i,:), imageSizes(j,:), sigmaHuber, colsMap{i}, colsMap{j}, ...
                opts);

        % Blocks (note: Ji,Jj already shaped as (2M × ci) and (2M × cj))
        Hii = Ji.'*Ji;    Hjj = Jj.'*Jj;    Hij = Ji.'*Jj;
        gi  = Ji.'*rij;   gj  = Jj.'*rij;

        out(p).bi  = colsMap{i};
        out(p).bj  = colsMap{j};
        out(p).Hii = Hii;   out(p).Hjj = Hjj;   out(p).Hij = Hij;
        out(p).gi  = gi;    out(p).gj  = gj;
        out(p).E   = E_ij;  out(p).R2sum = r2sum; out(p).Rcnt = rcnt;
    end

    % --- Serial reduction (triplets -> single sparse assembly) ---
    P = size(H,1);
    I = zeros(0,1); J = zeros(0,1); V = zeros(0,1);
    gg = zeros(P,1);
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
            I = [I; bi(ii(:))];   J = [J; bj(jj(:))];   V = [V; vv(:)];
            I = [I; bj(jj(:))];   J = [J; bi(ii(:))];   V = [V; vv(:)];
        end
    
        % RHS
        gg(bi) = gg(bi) + out(p).gi(:);
        gg(bj) = gg(bj) + out(p).gj(:);

         E       = E        + out(p).E;
        R2sum    = R2sum    + out(p).R2sum;
        Rcnt     = Rcnt     + out(p).Rcnt;
    end
    
    H = sparse(I, J, V, P, P);
    g = gg;

    rmse = sqrt(max(R2sum,0) / max(Rcnt,1));
end


function [r_stacked, Ji, Jj, E_sum, r2sum, rcnt] = jacobian_pair( ...
    Ui, Uj, cam_i_base, cam_j_base, cam_i_lin, cam_j_lin, ...
    size_i, size_j, sigmaHuber, cols_i, cols_j, opts)
% JACOBIAN_PAIR  Build residual/Jacobians for a pair (i,j), optionally one-direction only.
%
% If opts.OneDirection==true, only uses j->i residuals (2 per match).
% Otherwise (default), uses both j->i and i->j (4 per match).
%
% Outputs:
%   r_stacked : (2M or 4M) x 1 residual vector (Huber-weighted)
%   Ji, Jj    : (2M or 4M) x (#cols_i/#cols_j) Jacobian blocks
%   E_sum     : scalar total energy 0.5*||r||^2
%   r2sum     : sum of squared residuals (for RMSE)
%   rcnt      : number of residual scalars (2M or 4M)

    if nargin < 12, opts = struct; end
    do_both = ~(isfield(opts,'OneDirection') && opts.OneDirection);

    M = size(Ui,1);
    rows_per_match = 2 * (1 + do_both);      % 2 (one-dir) or 4 (both)
    R = rows_per_match * M;

    % Preallocate outputs
    r_stacked = zeros(R,1);
    Ji = zeros(R, numel(cols_i));
    Jj = zeros(R, numel(cols_j));

    E_sum = 0; r2sum = 0; rcnt = 0;

    % Tight loop with preallocated row pointer
    rp = 1;
    for k = 1:M
        ui = Ui(k,:).';
        uj = Uj(k,:).';

        % ---------- j -> i ----------
        % Jacobian at base cams, residual at lin cams
        [rij_base,  Jij_i, Jij_j] = computeSingleResidual( ...
            ui, uj, cam_i_base, cam_j_base, size_i, size_j);
        [rij_lin,   ~,      ~    ] = computeSingleResidual( ...
            ui, uj, cam_i_lin,  cam_j_lin,  size_i, size_j);

        w_ij  = huberWeight(norm(rij_lin), sigmaHuber);
        sw_ij = sqrt(w_ij);

        % Place j->i rows
        r_stacked(rp:rp+1) = sw_ij * rij_lin;
        Ji(rp:rp+1, :)     = sw_ij * Jij_i(:, 1:numel(cols_i));
        Jj(rp:rp+1, :)     = sw_ij * Jij_j(:, 1:numel(cols_j));

        E_sum = E_sum + 0.5 * (sw_ij^2) * (rij_lin.'*rij_lin);
        r2sum = r2sum + (sw_ij^2) * (rij_lin.'*rij_lin);
        rcnt  = rcnt  + 2;

        rp = rp + 2;

        % ---------- i -> j (optional) ----------
        if do_both
            [rji_base,  Jji_j, Jji_i] = computeSingleResidual( ...
                uj, ui, cam_j_base, cam_i_base, size_j, size_i);
            [rji_lin,   ~,     ~    ] = computeSingleResidual( ...
                uj, ui, cam_j_lin,  cam_i_lin,  size_j, size_i);

            w_ji  = huberWeight(norm(rji_lin), sigmaHuber);
            sw_ji = sqrt(w_ji);

            r_stacked(rp:rp+1) = sw_ji * rji_lin;
            % Note role swap: J wrt camera-i params uses Jji_i; wrt camera-j uses Jji_j
            Ji(rp:rp+1, :)     = sw_ji * Jji_i(:, 1:numel(cols_i));
            Jj(rp:rp+1, :)     = sw_ji * Jji_j(:, 1:numel(cols_j));

            E_sum = E_sum + 0.5 * (sw_ji^2) * (rji_lin.'*rji_lin);
            r2sum = r2sum + (sw_ji^2) * (rji_lin.'*rji_lin);
            rcnt  = rcnt  + 2;

            rp = rp + 2;
        end
    end
end

function [x, state] = solve_spd(A, b, state)
%SOLVE_SPD Cached solver for (near) SPD sparse systems.
% state is an optional struct you can keep & pass back in/out.

    persistent cache
    if nargin < 3 || isempty(state), state = struct; end
    key = [];
    if issparse(A)
        % Hash pattern (rows, cols, nnz) — cheap heuristic
        key = [size(A,1), size(A,2), nnz(A)];
    end

    use_cache = issparse(A) && isfield(cache,'key') && isequal(cache.key, key);

    if issparse(A)
        if ~use_cache
            % Build & cache permutation and preconditioner
            p = symamd(A); Ap = A(p,p);
            cache.key = key;  cache.p = p;

            % Try Cholesky once on the pattern
            [R, flag] = chol(Ap);
            if flag == 0
                cache.method = 'chol'; cache.R = R;
            else
                setup = struct('type','ict','droptol',1e-3,'diagcomp',0.01);
                try
                    L = ichol(Ap, setup);
                    cache.method = 'pcg'; cache.L = L;
                catch
                    cache.method = 'slash'; % fallback
                end
            end
        else
            p = cache.p; Ap = A(p,p);
        end

        bp = b(p);
        switch cache.method
            case 'chol'
                R = chol(Ap);   % numeric chol on same ordering is fast
                y = R \ (R' \ bp);
            case 'pcg'
                [y, flag] = pcg(Ap, bp, 1e-6, 200, cache.L, cache.L');
                if flag ~= 0, y = Ap \ bp; end
            otherwise
                y = Ap \ bp;
        end
        x      = zeros(size(b));
        x(p)   = y;
        if nargout>2, state.solver_cache = cache; end
    else
        x = A \ b;
    end

    % after solving:
    state.last_n = size(A,1);
    state.last_nz =nnz(A);
end


function delta_c = cap_per_camera_step(delta, pmap, cameras, camList, seed, theta_cap, frac_df)
    delta_c = delta;
    for k = 1:numel(pmap)
        s = pmap(k).startIdx;
        i = pmap(k).camIdx;
        if pmap(k).isSeed
            % clamp df
            f = cameras(i).f;
            df = delta_c(s);
            df = max(-frac_df*f, min(frac_df*f, df));
            delta_c(s) = df;
        else
            dth = delta_c(s:s+2);
            a   = norm(dth);
            if a > theta_cap
                dth = dth * (theta_cap / a);
                delta_c(s:s+2) = dth;
            end
            f = cameras(i).f;
            df = delta_c(s+3);
            df = max(-frac_df*f, min(frac_df*f, df));
            delta_c(s+3) = df;
        end
    end
end

function [Ui_out, Uj_out] = subsample_matches( ...
        Ui, Uj, cam_i, cam_j, size_i, size_j, opts)
%SUBSAMPLE_MATCHES  Cap correspondences per edge with controlled sampling.
% Ui, Uj: M×2 pixels.  cam_i.K/cam_j.K provide [cx,cy] if present.
% Supports modes: 'random' | 'grid' | 'polar' (see opts.* fields).

    M = size(Ui,1);
    cap = opts.MaxMatches;
    if M <= cap
        Ui_out = Ui; Uj_out = Uj; return;
    end

    switch lower(opts.SubsampleMode)
        case 'random'
            idx = randperm_per_pair(M, cap, cam_i, cam_j);
            Ui_out = Ui(idx,:); Uj_out = Uj(idx,:);

        case 'grid'
            % Stratify by a uniform grid on image i (can also average i/j)
            bins = opts.SubsampleGridBins;  % [rows cols]
            idx = grid_stratified(Ui, size_i, cap, bins, cam_i, cam_j);
            Ui_out = Ui(idx,:); Uj_out = Uj(idx,:);

        case 'polar'
            % Stratify by angle & radius around principal point on image i
            bins = opts.SubsamplePolarBins; % [nAngles nRadii]
            idx = polar_stratified(Ui, size_i, cap, bins, cam_i, cam_j);
            Ui_out = Ui(idx,:); Uj_out = Uj(idx,:);

        otherwise
            % Fallback to random
            idx = randperm_per_pair(M, cap, cam_i, cam_j);
            Ui_out = Ui(idx,:); Uj_out = Uj(idx,:);
    end
end

function idx = randperm_per_pair(M, K, cam_i, cam_j)
% deterministic per-pair permutation using a local stream (parfor-safe)
    % Build a cheap hash/seed from camera pointers (R address changes; use K,cx,cy)
    ci = double(round(1e3 * cam_i.K(1,3) + 2e3 * cam_i.K(2,3) + cam_i.K(1,1)));
    cj = double(round(1e3 * cam_j.K(1,3) + 2e3 * cam_j.K(2,3) + cam_j.K(1,1)));
    seed = mod(uint32(1664525)*uint32(ci) + uint32(1013904223)*uint32(cj), uint32(2^31-1));
    if seed == 0, seed = uint32(1); end

    try
        rs = RandStream('threefry','Seed',double(seed));     % fast & stateless
        idx = randperm(rs, M, K);
    catch
        % Fallback if 'threefry' not available
        rs = RandStream('mt19937ar','Seed',double(seed));
        idx = randperm(rs, M, K);
    end
end


function idx = grid_stratified(Ui, size_i, Kcap, bins, cam_i, cam_j)
% Ui: M×2 pixels on image i. bins=[rows cols].
    M = size(Ui,1);
    rows = max(1, bins(1)); cols = max(1, bins(2));
    H = size_i(1); W = size_i(2);

    % Bin each point
    rbin = min(rows, max(1, ceil(Ui(:,2) * rows / H)));
    cbin = min(cols, max(1, ceil(Ui(:,1) * cols / W)));
    binId = (rbin-1)*cols + cbin;      % 1..rows*cols

    % Quota per bin (at least 1 if points exist)
    nBins = rows*cols;
    counts = accumarray(binId, 1, [nBins,1], @sum, 0);
    nonEmpty = find(counts > 0);

    % Distribute cap approximately proportional to counts (with min 1)
    q = zeros(nBins,1);
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
    idx = zeros(0,1);
    for b = 1:nBins
        if q(b) == 0, continue; end
        members = find(binId == b);
        if numel(members) <= q(b)
            idx = [idx; members(:)];
        else
            % per-bin stream using bin id (deterministic)
            seed = uint32(2654435761) * uint32(b);
            try
                rs = RandStream('threefry','Seed',double(bitand(seed,2^31-1)));
            catch
                rs = RandStream('mt19937ar','Seed',double(bitand(seed,2^31-1)));
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

function idx = polar_stratified(Ui, size_i, Kcap, bins, cam_i, cam_j)
% bins = [nAngles nRadii]; center at principal point if K exists else image center.
    M = size(Ui,1);
    nA = max(1, bins(1)); nR = max(1, bins(2));

    % Center: use intrinsics if available, else image center
    if isfield(cam_i,'K') && ~isempty(cam_i.K)
        cx = cam_i.K(1,3); cy = cam_i.K(2,3);
    else
        cx = size_i(2)/2; cy = size_i(1)/2;
    end
    d = Ui - [cx, cy];                 % M×2
    ang = atan2(d(:,2), d(:,1));       % [-pi, pi]
    ang = mod(ang, 2*pi);              % [0, 2pi)
    rad = hypot(d(:,1), d(:,2));       % [0, ~max radius]
    % Normalize radius to [0,1] by max possible extent:
    rmax = hypot(max(cx, size_i(2)-cx), max(cy, size_i(1)-cy));
    rnorm = min(1, rad / max(rmax, eps));

    abin = min(nA, max(1, floor(ang / (2*pi/nA)) + 1));
    rbin = min(nR, max(1, floor(rnorm * nR) + 1));
    binId = (abin-1)*nR + rbin;
    nBins = nA*nR;

    counts = accumarray(binId, 1, [nBins,1], @sum, 0);
    nonEmpty = find(counts > 0);

    % Quotas (like grid_stratified)
    q = zeros(nBins,1);
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
    idx = zeros(0,1);
    for b = 1:nBins
        if q(b) == 0, continue; end
        members = find(binId == b);
        if numel(members) <= q(b)
            idx = [idx; members(:)];
        else
            seed = uint32(2166136261) * uint32(b);   % FNV-ish
            try
                rs = RandStream('threefry','Seed',double(bitand(seed,2^31-1)));
            catch
                rs = RandStream('mt19937ar','Seed',double(bitand(seed,2^31-1)));
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
    % All-zero increments around current cameras
    pmap = struct('camIdx',{},'startIdx',{},'isSeed',false);
    Phi  = [];
    idx  = 1;
    for k = 1:numel(camList)
        i = camList(k);
        pmap(k).camIdx  = i;
        pmap(k).startIdx = idx;
        if i == seed
            pmap(k).isSeed = true;
            Phi = [Phi; 0];           % df only
            idx = idx + 1;
        else
            pmap(k).isSeed = false;
            Phi = [Phi; 0;0;0; 0];    % [dθx dθy dθz df]
            idx = idx + 4;
        end
    end
end


function cams_out = applyIncrements(cameras, Phi, pmap, camList, seed)
% Fast apply: cache principal point (cx,cy) and only update f and R.

    cams_out = cameras;

    for k = 1:numel(pmap)
        i = pmap(k).camIdx;
        s = pmap(k).startIdx;

        % Ensure cx,cy are cached (in case older structs slip in)
        if ~isfield(cams_out(i),'cx') || isempty(cams_out(i).cx)
            % Fallback to current K if present, else image center is needed;
            % best practice: call prepareCameraCache(...) once before LM.
            if isfield(cams_out(i),'K') && ~isempty(cams_out(i).K)
                cams_out(i).cx = cams_out(i).K(1,3);
                cams_out(i).cy = cams_out(i).K(2,3);
            else
                % last-resort defaults (won't be used if you pre-cache properly)
                cams_out(i).cx = 0; cams_out(i).cy = 0;
            end
        end
        cx = cams_out(i).cx; cy = cams_out(i).cy;

        if pmap(k).isSeed
            % Seed: df only
            df = Phi(s);
             oldf = cams_out(i).f;
            f = max(100, min(5000, oldf + df));
            if abs(f - oldf) > 1e-9
                cams_out(i).f = f;
                Ki = cams_out(i).K;
                if isempty(Ki), Ki = eye(3); end
                Ki(1,1) = f; Ki(2,2) = f;
                cams_out(i).K = Ki;  % cx,cy unchanged
            end

        else
            % Non-seed: [dθx dθy dθz df]
            dth = Phi(s:s+2);
            df  = Phi(s+3);

            % SO(3) increment (left-multiplicative)
            a = norm(dth);
            if a < 1e-12
                Rupd = eye(3) + skewSymmetric(dth);
            else
                v = dth / a; K = skewSymmetric(v);
                Rupd = eye(3) + sin(a)*K + (1-cos(a))*(K*K);
            end

            cams_out(i).R = Rupd * cams_out(i).R;

            % --- Micro-opt: only update K if f changed significantly ---
            oldf = cams_out(i).f;
            f = max(100, min(5000, oldf + df));
            if abs(f - oldf) > 1e-9
               cams_out(i).f = f;
               Ki = cams_out(i).K;
               if isempty(Ki), Ki = eye(3); end
               Ki(1,1) = f; Ki(2,2) = f;
               cams_out(i).K = Ki;   % cx,cy unchanged
           end
        end
    end
end


function Cp_inv = buildBrownLowePrior(camList, seed, cameras)
    % Brown–Lowe units-balancing "prior"
    % σθ = π/16 (all axes), σf = mean(f)/10  (recomputed every iteration)
    fbar  = mean([cameras(camList).f]);
    sigth = pi/16;
    sigf  = max(1e-6, fbar/10);  % guard

    % Parameter layout: seed -> [df], others -> [dθ(3); df]
    n = 0;
    for k = 1:numel(camList), n = n + (camList(k)==seed) + 4*(camList(k)~=seed); end

    diagvals = zeros(n,1); idx = 1;
    for k = 1:numel(camList)
        i = camList(k);
        if i == seed
            diagvals(idx) = 1/(sigf^2); idx = idx+1;
        else
            diagvals(idx:idx+2) = 1/(sigth^2); idx = idx+3;
            diagvals(idx)       = 1/(sigf^2);  idx = idx+1;
        end
    end
    Cp_inv = spdiags(diagvals,0,n,n);
end

%---------------------------- Minimal to no for loops ------------------------------------
% Single residual and Jacobian
function [r, J_obs, J_src] = computeSingleResidual(u_obs, u_src, cam_obs, cam_src, ...
                                     imageSize_obs, imageSize_src)
    % Compute residual and Jacobians for one correspondence
    % Brown-Lowe Eq. 14-15: r = u - p, where p = K R R^T K^{-1} u
    
    u_src_h = [u_src; 1];  % Homogeneous coordinates
    
    % Project from src to obs: Brown-Lowe Eq. 15
    p_h = cam_obs.K * cam_obs.R * cam_src.R' * (cam_src.K \ u_src_h);
    
    % Dehomogenize
    if abs(p_h(3)) < 1e-10
        p_h(3) = 1e-10;
    end
    p = p_h(1:2) / p_h(3);
    
    % Residual: Brown-Lowe Eq. 14
    r = u_obs - p;
    
    % Jacobians: Brown-Lowe Eq. 20-23
    J_obs = computeJacobianWrtCamera(u_src_h, cam_obs, cam_src, p_h, 'obs');
    J_src = computeJacobianWrtCamera(u_src_h, cam_obs, cam_src, p_h, 'src');
end

% Jacobian w.r.t. camera parameters
function J = computeJacobianWrtCamera(u_src_h, cam_obs, cam_src, p_h, type)
    % Brown-Lowe Eq. 20-23: Jacobian computation via chain rule
    
    x = p_h(1);
    y = p_h(2);
    z = p_h(3);
    
    if abs(z) < 1e-10
        z = 1e-10;
    end
    
    % Dehomogenization Jacobian: Brown-Lowe Eq. 21
    J_dehom = [1/z,  0,  -x/z^2;
               0,  1/z,  -y/z^2];
    
    % Residual Jacobian: dr/dp = -I
    J_chain = -J_dehom;
    
    if strcmp(type, 'obs')
        % Observation camera Jacobian
        
        % Rotation Jacobian: Brown-Lowe Eq. 22-23
        J_rot = zeros(2, 3);
        for m = 1:3
            e_m = zeros(3, 1);
            e_m(m) = 1;
            skew_em = skewSymmetric(e_m);
            
            % ∂R/∂θ = R [e_m]×
            dp_h_dtheta = cam_obs.K * cam_obs.R * skew_em * cam_src.R' * ...
                (cam_src.K \ u_src_h);
            
            J_rot(:, m) = J_chain * dp_h_dtheta;
        end
        
        % Focal length Jacobian
        dK_df = [1, 0, 0; 0, 1, 0; 0, 0, 0];
        dp_h_df = dK_df * cam_obs.R * cam_src.R' * (cam_src.K \ u_src_h);
        J_f = J_chain * dp_h_df;
        
        J = [J_rot, J_f];  % 2×4
        
    else
        % Source camera Jacobian
        
        J_rot = zeros(2, 3);
        for m = 1:3
            e_m = zeros(3, 1);
            e_m(m) = 1;
            skew_em = skewSymmetric(e_m);
            
            % ∂(R^T)/∂θ = -R^T [e_m]×
            dp_h_dtheta = cam_obs.K * cam_obs.R * (-cam_src.R' * skew_em) * ...
                (cam_src.K \ u_src_h);
            
            J_rot(:, m) = J_chain * dp_h_dtheta;
        end
        
        % Focal length Jacobian (through K^{-1})
        dKinv_df = [-1/cam_src.f^2, 0, 0;
                    0, -1/cam_src.f^2, 0;
                    0, 0, 0];
        dp_h_df = cam_obs.K * cam_obs.R * cam_src.R' * dKinv_df * u_src_h;
        J_f = J_chain * dp_h_df;
        
        J = [J_rot, J_f];  % 2×4
    end
end

% Skew-symmetric matrix
function S = skewSymmetric(v)
    S = [0,    -v(3),  v(2);
         v(3),  0,    -v(1);
        -v(2),  v(1),  0   ];
end

% Huber weight function
function w = huberWeight(residual_norm, sigma)
    % Brown-Lowe Eq. 17: Huber robust error function
    if residual_norm < sigma
        w = 1;  % L2 for inliers
    else
        w = sigma / residual_norm;  % L1 for outliers
    end
end

function cameras = prepareCameraCache(cameras, imageSizes)
% Cache principal point per camera (cx,cy) once.
% If K already has cx,cy, keep them; otherwise, use image center.

    N = numel(cameras);
    for i = 1:N
        if ~isfield(cameras(i),'cx') || isempty(cameras(i).cx)
            if isfield(cameras(i),'K') && ~isempty(cameras(i).K)
                cx = cameras(i).K(1,3);  cy = cameras(i).K(2,3);
            else
                % imageSizes(i,:) = [H W ...]
                cx = imageSizes(i,2) / 2;
                cy = imageSizes(i,1) / 2;
            end
            cameras(i).cx = cx;
            cameras(i).cy = cy;

            % normalize K to use these cached cx,cy
            if isfield(cameras(i),'f') && ~isempty(cameras(i).f)
                f = cameras(i).f;
                cameras(i).K = [f, 0, cx; 0, f, cy; 0, 0, 1];
            end
        end
    end
end


% Build intrinsic matrix
function K = buildIntrinsicMatrix(f, imageSize)
    cx = imageSize(2) / 2;
    cy = imageSize(1) / 2;
    K = [f, 0, cx; 
         0, f, cy; 
         0, 0, 1];
end

% Extract Euler angles from rotation matrix
function [yaw, pitch, roll] = extractEulerAngles(R)
    % ZYX Euler convention
    sy = sqrt(R(1,1)^2 + R(2,1)^2);
    
    if sy > 1e-6
        yaw = atan2(R(3,2), R(3,3));
        pitch = atan2(-R(3,1), sy);
        roll = atan2(R(2,1), R(1,1));
    else
        yaw = atan2(-R(2,3), R(2,2));
        pitch = atan2(-R(3,1), sy);
        roll = 0;
    end
end

% 
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
        imageSizes (:, 3) double {mustBeFinite}
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
    if isempty(Hji) && isfield(pair,'Ui') && isfield(pair,'Uj') ...
       && ~isempty(pair.Ui) && ~isempty(pair.Uj) ...
       && size(pair.Ui,1) >= 4 && size(pair.Uj,1) >= 4
        disp('XXXXXXXXXXXXXXXXXX Estimate from matches XXXXXXXXXXXXXXXXXXX')
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

%
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