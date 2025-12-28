function G = bundleAdjustmentH(input, pairs, N, seed, varargin)
    % BUNDLEADJUSTMENTH  Jointly refine absolute per-image homographies G{i}.
    %
    %   G = bundleAdjustmentH(input, pairs, N, seed, 'Name',Value,...)
    %
    % Minimizes symmetric 2D error on a common canvas:
    %   r_k = Pi(G{i}*[u_i;v_i;1]) - Pi(G{j}*[u_j;v_j;1]),  for all matches k
    % where Pi([x;y;z]) = [x/z; y/z]. Gauge: G{seed} = I.
    %
    % INPUTS
    %   input : optional user data (kept for backward compatibility)
    %   pairs : 1xE struct with fields:
    %           .i, .j              image indices (1..N)
    %           .Ui (Kx2), .Uj(Kx2) matched pixel coords (x,y) in images i,j
    %           (optional) .w (Kx1) per-correspondence weights
    %   N     : number of images
    %   seed  : gauge image index (fixed to identity)
    %
    % NAME/VALUE OPTIONS (all optional)
    %   'G0'           : 1xN cell of initial absolute homographies (3x3). If empty,
    %                    they are built by chaining pairwise maps (robust BFS).
    %   'ImageSizes'   : image sizes
    %   'OneDirection' : if true, only use one-direction residuals in BA (default: true)
    %   'MaxMatches'   : cap max matches per pair (default: Inf)
    %   'SubsampleMode': 'random'|'grid'|'polar' (default: 'random')
    %   'MaxIters'     : default 50 (GN/LM iterations)
    %   'Huber'        : default 1.0 (pixels); set 0 to disable robust loss
    %   'Lambda'       : default 1e-3 (LM damping)
    %   'RegProj'      : default 1e-4; L2 on g,h entries (H(3,1), H(3,2))
    %   'RegDet'       : default 0; L2 on (log(det2x2)-0) of top-left 2x2 block
    %   'UseLSQ'       : default true if lsqnonlin exists; otherwise false
    %   'Verbose'      : default true
    %
    % OUTPUT
    %   G : 1xN cell, refined absolute homographies, normalized to H(3,3)=1
    %
    % NOTES
    % - Parameterization per image (except seed): 8-DOF vector p such that
    %     H(p) = [a b c; d e f; g h 1], row-major p=[a b c d e f g h]
    % - Residuals are 2D per correspondence (symmetric: both sides to canvas).
    % - If the pipeline already has chained absolute warps (e.g., from imwarp
    %   setup), pass them as 'G0' for faster & better convergence.

    arguments
        input
        pairs (1, :) struct
        N (1, 1) double {mustBeInteger, mustBePositive}
        seed (1, 1) double {mustBeInteger, mustBePositive}
    end

    arguments (Repeating)
        varargin
    end

    opts = parseOpts(varargin{:});

    % Seed gauge & initial G
    G = cell(1, N);

    for i = 1:N
        G{i} = normalizeH(opts.G0{i});
    end

    G{seed} = eye(3);

    % Subsample matches per pair (if requested)
    if isfield(opts, 'MaxMatches') && opts.MaxMatches < Inf

        for e = 1:numel(pairs)
            M = size(pairs(e).Ui, 1);

            if M > opts.MaxMatches
                % Extract image sizes from imageSizes [Nx3]
                if isfield(opts, 'ImageSizes') && ~isempty(opts.ImageSizes)
                    % imageSizes is Nx3: [height, width, channels]
                    sizeI = opts.ImageSizes(pairs(e).i, 1:2);
                    sizeJ = opts.ImageSizes(pairs(e).j, 1:2);
                else
                    % Fallback: estimate from max coords
                    sizeI = [max(pairs(e).Ui(:, 2)) * 1.2, max(pairs(e).Ui(:, 1)) * 1.2];
                    sizeJ = [max(pairs(e).Uj(:, 2)) * 1.2, max(pairs(e).Uj(:, 1)) * 1.2];
                end

                [pairs(e).Ui, pairs(e).Uj] = subsampleMatches( ...
                    pairs(e).Ui, pairs(e).Uj, ...
                    pairs(e).i, pairs(e).j, ...
                    sizeI, sizeJ, opts);

                if isfield(pairs(e), 'w') && ~isempty(pairs(e).w)
                    pairs(e).w = ones(size(pairs(e).Ui, 1), 1);
                end

            end

        end

        if opts.Verbose
            fprintf('[BA] Subsampled matches: cap=%d, mode=%s\n', ...
                opts.MaxMatches, opts.SubsampleMode);
        end

    end

    % Pack parameters (exclude seed)
    mask = true(1, N); mask(seed) = false;
    idToBlk = zeros(1, N); idToBlk(mask) = 1:nnz(mask);

    p0 = zeros(8 * nnz(mask), 1);

    for i = 1:N
        if ~mask(i), continue; end
        p0(8 * (idToBlk(i) - 1) + (1:8)) = hom2param(G{i});
    end

    % Choose optimizer
    useLSQ = opts.UseLSQ && exist('lsqnonlin', 'file') == 2;

    if useLSQ
        if opts.Verbose, fprintf('[BA] Using lsqnonlin\n'); end
        fun = @(p) residualsJacobian(p, G, pairs, mask, idToBlk, seed, opts, false); % no J; let lsqnonlin numdiff
        lsqOpts = optimoptions('lsqnonlin', 'SpecifyObjectiveGradient', true, ...
            'Display', 'iter-detailed', ...
            'MaxIterations', opts.MaxIters, ...
            'StepTolerance', 1e-10, 'FunctionTolerance', 1e-12, ...
            'OptimalityTolerance', 1e-10, ...
            'Display', 'off', 'UseParallel', true);
        p = lsqnonlin(fun, p0, [], [], lsqOpts);
    else
        if opts.Verbose, fprintf('[BA] Using Gauss-Newton + LM (analytic J)\n'); end
        p = adaptiveLM(p0, G, pairs, mask, idToBlk, seed, opts);
    end

    % Unpack & renormalize
    for i = 1:N

        if mask(i)
            pp = p(8 * (idToBlk(i) - 1) + (1:8));
            G{i} = normalizeH(param2hom(pp));
        else
            G{i} = eye(3);
        end

    end

end % globalHomographyBA

function p = adaptiveLM(p0, G, pairs, mask, idToBlk, seed, opts)
    % ADAPTIVELM  Levenberg-Marquardt optimizer with adaptive damping.
    %
    %   p = adaptiveLM(p0, G, pairs, mask, idToBlk, seed, opts)
    %
    % Inputs:
    %   p0      - initial parameter vector (8*numActive x 1)
    %   G       - cell array of absolute homographies (preallocated)
    %   pairs   - pairwise match struct array
    %   mask    - logical vector marking active images
    %   idToBlk - mapping from image id to parameter block index
    %   seed    - gauge image index
    %   opts    - options struct (see parseOpts)
    %
    % Outputs:
    %   p       - optimized parameter vector (same size as p0)

    arguments
        p0 (:, 1) double
        G cell
        pairs struct
        mask (:, 1) logical
        idToBlk (:, 1) double
        seed (1, 1) double {mustBeInteger, mustBePositive}
        opts struct
    end

    p = p0;
    lambda = opts.Lambda;
    nu = 2.0; % lambda scaling factor (standard LM)

    % Initial evaluation
    [r, J] = residualsJacobian(p, G, pairs, mask, idToBlk, seed, opts, false);
    error = 0.5 * (r.' * r);

    if opts.Verbose
        fprintf('  iter %2s  E=%12.6g  |r|=%10.6g  lambda=%10.3e  status\n', ...
            'init', error, norm(r), lambda);
    end

    for it = 1:opts.MaxIters
        % Compute damped normal equations
        H = J.' * J + lambda * speye(size(J, 2));
        g = J.' * r;

        % Solve for step
        dp = -H \ g;

        % Check for convergence (small step)
        stepNorm = norm(dp);

        if stepNorm <= 1e-8 * (1 + norm(p))

            if opts.Verbose
                fprintf('  iter %2d  Converged: step norm %.3e\n', it, stepNorm);
            end

            break;
        end

        % Candidate parameters
        pNew = p + dp;

        % Evaluate at candidate point
        rNew = residualsJacobian(pNew, G, pairs, mask, idToBlk, seed, opts, true);
        eNew = 0.5 * (rNew.' * rNew);

        % Compute gain ratio (actual reduction / predicted reduction)
        % Predicted reduction: ΔE_pred = -g'*dp - 0.5*dp'*H*dp
        % For LM this simplifies nicely
        dEActual = error - eNew;
        dEPred = -g.' * dp - 0.5 * dp.' * (lambda * dp); % linear + quadratic

        % Alternative formulation (more numerically stable):
        % dEPred = 0.5 * dp.' * (lambda * dp - g);

        rho = dEActual / (abs(dEPred) + eps);

        % Step acceptance/rejection and lambda update
        if rho > 0 % Step reduces error
            % Accept step
            p = pNew;
            r = rNew;
            error = eNew;

            % Recompute Jacobian at new point
            [~, J] = residualsJacobian(p, G, pairs, mask, idToBlk, seed, opts, false);

            % Decrease lambda (move toward Gauss-Newton)
            % Standard LM update rule
            lambda = lambda * max(1/3, 1 - (2 * rho - 1) ^ 3);
            nu = 2.0;

            status = 'accept';
        else % Step increases error
            % Reject step, increase lambda (move toward steepest descent)
            lambda = lambda * nu;
            nu = 2 * nu;

            status = 'REJECT';
        end

        % Display progress
        if opts.Verbose
            fprintf('  iter %2d  E=%12.6g  |r|=%10.6g  lambda=%10.3e  rho=%7.3f  %s\n', ...
                it, error, norm(r), lambda, rho, status);
        end

        % Check for convergence (small gradient)
        gradNorm = norm(g);

        if gradNorm <= 1e-10 * (1 + error)

            if opts.Verbose
                fprintf('  iter %2d  Converged: gradient norm %.3e\n', it, gradNorm);
            end

            break;
        end

        % Check for lambda explosion (too many rejections)
        if lambda > 1e12

            if opts.Verbose
                fprintf('  iter %2d  Lambda too large (%.3e), stopping\n', it, lambda);
            end

            break;
        end

    end

end

% ---------------- Residuals & Jacobians --------------------
function [r, J] = residualsJacobian(p, G, pairs, mask, idToBlk, seed, opts, noJac)
    % RESIDUALSJACOBIAN  Build absolute homographies and compute residuals/Jacobian.
    %
    %   [r, J] = residualsJacobian(p, G, pairs, mask, idToBlk, seed, opts, noJac)
    %
    % Inputs:
    %   p       - parameter vector (8*numActive x 1)
    %   G       - cell array of absolute homographies (initial guesses)
    %   pairs   - pairwise match struct array
    %   mask    - logical vector marking active images
    %   idToBlk - mapping from image id to parameter block index
    %   seed    - gauge image index
    %   opts    - options struct
    %   noJac   - logical flag; if true, do not compute J (only r)
    %
    % Outputs:
    %   r       - stacked residual vector
    %   J       - sparse Jacobian matrix (empty if noJac==true)

    arguments
        p (:, 1) double
        G cell
        pairs struct
        mask (:, 1) logical
        idToBlk (:, 1) double
        seed (1, 1) double {mustBeInteger, mustBePositive}
        opts struct
        noJac (1, 1) logical = false
    end

    % Unpack homographies
    Habs = G;

    for i = 1:numel(G)

        if mask(i)
            pp = p(8 * (idToBlk(i) - 1) + (1:8));
            Habs{i} = param2hom(pp);
        else
            Habs{i} = eye(3);
        end

    end

    for i = 1:numel(Habs), Habs{i} = normalizeH(Habs{i}); end

    % Count residuals
    E = numel(pairs);
    Mtot = 0;

    for e = 1:E
        Mtot = Mtot + size(pairs(e).Ui, 1);
    end

    if opts.OneDirection
        Mdata = 2 * Mtot;
    else
        Mdata = 4 * Mtot;
    end

    nActive = nnz(mask);
    nRegProj = 2 * nActive * (opts.RegProj > 0);
    nRegDet = 1 * nActive * (opts.RegDet > 0);
    Mtotal = Mdata + nRegProj + nRegDet;

    r = zeros(Mtotal, 1, 'double');

    if ~noJac
        % Better number of non-zeros (NNZ) estimate: each residual row has up to 16 entries (2 images × 8 params)
        maxNnz = Mdata * 16 + nRegProj + nRegDet * 6;
        iRow = zeros(maxNnz, 1);
        jCol = zeros(maxNnz, 1);
        vals = zeros(maxNnz, 1);
        nnzCount = 0;
    else
        J = [];
    end

    row = 0;
    delta = max(0, opts.Huber);

    % Process each pair with vectorized operations
    for e = 1:E
        i = pairs(e).i;
        j = pairs(e).j;
        Ui = pairs(e).Ui;
        Uj = pairs(e).Uj;
        M = size(Ui, 1);

        if isfield(pairs(e), 'w') && ~isempty(pairs(e).w)
            w = pairs(e).w(:);
        else
            w = ones(M, 1);
        end

        Hi = Habs{i};
        Hj = Habs{j};

        % Vectorize: process all M matches at once
        % Xi: 3×M, Xj: 3×M
        Xi = [Ui, ones(M, 1)]'; % 3×M
        Xj = [Uj, ones(M, 1)]'; % 3×M

        if ~opts.OneDirection
            % Bidirectional residuals (vectorized)
            [resAll, jiAll, jjAll] = computeBidirResiduals(Hi, Hj, Xi, Xj, w, delta, mask(i), mask(j), noJac);

            numRes = 4 * M;
            r(row + (1:numRes)) = resAll;

            if ~noJac

                if mask(i)
                    bi = idToBlk(i);
                    cols = 8 * (bi - 1) + (1:8);
                    [iRow, jCol, vals, nnzCount] = addJacobianBlock(iRow, jCol, vals, nnzCount, jiAll, row, cols, M, 4);
                end

                if mask(j)
                    bj = idToBlk(j);
                    cols = 8 * (bj - 1) + (1:8);
                    [iRow, jCol, vals, nnzCount] = addJacobianBlock(iRow, jCol, vals, nnzCount, jjAll, row, cols, M, 4);
                end

            end

            row = row + numRes;
        else
            % Unidirectional residuals (vectorized)
            [resAll, jiAll, jjAll] = computeUnidirResiduals(Hi, Hj, Xi, Xj, w, delta, mask(i), mask(j), noJac);

            numRes = 2 * M;
            r(row + (1:numRes)) = resAll;

            if ~noJac

                if mask(i)
                    bi = idToBlk(i);
                    cols = 8 * (bi - 1) + (1:8);
                    [iRow, jCol, vals, nnzCount] = addJacobianBlock(iRow, jCol, vals, nnzCount, jiAll, row, cols, M, 2);
                end

                if mask(j)
                    bj = idToBlk(j);
                    cols = 8 * (bj - 1) + (1:8);
                    [iRow, jCol, vals, nnzCount] = addJacobianBlock(iRow, jCol, vals, nnzCount, -jjAll, row, cols, M, 2);
                end

            end

            row = row + numRes;
        end

    end

    % Regularizers (unchanged, already fast)
    if (opts.RegProj > 0 || opts.RegDet > 0)

        for i = 1:numel(Habs)
            if ~mask(i), continue; end
            H = Habs{i};
            bi = idToBlk(i);
            cols = 8 * (bi - 1) + (1:8);

            if opts.RegProj > 0
                row = row + 1;
                r(row) = sqrt(opts.RegProj) * H(3, 1);

                if ~noJac
                    nnzCount = nnzCount + 1;
                    iRow(nnzCount) = row;
                    jCol(nnzCount) = cols(7);
                    vals(nnzCount) = sqrt(opts.RegProj);
                end

                row = row + 1;
                r(row) = sqrt(opts.RegProj) * H(3, 2);

                if ~noJac
                    nnzCount = nnzCount + 1;
                    iRow(nnzCount) = row;
                    jCol(nnzCount) = cols(8);
                    vals(nnzCount) = sqrt(opts.RegProj);
                end

            end

            if opts.RegDet > 0
                a = H(1, 1); b = H(1, 2);
                d = H(2, 1); e = H(2, 2);
                det2 = a * e - b * d;

                row = row + 1;
                r(row) = sqrt(opts.RegDet) * log(max(1e-8, abs(det2)));

                if ~noJac
                    sgn = sign(det2); if sgn == 0, sgn = 1; end
                    gdet = sgn / max(1e-8, abs(det2));
                    grad = gdet * [e, -d, 0, -b, a, 0, 0, 0];

                    for cc = 1:8

                        if abs(grad(cc)) > eps
                            nnzCount = nnzCount + 1;
                            iRow(nnzCount) = row;
                            jCol(nnzCount) = cols(cc);
                            vals(nnzCount) = sqrt(opts.RegDet) * grad(cc);
                        end

                    end

                end

            end

        end

    end

    if ~noJac
        iRow = iRow(1:nnzCount);
        jCol = jCol(1:nnzCount);
        vals = vals(1:nnzCount);
        J = sparse(iRow, jCol, vals, Mtotal, 8 * nActive);
    end

end

% ========== Helper Functions ==========

function [resAll, jiAll, jjAll] = computeUnidirResiduals(Hi, Hj, Xi, Xj, w, delta, needJi, needJj, noJac)
    % COMPUTEUNIDIRRESIDUALS  Vectorized unidirectional residuals for matches.
    %
    %   [resAll, jiAll, jjAll] = computeUnidirResiduals(Hi, Hj, Xi, Xj, w, delta, needJi, needJj, noJac)
    %
    % Inputs:
    %   Hi, Hj - 3×3 homography matrices for images i and j
    %   Xi, Xj - 3×M arrays of homogeneous coordinates (columns are points)
    %   w      - 1×M weight vector per match
    %   delta  - scalar Huber threshold (0 disables)
    %   needJi - logical, compute Jacobian w.r.t. Hi
    %   needJj - logical, compute Jacobian w.r.t. Hj
    %   noJac  - logical, if true skip computing Jacobians
    %
    % Outputs:
    %   resAll - stacked residuals (2M×1)
    %   jiAll  - Jacobian block w.r.t Hi ((2M)×8) or []
    %   jjAll  - Jacobian block w.r.t Hj ((2M)×8) or []

    arguments
        Hi (3, 3) double
        Hj (3, 3) double
        Xi (3, :) double
        Xj (3, :) double
        w (1, :) double
        delta (1, 1) double = 0
        needJi (1, 1) logical = true
        needJj (1, 1) logical = true
        noJac (1, 1) logical = false
    end

    M = size(Xi, 2);

    % Transform all points at once
    Yi = Hi * Xi; % 3×M
    Yj = Hj * Xj; % 3×M

    % Normalize
    Yi = Yi ./ Yi(3, :); % 3×M
    Yj = Yj ./ Yj(3, :); % 3×M

    % Residuals: 2×M
    res = Yi(1:2, :) - Yj(1:2, :); % 2×M

    % Apply Huber weights
    if delta > 0
        resNorms = sqrt(sum(res .^ 2, 1)); % 1×M
        wHuber = ones(1, M);
        outliers = resNorms >= delta;
        wHuber(outliers) = delta ./ resNorms(outliers);
        w = w(:)' .* wHuber; % 1×M
    else
        w = w(:)'; % 1×M
    end

    % Apply weights and reshape: interleave [u1 v1 u2 v2 ...]
    resWeighted = res .* w; % 2×M
    resAll = reshape(resWeighted, [], 1); % 2M×1

    % Jacobians
    if noJac
        jiAll = []; jjAll = [];
    else

        if needJi
            jiAll = computeJacobianBatch(Hi, Xi, w); % (2M)×8
        else
            jiAll = [];
        end

        if needJj
            jjAll = computeJacobianBatch(Hj, Xj, w); % (2M)×8
        else
            jjAll = [];
        end

    end

end

function [resAll, jiAll, jjAll] = computeBidirResiduals(Hi, Hj, Xi, Xj, w, delta, needJi, needJj, noJac)
    % COMPUTEBIDIRRESIDUALS  Vectorized bidirectional residuals for matches.
    %
    %   [resAll, jiAll, jjAll] = computeBidirResiduals(Hi, Hj, Xi, Xj, w, delta, needJi, needJj, noJac)
    %
    % Inputs:
    %   Hi, Hj - 3×3 homography matrices for images i and j
    %   Xi, Xj - 3×M arrays of homogeneous coordinates (columns are points)
    %   w      - 1×M weight vector per match
    %   delta  - scalar Huber threshold (0 disables)
    %   needJi - logical, compute Jacobian w.r.t. Hi
    %   needJj - logical, compute Jacobian w.r.t. Hj
    %   noJac  - logical, if true skip computing Jacobians
    %
    % Outputs:
    %   resAll - stacked residuals (4M×1)
    %   jiAll  - Jacobian block w.r.t Hi ((4M)×8) or []
    %   jjAll  - Jacobian block w.r.t Hj ((4M)×8) or []

    arguments
        Hi (3, 3) double
        Hj (3, 3) double
        Xi (3, :) double
        Xj (3, :) double
        w (1, :) double
        delta (1, 1) double = 0
        needJi (1, 1) logical = true
        needJj (1, 1) logical = true
        noJac (1, 1) logical = false
    end

    M = size(Xi, 2);

    % Forward: Hj^-1 * Hi * Xi
    yiInI = Hi * Xi; % 3×M
    yiInI = yiInI ./ yiInI(3, :);

    yiInJ = Hj \ yiInI; % 3×M
    yiInJ = yiInJ ./ yiInJ(3, :);

    resFwd = Xj(1:2, :) - yiInJ(1:2, :); % 2×M

    % Backward: Hi^-1 * Hj * Xj
    yjInJ = Hj * Xj; % 3×M
    yjInJ = yjInJ ./ yjInJ(3, :);

    yjInI = Hi \ yjInJ; % 3×M
    yjInI = yjInI ./ yjInI(3, :);

    resBwd = Xi(1:2, :) - yjInI(1:2, :); % 2×M

    % Stack: [resFwd; resBwd] = 4×M
    res = [resFwd; resBwd]; % 4×M

    % Huber weights
    if delta > 0
        resNorms = sqrt(sum(res .^ 2, 1)); % 1×M (4D norm per match)
        wHuber = ones(1, M);
        outliers = resNorms >= delta;
        wHuber(outliers) = delta ./ resNorms(outliers);
        w = w(:)' .* wHuber;
    else
        w = w(:)';
    end

    resWeighted = res .* w; % 4×M
    resAll = reshape(resWeighted, [], 1); % 4M×1

    if noJac
        jiAll = []; jjAll = [];
    else

        if needJi
            % Compute Jacobians for bidirectional
            JiFwd = computeJacobianBatchBidir(Hj, Hi, Xi, w, false, false); % (2M)×8
            JiBwd = computeJacobianBatchBidir(Hi, Hj, Xj, w, true, true); % (2M)×8
            jiAll = [JiFwd; JiBwd]; % (4M)×8
        else
            jiAll = [];
        end

        if needJj
            JjFwd = computeJacobianBatchBidir(Hj, Hi, Xi, w, true, true); % (2M)×8
            JjBwd = computeJacobianBatchBidir(Hi, Hj, Xj, w, false, false); % (2M)×8
            jjAll = [JjFwd; JjBwd]; % (4M)×8
        else
            jjAll = [];
        end

    end

end

function jBatch = computeJacobianBatch(H, X, w)
    % COMPUTEJACOBIANBATCH  Compute Jacobian for all M points at once.
    %
    %   jBatch = computeJacobianBatch(H, X, w)
    %
    % Inputs:
    %     H - 3×3 homography matrix
    %     X - 3×M homogeneous points (columns)
    %     w - 1×M weights
    %
    % Outputs:
    %     jBatch - (2M)×8 Jacobian matrix (rows interleaved u,v)

    arguments
        H (3, 3) double
        X (3, :) double
        w (1, :) double
    end

    M = size(X, 2);
    a = H(1, 1); b = H(1, 2); c = H(1, 3);
    d = H(2, 1); e = H(2, 2); f = H(2, 3);
    g = H(3, 1); h = H(3, 2);

    u = X(1, :); % 1×M
    v = X(2, :); % 1×M

    Y1 = a * u + b * v + c; % 1×M
    Y2 = d * u + e * v + f; % 1×M
    Y3 = g * u + h * v + 1; % 1×M

    y3Sq = Y3 .^ 2;

    % Derivatives (each row is 1×M, represents derivative for all M points)
    dY1Dp = [u; v; ones(1, M); zeros(1, M); zeros(1, M); zeros(1, M); zeros(1, M); zeros(1, M)]; % 8×M
    dY2Dp = [zeros(1, M); zeros(1, M); zeros(1, M); u; v; ones(1, M); zeros(1, M); zeros(1, M)]; % 8×M
    dY3Dp = [zeros(1, M); zeros(1, M); zeros(1, M); zeros(1, M); zeros(1, M); zeros(1, M); u; v]; % 8×M

    % du/dp = (dY1/dp * Y3 - Y1 * dY3/dp) / Y3^2
    duDp = (dY1Dp .* Y3 - Y1 .* dY3Dp) ./ y3Sq; % 8×M

    % dv/dp = (dY2/dp * Y3 - Y2 * dY3/dp) / Y3^2
    dvDp = (dY2Dp .* Y3 - Y2 .* dY3Dp) ./ y3Sq; % 8×M

    % Apply weights
    duDp = duDp .* w; % 8×M
    dvDp = dvDp .* w; % 8×M

    % Reshape to (2M)×8: [u1_row; v1_row; u2_row; v2_row; ...]
    jBatch = zeros(2 * M, 8);
    jBatch(1:2:end, :) = duDp'; % M×8 for u
    jBatch(2:2:end, :) = dvDp'; % M×8 for v
end

function JBatch = computeJacobianBatchBidir(Houter, Hinner, X, w, outerParam, negate)
    % COMPUTEJACOBIANBATCHBIDIR  Batch bidirectional Jacobian computation.
    %
    %   JBatch = computeJacobianBatchBidir(Houter, Hinner, X, w, outerParam, negate)
    %
    % Inputs:
    %   Houter, Hinner - 3×3 homographies in outer/inner order
    %   X              - 3×M homogeneous points
    %   w              - 1×M weights
    %   outerParam     - logical, differentiate wrt outer H when true
    %   negate         - logical, negate Jacobian when true
    %
    % Outputs:
    %   JBatch - (2M)×8 Jacobian matrix

    arguments
        Houter (3, 3) double
        Hinner (3, 3) double
        X (3, :) double
        w (1, :) double
        outerParam (1, 1) logical = false
        negate (1, 1) logical = false
    end

    M = size(X, 2);

    % Transform inner
    Yinner = Hinner * X;
    Yinner = Yinner ./ Yinner(3, :);

    if outerParam
        % d/dHouter
        Hinv = inv(Houter);
        JBatch = computeJacobianBatch(Hinv, Yinner, w);

        if negate
            JBatch = -JBatch;
        end

    else
        % d/dHinner (simplified: use chain rule approximation)
        % This is more complex, fallback to loop if needed
        JBatch = zeros(2 * M, 8);

        for k = 1:M
            x = X(:, k);

            % Use original localHomJacBidir
            Jk = localHomJacBidir(Houter, Hinner, x, outerParam);
            JBatch(2 * (k - 1) + 1:2 * k, :) = w(k) * Jk;
        end

        if negate
            JBatch = -JBatch;
        end

    end

end

function [iRow, jCol, vals, nnzCount] = addJacobianBlock(iRow, jCol, vals, nnzCount, jBlock, rowOffset, cols, M, resPerMatch)
    % ADDJACOBIANBLOCK  Append block entries to sparse triplet arrays.
    %
    %   [iRow, jCol, vals, nnzCount] = addJacobianBlock(iRow, jCol, vals, nnzCount, jBlock, rowOffset, cols, M, resPerMatch)
    %
    % Inputs:
    %   iRow, jCol, vals - preallocated triplet arrays
    %   nnzCount         - current nnz count
    %   jBlock           - block of Jacobian values ((resPerMatch*M)×8)
    %   rowOffset        - row offset where block should be placed
    %   cols             - column indices for the block (8 values)
    %   M                - number of matches
    %   resPerMatch      - residuals per match (2 or 4)
    %
    % Outputs:
    %   iRow, jCol, vals - updated triplet arrays
    %   nnzCount         - updated nnz count

    arguments
        iRow (:, 1) double
        jCol (:, 1) double
        vals (:, 1) double
        nnzCount (1, 1) double {mustBeInteger, mustBeNonnegative}
        jBlock (:, :) double
        rowOffset (1, 1) double {mustBeInteger, mustBeNonnegative}
        cols (:, 1) double {mustBeInteger, mustBePositive}
        M (1, 1) double {mustBeInteger, mustBeNonnegative}
        resPerMatch (1, 1) double {mustBeInteger, mustBePositive}
    end

    if isempty(jBlock), return; end

    nRows = resPerMatch * M;

    % Generate indices
    rows = rowOffset + (1:nRows)';
    [I, C] = ndgrid(rows, cols);

    % Flatten
    I = I(:);
    C = C(:);
    V = jBlock(:);

    % Add to triplets
    n = numel(I);
    iRow(nnzCount + (1:n)) = I;
    jCol(nnzCount + (1:n)) = C;
    vals(nnzCount + (1:n)) = V;
    nnzCount = nnzCount + n;
end

function J = localHomJacBidir(Houter, Hinner, x, outerParam)
    % LOCALHOMJACBIDIR  Compute local Jacobian for bidirectional homography chain.
    %
    %   J = localHomJacBidir(Houter, Hinner, x, outerParam)
    %
    % Inputs:
    %     Houter, Hinner - 3×3 homographies
    %     x              - 3×1 homogeneous point
    %     outerParam     - logical, true to differentiate wrt Houter
    %
    % Outputs:
    %     J - 2×8 local Jacobian matrix

    arguments
        Houter (3, 3) double
        Hinner (3, 3) double
        x (3, 1) double
        outerParam (1, 1) logical = false
    end

    yInner = Hinner * x;
    yInner = yInner ./ yInner(3);

    if outerParam
        Hinv = inv(Houter);
        J = -localHomJac(Hinv, yInner);
    else
        yFinal = Houter \ yInner;
        Y1 = yFinal(1); Y2 = yFinal(2); Y3 = yFinal(3);
        dProjDy = [1 / Y3, 0, -Y1 / Y3 ^ 2;
                   0, 1 / Y3, -Y2 / Y3 ^ 2];
        jInner = localHomJac(Hinner, x);
        J = dProjDy * (Houter \ [jInner(1, :); jInner(2, :); zeros(1, 8)]);
        J = J(1:2, :);
    end

end

function J = localHomJac(H, x)
    % LOCALHOMJAC  Analytical Jacobian of 2D projection for a single homography.
    %
    %   J = localHomJac(H, x)
    %
    % Inputs:
    %   H - 3×3 homography
    %   x - 3×1 homogeneous coordinate
    %
    % Outputs:
    %   J - 2×8 Jacobian (du/dparams; dv/dparams)

    arguments
        H (3, 3) double
        x (3, 1) double
    end

    a = H(1, 1); b = H(1, 2); c = H(1, 3);
    d = H(2, 1); e = H(2, 2); f = H(2, 3);
    g = H(3, 1); h = H(3, 2);
    u = x(1); v = x(2);

    Y1 = a * u + b * v + c;
    Y2 = d * u + e * v + f;
    Y3 = g * u + h * v + 1;

    dY1 = [u, v, 1, 0, 0, 0, 0, 0];
    dY2 = [0, 0, 0, u, v, 1, 0, 0];
    dY3 = [0, 0, 0, 0, 0, 0, u, v];

    du = (dY1 * Y3 - Y1 * dY3) / (Y3 ^ 2);
    dv = (dY2 * Y3 - Y2 * dY3) / (Y3 ^ 2);
    J = [du; dv];
end

% ---------------- Utilities --------------------------------
function p = hom2param(H)
    % HOM2PARAM  Convert 3×3 homography to 8-parameter vector [a b c d e f g h].
    %
    %   p = hom2param(H)
    %
    % Inputs:
    %   H - 3×3 homography matrix
    %
    % Outputs:
    %   p - 8×1 vector [a b c d e f g h]'

    arguments
        H (3, 3) double
    end

    H = normalizeH(H);
    p = [H(1, 1), H(1, 2), H(1, 3), H(2, 1), H(2, 2), H(2, 3), H(3, 1), H(3, 2)].';
end

function H = param2hom(p)
    % PARAM2HOM  Convert 8-parameter vector back to 3×3 homography.
    %
    %   H = param2hom(p)
    %
    % Inputs:
    %   p - 8×1 parameter vector
    %
    % Outputs:
    %   H - 3×3 homography matrix with H(3,3)=1

    arguments
        p (8, 1) double
    end

    H = [p(1) p(2) p(3);
         p(4) p(5) p(6);
         p(7) p(8) 1];
end

function H = normalizeH(H)
    % NORMALIZEH  Normalize homography so H(3,3)=1 (fallback scaling for singular H).
    %
    %   Hout = normalizeH(Hin)
    %
    % Inputs:
    %   H - 3×3 homography matrix
    %
    % Outputs:
    %   H - normalized 3×3 homography with H(3,3)=1 (if possible)

    arguments
        H (3, 3) double
    end

    if H(3, 3) == 0, s = sign(det(H)) * nthroot(max(eps, abs(det(H))), 3); H = H / s; end
    if H(3, 3) ~= 0, H = H / H(3, 3); end
end

function opts = parseOpts(varargin)
    % PARSEOPTS  Parse name/value options for bundleAdjustmentH.
    %
    %   opts = parseOpts(varargin) accepts Name/Value pairs and returns a
    %   struct with default fields used by the bundle adjustment routine.

    arguments (Repeating)
        varargin
    end

    % Basic validation of name/value formatting is performed below.
    opts.G0 = [];
    opts.MaxIters = 50;
    opts.Huber = 1.0;
    opts.Lambda = 1e-3;
    opts.RegProj = 1e-4;
    opts.RegDet = 0;
    opts.UseLSQ = true;
    opts.Verbose = true;
    opts.OneDirection = true;

    % Subsampling options
    opts.ImageSizes = []; % Nx3 matrix [height, width, channels]
    opts.MaxMatches = Inf; % No cap by default
    opts.SubsampleMode = 'random'; % 'random' | 'grid' | 'polar'
    opts.SubsampleGridBins = [4 4]; % [rows cols]
    opts.SubsamplePolarBins = [12 5]; % [nAngles nRadii]

    if mod(numel(varargin), 2) ~= 0
        error('Name/Value arguments expected.');
    end

    for k = 1:2:numel(varargin)
        opts.(varargin{k}) = varargin{k + 1};
    end

end

function [uiOut, ujOut, idx] = subsampleMatches( ...
        Ui, Uj, imgI, imgJ, sizeI, sizeJ, opts)
    %SUBSAMPLEMATCHES  Cap correspondences for homography BA (no camera deps)
    %   SUBSAMPLEMATCHES(MATCHES, IMG_I, IMG_J, SIZE_I, SIZE_J, OPTS) returns a
    %   reduced set of putative correspondences suitable for homography-only
    %   bundle adjustment. This routine limits the number of matches, can enforce
    %   a spatially balanced subsample, and uses a deterministic random seed
    %   derived from the integer image indices IMG_I and IMG_J so repeated runs
    %   on the same image pair produce the same subset.
    %
    %   INPUTS
    %     MATCHES  - Nx4 numeric array of putative correspondences. Each row is
    %                expected to be [x_i, y_i, x_j, y_j] (coordinates in image
    %                pixel units). Other compatible formats may be accepted by
    %                the implementation.
    %     IMG_I    - Integer index of the first image (used for deterministic
    %                seeding of any randomized selection).
    %     IMG_J    - Integer index of the second image (used for deterministic
    %                seeding).
    %     SIZE_I   - Two-element vector [height, width] for the first image.
    %     SIZE_J   - Two-element vector [height, width] for the second image.
    %     OPTS     - (optional) struct or name/value pairs controlling behavior:
    %                .maxMatches  - maximum number of correspondences to keep
    %                               (default: implementation-dependent, e.g. 2000)
    %                .method      - 'uniform' | 'grid' | 'random' selection
    %                .gridSize    - [rows,cols] for spatial binning when using
    %                               grid-based subsampling
    %                .preserve    - logical mask or indices of matches that must
    %                               be preserved (e.g. known inliers)
    %
    %   OUTPUTS
    %     MATCHES_SUB - Kx4 array of retained correspondences (same format as MATCHES)
    %     INFO         - struct with diagnostic fields (typical fields):
    %                    .originalCount - number of input matches
    %                    .keptCount     - number of matches returned
    %                    .seed          - deterministic seed computed from IMG_I/IMG_J
    %                    .method        - actual method used for subsampling
    %                    .binsUsed      - binning statistics when grid sampling
    arguments
        Ui (:, 2) double {mustBeFinite}
        Uj (:, 2) double {mustBeFinite}
        imgI (1, 1) double {mustBeInteger, mustBePositive}
        imgJ (1, 1) double {mustBeInteger, mustBePositive}
        sizeI (1, 2) double {mustBeFinite}
        sizeJ (1, 2) double {mustBeFinite}
        opts struct
    end

    M = size(Ui, 1);
    cap = opts.MaxMatches;

    if M <= cap
        uiOut = Ui; ujOut = Uj; idx = (1:M)'; return;
    end

    switch lower(opts.SubsampleMode)
        case 'random'
            idx = randPermutationPair(M, cap, imgI, imgJ);

        case 'grid'
            bins = opts.SubsampleGridBins; % [rows cols]
            idx = gridStratified(Ui, sizeI, cap, bins, imgI, imgJ);

        case 'polar'
            bins = opts.SubsamplePolarBins; % [nAngles nRadii]
            idx = polarStratified(Ui, sizeI, cap, bins, imgI, imgJ);

        otherwise
            idx = randPermutationPair(M, cap, imgI, imgJ);
    end

    uiOut = Ui(idx, :);
    ujOut = Uj(idx, :);
end

function idx = randPermutationPair(M, kCap, imgI, imgJ)
    % RANDPERMUTATIONPAIR  Create random correspondence index pairs for two images
    %
    %   IDX = randPermutationPair(M, KCAP, IMGI, IMGJ) returns a set of random
    %   index pairs to be used as correspondences (matches) between two images
    %   identified by IMGI and IMGJ. The function treats the index pool as the
    %   integers 1..M, randomly permutes that pool, and selects up to KCAP
    %   elements to form correspondences between the two images.
    %
    %   Inputs
    %     M     - Positive integer scalar. Size of the index pool (indices 1:M).
    %     KCAP  - Positive integer scalar. Maximum number of pairs to return.
    %             If KCAP > M, the function will return at most M pairs.
    %     IMGI  - Scalar (numeric or integer) identifying the first image in the
    %             pair. Used for bookkeeping and does not affect the numeric
    %             indices returned.
    %     IMGJ  - Scalar (numeric or integer) identifying the second image.
    %
    %   Output
    %     IDX   - N-by-2 integer array of index pairs, where N = min(M, KCAP).
    %             Column 1 contains indices for image IMGI and column 2
    %             contains indices for image IMGJ. Each row represents a single
    %             randomized correspondence (match) between IMGI and IMGJ.
    arguments
        M (1, 1) double {mustBeInteger, mustBeNonnegative}
        kCap (1, 1) double {mustBeInteger, mustBeNonnegative}
        imgI (1, 1) double {mustBeInteger}
        imgJ (1, 1) double {mustBeInteger}
    end

    % Stable hash from image indices
    seed = mod(uint32(1664525) * uint32(imgI) + ...
        uint32(1013904223) * uint32(imgJ), uint32(2 ^ 31 - 1));
    if seed == 0, seed = uint32(1); end

    try
        rs = RandStream('threefry', 'Seed', double(seed));
        idx = randperm(rs, M, kCap);
    catch
        rs = RandStream('mt19937ar', 'Seed', double(seed));
        idx = randperm(rs, M, kCap);
    end

end

function idx = gridStratified(Ui, sizeI, kCap, bins, imgI, imgJ)
    % GRIDSTRATIFIED Select a stratified subset of feature points on a 2-D image grid
    %
    % IDX = gridStratified(UI, SIZEI, KCAP, BINS)
    % IDX = gridStratified(UI, SIZEI, KCAP, BINS, IMGI, IMGJ)
    %
    % Divides the image domain into a regular grid and selects up to KCAP feature
    % points per grid cell from the set UI. The function is useful to produce a
    % spatially balanced subset of interest points or correspondences prior to
    % downstream processing (e.g. matching, bundle adjustment).
    %
    % Inputs
    %   UI    - N-by-2 (or 2-by-N) array of feature coordinates [x y] in pixel
    %           units (image coordinate system). Rows correspond to points.
    %   SIZEI - Two-element vector [height, width] specifying the image size in
    %           pixels.
    %   KCAP  - Nonnegative scalar specifying the maximum number of points to
    %           keep per grid cell.
    %   BINS  - Scalar or two-element vector [nRows, nCols] specifying the grid
    %           resolution. If scalar, the same number of bins is used for both
    %           dimensions.
    %   IMGI  - (optional) Vector of length N containing image indices, point
    %           identifiers or correspondence labels. When provided, the
    %           selection procedure may use these values to preserve pair-wise
    %           constraints or to balance selection across images. Pass [] to
    %           ignore.
    %   IMGJ  - (optional) Same semantics as IMGI for the matching/paired image.
    %
    % Output
    %   IDX   - Column vector of indices into UI selecting the chosen subset of
    %           feature points. The number of returned indices is <= KCAP * numel(BINS).
    arguments
        Ui (:, 2) double {mustBeFinite}
        sizeI (1, 2) double {mustBeFinite}
        kCap (1, 1) double {mustBeInteger, mustBeNonnegative}
        bins (1, 2) double {mustBeInteger, mustBePositive}
        imgI (1, 1) double {mustBeInteger}
        imgJ (1, 1) double {mustBeInteger}
    end

    M = size(Ui, 1);
    rows = max(1, bins(1)); cols = max(1, bins(2));
    H = sizeI(1); W = sizeI(2);

    % Bin assignment (same as before)
    rbin = min(rows, max(1, ceil(Ui(:, 2) * rows / H)));
    cbin = min(cols, max(1, ceil(Ui(:, 1) * cols / W)));
    binId = (rbin - 1) * cols + cbin;

    nBins = rows * cols;
    counts = accumarray(binId, 1, [nBins, 1], @sum, 0);
    nonEmpty = find(counts > 0);

    % Quota distribution (same logic)
    q = zeros(nBins, 1);

    if ~isempty(nonEmpty)
        prop = counts(nonEmpty) / sum(counts(nonEmpty));
        q(nonEmpty) = max(1, round(prop * kCap));
        over = sum(q) - kCap;

        if over > 0
            [~, ord] = sort(q(nonEmpty), 'descend');
            t = nonEmpty(ord); k = 1;

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

    % Sample per bin with deterministic seed
    idx = zeros(0, 1);

    for b = 1:nBins
        if q(b) == 0, continue; end
        members = find(binId == b);

        if numel(members) <= q(b)
            idx = [idx; members(:)];
        else
            % Combine pair indices with bin for unique seed
            seed = uint32(2654435761) * uint32(b) + uint32(imgI * 97 + imgJ);

            try
                rs = RandStream('threefry', 'Seed', double(bitand(seed, 2 ^ 31 - 1)));
            catch
                rs = RandStream('mt19937ar', 'Seed', double(bitand(seed, 2 ^ 31 - 1)));
            end

            pick = members(randperm(rs, numel(members), q(b)));
            idx = [idx; pick(:)];
        end

    end

    if numel(idx) > kCap, idx = idx(1:kCap); end
end

function idx = polarStratified(Ui, sizeI, kCap, bins, imgI, imgJ)
    % POLARSTRATIFIED Select indices of features using polar stratified sampling
    %
    %   IDX = polarStratified(Ui, SIZEI, KCAP, BINS, IMGI, IMGJ)
    %
    %   Returns a vector of indices IDX selecting a subset of feature points from
    %   Ui by dividing the image domain into polar (radial x angular) bins and
    %   taking up to KCAP points per bin. The selection is performed in a
    %   stratified manner so that features are spatially distributed around the
    %   image center.
    %
    %   Inputs
    %     Ui    - N-by-2 or N-by-3 numeric array of feature coordinates. The
    %             first two columns are interpreted as [x y] image coordinates.
    %             If a third column is present it is treated as a confidence/
    %             score value used to prioritize selections within each bin.
    %     sizeI - Two-element vector [height, width] specifying the reference
    %             image size used to compute the image center. If empty, the
    %             size is inferred from IMGI (if provided).
    %     kCap  - Positive integer scalar: maximum number of features to keep
    %             per polar bin.
    %     bins  - Specification of the polar binning. Accepted forms:
    %               * two-element vector [Rbins, Thetabins] giving number of
    %                 radial and angular bins, or
    %               * struct with fields describing radial and angular bin
    %                 edges (e.g. radialEdges, angularEdges) or counts.
    %             The function will map each feature to a (radial,angular) bin
    %             according to its polar coordinates about the image center.
    %     imgI, imgJ - (optional) image matrices for the two views. If provided
    %             they may be used to infer sizeI or to mask-out features that
    %             fall outside image bounds. Either can be empty if not used.
    %
    %   Output
    %     idx   - Column vector of indices into Ui of the selected features.
    %             Indices are returned in no particular global order; within
    %             each bin selected features are ordered by descending score
    %             when Ui contains a score column, otherwise by the original
    %             order.
    %
    %   Behavior / Algorithm (summary)
    %     1) Compute the image center from SIZEI (or IMGI).
    %     2) Convert Ui coordinates to polar coordinates (radius, angle)
    %        relative to the center.
    %     3) Quantize radius and angle according to BINS to obtain bin ids.
    %     4) For each bin, select up to KCAP features. If Ui contains a score
    %        column, features are sorted by score (descending) inside each
    %        bin before selection; otherwise the original order is used.
    %     5) Points outside the valid image region or falling into no bin are
    %        ignored.
    %
    %   Notes
    %     - If KCAP <= 0 the function returns an empty index vector.
    %     - If BINS is specified as counts [Rbins,Thetabins], radial bin
    %       boundaries are computed from the image center to the image corner.
    %     - The function performs integer/edge handling such that points lying
    %       exactly on bin boundaries are assigned consistently (implementation
    %       dependent).
    arguments
        Ui (:, 2) double {mustBeFinite}
        sizeI (1, 2) double {mustBeFinite}
        kCap (1, 1) double {mustBeInteger, mustBeNonnegative}
        bins (1, 2) double {mustBeInteger, mustBePositive}
        imgI (1, 1) double {mustBeInteger}
        imgJ (1, 1) double {mustBeInteger}
    end

    % Number of bins
    nA = max(1, bins(1)); nR = max(1, bins(2));

    % Use image center (no K matrix needed)
    cx = sizeI(2) / 2;
    cy = sizeI(1) / 2;

    d = Ui - [cx, cy];
    ang = atan2(d(:, 2), d(:, 1));
    ang = mod(ang, 2 * pi);
    rad = hypot(d(:, 1), d(:, 2));
    rmax = hypot(max(cx, sizeI(2) - cx), max(cy, sizeI(1) - cy));
    rnorm = min(1, rad / max(rmax, eps));

    abin = min(nA, max(1, floor(ang / (2 * pi / nA)) + 1));
    rbin = min(nR, max(1, floor(rnorm * nR) + 1));
    binId = (abin - 1) * nR + rbin;
    nBins = nA * nR;

    counts = accumarray(binId, 1, [nBins, 1], @sum, 0);
    nonEmpty = find(counts > 0);

    % Quota (same as grid)
    q = zeros(nBins, 1);

    if ~isempty(nonEmpty)
        prop = counts(nonEmpty) / sum(counts(nonEmpty));
        q(nonEmpty) = max(1, round(prop * kCap));
        over = sum(q) - kCap;

        if over > 0
            [~, ord] = sort(q(nonEmpty), 'descend');
            t = nonEmpty(ord); k = 1;

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

    idx = zeros(0, 1);

    for b = 1:nBins
        if q(b) == 0, continue; end
        members = find(binId == b);

        if numel(members) <= q(b)
            idx = [idx; members(:)];
        else
            seed = uint32(2166136261) * uint32(b) + uint32(imgI * 89 + imgJ);

            try
                rs = RandStream('threefry', 'Seed', double(bitand(seed, 2 ^ 31 - 1)));
            catch
                rs = RandStream('mt19937ar', 'Seed', double(bitand(seed, 2 ^ 31 - 1)));
            end

            pick = members(randperm(rs, numel(members), q(b)));
            idx = [idx; pick(:)];
        end

    end

    if numel(idx) > kCap, idx = idx(1:kCap); end
end
