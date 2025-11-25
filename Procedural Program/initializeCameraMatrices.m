function cameras = initializeCameraMatrices(input, pairs, matches, imageSizes, initialTforms, ...
        seed, numImages, numMatches)
    % INITIALIZECAMERAMATRICES Build initial camera struct array (K,R,f,initialized).
    %INITIALIZECAMERAMATRICES Initialize per-image camera matrices for panorama stitching.
    %   CAMERAS = INITIALIZECAMERAMATRICES(INPUT, PAIRS, IMAGESIZES, INITIALTFORMS, SEED, NUMIMAGES)
    %   builds an initial set of camera intrinsics and extrinsics for a set of
    %   images connected by pairwise geometric transforms. The result is suitable
    %   as a starting point for global pose refinement (e.g., bundle adjustment)
    %   in an automatic panorama stitching pipeline.
    %
    % Syntax
    %   cameras = initializeCameraMatrices(input, pairs, imageSizes, initialTforms, seed, numImages)
    %
    % Description
    %   The function constructs a view graph from PAIRS and propagates relative
    %   transforms in INITIALTFORMS from a chosen SEED view to all reachable views.
    %   Intrinsic parameters are initialized from INPUT and IMAGESIZES (or set to
    %   sensible defaults), and per-view extrinsics (rotation/translation) are
    %   composed to express each camera pose in the seed frame. The output is a
    %   per-image collection of camera matrices/parameters for subsequent global
    %   optimization and warping.
    %
    % Inputs
    %   input          - Structure or object with initialization options. Typical
    %                    fields may include:
    %                      • Intrinsics (cameraParameters, K, focal length/FOV)
    %                      • Normalization/centering options
    %                      • Projection model ('planar','cylindrical','spherical')
    %                      • Reference view policy and scale settings
    %                    If empty, reasonable defaults are used.
    %   pairs          - M-by-2 array of image indices describing the adjacency
    %                    graph; each row [i j] denotes a relation between images i
    %                    and j for which a relative transform is provided.
    %   imageSizes     - NUMIMAGES-by-3 array of [height width] for each image.
    %   initialTforms  - Collection of pairwise transforms relating images in PAIRS.
    %                    Supported forms typically include:
    %                      • M-by-1 cell array of projective2d/affine2d objects
    %                        where initialTforms{k} maps points from image j to i
    %                        for PAIRS(k,:) = [i j].
    %                      • or a struct/map keyed by pairs with corresponding
    %                        3x3 homographies (up to scale).
    %   seed           - Scalar index of the seed (reference) image used as the
    %                    origin of the pose graph. The seed view is assigned
    %                    identity rotation and zero translation.
    %   numImages     - Total number of images/views in the panorama.
    %
    % Output
    %   cameras        - NUMIMAGES-by-1 container of per-image camera parameters.
    %                    The exact type depends on the implementation; commonly a
    %                    struct array with fields such as:
    %                      • K  (3x3) Intrinsic calibration matrix
    %                      • R  (3x3) Rotation from world to camera
    %                      • t  (3x1) Translation of camera center in world coords
    %                      • P  (3x4) Projection matrix P = K * [R | t]
    %                      • ImageSize ([H W]) Original image size
    %                      • Id (scalar) Image index
    %                    Entries corresponding to images unreachable from SEED
    %                    may be empty or omitted.
    %
    % Notes
    %   - The function traverses the view graph induced by PAIRS starting at SEED
    %     and composes INITIALTFORMS to express each view relative to the seed.
    %   - If the view graph is disconnected or required transforms are missing,
    %     an error is thrown or unreachable views are left uninitialized.
    %   - Intrinsics are estimated from INPUT and IMAGESIZES; if not provided,
    %     focal lengths may be initialized heuristically (e.g., using image
    %     diagonal and an assumed field of view).
    %   - Use the returned CAMERAS as initialization for global refinement such as
    %     bundleAdjustment or nonlinear homography optimization.
    %
    % Example
    %   % Example data
    %   pairs       = [1 2; 2 3; 3 4];
    %   imageSizes  = repmat([1080 1920], 4, 1);
    %   initialTforms = {
    %       projective2d(eye(3));
    %       projective2d(eye(3));
    %       projective2d(eye(3))};
    %   seed        = 2;
    %   numImages  = 4;
    %
    %   % Initialization options (intrinsics, projection model, etc.)
    %   input = struct('Projection','planar');  % add fields as needed
    %
    %   % Build initial camera matrices
    %   cameras = initializeCameraMatrices(input, pairs, imageSizes, initialTforms, seed, numImages);
    %
    % See also
    %   projective2d, affine2d, estimateGeometricTransform2D, viewSet, bundleAdjustment

    arguments
        input (1, 1) struct
        pairs (1, :) struct
        matches (:, :) cell
        imageSizes (:, 3) double {mustBeFinite}
        initialTforms
        seed (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
        numImages (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
        numMatches double
    end

    % You already have: pairs, imageSizes (N×2), numImages = N, and seed
    [K, R, fUsed, H2seed, noRotation] = initializeKRf(input, pairs, matches, imageSizes, numImages, seed, initialTforms, numMatches);

    if isscalar(fUsed)
        fVec = repmat(fUsed, numImages, 1);
    else
        fVec = fUsed(:);
    end

    % replicate scalar flag per camera
    noRotVec = repmat(logical(noRotation), numImages, 1);

    cameras = struct( ...
        'f', num2cell(fVec), ...
        'K', K(:), ...
        'R', R(:), ...
        'H2seed', H2seed(:), ... % per-camera chain homography
        'noRotation', num2cell(noRotVec), ... % same flag in every camera
        'initialized', num2cell(false(numImages, 1)));

    cameras = cameras'; % now 1×N instead of N×1

    % Show rotation determinants and angles
    if input.verboseInitRKf

        for i = 1:numImages
            fprintf('Cam %2d  |  det(R)=%.3f  |  angle=%7.2f°\n', ...
                i, det(R{i}), ...
                acos(max(-1, min(1, (trace(R{i}) - 1) / 2))) * 180 / pi);
        end

    end

end

function [K, R, fUsed, H2seed, noRotation] = initializeKRf(input, pairs, matches, imageSizes, ...
        numImages, seed, initialTforms, numMatches)
    % INITIALIZEKRF Initialize intrinsics K, rotations R and focal(s) from H list.
    %   [K,R,fUsed] = initializeKRf(input, pairs, imageSizes, numImages, seed, initialTforms)
    % initializeKRf
    % Robustly initialize intrinsics K, rotations R, and focal(s) for panorama BA.
    %
    % Inputs
    %   pairs          : struct array with fields:
    %                    .i, .j               (image indices, i<j recommended)
    %                    .Ui (M×2), .Uj (M×2) matched pixel coords (x,y)
    %   imageSizes     : N×2 [H W]
    %   numImages     : N
    %   seed           : chosen seed image index (gauge fix)
    %   initialTforms  : [] OR either:
    %                    - cell NxN with projective2d or 3×3 H mapping j->i
    %                    - struct array with fields .i, .j, .H (j->i)
    %
    % Outputs
    %   K      : 1×N cell, K{i} = [f 0 cx; 0 f cy; 0 0 1]
    %   R      : 1×N cell, absolute rotations (w2c) with R{seed}=I
    %   fUsed : scalar if estimated globally; otherwise N×1 vector of per-image fallback focals

    arguments
        input (1, 1) struct
        pairs (1, :) struct
        matches (:, :) cell
        imageSizes (:, 3) double {mustBeFinite}
        numImages (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
        seed (1, 1) double {mustBeInteger, mustBePositive, mustBeFinite}
        initialTforms cell
        numMatches double
    end

    N = numImages;
    K = cell(1, N);
    R = repmat({eye(3)}, 1, N);

    % ---------- (A) Estimate focal lengths
    switch input.focalEstimateMethod
        case 'wConstraint'
            % ---------- (B) Estimate global focal from homographies ----------
            ws = []; % collect positive candidates for w = 1/f^2

            for k = 1:numel(pairs)
                i = pairs(k).i;
                j = pairs(k).j;
                H = pairs(k).Hij;

                % i, j are the images in H = H(j->i)
                Hi = imageSizes(i, 1); Wi = imageSizes(i, 2);
                Hj = imageSizes(j, 1); Wj = imageSizes(j, 2);

                [cxi, cyi] = ppCenter(Wi, Hi);
                [cxj, cyj] = ppCenter(Wj, Hj);

                Ci = [1 0 cxi; 0 1 cyi; 0 0 1]; % principal shift for i
                Cj = [1 0 cxj; 0 1 cyj; 0 0 1]; % principal shift for j

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
                    fCands = 1 ./ sqrt(ws);

                    % plausible focal band relative to image sizes
                    base = median(max(imageSizes, [], 2)); % typical "longer side"
                    fLo = 0.3 * base; % generous low bound
                    fHi = 6.0 * base; % generous high bound

                    fCands = fCands(isfinite(fCands) & fCands >= fLo & fCands <= fHi);

                    if isempty(fCands)
                        haveFocal = false;
                    else
                        fUsed = median(fCands); % robust pick
                        fprintf('Estimated focal length (robust): %.4f pixels\n', fUsed);
                    end

                end

            end

            if ~haveFocal
                % ---------- (C) Fallback focal(s): 0.8 * max(H,W) per image ----------
                fFallback = 0.8 * max(imageSizes, [], 2); % N×1
                fUsed = median(fFallback); % vector
                fprintf(['Cannot estimate focal lengths, %s motion model is used! ', ...
                         'Therefore, using the max(h,w) x 0.8 | f used: %.4f pixels\n'], input.transformationType, fUsed);
            end

        case 'shumSzeliskiOneHPaper'
            % pairs(e).H is already j->i (column-form x_i ~ H * x_j)
            Hc = cell(1, numel(pairs));

            for e = 1:numel(pairs)
                i = pairs(e).i; j = pairs(e).j;

                Hi = imageSizes(i, 1); Wi = imageSizes(i, 2);
                Hj = imageSizes(j, 1); Wj = imageSizes(j, 2);

                % Center and determinant-normalize (expects j->i)
                H = pairs(e).Hij; % ensure bottom-right = 1
                Hc{e} = centerNormalizeH(H, Wi, Hi, Wj, Hj);
            end

            % Drop empties / ill-conditioned
            Hc = Hc(~cellfun(@isempty, Hc));

            % Optionally also use the opposite direction (AFTER centering)
            HcBoth = [Hc, cellfun(@(M) inv(M), Hc, 'UniformOutput', false)];

            % Per-H focal (guarded)
            fvec = cellfun(@focalsHomographyShumsz, HcBoth);
            fvec = fvec(isfinite(fvec) & fvec > 0 & fvec < 5e4); % sanity cap

            if ~isempty(fvec)
                fUsed = median(fvec);
                fprintf('Estimated focal length Shum–Szeliski (single homography H): %.4f pixels\n', fUsed);
            else
                % Fallback: 0.8*max(h,w) per image -> global median
                fFallback = 0.8 * max(imageSizes, [], 2); % N×1
                fUsed = median(fFallback);
                fprintf(['Cannot estimate focal lengths, %s motion model is used! ', ...
                         'Therefore, using the max(h,w) x 0.8 | f used: %.4f pixels\n'], ...
                    input.transformationType, fUsed);
            end

        otherwise
            error('Require one focal estimate method.')
    end

    % ---------- (B) Build K for all images  (consistent pp) ----------
    parfor i = 1:N
        Hi = imageSizes(i, 1); Wi = imageSizes(i, 2);
        [cx, cy] = ppCenter(Wi, Hi);
        fi = fUsed;
        K{i} = [fi 0 cx; 0 fi cy; 0 0 1];
    end

    % ---------- (C) Build MST from MATCH COUNTS (SIMPLIFIED) ----------
    % Get MST based on match counts (like working code)
    tree = maximumSpanningTree(numMatches);

    % ---------- (D) Extract rotations from MST edges ----------
    % Now compute relative rotations for edges in the tree
    % Vectorized: Extract edges from upper triangle of tree
    [iVec, jVec] = find(triu(tree, 1));
    treeEdges = [iVec, jVec];

    % Propagate rotations along tree from seed
    visited = false(N, 1);
    visited(seed) = true;
    R{seed} = eye(3);
    queue = seed;

    while ~isempty(queue)
        u = queue(1); queue(1) = [];

        % Find neighbors in tree
        for e = 1:size(treeEdges, 1)
            i = treeEdges(e, 1);
            j = treeEdges(e, 2);

            if i == u && ~visited(j)
                % u -> j
                Hij = initialTforms{i, j}; % i <- j

                if ~isempty(Hij)
                    Wi = imageSizes(i, 2); Hi = imageSizes(i, 1);
                    Wj = imageSizes(j, 2); Hj = imageSizes(j, 1);
                    Rrel = relativeRotHij(Hij, Wi, Hi, Wj, Hj, fUsed); % R_j R_i^T
                    R{j} = projectToSO3(Rrel' * R{i});
                else
                    R{j} = R{i}; % fallback
                end

                visited(j) = true;
                queue = [queue, j];

            elseif j == u && ~visited(i)
                % u -> i
                Hji = initialTforms{j, i}; % j <- i

                if ~isempty(Hji)
                    Wi = imageSizes(i, 2); Hi = imageSizes(i, 1);
                    Wj = imageSizes(j, 2); Hj = imageSizes(j, 1);
                    Rrel = relativeRotHij(Hji, Wi, Hi, Wj, Hj, fUsed); % R_i R_j^T
                    R{i} = projectToSO3(Rrel' * R{j});
                else
                    R{i} = R{j}; % fallback
                end

                visited(i) = true;
                queue = [queue, i];
            end

        end

    end

    % ---------- (E) Check rotation consistency ----------
    [noRotation, meanAE, medAE, maxAE] = rotationConsistency(input, pairs, imageSizes, R, fUsed);

    fprintf('Init rotation consistency: mean=%.2f°, median=%.2f°, max=%.2f°, noRot=%i\n', ...
        meanAE, medAE, maxAE, noRotation);

    if noRotation
        fprintf('Panorama classified as NON-rotational (planar / translating camera).\n');
    else
        fprintf('Panorama classified as rotational (approx. pure rotation about center).\n');
    end

    % ---------- (F) Chain homographies to seed using getTforms pattern ----------
    if noRotation || input.forcePlanarScan
        % Chain homographies using clean pattern
        H2seed = chainedHomographies(tree, seed, initialTforms, N);
    else
        H2seed = repmat({eye(3)}, 1, N);
    end

end

% ---------- Helper Functions ----------
function tree = maximumSpanningTree(G)
    % MAXIMUMSPANNINGTREE Kruskal maximum spanning tree from match counts.
    %   tree = maximumSpanningTree(G) returns an undirected graph adjacency
    %   matrix `tree` (NxN) representing the maximum spanning tree computed
    %   from the input weight matrix G. Higher weights are preferred.
    %
    % Inputs
    %   G    - NxN numeric symmetric weight matrix (e.g., match counts).
    %
    % Outputs
    %   tree - NxN numeric adjacency matrix containing selected MST edges.

    arguments
        G (:, :) {mustBeNumeric, mustBeFinite}
    end

    n = size(G, 1);
    ccs = (1:n)';
    components = cell(n, 1);

    for i = 1:n
        components{i} = i;
    end

    tree = zeros(n);
    numEdges = 0;
    [values, indices] = sort(G(:), 'descend');

    for k = 1:length(indices)

        if values(k) > 0
            i = mod(indices(k) - 1, n) + 1;
            j = ceil(indices(k) / n);

            if ccs(i) ~= ccs(j)
                tree(i, j) = values(k);
                tree(j, i) = values(k);
                components{ccs(i)} = [components{ccs(i)}; components{ccs(j)}];
                ccs(components{ccs(j)}) = ccs(i);
                numEdges = numEdges + 1;
            end

        end

        if numEdges == n - 1
            break;
        end

    end

end

function tforms = chainedHomographies(G, i, Tforms, n)
    % CHAINEDHOMOGRAPHIES Chain homographies from a seed view to all views.
    %   tforms = chainedHomographies(G, i, Tforms, n) composes homographies
    %   stored in Tforms using connectivity encoded by G starting at seed i.
    %
    % Inputs
    %   G      - NxN numeric adjacency matrix (nonzero entries indicate available transforms)
    %   i      - seed index (scalar integer)
    %   Tforms - cell NxN of homography matrices (or transform objects)
    %   n      - number of views (N)
    %
    % Outputs
    %   tforms - 1xN cell array of homographies mapping each view to the seed.

    arguments
        G (:, :) {mustBeNumeric}
        i (1, 1) {mustBeInteger, mustBePositive, mustBeFinite}
        Tforms cell
        n (1, 1) {mustBeInteger, mustBePositive, mustBeFinite}
    end

    visited = zeros(n, 1);
    tforms = repmat({eye(3)}, 1, n);
    tforms{i} = eye(3); % seed
    tforms = updateTforms(G, i, visited, tforms, Tforms, n);
end

function tforms = updateTforms(G, i, visited, tforms, Tforms, n)
    % UPDATETFORMS Recursively update chain homographies (helper for chaining).
    %   tforms = updateTforms(G, i, visited, tforms, Tforms, n) recursively
    %   walks neighbors of node i and composes homographies into `tforms`.
    %
    % Inputs
    %   G       - NxN numeric adjacency matrix
    %   i       - current seed/index (scalar)
    %   visited - Nx1 logical/numeric visited mask
    %   tforms  - 1xN cell of current homography chains
    %   Tforms  - NxN cell of per-edge transforms
    %   n       - number of views
    %
    % Outputs
    %   tforms  - updated 1xN cell array of homographies

    arguments
        G (:, :) {mustBeNumeric}
        i (1, 1) {mustBeInteger, mustBePositive, mustBeFinite}
        visited (:, 1)
        tforms cell
        Tforms cell
        n (1, 1) {mustBeInteger, mustBePositive, mustBeFinite}
    end

    visited(i) = 1;

    for j = 1:n

        if G(i, j) > 0 && ~visited(j)
            tform = Tforms{i, j};
            tform = (tform' * tforms{i}')';
            tforms{j} = (tform' ./ tform(3, 3)')';
            [tforms] = updateTforms(G, j, visited, tforms, Tforms, n);
        end

    end

end

function [noRotation, meanAE, medAE, maxAE] = rotationConsistency(input, pairs, imageSizes, R, fUsed)
    % ROTATIONCONSISTENCY Compute angular error statistics between relative rotations.
    %   [noRotation, meanAE, medAE, maxAE] = rotationConsistency(input, pairs,
    %       imageSizes, R, fUsed) computes per-pair angle errors comparing
    %   estimated rotations R to those implied by homographies in pairs.
    %
    % Inputs
    %   input      - options struct (controls verbose output)
    %   pairs      - struct array with fields .i, .j and .Hij (homographies)
    %   imageSizes - N×2 array of image sizes
    %   R          - 1×N cell array of rotation matrices
    %   fUsed      - scalar or vector of focal lengths used for relative rot calc
    %
    % Outputs
    %   noRotation - logical scalar: true if content classified as non-rotational
    %   meanAE     - mean angular error (degrees)
    %   medAE      - median angular error (degrees)
    %   maxAE      - maximum angular error (degrees)

    arguments
        input struct
        pairs struct
        imageSizes (:, :) {mustBeNumeric}
        R cell
        fUsed {mustBeNumeric}
    end

    angleErr = zeros(numel(pairs), 1);

    for e = 1:numel(pairs)
        i = pairs(e).i; j = pairs(e).j; H = pairs(e).Hij;
        Wi = imageSizes(i, 2); Hi = imageSizes(i, 1);
        Wj = imageSizes(j, 2); Hj = imageSizes(j, 1);
        Rrel = relativeRotHij(H, Wi, Hi, Wj, Hj, fUsed);
        D = R{i} * R{j}'; % should match Rrel
        c = max(-1, min(1, (trace(D' * Rrel) - 1) / 2));
        angleErr(e) = acos(c); % radians
    end

    if input.verboseInitRKf
        fprintf('Init rotation consistency: mean=%.2f°, median=%.2f°, max=%.2f°\n', ...
            mean(angleErr) * 180 / pi, median(angleErr) * 180 / pi, max(angleErr) * 180 / pi);
    end

    meanAE = mean(angleErr) * 180 / pi;
    medAE = median(angleErr) * 180 / pi;
    maxAE = max(angleErr) * 180 / pi;

    noRotation = medAE > 0.6 && maxAE > 100;
end

function [cx, cy] = ppCenter(W, H)
    % PCCENTER Compute image principal-point center coordinates.
    %   [cx, cy] = ppCenter(W, H) returns the (cx,cy) image center coordinates
    %   used for centering homographies.
    %
    % Inputs
    %   W - image width (positive scalar)
    %   H - image height (positive scalar)
    %
    % Outputs
    %   cx, cy - center coordinates (scalars)

    arguments
        W (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        H (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    cx = W / 2; cy = H / 2;
end

function relativeRot = relativeRotHij(Hij, Wi, Hi, Wj, Hj, f)
    % RELATIVEROTHIJ Compute relative rotation from centered homography Hij.
    %   relativeRot = relativeRotHij(Hij, Wi, Hi, Wj, Hj, f) computes an
    %   approximation of R_i * R_j^T from the homography mapping j->i.
    %
    % Inputs
    %   Hij - 3x3 homography mapping points in image j to image i
    %   Wi,Hi, Wj,Hj - image widths and heights for images i and j
    %   f   - focal length (scalar)
    %
    % Outputs
    %   relativeRot - 3x3 rotation matrix approximating relative rotation

    arguments
        Hij (3, 3) {mustBeNumeric}
        Wi (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        Hi (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        Wj (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        Hj (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        f {mustBeNumeric}
    end

    [cxi, cyi] = ppCenter(Wi, Hi);
    [cxj, cyj] = ppCenter(Wj, Hj);
    Ci = [1 0 cxi; 0 1 cyi; 0 0 1];
    Cj = [1 0 cxj; 0 1 cyj; 0 0 1];

    Hc = Ci \ Hij * Cj; % j -> i, centered
    s = sign(det(Hc)) * nthroot(abs(det(Hc)) + eps, 3); % det-normalize
    Hn = Hc / s;

    K0 = diag([f, f, 1]); % zero-centered K
    relativeRot = projectToSO3(K0 \ Hn * K0); % ≈ R_i R_j^T
end

function f = focalsHomographyShumsz(H)
    % FOCALSHOMOGRAPHYSHUMSZ  Estimate focal length from a single homography
    %   f = focalsHomographyShumsz(H)
    %   Computes an estimate of the focal length (in pixels) from a single
    %   3x3 homography matrix. The method follows Shum & Szeliski-style
    %   algebraic constraints and returns NaN when a valid estimate cannot be
    %   derived from the provided matrix.
    %
    % Inputs
    %   H  - 3x3 numeric homography matrix (may be unnormalized)
    %
    % Outputs
    %   f  - scalar focal length estimate in pixels (double). Returns NaN
    %        if the estimate is invalid or cannot be computed.

    % Reference: Shum, HY., Szeliski, R. (2001). Construction of
    % Panoramic Image Mosaics with Global and Local Alignment.
    % In: Benosman, R., Kang, S.B. (eds) Panoramic Vision.
    % Monographs in Computer Science. Springer, New York,
    % NY. https://doi.org/10.1007/978-1-4757-3482-9_13

    arguments
        H (3, 3) double {mustBeFinite}
    end

    if isempty(H)
        f = NaN; return;
    end

    % --- f1 ---
    d1 = H(3, 1) * H(3, 2);
    d2 = (H(3, 2) - H(3, 1)) * (H(3, 2) + H(3, 1));
    v1 =- (H(1, 1) * H(1, 2) + H(2, 1) * H(2, 2)) / d1;
    v2 = (H(1, 1) ^ 2 + H(2, 1) ^ 2 - H(1, 2) ^ 2 - H(2, 2) ^ 2) / d2;

    if v1 < v2
        tmp = v1; v1 = v2; v2 = tmp;
    end

    if v1 > 0 && v2 > 0
        f1 = sqrt(v1 * (abs(d1) > abs(d2)) + v2 * (abs(d1) <= abs(d2)));
    elseif v1 > 0
        f1 = sqrt(v1);
    else
        f = NaN; return;
    end

    % --- f0 ---
    d1 = H(1, 1) * H(2, 1) + H(1, 2) * H(2, 2);
    d2 = H(1, 1) ^ 2 + H(1, 2) ^ 2 - H(2, 1) ^ 2 - H(2, 2) ^ 2;
    v1 = -H(1, 3) * H(2, 3) / d1;
    v2 = (H(2, 3) ^ 2 - H(1, 3) ^ 2) / d2;

    if v1 < v2
        tmp = v1; v1 = v2; v2 = tmp;
    end

    if v1 > 0 && v2 > 0
        f0 = sqrt(v1 * (abs(d1) > abs(d2)) + v2 * (abs(d1) <= abs(d2)));
    elseif v1 > 0
        f0 = sqrt(v1);
    else
        f = NaN; return;
    end

    f = sqrt(f1 * f0); % geometric mean
end

function Hn = centerNormalizeH(H, Wi, Hi, Wj, Hj)
    % CENTERNORMALIZEH  Center homography and normalize determinant to 1
    %   Hn = centerNormalizeH(H, Wi, Hi, Wj, Hj)
    %   Transforms the input homography (mapping points from image j to i)
    %   by shifting principal points to image centers and scaling the result
    %   so that its determinant has unit magnitude. Returns empty when the
    %   input is degenerate or non-finite.
    %
    % Inputs
    %   H   - 3x3 homography matrix mapping points from image j to image i
    %   Wi  - width of image i (positive scalar)
    %   Hi  - height of image i (positive scalar)
    %   Wj  - width of image j (positive scalar)
    %   Hj  - height of image j (positive scalar)
    %
    % Outputs
    %   Hn  - 3x3 centered and determinant-normalized homography. Returns
    %         empty [] if the input is invalid or normalization is not
    %         possible (e.g., non-finite determinant).

    arguments
        H (3, 3) double {mustBeFinite}
        Wi (1, 1) double {mustBeFinite, mustBePositive}
        Hi (1, 1) double {mustBeFinite, mustBePositive}
        Wj (1, 1) double {mustBeFinite, mustBePositive}
        Hj (1, 1) double {mustBeFinite, mustBePositive}
    end

    % Column-form H (j->i). Returns centered (Ci^{-1} H Cj) and det-normalized.

    [cxi, cyi] = ppCenter(Wi, Hi);
    [cxj, cyj] = ppCenter(Wj, Hj);

    Ci = [1 0 cxi; 0 1 cyi; 0 0 1];
    Cj = [1 0 cxj; 0 1 cyj; 0 0 1];

    Hc = (Ci \ H) * Cj;

    d = det(Hc);
    if ~isfinite(d) || d == 0, Hn = []; return; end

    s = sign(d) * nthroot(abs(d), 3);
    Hn = Hc / s;
end

function rotation = projectToSO3(M)
    % PROJECTTOSO3  Project a 3x3 matrix to the nearest proper rotation
    %   rotation = projectToSO3(M)
    %   Finds the closest proper rotation matrix (element of SO(3)) to the
    %   input 3x3 matrix using the singular value decomposition (SVD). The
    %   resulting matrix has determinant +1.
    %
    % Inputs
    %   M  - 3x3 numeric matrix to be projected
    %
    % Outputs
    %   rotation - 3x3 proper rotation matrix with det(rotation) == +1

    arguments
        M (3, 3) double {mustBeFinite}
    end

    % SVD projection to the closest proper rotation.
    [U, ~, V] = svd(M);
    rotation = U * diag([1, 1, sign(det(U * V'))]) * V';
end
