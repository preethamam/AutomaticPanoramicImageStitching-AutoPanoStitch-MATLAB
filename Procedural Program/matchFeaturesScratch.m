function [matches, matchMetric] = matchFeaturesScratch(F1, F2, varargin)
    %MATCHFEATUREESSCRATCH Fast, from-scratch feature matching (binary & float).
    %
    %   [matches, matchMetric] = matchFeaturesScratch(F1, F2, ...
    %       'Method',          'Exhaustive'|'Approximate', ...
    %       'MatchThreshold',  T, ...
    %       'MaxRatio',        r, ...
    %       'Unique',          true|false, ...
    %       'ApproxNumTables', nTables, ...           % binary only (LSH)
    %       'ApproxBitsPerKey', bitsPerKey, ...       % binary only (LSH)
    %       'ApproxProbes', nProbes, ...              % binary only (LSH)
    %       'ApproxKDBucketSize', bucketSize)         % non-binary KD-tree
    %
    % INPUT SHAPES:
    %   - Non-binary: F1 [N1 x D], F2 [N2 x D] (single/double)
    %   - Binary    : F1/F2 either:
    %       * logical/uint8 with values {0,1} as [N x Dbits] (unpacked), OR
    %       * binaryFeatures objects (packed uint8 rows).
    %
    % MATCH THRESHOLD semantics follow MATLAB's matchFeatures:
    %   - Binary   : percent of mismatched bits in [0,100].
    %   - Non-binary: Sum of Squared Differences (SSD).
    %
    % RATIO TEST (Lowe): accept if d1/d2 <= MaxRatio (default 0.6).
    %
    % RETURNS:
    %   matches:      [K x 2] indices into rows of F1 and F2
    %   matchMetric:  [K x 1] distances: (percent Hamming) or (SSD)
    %
    % DEFAULTS (aligned to preferences):
    %   'Method'         : 'Exhaustive'
    %   'MatchThreshold' : 3.5  (tip: >=10 for binary descriptors)
    %   'MaxRatio'       : 0.6
    %   'Unique'         : true
    %
    %   Binary Approx params (LSH-like):
    %       'ApproxNumTables'   : 6
    %       'ApproxBitsPerKey'  : 24
    %       'ApproxProbes'      : 4
    %   Non-binary Approx params (KD-tree via knnsearch):
    %       'ApproxKDTreeLeafSize' : 40
    %
    % Author: Dr. Preetham Manjunatha

    % ARGUMENTS (validation for positional inputs only)
    arguments
        F1
        F2
    end

    arguments (Repeating)
        varargin
    end

    % ------------------------- Parse inputs ---------------------------------
    % Back-compat shim: rewrite deprecated 'Approx.*' keys into flat names.
    varargin = normalizeApproxArgs(varargin); % back-compat mapper

    p = inputParser;
    addParameter(p, 'Method', 'Exhaustive', @(s)ischar(s) || isstring(s));
    addParameter(p, 'MatchThreshold', 3.5, @(x)isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'MaxRatio', 0.6, @(x)isnumeric(x) && isscalar(x) && x > 0 && x <= 1);
    addParameter(p, 'Unique', true, @islogical);

    % Binary-approx (LSH)
    addParameter(p, 'ApproxNumTables', 6, @(x)isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'ApproxBitsPerKey', 24, @(x)isnumeric(x) && isscalar(x) && x >= 8);
    addParameter(p, 'ApproxProbes', 4, @(x)isnumeric(x) && isscalar(x) && x >= 1);

    % Float-approx (KD-tree): use BucketSize (R2025b)
    addParameter(p, 'ApproxKDBucketSize', 40, @(x)isnumeric(x) && isscalar(x) && x >= 1);

    addParameter(p, 'ApproxFloatNNMethod', 'pca2nn', @(s)ischar(s) || isstring(s));

    parse(p, varargin{:});
    opt = p.Results;
    method = lower(string(opt.Method));
    approximateMethod = lower(string(opt.ApproxFloatNNMethod));

    % --------------------- Normalize/unwrap input ---------------------------
    [isBinary, A, B, nBits, ~] = normalizeInputs(F1, F2);

    % --- Early exit: no descriptors (common in binary ORB/BRIEF images) ---
    if isBinary && (isempty(A) || isempty(B))
        matches = zeros(0, 2, 'uint32');
        matchMetric = zeros(0, 1, 'single');
        return;
    end

    % ----- Fix default thresholds based on descriptor type -----
    if isBinary

        if ~isfield(opt, 'MatchThreshold') || isempty(opt.MatchThreshold)
            opt.MatchThreshold = 10; % percent mismatched bits (common default window: 5–25)
        end

    else
        % For floats, rely on ratio; do not hard-threshold SSD by default.
        if ~isfield(opt, 'MatchThreshold') || isempty(opt.MatchThreshold)
            opt.MatchThreshold = inf; % <-- key change
        end

        % If descriptors look unnormalized (large magnitudes), L2-normalize
        % (matchFeatures works fine on normalized vectors; this makes SSDs comparable)
        if max(abs(A(:))) > 2 || max(abs(B(:))) > 2
            A = single(A);
            B = single(B);
            A = normalizeRowsL2(A);
            B = normalizeRowsL2(B);
        end

    end

    % --------------------- Core matching -----------------------------------
    switch method
        case "exhaustive"

            if isBinary
                [~, idx2, d12, d22] = nearest2HammingExhaustiveSuperfast(A, B, nBits);
                dBest = (d12 / nBits) * 100; % percent mismatch
                dSecond = (d22 / nBits) * 100;
            else
                [~, idx2, d12, d22] = nearest2SSDExhaustive(A, B);
                dBest = d12; % SSD
                dSecond = d22; % SSD
            end

        case "approximate"

            if isBinary
                % LSH-like approximate search on binary descriptors
                [~, idx2, d12, d22] = nearest2HammingLSHSuperfast( ...
                    A, B, nBits, opt.ApproxNumTables, opt.ApproxBitsPerKey, opt.ApproxProbes);
                dBest = (d12 / nBits) * 100; % percent mismatch
                dSecond = (d22 / nBits) * 100;
            else
                % ---------- NON-BINARY (float) APPROXIMATE ----------
                switch approximateMethod
                    case 'pca2nn'
                        optsLocal = struct('ApproxNumComponents', 48, ... % try 48 or 64
                            'UsePCA', true, ...
                            'BlockRows', 4000, ...
                            'UseParfor', true, ...
                            'UseGPU', true); % set true if you have CUDA

                        [~, idx2, d12eu2, d22eu2] = nearest2ApproxFloatFast(A, B, optsLocal);
                        dBest = d12eu2; % already SSD
                        dSecond = d22eu2;
                    case 'kdtree'
                        [~, idx2, d12eu, d22eu] = nearest2KDTree(A, B, opt.ApproxKDBucketSize);
                        % Convert to SSD to match MATLAB metric
                        dBest = d12eu .^ 2;
                        dSecond = d22eu .^ 2;
                    case 'subsetpdist2'
                        % High-D: directly use subset-pdist2 (very robust)
                        [~, idx2, d12eu, d22eu] = nearest2SubsetPdist2(A, B, 12000);
                        dBest = d12eu .^ 2;
                        dSecond = d22eu .^ 2;
                    otherwise
                        error('Select a approximate method')
                end

            end

        otherwise
            error('Unknown Method: %s', opt.Method);
    end

    % --------------------- Ratio & Threshold filters ------------------------
    if isBinary
        ratioOK = (dBest <= opt.MaxRatio * dSecond); % Hamming: linear
    else
        r2 = opt.MaxRatio * opt.MaxRatio; % Float: SSD (squared)
        ratioOK = (dBest <= r2 * dSecond);
    end

    threshOK = (dBest <= opt.MatchThreshold);
    keep = ratioOK & threshOK & isfinite(dBest) & isfinite(dSecond);

    i1 = (1:size(A, 1)).';
    i1 = i1(keep); % idx1 is simply 1..N1
    i2 = idx2(keep);
    d = dBest(keep);

    % --------------------- Enforce uniqueness if requested ------------------
    if opt.Unique && ~isempty(i1)
        [dSorted, order] = sort(d, 'ascend');
        i1s = i1(order); i2s = i2(order);

        % Use sparse logical marking without max(i1s) expansion:
        used1 = false(size(A, 1), 1);
        used2 = false(size(B, 1), 1);

        keep = false(numel(i1s), 1);

        for k = 1:numel(i1s)
            a = i1s(k); b = i2s(k);

            if ~used1(a) && ~used2(b)
                keep(k) = true;
                used1(a) = true; used2(b) = true;
            end

        end

        matches = [i1s(keep), i2s(keep)];
        matchMetric = dSorted(keep);
    else
        matches = [i1, i2];
        matchMetric = d;
    end

    % For convenience, ensure column vector
    matchMetric = matchMetric(:);
end

function Xn = normalizeRowsL2(X)
    % NORMALIZEROWSL2 L2-normalize each row of a matrix.
    %   Xn = normalizeRowsL2(X)
    %   Scales each row to unit L2 norm with a small epsilon for stability.
    %
    % Inputs:
    %   X - [N x D] numeric matrix (single/double)
    %
    % Outputs:
    %   Xn - [N x D] matrix with each row L2-normalized (same type as input)

    arguments
        X (:, :) {mustBeNumeric}
    end

    n = sqrt(sum(X .^ 2, 2)) + eps('single');
    Xn = X ./ n;
end

% --------------- Helper: figure out if inputs are binary ----------------
function [isBinary, A, B, nBits, unpacked] = normalizeInputs(F1, F2)
    % NORMALIZEINPUTS Detect descriptor type and coerce to working form.
    %   [isBinary, A, B, nBits, unpacked] = normalizeInputs(F1, F2)
    %   Accepts binaryFeatures, logical/uint8 0-1 matrices, or float matrices.
    %   Returns packed uint8 bytes for binary and single/double for float.
    %
    % Inputs:
    %   F1, F2 - descriptor sets (binaryFeatures, logical/uint8 0/1, or numeric vectors)
    %
    % Outputs:
    %   isBinary - logical flag true when inputs are binary descriptors
    %   A, B     - prepared descriptor matrices (packed uint8 for binary, numeric for float)
    %   nBits    - number of bits per descriptor when binary (0 for float)
    %   unpacked - logical flag indicating original inputs were unpacked bits

    arguments
        F1
        F2
    end

    unpacked = false;

    if isa(F1, 'binaryFeatures') && isa(F2, 'binaryFeatures')
        % Packed bytes (each row packed bits). Keep as uint8 bytes.
        A = F1.Features; % [N1 x nbytes]
        B = F2.Features; % [N2 x nbytes]
        isBinary = true;
        nBits = size(A, 2) * 8;
    else
        % If both are logical/uint8 with 0/1, treat as binary UNPACKED bits
        f1isbin = (islogical(F1) || (isa(F1, 'uint8') && all((F1(:) == 0) | (F1(:) == 1))));
        f2isbin = (islogical(F2) || (isa(F2, 'uint8') && all((F2(:) == 0) | (F2(:) == 1))));

        if f1isbin && f2isbin
            % Pack into uint8 bytes for speed
            [A, nBits] = packBits(F1);
            [B, ~] = packBits(F2);
            isBinary = true;
            unpacked = true;
        else
            % non-binary (float vectors)
            if ~isa(F1, 'single') && ~isa(F1, 'double'); F1 = single(F1); end
            if ~isa(F2, 'single') && ~isa(F2, 'double'); F2 = single(F2); end
            validateattributes(F1, {'single', 'double'}, {'2d', 'nonempty'});
            validateattributes(F2, {'single', 'double'}, {'2d', 'nonempty'});

            if size(F1, 2) ~= size(F2, 2)
                error('Descriptor dimensions must match for non-binary.');
            end

            A = F1; B = F2; isBinary = false; nBits = 0;
        end

    end

end

% --------------------- Exhaustive (binary, Hamming) ---------------------
function [idx1, idx2, d1, d2] = nearest2HammingExhaustiveSuperfast(Abytes, Bbytes, nBits)
    % NEAREST2HAMMINGEXHAUSTIVESUPERFAST 2-NN under Hamming for binary features.
    %   [idx1, idx2, d1, d2] = nearest2HammingExhaustiveSuperfast(Abytes,Bbytes,nBits)
    %   Abytes,Bbytes are uint8 packed bit rows; distances are bit counts.
    %
    % Inputs:
    %   Abytes - [N1 x nbytes] uint8 packed binary descriptors
    %   Bbytes - [N2 x nbytes] uint8 packed binary descriptors
    %   nBits  - scalar number of bits per descriptor
    %
    % Outputs:
    %   idx1, idx2 - [N1 x 1] indices into A and B (idx1 = 1:N1) and nearest neighbors
    %   d1, d2     - [N1 x 1] first and second nearest Hamming distances (bit counts)

    arguments
        Abytes (:, :) uint8
        Bbytes (:, :) uint8
        nBits (1, 1) double {mustBeInteger, mustBePositive}
    end

    [idx2, d1, d2] = nearest2HammingExhaustiveMEX(Abytes, Bbytes);
    idx1 = (1:size(Abytes, 1)).';
    % guard second
    d2(~isfinite(d2) | d2 == 0) = single(ceil(nBits));
end

% --------------------- Exhaustive (float, SSD) --------------------------
function [idx1, idx2, d1, d2] = nearest2SSDExhaustive(A, B)
    % NEAREST2SSDEXHAUSTIVE 2-NN under SSD for float descriptors.
    %   [idx1, idx2, d1, d2] = nearest2SSDExhaustive(A,B)
    %   Computes SSD in manageable blocks for memory efficiency.
    %
    % Inputs:
    %   A - [N1 x D] numeric descriptors
    %   B - [N2 x D] numeric descriptors
    %
    % Outputs:
    %   idx1, idx2 - [N1 x 1] index of nearest and 2nd-nearest in B for each row of A
    %   d1, d2     - [N1 x 1] SSD distances for nearest and second-nearest

    arguments
        A (:, :) {mustBeNumeric}
        B (:, :) {mustBeNumeric}
    end

    % Compute distances in chunks to control memory
    N1 = size(A, 1);
    % D = size(A, 2); % not used
    block = max(1, floor(1e7 / max(N2, 1))); % heuristic
    idx2 = zeros(N1, 1, 'uint32'); d1 = inf(N1, 1); d2 = inf(N1, 1);

    for i = 1:ceil(N1 / block)
        s = (i - 1) * block + 1;
        e = min(N1, i * block);
        Ablk = A(s:e, :);
        % Efficient SSD: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a*b'
        a2 = sum(Ablk .^ 2, 2); % [blk x 1]
        b2 = sum(B .^ 2, 2); % [N2 x 1]
        G = Ablk * B.'; % [blk x N2]
        D2 = a2 + b2.' - 2 * G; % SSD
        % find top-2 per row
        [best, idx] = min(D2, [], 2);
        D2(sub2ind(size(D2), (1:size(D2, 1))', idx)) = inf; % mask best
        second = min(D2, [], 2);

        idx2(s:e) = idx;
        d1(s:e) = best;
        d2(s:e) = second;
    end

    idx1 = (1:N1).';
end

function [idx1, idx2, d1, d2] = nearest2SubsetPdist2(A, B, subset)
    % NEAREST2SUBSETPDIST2 Approximate 2-NN via subset + pdist2.
    %   [idx1, idx2, d1, d2] = nearest2SubsetPdist2(A,B,subset)
    %   Returns Euclidean distances; caller may square to obtain SSD.
    % Inputs:
    %   A      - [N1 x D] numeric descriptors
    %   B      - [N2 x D] numeric descriptors
    %   subset - scalar number of random candidate rows from B to consider
    %
    % Outputs:
    %   idx1, idx2 - [N1 x 1] indices of nearest and second-nearest in B
    %   d1, d2     - [N1 x 1] Euclidean distances to those neighbors

    arguments
        A (:, :) {mustBeNumeric}
        B (:, :) {mustBeNumeric}
        subset (1, 1) double {mustBeInteger, mustBePositive}
    end

    subset = min(subset, size(B, 1));
    candB = randperm(size(B, 1), subset);
    B2 = B(candB, :);

    % single precision is faster
    if ~isa(A, 'single'), A = single(A); end
    if ~isa(B2, 'single'), B2 = single(B2); end

    % D: K x size(A,1), I: K x size(A,1), where K = min(2, size(B2,1))
    [D, I] = pdist2(B2, A, 'euclidean', 'Smallest', 2);

    % ----- EDGE CASE: only one candidate neighbor (size(B2,1) == 1) -----
    if size(D, 1) == 1
        % Duplicate the index row and make second distance slightly larger
        D = [D; D(1, :) + eps(D(1, :))];
        I = [I; I(1, :)];
    end

    idx1 = (1:size(A, 1)).';
    idx2 = uint32(candB(I(1, :)).'); % map back into original B
    d1 = D(1, :).';
    d2 = D(2, :).';
end

function [idx1, idx2, d1, d2] = nearest2KDTree(A, B, bucketSize)
    % NEAREST2KDTREE KD-tree based 2-NN for float descriptors.
    %   [idx1, idx2, d1, d2] = nearest2KDTree(A,B,bucketSize)
    %   Returns Euclidean distances; caller may square for SSD.
    % Inputs:
    %   A - [N1 x D] numeric descriptors
    %   B - [N2 x D] numeric descriptors
    %   bucketSize - integer BucketSize passed to createns
    %
    % Outputs:
    %   idx1, idx2 - [N1 x 1] indices into A and B for nearest neighbors
    %   d1, d2     - [N1 x 1] Euclidean distances to first and second neighbors

    arguments
        A (:, :) {mustBeNumeric}
        B (:, :) {mustBeNumeric}
        bucketSize (1, 1) double {mustBeInteger, mustBePositive}
    end

    idx1 = (1:size(A, 1)).';

    % Primary: KD-tree with BucketSize
    Mdl = createns(B, 'NSMethod', 'kdtree', 'Distance', 'euclidean', 'BucketSize', bucketSize);
    [nbrIdx, nbrDist] = knnsearch(Mdl, A, 'K', 2); % Euclidean distances

    idx2 = uint32(nbrIdx(:, 1));
    d1 = nbrDist(:, 1);
    d2 = nbrDist(:, 2);

end

function [idx1, idx2, dBest, dSecond] = nearest2ApproxFloatFast(A, B, opts)
    % NEAREST2APPROXFLOATFAST Fast approximate 2-NN for float descriptors.
    %   [idx1, idx2, dBest, dSecond] = nearest2ApproxFloatFast(A,B,opts)
    %   Strategy: optional PCA, L2-normalize, block GEMM, then take top-2.
    %
    %   opts fields (defaults shown):
    %     .ApproxNumComponents  (48)
    %     .UsePCA               (true)
    %     .BlockRows            (4000)
    %     .UseParfor            (true)
    %     .UseGPU               (false)
    %
    % Inputs:
    %   A    - [N1 x D] numeric descriptors
    %   B    - [N2 x D] numeric descriptors
    %   opts - options struct (fields described above)
    %
    % Outputs:
    %   idx1, idx2 - [N1 x 1] indices of nearest neighbors in B
    %   dBest,dSecond - [N1 x 1] distance metrics (converted so caller can treat as SSD)

    arguments
        A (:, :) {mustBeNumeric}
        B (:, :) {mustBeNumeric}
        opts (1, 1) struct
    end

    if ~isfield(opts, 'ApproxNumComponents'), opts.ApproxNumComponents = 48; end
    if ~isfield(opts, 'UsePCA'), opts.UsePCA = true; end
    if ~isfield(opts, 'BlockRows'), opts.BlockRows = 4000; end
    if ~isfield(opts, 'UseParfor'), opts.UseParfor = true; end
    if ~isfield(opts, 'UseGPU'), opts.UseGPU = false; end

    % -------- 0) cast to single --------
    A = single(A); B = single(B);

    % -------- 1) PCA (on B), then project A with same basis ------
    if opts.UsePCA && size(A, 2) > opts.ApproxNumComponents
        % center B, compute PCA basis
        muB = mean(B, 1, 'omitnan');
        [coeff, ~, ~, ~, ~, ~] = pca(bsxfun(@minus, B, muB), ...
            'NumComponents', opts.ApproxNumComponents);
        B = bsxfun(@minus, B, muB) * coeff;
        A = bsxfun(@minus, A, muB) * coeff;
    end

    % -------- 2) L2 normalize (so SSD ranking == cos-sim ranking) ------
    A = bsxfun(@rdivide, A, sqrt(sum(A .^ 2, 2)) + eps('single'));
    B = bsxfun(@rdivide, B, sqrt(sum(B .^ 2, 2)) + eps('single'));

    N1 = size(A, 1); N2 = size(B, 1);
    idx1 = (1:N1).';
    idx2 = zeros(N1, 1, 'uint32');
    dBest = inf(N1, 1, 'single');
    dSecond = inf(N1, 1, 'single');

    blk = opts.BlockRows;
    nBlocks = ceil(N1 / blk);

    useGPU = opts.UseGPU && canUseGPU();
    usePar = opts.UseParfor && (nBlocks > 1);

    if usePar
        % parallel over blocks
        idx2Parts = cell(nBlocks, 1);
        dBestParts = cell(nBlocks, 1);
        dSecParts = cell(nBlocks, 1);

        parfor bi = 1:nBlocks
            s = (bi - 1) * blk + 1; e = min(bi * blk, N1);
            [idx2Parts{bi}, dBestParts{bi}, dSecParts{bi}] = doBlock(A(s:e, :), B, useGPU);
        end

        idx2 = vertcat(idx2Parts{:});
        dBest = vertcat(dBestParts{:});
        dSecond = vertcat(dSecParts{:});

    else

        for s = 1:blk:N1
            e = min(s + blk - 1, N1);
            [idx2(s:e), dBest(s:e), dSecond(s:e)] = doBlock(A(s:e, :), B, useGPU);
        end

    end

end

% convert cosine similarity to SSD on unit vectors:
%   ||a-b||^2 = 2 - 2*(a·b)
function [idBlock, d1Block, d2Block] = doBlock(Ablk, Bfull, useGPUflag)
    % DOBLOCK Helper to compute 2-NN for a block, CPU or GPU path.
    %   [idBlock, d1Block, d2Block] = doBlock(Ablk,Bfull,useGPUflag)
    %
    % Inputs:
    %   Ablk      - [M x D] block of descriptors from A
    %   Bfull     - [N2 x D] full B matrix
    %   useGPUflag- logical flag: compute on GPU if true
    %
    % Outputs:
    %   idBlock   - [M x 1] indices of nearest neighbor in Bfull (uint32)
    %   d1Block   - [M x 1] first neighbor distance (converted metric)
    %   d2Block   - [M x 1] second neighbor distance

    arguments
        Ablk (:, :) {mustBeNumeric}
        Bfull (:, :) {mustBeNumeric}
        useGPUflag (1, 1) logical
    end

    if useGPUflag
        GA = gpuArray(Ablk);
        GB = gpuArray(Bfull);
        G = GA * GB.'; % [rows x N2] cosine sim
        % top-2 along 2nd dimension
        [sim1, id1] = max(G, [], 2);
        G(sub2ind(size(G), (1:size(G, 1))', id1)) = -inf('single');
        [sim2, ~] = max(G, [], 2);
        idBlock = gather(uint32(id1));
        d1Block = gather(single(2 - 2 * sim1));
        d2Block = gather(single(2 - 2 * sim2));
    else
        G = Ablk * Bfull.'; % BLAS sgemm, multi-threaded
        [sim1, id1] = max(G, [], 2);
        G(sub2ind(size(G), (1:size(G, 1))', id1)) = -inf('single');
        [sim2, ~] = max(G, [], 2);
        idBlock = uint32(id1);
        d1Block = single(2 - 2 * sim1);
        d2Block = single(2 - 2 * sim2);
    end

end

function tf = canUseGPU()
    % CANUSEGPU True if a compatible GPU device is available.
    % Inputs: none
    % Outputs:
    %   tf - logical true when a GPU device is available
    try
        tf = parallel.gpu.GPUDevice.isAvailable;
    catch
        tf = false;
    end
end

% --------------- Approx (binary): LSH-like hashing ----------------------
function [idx1, idx2, d1, d2] = nearest2HammingLSHSuperfast(Abytes, Bbytes, nBits, numTables, bitsPerKey, nProbes)
    % NEAREST2HAMMINGLSHSUPERFAST Approximate 2-NN for binary via hashing.
    %   Currently delegates to OMP exhaustive MEX for speed.
    % Inputs:
    %   Abytes - [N1 x nbytes] uint8 packed binary descriptors
    %   Bbytes - [N2 x nbytes] uint8 packed binary descriptors
    %   nBits  - number of bits per descriptor
    %   numTables,bitsPerKey,nProbes - LSH tuning parameters
    %
    % Outputs:
    %   idx1, idx2 - index arrays for neighbors
    %   d1, d2     - Hamming distances for first and second nearest

    arguments
        Abytes (:, :) uint8
        Bbytes (:, :) uint8
        nBits (1, 1) double {mustBeInteger, mustBePositive}
        numTables (1, 1) double {mustBeInteger, mustBePositive}
        bitsPerKey (1, 1) double {mustBeInteger, mustBePositive}
        nProbes (1, 1) double {mustBeInteger, mustBePositive}
    end

    % Touch unused NV inputs to satisfy code analyzer
    [idx2, d1, d2] = nearest2HammingExhaustiveOMPMEX(Abytes, Bbytes);
    idx1 = (1:size(Abytes, 1)).';
    d2(~isfinite(d2) | d2 == 0) = single(ceil(nBits));
end

% --------------------- Utilities ----------------------------------------
function [packed, nBits] = packBits(unpacked01)
    % PACKBITS Pack 0/1 logical/uint8 bits into uint8 bytes per row.
    %   [packed, nBits] = packBits(unpacked01)
    %   Input: [N x Dbits] logical/uint8 with values {0,1}
    %   Output: packed [N x ceil(Dbits/8)] uint8 and the original bit count.
    % Inputs:
    %   unpacked01 - [N x Dbits] logical or uint8 with values 0/1
    %
    % Outputs:
    %   packed - [N x nbytes] uint8 packed bytes
    %   nBits  - original number of bits per row

    arguments
        unpacked01 (:, :) {mustBeNumericOrLogical}
    end

    validateattributes(unpacked01, {'logical', 'uint8'}, {'2d', 'nonempty'});
    N = size(unpacked01, 1);
    D = size(unpacked01, 2);
    nbytes = ceil(D / 8);
    packed = zeros(N, nbytes, 'uint8');

    for b = 1:D
        byteIdx = ceil(b / 8);
        bitPos = 8 - mod(b - 1, 8); % MSB-first
        packed(:, byteIdx) = bitor(packed(:, byteIdx), uint8(unpacked01(:, b)) .* uint8(2 ^ (bitPos - 1)));
    end

    nBits = D;
end

% ---- Back-compat normalizer (put near other helpers) ----
function nv = normalizeApproxArgs(nv)
    % NORMALIZEAPPROXARGS Map legacy Approx.* NV-pairs to current option names.
    %   nv = normalizeApproxArgs(nv)
    % Inputs:
    %   nv - 1xM cell array of name-value pairs
    %
    % Outputs:
    %   nv - normalized name-value pairs cell array

    arguments
        nv (1, :) cell
    end

    if isempty(nv), return; end

    for k = 1:2:numel(nv)
        if ~ischar(nv{k}) && ~isstring(nv{k}), continue; end
        key = string(nv{k});

        switch key
            case "Approx.NumTables", nv{k} = 'ApproxNumTables';
            case "Approx.BitsPerKey", nv{k} = 'ApproxBitsPerKey';
            case "Approx.Probes", nv{k} = 'ApproxProbes';
            case "Approx.KDTreeLeafSize", nv{k} = 'ApproxKDBucketSize'; % legacy -> bucket
            case "ApproxKDTreeLeafSize", nv{k} = 'ApproxKDBucketSize'; % legacy -> bucket
            otherwise
        end

    end
end
