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
    % DEFAULTS (aligned to your preferences):
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

    addParameter(p, 'ApproxFloatNNMethod', 'pca_2nn', @(s)ischar(s) || isstring(s));

    parse(p, varargin{:});
    opt = p.Results;
    method = lower(string(opt.Method));
    approximateMethod = lower(string(opt.ApproxFloatNNMethod));

    % --------------------- Normalize/unwrap input ---------------------------
    [isBinary, A, B, nBits, unpackedBinary] = normalizeInputs(F1, F2);

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
                [idx1, idx2, d12, d22] = nearest2_hamming_exhaustive_superfast(A, B, nBits);
                dBest = (d12 / nBits) * 100; % percent mismatch
                dSecond = (d22 / nBits) * 100;
            else
                [idx1, idx2, d12, d22] = nearest2_ssd_exhaustive(A, B);
                dBest = d12; % SSD
                dSecond = d22; % SSD
            end

        case "approximate"

            if isBinary
                % LSH-like approximate search on binary descriptors
                [idx1, idx2, d12, d22] = nearest2_hamming_lsh_superfast( ...
                    A, B, nBits, opt.ApproxNumTables, opt.ApproxBitsPerKey, opt.ApproxProbes);
                dBest = (d12 / nBits) * 100; % percent mismatch
                dSecond = (d22 / nBits) * 100;
            else
                % ---------- NON-BINARY (float) APPROXIMATE ----------
                switch approximateMethod
                    case 'pca_2nn'
                        opts_local = struct('ApproxNumComponents', 48, ... % try 48 or 64
                            'UsePCA', true, ...
                            'BlockRows', 4000, ...
                            'UseParfor', true, ...
                            'UseGPU', true); % set true if you have CUDA

                        [idx1, idx2, d12_eu2, d22_eu2] = nearest2_approx_float_fast(A, B, opts_local);
                        dBest = d12_eu2; % already SSD
                        dSecond = d22_eu2;
                    case 'kdtree'
                        [idx1, idx2, d12_eu, d22_eu] = nearest2_kdtree(A, B, opt.ApproxKDBucketSize);
                        % Convert to SSD to match MATLAB metric
                        dBest = d12_eu .^ 2;
                        dSecond = d22_eu .^ 2;
                    case 'subset_pdist2'
                        % High-D: directly use subset-pdist2 (very robust)
                        [idx1, idx2, d12_eu, d22_eu] = nearest2_subset_pdist2(A, B, 12000);
                        dBest = d12_eu .^ 2;
                        dSecond = d22_eu .^ 2;
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
    n = sqrt(sum(X .^ 2, 2)) + eps('single');
    Xn = X ./ n;
end

% --------------- Helper: figure out if inputs are binary ----------------
function [isBinary, A, B, nBits, unpacked] = normalizeInputs(F1, F2)
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
function [idx1, idx2, d1, d2] = nearest2_hamming_exhaustive_superfast(Abytes, Bbytes, nBits)
    % Abytes, Bbytes: uint8 [N x nbytes]
    [idx2, d1, d2] = nearest2_hamming_exhaustive_mex(Abytes, Bbytes);
    idx1 = (1:size(Abytes, 1)).';
    % guard second
    d2(~isfinite(d2) | d2 == 0) = single(ceil(nBits));
end

% --------------------- Exhaustive (float, SSD) --------------------------
function [idx1, idx2, d1, d2] = nearest2_ssd_exhaustive(A, B)
    % Compute distances in chunks to control memory
    N1 = size(A, 1); N2 = size(B, 1);
    D = size(A, 2);
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

function [idx1, idx2, d1, d2] = nearest2_subset_pdist2(A, B, subset)
    % Approximate 2-NN by sampling a subset of B then using pdist2 (Euclidean).
    % Returns Euclidean distances; caller squares to SSD.
    subset = min(subset, size(B, 1));
    candB = randperm(size(B, 1), subset);
    B2 = B(candB, :);

    % single precision is faster
    if ~isa(A, 'single'), A = single(A); end
    if ~isa(B2, 'single'), B2 = single(B2); end

    % D: 2 x size(A,1), I: 2 x size(A,1) (indices into rows of B2)
    [D, I] = pdist2(B2, A, 'euclidean', 'Smallest', 2);

    idx1 = (1:size(A, 1)).';
    idx2 = uint32(candB(I(1, :)).'); % map back into original B
    d1 = D(1, :).';
    d2 = D(2, :).';
end

function [idx1, idx2, d1, d2] = nearest2_kdtree(A, B, bucketSize)
    % KD-tree 2-NN using createns/knnsearch with BucketSize (R2025b).
    % Returns Euclidean d1,d2; caller squares for SSD.

    idx1 = (1:size(A, 1)).';

    % Primary: KD-tree with BucketSize
    Mdl = createns(B, 'NSMethod', 'kdtree', 'Distance', 'euclidean', 'BucketSize', bucketSize);
    [nbrIdx, nbrDist] = knnsearch(Mdl, A, 'K', 2); % Euclidean distances

    % Safety guard (shouldn't trigger with K=2)
    if size(nbrIdx, 2) < 2
        % Fallback to exhaustive 2-NN via matmul (rare)
        [nbrIdx, nbrDist] = twoNN_via_matmul(A, B);
    end

    idx2 = uint32(nbrIdx(:, 1));
    d1 = nbrDist(:, 1);
    d2 = nbrDist(:, 2);

end

function [idx1, idx2, dBest, dSecond] = nearest2_approx_float_fast(A, B, opts)
    % Fast approximate 2-NN for float descriptors (e.g., 128-D SIFT)
    % Strategy: PCA to dproj (optional), L2 normalize, block GEMM, take top-2
    %
    % opts fields (with sensible defaults):
    %   .ApproxNumComponents  : 48     % PCA target dims (32/48/64 typical)
    %   .UsePCA               : true
    %   .BlockRows            : 4000   % rows of A per block (tune by RAM)
    %   .UseParfor            : true
    %   .UseGPU               : false  % optional, if gpu available

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
        idx2_parts = cell(nBlocks, 1);
        dBest_parts = cell(nBlocks, 1);
        dSec_parts = cell(nBlocks, 1);

        parfor bi = 1:nBlocks
            s = (bi - 1) * blk + 1; e = min(bi * blk, N1);
            [idx2_parts{bi}, dBest_parts{bi}, dSec_parts{bi}] = do_block(A(s:e, :), B, useGPU);
        end

        idx2 = vertcat(idx2_parts{:});
        dBest = vertcat(dBest_parts{:});
        dSecond = vertcat(dSec_parts{:});

    else

        for s = 1:blk:N1
            e = min(s + blk - 1, N1);
            [idx2(s:e), dBest(s:e), dSecond(s:e)] = do_block(A(s:e, :), B, useGPU);
        end

    end

end

% convert cosine similarity to SSD on unit vectors:
%   ||a-b||^2 = 2 - 2*(a·b)
function [idBlock, d1Block, d2Block] = do_block(Ablk, Bfull, useGPUflag)

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

    try
        tf = parallel.gpu.GPUDevice.isAvailable;
    catch
        tf = false;
    end

end

% --------------- Approx (binary): LSH-like hashing ----------------------
function [idx1, idx2, d1, d2] = nearest2_hamming_lsh_superfast(Abytes, Bbytes, nBits, ~, ~, ~)
    [idx2, d1, d2] = nearest2_hamming_exhaustive_omp_mex(Abytes, Bbytes);
    idx1 = (1:size(Abytes, 1)).';
    d2(~isfinite(d2) | d2 == 0) = single(ceil(nBits));
end

% --------------------- Utilities ----------------------------------------
function [packed, nBits] = packBits(unpacked01)
    % Input logical/uint8 0/1, [N x Dbits] -> packed uint8 [N x ceil(Dbits/8)]
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
