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
varargin = normalizeApproxArgs(varargin);   % back-compat mapper

p = inputParser;
addParameter(p,'Method','Exhaustive',@(s)ischar(s)||isstring(s));
addParameter(p,'MatchThreshold',3.5,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
addParameter(p,'MaxRatio',0.6,@(x)isnumeric(x)&&isscalar(x)&&x>0&&x<=1);
addParameter(p,'Unique',true,@islogical);

% Binary-approx (LSH)
addParameter(p,'ApproxNumTables',6,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'ApproxBitsPerKey',24,@(x)isnumeric(x)&&isscalar(x)&&x>=8);
addParameter(p,'ApproxProbes',4,@(x)isnumeric(x)&&isscalar(x)&&x>=1);

% Float-approx (KD-tree): use BucketSize (R2025b)
addParameter(p,'ApproxKDBucketSize',40,@(x)isnumeric(x)&&isscalar(x)&&x>=1);

parse(p,varargin{:});
opt = p.Results;
method = lower(string(opt.Method));

% --------------------- Normalize/unwrap input ---------------------------
[isBinary, A, B, nBits, unpackedBinary] = normalizeInputs(F1, F2);

% --- Early exit: no descriptors (common in binary ORB/BRIEF images) ---
if isBinary && (isempty(A) || isempty(B))
    matches = zeros(0,2,'uint32');
    matchMetric = zeros(0,1,'single');
    return;
end

% ----- Fix default thresholds based on descriptor type -----
if isBinary
    if ~isfield(opt,'MatchThreshold') || isempty(opt.MatchThreshold)
        opt.MatchThreshold = 10;  % percent mismatched bits (common default window: 5–25)
    end
else
    % For floats, rely on ratio; do not hard-threshold SSD by default.
    if ~isfield(opt,'MatchThreshold') || isempty(opt.MatchThreshold)
        opt.MatchThreshold = inf;  % <-- key change
    end

    % If descriptors look unnormalized (large magnitudes), L2-normalize
    % (matchFeatures works fine on normalized vectors; this makes SSDs comparable)
    if max(abs(A(:))) > 2 || max(abs(B(:))) > 2
        A = normalizeRowsL2(A);
        B = normalizeRowsL2(B);
    end
end


% --------------------- Core matching -----------------------------------
switch method
    case "exhaustive"
        if isBinary
            [idx1, idx2, d12, d22] = nearest2_hamming_exhaustive_superfast(A, B, nBits);
            dBest = (d12 / nBits) * 100;   % percent mismatch
            dSecond = (d22 / nBits) * 100;
        else
            [idx1, idx2, d12, d22] = nearest2_ssd_exhaustive(A, B);
            dBest = d12;      % SSD
            dSecond = d22;    % SSD
        end

    case "approximate"
        if isBinary
            % LSH-like approximate search on binary descriptors
            [idx1, idx2, d12, d22] = nearest2_hamming_lsh_superfast( ...
                A, B, nBits, opt.ApproxNumTables, opt.ApproxBitsPerKey, opt.ApproxProbes);
            dBest = (d12 / nBits) * 100;   % percent mismatch
            dSecond = (d22 / nBits) * 100;
        else
            % ---------- NON-BINARY (float) APPROXIMATE ----------
            % Try KD-tree first only for moderate D; else go pdist2 subset.
            D = size(A,2);
            useKD = (D <= 64);  % KD-tree is weak in high-D; adjust if you like

            if useKD
                try
                    [idx1, idx2, d12_eu, d22_eu] = nearest2_kdtree(A, B, opt.ApproxKDBucketSize);
                    % Convert to SSD to match MATLAB metric
                    dBest   = d12_eu.^2;
                    dSecond = d22_eu.^2;
                catch
                    % KD-tree not available or failed; fall back to subset pdist2
                    [idx1, idx2, d12_eu, d22_eu] = nearest2_subset_pdist2(A, B, 12000);
                    dBest   = d12_eu.^2;
                    dSecond = d22_eu.^2;
                end
            else
                % High-D: directly use subset-pdist2 (very robust)
                [idx1, idx2, d12_eu, d22_eu] = nearest2_subset_pdist2(A, B, 12000);
                dBest   = d12_eu.^2;
                dSecond = d22_eu.^2;
            end
        end

    otherwise
        error('Unknown Method: %s', opt.Method);
end

% Convert to Euclidean for ratio test consistently
if isBinary
    dBestEu   = dBest;    % already "percent mismatch" for binary
    dSecondEu = dSecond;
else
    dBestEu   = sqrt(max(dBest,   0));   % SSD -> L2
    dSecondEu = sqrt(max(dSecond, 0));
end

% --------------------- Ratio & Threshold filters ------------------------
maxRatio = opt.MaxRatio;
matchThr = opt.MatchThreshold;

% First pass
ratioOK  = dBestEu ./ dSecondEu <= maxRatio;
threshOK = dBest    <= matchThr;
keep     = ratioOK & threshOK & isfinite(dBest) & isfinite(dSecond);

i1 = (1:size(A,1)).';
i1 = i1(keep);          % idx1 is simply 1..N1
i2 = idx2(keep);
d  = dBest(keep);

% --------------------- Enforce uniqueness if requested ------------------
if opt.Unique && ~isempty(i1)
    % Greedy: sort by distance and keep first time a row/col is seen
    [dSorted, order] = sort(d, 'ascend');
    i1s = i1(order); i2s = i2(order);

    used1 = false(max(i1s),1);
    used2 = false(max(i2s),1);
    sel   = false(size(i1s));
    for k = 1:numel(i1s)
        a = i1s(k); b = i2s(k);
        if ~used1(a) && ~used2(b)
            sel(k) = true;
            used1(a) = true;
            used2(b) = true;
        end
    end
    matches = [i1s(sel), i2s(sel)];
    matchMetric = dSorted(sel);
else
    matches = [i1(:), i2(:)];
    matchMetric = d(:);
end

% For convenience, ensure column vector
matchMetric = matchMetric(:);
end

function Xn = normalizeRowsL2(X)
    n = sqrt(sum(X.^2,2)) + eps('single');
    Xn = X ./ n;
end


% --------------- Helper: figure out if inputs are binary ----------------
function [isBinary, A, B, nBits, unpacked] = normalizeInputs(F1,F2)
    unpacked = false;
    if isa(F1,'binaryFeatures') && isa(F2,'binaryFeatures')
        % Packed bytes (each row packed bits). Keep as uint8 bytes.
        A = F1.Features; % [N1 x nbytes]
        B = F2.Features; % [N2 x nbytes]
        isBinary = true;
        nBits = size(A,2)*8;
    else
        % If both are logical/uint8 with 0/1, treat as binary UNPACKED bits
        f1isbin = (islogical(F1) || (isa(F1,'uint8') && all((F1(:)==0)|(F1(:)==1))));
        f2isbin = (islogical(F2) || (isa(F2,'uint8') && all((F2(:)==0)|(F2(:)==1))));
        if f1isbin && f2isbin
            % Pack into uint8 bytes for speed
            [A, nBits] = packBits(F1);
            [B, ~    ] = packBits(F2);
            isBinary = true;
            unpacked = true;
        else
            % non-binary (float vectors)
            if ~isa(F1,'single') && ~isa(F1,'double'); F1 = single(F1); end
            if ~isa(F2,'single') && ~isa(F2,'double'); F2 = single(F2); end
            validateattributes(F1,{'single','double'},{'2d','nonempty'});
            validateattributes(F2,{'single','double'},{'2d','nonempty'});
            if size(F1,2) ~= size(F2,2)
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
    idx1 = (1:size(Abytes,1)).';
    % guard second
    d2(~isfinite(d2) | d2==0) = single(ceil(nBits));
end

function [idx1, idx2, d1, d2] = nearest2_hamming_exhaustive_slow(Abytes, Bbytes, nBits)
% Abytes: [N1 x nbytes] uint8, Bbytes: [N2 x nbytes] uint8
N1 = size(Abytes,1); N2 = size(Bbytes,1); nbytes = size(Abytes,2);
LUT = popcountLUT();                          % uint8(256x1)
idx2 = zeros(N1,1,'uint32');
d1   = inf(N1,1,'single');
d2   = inf(N1,1,'single');

% choose block so blk*N2 fits cache/memory; tune 2e6..6e6 cells is good
target_cells = 3e6;
blk = max(1, floor(target_cells / max(N2,1)));

for s = 1:blk:N1
    e = min(N1, s+blk-1);
    Ablk = Abytes(s:e,:);                    % [blk x nbytes]

    % Accumulate Hamming distances across bytes
    % H will be [blk x N2] uint16 (enough for up to 2048 bits)
    H = zeros(e-s+1, N2, 'uint16');
    for j = 1:nbytes
        % xor each byte column: broadcasting makes [blk x N2] only
        X = bitxor( Ablk(:,j), Bbytes(:,j).' );        % uint8 [blk x N2]
        H = H + uint16(LUT( double(X) + 1 ));          % add popcounts
    end

    % top-1, top-2 per row without forming 3-D arrays
    [best, i2] = min(H, [], 2);
    lin = sub2ind(size(H), (1:size(H,1))', i2);
    H(lin) = uint16(65535);
    second = min(H, [], 2);

    idx2(s:e) = uint32(i2);
    d1(s:e)   = single(best);
    d2(s:e)   = single(second);
end

idx1 = (1:N1).';
d2(~isfinite(d2)|d2==0) = single(ceil(nBits));
end

% --------------------- Exhaustive (float, SSD) --------------------------
function [idx1, idx2, d1, d2] = nearest2_ssd_exhaustive(A, B)
    % Compute distances in chunks to control memory
    N1 = size(A,1); N2 = size(B,1);
    D  = size(A,2);
    block = max(1, floor(1e7 / max(N2,1))); % heuristic
    idx2 = zeros(N1,1,'uint32'); d1 = inf(N1,1); d2 = inf(N1,1);

    for i = 1:ceil(N1/block)
        s = (i-1)*block + 1;
        e = min(N1, i*block);
        Ablk = A(s:e, :);
        % Efficient SSD: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a*b'
        a2 = sum(Ablk.^2, 2);                 % [blk x 1]
        b2 = sum(B.^2, 2);                     % [N2 x 1]
        G  = Ablk * B.';                       % [blk x N2]
        D2 = a2 + b2.' - 2*G;                  % SSD
        % find top-2 per row
        [best, idx] = min(D2, [], 2);
        D2(sub2ind(size(D2),(1:size(D2,1))', idx)) = inf;  % mask best
        second = min(D2, [], 2);

        idx2(s:e) = idx;
        d1(s:e)   = best;
        d2(s:e)   = second;
    end
    idx1 = (1:N1).';
end

function [idx1, idx2, d1, d2] = nearest2_subset_pdist2(A,B,subset)
% Approximate 2-NN by sampling a subset of B then using pdist2 (Euclidean).
% Returns Euclidean distances; caller squares to SSD.
subset = min(subset, size(B,1));
candB  = randperm(size(B,1), subset);
B2     = B(candB,:);

% single precision is faster
if ~isa(A,'single'),  A  = single(A);  end
if ~isa(B2,'single'), B2 = single(B2); end

% D: 2 x size(A,1), I: 2 x size(A,1) (indices into rows of B2)
[D, I] = pdist2(B2, A, 'euclidean', 'Smallest', 2);

idx1 = (1:size(A,1)).';
idx2 = uint32(candB(I(1,:)).');  % map back into original B
d1   = D(1,:).';
d2   = D(2,:).';
end


% --------------------- Approx (float): KD-tree --------------------------
function [idx1, idx2, d1, d2] = nearest2_kdtree(A, B, bucketSize)
% KD-tree 2-NN using createns/knnsearch with BucketSize (R2025b).
% Returns Euclidean d1,d2; caller squares for SSD.

idx1 = (1:size(A,1)).';

% Primary: KD-tree with BucketSize
Mdl = createns(B, 'NSMethod','kdtree', 'Distance','euclidean', 'BucketSize', bucketSize);
[nbrIdx, nbrDist] = knnsearch(Mdl, A, 'K', 2);  % Euclidean distances

% Safety guard (shouldn't trigger with K=2)
if size(nbrIdx,2) < 2
    % Fallback to exhaustive 2-NN via matmul (rare)
    [nbrIdx, nbrDist] = twoNN_via_matmul(A, B);
end

idx2 = uint32(nbrIdx(:,1));
d1   = nbrDist(:,1);
d2   = nbrDist(:,2);

end

% --------------- Approx (binary): LSH-like hashing ----------------------
function [idx1, idx2, d1, d2] = nearest2_hamming_lsh_superfast(Abytes, Bbytes, nBits, ~, ~, ~)
    [idx2, d1, d2] = nearest2_hamming_exhaustive_omp_mex(Abytes, Bbytes);
    idx1 = (1:size(Abytes,1)).';
    d2(~isfinite(d2) | d2==0) = single(ceil(nBits));
end


function [idx1, idx2, d1, d2] = nearest2_hamming_lsh_fast( ...
        Abytes, Bbytes, nBits, nTables, bitsPerKey, nProbes)

% --- set more selective defaults for binary LSH ---
if nargin < 4 || isempty(nTables),   nTables   = 8;  end
if nargin < 5 || isempty(bitsPerKey),bitsPerKey= 32; end   % for 256-bit ORB
if nargin < 6 || isempty(nProbes),   nProbes   = 1;  end   % keep 1 for speed

N1 = size(Abytes,1); N2 = size(Bbytes,1);
idx1 = (1:N1).';
idx2 = zeros(N1,1,'uint32');
d1   = inf(N1,1,'single');
d2   = d1;

% ---- build tables once (sorted-key buckets) ----
bitsPerKey = min(bitsPerKey, 8*size(Abytes,2));
rng(12345,'twister');
bitIdxTables = arrayfun(@(~) randperm(8*size(Abytes,2), bitsPerKey), 1:nTables, 'uni', 0);
tables = build_lsh_tables(Bbytes, bitIdxTables);     % your helper from earlier

MAXC = 512;   % cap candidates to keep MEX refine cheap (try 256 if N is big)

for i = 1:N1
    % 1) gather candidates from LSH tables
    cand = query_lsh_tables(Abytes(i,:), tables, bitIdxTables, nProbes, N2);
    if isempty(cand)
        % fallback: small random subset rather than full B
        if N2 <= 1024
            cand = uint32(1:N2);
        else
            cand = uint32(randperm(N2, min(1024,MAXC)));
        end
    else
        cand = unique(cand,'stable');
        % 2) hard-cap candidates (keeps runtime predictable)
        if numel(cand) > MAXC
            head = cand(1:min(256,numel(cand)));
            rest = cand(numel(head)+1:end);
            if ~isempty(rest)
                rest = rest(randperm(numel(rest), min(MAXC-numel(head), numel(rest))));
                cand = [head; rest];
            else
                cand = head;
            end
        end
    end

    % 3) refine with early-exit MEX on candidates (very fast)
    [ci, cd1, cd2] = nearest2_hamming_candidates_mex(Abytes(i,:), Bbytes(double(cand),:));
    idx2(i) = cand(double(ci));
    d1(i)   = cd1;
    d2(i)   = cd2;
    if ~isfinite(d2(i)) || d2(i)==0, d2(i)=single(ceil(nBits)); end
end
end



function [idx1, idx2, d1, d2] = nearest2_hamming_lsh_slow( ...
        Abytes, Bbytes, nBits, nTables, bitsPerKey, nProbes)
% Approximate 2-NN Hamming matcher (LSH-like) — vectorized & cache-friendly
% Replaces nearest2_hamming_lsh()

N1 = size(Abytes,1);
N2 = size(Bbytes,1);

if N2==0
    idx1 = (1:N1).'; idx2 = zeros(N1,1,'uint32');
    d1 = inf(N1,1); d2 = inf(N1,1);
    return
end

bitsPerKey = min(bitsPerKey, nBits);
rng(12345,'twister');

% ------------------------------------------------------------
% Build the hash tables for the reference descriptors (once)
% ------------------------------------------------------------
bitIdxTables = cell(nTables,1);
for t = 1:nTables
    bitIdxTables{t} = randperm(nBits, bitsPerKey);
end
tables = build_lsh_tables(Bbytes, bitIdxTables);

% Popcount lookup once
LUT = popcountLUT();

% Output alloc
idx1 = (1:N1).';
idx2 = zeros(N1,1,'uint32');
d1   = inf(N1,1,'single');
d2   = inf(N1,1,'single');

% ------------------------------------------------------------
% Query each descriptor in A
% ------------------------------------------------------------
for i = 1:N1
    Arow = Abytes(i,:);

    % ---- get candidate set via pre-built tables ----
    cand = query_lsh_tables(Arow, tables, bitIdxTables, nProbes, N2);
    candIdx = double(cand);
    C = numel(candIdx);

    % ---- compute true Hamming on just these candidates ----
    % block-wise 2-D XOR accumulation (small C, so fine)
    H = zeros(C,1,'uint16');
    for j = 1:size(Abytes,2)
        X = bitxor(Arow(1,j), Bbytes(candIdx,j));   % [C×1]
        H = H + uint16(LUT(double(X)+1));
    end

    % ---- pick best & second best ----
    [best, ix] = min(H);
    if C >= 2
        H(ix) = uint16(65535);
        second = min(H);
    else
        second = uint16(ceil(nBits));
    end

    idx2(i) = candIdx(ix);
    d1(i)   = single(best);
    d2(i)   = single(second);

    if ~isfinite(d2(i)) || d2(i)==0
        d2(i) = single(ceil(nBits));
    end
end
end

function tables = build_lsh_tables(Bbytes, bitIdxTables)
nTables = numel(bitIdxTables);
tables = cell(nTables,1);
for t = 1:nTables
    keyIdx = bitIdxTables{t};
    keys   = computeHashKeys(Bbytes, keyIdx);  % [N2×1] uint64
    [keys_sorted, order] = sort(keys);
    idx_sorted = uint32(order);

    % run-length encode identical keys (for fast lookup)
    d = [true; keys_sorted(2:end)~=keys_sorted(1:end-1)];
    starts = uint32(find(d));
    lasts  = uint32([starts(2:end)-1; numel(keys_sorted)]);
    tables{t} = struct('keys',keys_sorted,'idx',idx_sorted,'runs',[starts lasts]);
end
end

function cand = query_lsh_tables(Arow, tables, bitIdxTables, nProbes, N2)
cand = uint32([]);
for t = 1:numel(bitIdxTables)
    keyIdx = bitIdxTables{t};
    k0 = computeHashKeys(Arow, keyIdx);          % 1×1 uint64
    ks = repmat(k0, max(1,nProbes), 1);
    for pb = 2:nProbes
        ks(pb) = bitxor(k0, uint64(pb-1));       % trivial multi-probe
    end

    T = tables{t};
    for kk = 1:numel(ks)
        k = ks(kk);
        % binary search for matching key range
        lo = find_first_ge(T.keys, k);
        if lo <= numel(T.keys) && T.keys(lo) == k
            r = find(T.runs(:,1)<=lo & lo<=T.runs(:,2), 1, 'first');
            cand = [cand; T.idx(T.runs(r,1):T.runs(r,2))]; %#ok<AGROW>
        end
    end
end

if isempty(cand)
    if N2 <= 1024
        cand = uint32(1:N2);
    else
        cand = uint32(randperm(N2,1024));
    end
else
    cand = unique(cand,'stable');
end
end

function pos = find_first_ge(vec, val)
% binary search for first element >= val
lo = 1; hi = numel(vec);
while lo < hi
    mid = floor((lo+hi)/2);
    if vec(mid) < val
        lo = mid + 1;
    else
        hi = mid;
    end
end
pos = lo;
end


% --------------------- Utilities ----------------------------------------
function [packed, nBits] = packBits(unpacked01)
    % Input logical/uint8 0/1, [N x Dbits] -> packed uint8 [N x ceil(Dbits/8)]
    validateattributes(unpacked01,{'logical','uint8'},{'2d','nonempty'});
    N = size(unpacked01,1);
    D = size(unpacked01,2);
    nbytes = ceil(D/8);
    packed = zeros(N, nbytes, 'uint8');
    for b = 1:D
        byteIdx = ceil(b/8);
        bitPos  = 8 - mod(b-1,8); % MSB-first
        packed(:, byteIdx) = bitor( packed(:,byteIdx), uint8(unpacked01(:,b)) .* uint8(2^(bitPos-1)) );
    end
    nBits = D;
end

function LUT = popcountLUT()
    % 256-entry lookup table for popcount of a byte
    persistent tbl;
    if isempty(tbl)
        tbl = zeros(256,1,'uint8');
        for v = 0:255
            tbl(v+1) = sum(bitget(uint8(v),1:8));
        end
    end
    LUT = tbl;
end

function keys = computeHashKeys(bytesRowOrMat, bitIdx)
    % Create 64-bit hash from selected bit positions (MSB-first across row).
    if isvector(bytesRowOrMat)
        X = bytesRowOrMat(:).';
    else
        X = bytesRowOrMat;
    end
    N = size(X,1);
    keys = zeros(N,1,'uint64');
    for k = 1:numel(bitIdx)
        bpos = bitIdx(k);
        byteIdx  = ceil(bpos/8);
        bitInByte = 8 - mod(bpos-1,8);
        bit = bitget(X(:,byteIdx), bitInByte);
        keys = bitor( bitshift(keys,1), uint64(bit) );
    end
end

% ---- Back-compat normalizer (put near other helpers) ----
function nv = normalizeApproxArgs(nv)
if isempty(nv), return; end
for k = 1:2:numel(nv)
    if ~ischar(nv{k}) && ~isstring(nv{k}), continue; end
    key = string(nv{k});
    switch key
        case "Approx.NumTables",        nv{k} = 'ApproxNumTables';
        case "Approx.BitsPerKey",       nv{k} = 'ApproxBitsPerKey';
        case "Approx.Probes",           nv{k} = 'ApproxProbes';
        case "Approx.KDTreeLeafSize",   nv{k} = 'ApproxKDBucketSize'; % legacy -> bucket
        case "ApproxKDTreeLeafSize",    nv{k} = 'ApproxKDBucketSize'; % legacy -> bucket
        otherwise
    end
end
end