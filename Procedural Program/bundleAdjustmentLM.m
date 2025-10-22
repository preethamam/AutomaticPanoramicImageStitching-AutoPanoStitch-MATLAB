function [cameras, concomps, imageNeighbors, center] = bundleAdjustmentLM(input, matches, keypoints, imageSizes, initialTforms, numMatches, varargin)
% INCREMENTAL_PANORAMA_BA
% Incremental bundle adjustment for panoramic stitching with rotations R_i and focal lengths f_i.
% Uses your initializeCameraMatrices() to seed cameras, then:
%  - Adds images one-by-one (best-connected next),
%  - Initializes the new image with the R and f of its best neighbor,
%  - Runs Levenberg–Marquardt with the original Huber robust error (no IRLS loop).
%  - Hard gauge fix: seed image rotation is fixed as identity for all LM steps.
%
% INPUTS
%   matches      : N x N cell; matches{i,j} is M_ij x 2 [idx_in_keypoints{i}, idx_in_keypoints{j}]
%   keypoints    : 1 x N cell; keypoints{i} = K_i x 2 [x,y] pixel coordinates (top-left origin)
%   imageSizes   : N x 2 [H, W]
%   initialTforms: array of initial pairwise transforms (your type), used only by your initializer
%   input        : struct passed to initializeCameraMatrices (contains .transformationType)
%
% Name-Value options
%   'SigmaHuber'      : Huber threshold (pixels). Default 2.0
%   'MaxLMIters'      : LM iterations per growth step. Default 15
%   'MaxGrow'         : Max number of growth steps. Default N
%   'Lambda0'         : Initial LM damping. Default 1e-3
%   'PriorSigmaF'     : Focal prior stdev. Default mean(active f)/10
%   'Verbose'         : 0/1. Default 1
%
% OUTPUT
%   cameras: struct with fields
%       R           : 1 x N cell, 3x3 rotations
%       f           : 1 x N double
%       K           : 1 x N cell, 3x3 intrinsics [f 0 cx; 0 f cy; 0 0 1]
%       initialized : 1 x N logical (stored as cell logicals to match your initializer)

% ------------------ Options ------------------
p = inputParser;
N = numel(keypoints);
p.addParameter('SigmaHuber', 3.0);
p.addParameter('MaxLMIters', 30);
p.addParameter('MaxGrow', N);
p.addParameter('Lambda0', 1e-3);
p.addParameter('PriorSigmaF', []);
p.addParameter('Verbose', 1);
p.parse(varargin{:});
opt = p.Results;
% userSeed = [];        % <-- set to a specific image index to fix manually, e.g., userSeed = 1;

%%***********************************************************************%
%*                   Automatic panorama stitching                       *%
%*                        Bundle adjustment                             *%
%*                                                                      *%
%* Code author: Preetham Manjunatha                                     *%
%* Github link: https://github.com/preethamam                           *%
%* Date: 01/27/2022                                                     *%
%************************************************************************%

% Find connected components of image matches
numMatchesG = graph(numMatches,'upper');
[concomps, ccBinSizes] = conncomp(numMatchesG);
panaromaCCs = find(ccBinSizes>=1);
ccnum = numel(panaromaCCs);
tree = getMST(numMatches);
indices = find(concomps == ccnum);
% ordering = getOrdering(indices, tree);

% Find images neighbors    
nearestFeaturesNum = input.nearestFeaturesNum;
unqCCs = unique(concomps,'stable');
imageNeighbors = cell(numel(unqCCs),1);
for j = 1:length(unqCCs)
    ccIdx = find(concomps == unqCCs(j));
    numMatchesCCs = numMatches(ccIdx, ccIdx);
    numMatchesGCCs = graph(numMatchesCCs,'upper');
    parfor i = 1:size(numMatchesCCs,1)
        nn = neighbors(numMatchesGCCs,i)';
        nn_dist = distances(numMatchesGCCs,i, nn);
        imageNeighborsTemp{i} = nn(nn_dist>nearestFeaturesNum);
    end
    imageNeighbors{j} = imageNeighborsTemp;
end

userSeed = [];  % or 3, or whichever image you want to define the world frame

% Debug input matches
[n1, n2] = size(matches);

% Verify matches cell array is square
assert(n1 == n2, 'Matches cell array must be square (NxN)');
num_images = n1;

% Verify keypoints array size
assert(size(keypoints, 2) >= num_images, 'Keypoints array must have entry for each image');

% ------------------ Build pair list (compact) ------------------
pairs = build_pairs_from_cells(matches, keypoints); % each pair has i,j, Ui (px), Uj (px); i<j

% ------------------ Choose seed (image with max degree / matches) ------------------
% Optionally override the automatic seed selection
if ~isempty(userSeed)
    seed = userSeed;
else
    % Automatic seed: image with maximum degree
    deg = zeros(1,N);
    for t=1:numel(pairs)
        M = size(pairs(t).Ui,1);
        deg(pairs(t).i) = deg(pairs(t).i) + M;
        deg(pairs(t).j) = deg(pairs(t).j) + M;
    end
    [~, seed] = max(deg);
end

center = seed;
initialTforms = getTforms(tree, center, initialTforms);

% ------------------ Initialize cameras via your function ------------------
cameras = initializeCameraMatrices(input, imageSizes, initialTforms, num_images); % (R, f, K, initialized=false)

% Normalize to struct array (1×N) with scalar fields
cameras = normalize_cameras_struct(cameras, num_images);

cameras(seed).initialized = true;

if opt.Verbose
    fprintf('Seed image: %d | initial f≈%.2f px\n', seed, cameras(seed).f);
end

% Internal rotation parameterization (axis-angle per image)
theta = zeros(N,3);

% Keep R matrices aligned with theta; seed is fixed to identity
for i=1:N
    if isempty(cameras(i).R), cameras(i).R = eye(3); end
end
cameras(seed).R = eye(3); theta(seed,:) = [0 0 0];

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
    cx = imageSizes(nextImg,2)/2; cy = imageSizes(nextImg,1)/2;
    cameras(nextImg).K = [cameras(nextImg).f, 0, cx; 0, cameras(nextImg).f, cy; 0, 0, 1];
    cameras(nextImg).initialized = true;
    theta(nextImg,:) = theta(bestNbr,:);

    if opt.Verbose
        fprintf('>> Added image %d (best neighbor %d). Active = %d\n', ...
            nextImg, bestNbr, nnz([cameras.initialized]));
    end

    % LM over all active (seed rotation hard-fixed)
    active = find([cameras.initialized]);
    [theta, cameras] = run_lm(active, theta, cameras, pairs, imageSizes, seed, opt);
    % cameras = canonicalize_global_orientation(cameras);

    growSteps = growSteps + 1;
end
end

% ======================================================================
% ======================  BUILD PAIR LIST FROM CELLS  ==================
% ======================================================================
function pairs = build_pairs_from_cells(matches, keypoints)
N = numel(keypoints);
plist = [];
for i=1:N
    for j=i+1:N
        Mij = matches{i,j}';
        if isempty(Mij), continue; end
        Ui = keypoints{i}(:,Mij(:,1))'; % M×2 pixels
        Uj = keypoints{j}(:,Mij(:,2))';
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
scores = zeros(1,N);
nbr    = zeros(1,N);
act    = find(initialized);
for t=1:numel(pairs)
    i = pairs(t).i; j = pairs(t).j; M = size(pairs(t).Ui,1);
    if initialized(i) && ~initialized(j)
        scores(j) = scores(j) + M; nbr(j) = i;
    elseif initialized(j) && ~initialized(i)
        scores(i) = scores(i) + M; nbr(i) = j;
    end
end
scores(initialized) = -inf;
[~, nxt] = max(scores);
if isinf(scores(nxt)), nxt = []; bestNbr = []; return; end
bestNbr = nbr(nxt); if bestNbr==0, bestNbr = act(1); end
end

% ======================================================================
% =============================  LM CORE  ==============================
% ======================================================================
function [thetaA, cameras] = run_lm(active, theta0, cameras, pairs, imageSizes, seed, opt)
thetaA = theta0;

lambda  = opt.Lambda0;
sigHub  = opt.SigmaHuber;

% adaptive prior for f (no rotation prior; we hard-fix seed instead)
actF = [cameras(active).f];
if isempty(opt.PriorSigmaF)
    sigF = max(1e-6, mean(actF)/10);
else
    sigF = opt.PriorSigmaF;
end

for it = 1:opt.MaxLMIters

    % if it <= 3, sigHub = Inf; else, sigHub = opt.SigmaHuber; end

    [H, g, Eold, nres] = accumulate_normal_equations(thetaA, cameras, pairs, active, imageSizes, sigHub);

    % --- apples-to-apples metrics like OpenPano ---
    [R_abs_mean, R_abs_max] = avg_abs_residual(thetaA, cameras, pairs, active, imageSizes);
    fprintf('   avg|r|=%.3f px  max|r|=%.3f px\n', R_abs_mean, R_abs_max);
    
    % --- focal prior (optional but helpful) ---
    if isempty(opt.PriorSigmaF), sigF = max(1e-6, mean([cameras(active).f])/10);
    else,                        sigF = opt.PriorSigmaF;
    end
    wF = 1/(sigF^2);
    f0 = cameras(seed).f;   % or median([cameras(active).f])
    for k = 1:numel(active)
        row = (k-1)*4 + 4;  % focal slot in the 4x4 block
        H(row,row) = H(row,row) + wF;
        g(row)     = g(row)     + wF * (cameras(active(k)).f - f0);
    end


    % -------- Hard gauge fix: keep seed rotation as identity --------
    seedPos = find(active==seed, 1);
    if ~isempty(seedPos)
        seedRows = (seedPos-1)*4 + (1:3);  % rotation rows of seed block
        H(seedRows,:) = 0; H(:,seedRows) = 0;
        H(sub2ind(size(H), seedRows, seedRows)) = 1e12; % strong pin
        g(seedRows) = 0;
    end


    % after you build H, g
    sigma_theta = pi/16;  wR = 1/(sigma_theta^2);
    for k = 1:numel(active)
        rowR = (k-1)*4 + (1:3);
        H(sub2ind(size(H), rowR, rowR)) = H(sub2ind(size(H), rowR, rowR)) + wR;
        % no bias term for rotations (zero-mean)
    end


    if it==1 && opt.Verbose
        jacobian_check(thetaA, cameras, pairs, active, imageSizes, sigHub, seed);
    end

    % ----------------------------------------------------------------
    
    % -------- keep copies for delta print --------
    theta_before = thetaA; 
    cams_before  = cameras;

    % LM damping (simple isotropic)
    H = H + lambda * speye(size(H));
    
    % Solve *with negative RHS*
    b = -g;                               % <--- FIX
    try
        L = chol(H,'lower');
        d = L'\(L\b);                     % <--- b, not g
    catch
        d = H\b;                          % <--- b, not g
    end

    % Apply update (seed rotation ignored)
    [thetaTrial, camsTrial] = apply_update(thetaA, cameras, d, active, imageSizes, seed);

    % Evaluate new Huber energy
    Enew = total_energy(thetaTrial, camsTrial, pairs, active, imageSizes, sigHub);

    % Gain ratio
    % Predicted reduction for gain ratio (LM theory)
    pred = -0.5 * (d'*(lambda*d - g));    % <--- use minus; was d'*(lambda*d+g)
    denom = max(1e-12, pred);
    rho = (Eold - Enew) / denom;          % unchanged, but now denom is correct
    if rho > 0
        thetaA  = thetaTrial;
        cameras = camsTrial;
        lambda  = max(1e-9, lambda * max(1/3, 1 - (2*rho - 1)^3));
        if opt.Verbose && mod(it,3)==1
            % fprintf('   it=%2d  E=%.3f  nres=%d  lambda=%.2e  accepted\n', it, Enew, nres, lambda);
            % print_step_deltas(theta_before, thetaA, cams_before, cameras, active);
        end
        if norm(d) < 1e-8 || abs(Eold - Enew) < 1e-6, break; end
    else
        lambda = min(1e9, lambda * 10);
        if opt.Verbose && mod(it,3)==1
            % fprintf('   it=%2d  E=%.3f  nres=%d  lambda=%.2e  REJECT\n', it, Eold, nres, lambda);
        end
    end
end
end


function jacobian_check(theta, cameras, pairs, active, imageSizes, sigma, seed)
rng(0); eps = 1e-6; picks = 5; % a few random residuals
activeMask = false(1,numel(cameras)); activeMask(active)=true;

% Build analytic J*g on a tiny subset
[H, g, ~, ~] = accumulate_normal_equations(theta, cameras, pairs, active, imageSizes, sigma);

% Pick a random active image and parameter slot (θx,θy,θz,f)
ai = active(randi(numel(active)));
slot = randi(4);
idx  = (find(active==ai)-1)*4 + slot;

% Finite difference gradient of total energy wrt that param
E0 = total_energy(theta, cameras, pairs, active, imageSizes, sigma);
d  = zeros(size(g)); d(idx)=eps;

% +eps
[thetaP, camsP] = apply_update(theta, cameras, d, active, imageSizes, seed); % ai only used to keep seed fixed
E1 = total_energy(thetaP, camsP, pairs, active, imageSizes, sigma);

% -eps
d(idx) = -eps;
[thetaM, camsM] = apply_update(theta, cameras, d, active, imageSizes, seed);
E2 = total_energy(thetaM, camsM, pairs, active, imageSizes, sigma);

g_num = (E1 - E2)/(2*eps);
g_ana = g(idx);  % note LM uses b=-g later

fprintf('J-check img %d slot %d: analytic g=%.4e  numeric g=%.4e  diff=%.2e\n', ...
        ai, slot, g_ana, g_num, abs(g_ana-g_num));
end


function [m, mx] = avg_abs_residual(theta, cameras, pairs, active, imageSizes)
    activeMask = false(1,numel(cameras)); activeMask(active)=true;
    tot = 0; cnt = 0; mx = 0;
    for t=1:numel(pairs)
        i=pairs(t).i; j=pairs(t).j;
        if ~(activeMask(i) && activeMask(j)), continue; end
        Ui=pairs(t).Ui; Uj=pairs(t).Uj; M=size(Ui,1);

        Ri=expm_hat(theta(i,:)); Rj=expm_hat(theta(j,:));
        [Ki,KiInv]=make_K_and_inv(cameras(i).f,imageSizes(i,:));
        [Kj,KjInv]=make_K_and_inv(cameras(j).f,imageSizes(j,:));

        % One direction only (j -> i), to match OpenPano logs
        yj  = [Uj,ones(M,1)]*KjInv.'; sJ = yj*Rj;
        XYZ = (sJ*Ri.')*Ki.'; pix=[XYZ(:,1)./XYZ(:,3), XYZ(:,2)./XYZ(:,3)];
        r = Ui - pix;

        a = abs(r); tot = tot + sum(a(:)); cnt = cnt + numel(a); mx = max(mx,max(a(:)));
    end
    m = tot/max(1,cnt);
end


% ======================================================================
% ==================  Accumulate J^T(α)J and J^T(α)r  =================
% ======================================================================
function [H, g, E, nres] = accumulate_normal_equations(theta, cameras, pairs, active, imageSizes, sigmaHuber)
m = numel(active);
blkSize = 4; % [theta(3), f(1)]

% map image id -> block index [0..m-1]
id2blk = containers.Map('KeyType','int32','ValueType','int32');
for k=1:m, id2blk(int32(active(k))) = int32(k-1); end

% init block accumulators
Hblocks = cell(m,m); gblocks = cell(m,1);
for a=1:m
    gblocks{a} = zeros(blkSize,1);
    for b=1:m, Hblocks{a,b} = zeros(blkSize, blkSize); end
end

E = 0; nres = 0;

% rotation generators
E1 = [0 0 0; 0 0 -1; 0 1 0];
E2 = [0 0 1; 0 0 0; -1 0 0];
E3 = [0 -1 0; 1 0 0; 0 0 0];

for t=1:numel(pairs)
    i = pairs(t).i; j = pairs(t).j;
    if ~isKey(id2blk, int32(i)) || ~isKey(id2blk, int32(j)), continue; end
    bi = id2blk(int32(i))+1; bj = id2blk(int32(j))+1;

    Ui = pairs(t).Ui; Uj = pairs(t).Uj; M = size(Ui,1);
    if M==0, continue; end

    % Camera params
    Ri = expm_hat(theta(i,:));  Rj = expm_hat(theta(j,:));
    fi = cameras(i).f;          fj = cameras(j).f;
    [Ki, KiInv, dKi_df, dKiInv_df] = make_K_and_inv(fi, imageSizes(i,:));
    [Kj, KjInv, dKj_df, dKjInv_df] = make_K_and_inv(fj, imageSizes(j,:));

    % --- j -> i (row-form, right-multiply) ---
    % Homogenize and back-project in j:
    yj  = [Uj, ones(M,1)] * KjInv.';    % Mx3  (rows)
    % Rotate world rays into camera i frame:
    sJ  = yj * Rj;                      % Mx3  (since s = Rj^T * y  -> rows: y * Rj)
    % Project in camera i (pre-divide):
    XYZ = (sJ * Ri.') * Ki.';           % Mx3  (rows: s^T -> (s * Ri^T) * Ki^T)
    x = XYZ(:,1); y = XYZ(:,2); z = XYZ(:,3);
    pix = [x./z, y./z];
    r   = Ui - pix;                          % residuals

    [alpha, rho] = huber_alpha_and_energy(r, sigmaHuber);
    E = E + sum(rho); nres = nres + 2*M;

    % Perspective-divide Jacobian terms
    invz = 1./z; invz2 = invz.^2;
    A11 = invz; A13 = -x.*invz2; A22 = invz; A23 = -y.*invz2;

    % 3D Jacobian pieces (all Mx3, computed by right-multiplying constants)
    % d/dθ_i: Ki*(Ri*E_k)*s   -> rows: s * (E_k^T * Ri^T * Ki^T)
    Ti1 = sJ * (E1.' * Ri.' * Ki.');   % Mx3
    Ti2 = sJ * (E2.' * Ri.' * Ki.');
    Ti3 = sJ * (E3.' * Ri.' * Ki.');
    
    % d/dθ_j: -Ki*Ri*(E_k*y)  -> rows: -y * (E_k^T * Ri^T * Ki^T)
    Tj1 = - yj * (E1.' * Ri.' * Ki.'); % Mx3
    Tj2 = - yj * (E2.' * Ri.' * Ki.');
    Tj3 = - yj * (E3.' * Ri.' * Ki.');

    % d/df_i: (dKi/df) * Ri * s        -> rows: s * (Ri^T * dKi_df^T)
    dfi = sJ * (Ri.' * dKi_df.');
    
    % d/df_j: Ki * Ri * Rj^T * (dKjInv/df) * [Uj;1]
    % rows: [Uj,1] * dKjInv_df^T * Rj * Ri^T * Ki^T
    dfj = ([Uj, ones(M,1)] * dKjInv_df.') * (Rj * Ri.' * Ki.');

    [Hii, Hij, Hjj, gi, gj] = accumulate_blocks_huber(A11,A13,A22,A23, r, alpha, ...
        Ti1,Ti2,Ti3, Tj1,Tj2,Tj3, dfi, dfj);

    Hblocks{bi,bi} = Hblocks{bi,bi} + Hii;
    Hblocks{bj,bj} = Hblocks{bj,bj} + Hjj;
    Hblocks{bi,bj} = Hblocks{bi,bj} + Hij;
    Hblocks{bj,bi} = Hblocks{bj,bi} + Hij.';
    gblocks{bi}    = gblocks{bi} + gi;
    gblocks{bj}    = gblocks{bj} + gj;

    % --- i -> j (row-form, right-multiply) ---
    yi   = [Ui, ones(M,1)] * KiInv.';      % Mx3
    sI   = yi * Ri;                         % Mx3   (s = Ri^T * y -> rows: y * Ri)
    XYZp = (sI * Rj.') * Kj.';              % Mx3
    x2 = XYZp(:,1); y2 = XYZp(:,2); z2 = XYZp(:,3);
    pix2 = [x2./z2, y2./z2];
    r2   = Uj - pix2;

    [alpha2, rho2] = huber_alpha_and_energy(r2, sigmaHuber);
    E = E + sum(rho2); nres = nres + 2*M;

    invz = 1./z2; invz2 = invz.^2;
    B11 = invz; B13 = -x2.*invz2; B22 = invz; B23 = -y2.*invz2;

    
    % d/dθ_j: Kj*(Rj*E_k)*s   -> rows: s * (E_k^T * Rj^T * Kj^T)
    Sj1 = sI * (E1.' * Rj.' * Kj.');  % Mx3
    Sj2 = sI * (E2.' * Rj.' * Kj.');
    Sj3 = sI * (E3.' * Rj.' * Kj.');
    
    % d/dθ_i: -Kj*Rj*(E_k*s)  -> rows: -s * (E_k^T * Rj^T * Kj^T)
    Si1 = - sI * (E1.' * Rj.' * Kj.'); % Mx3
    Si2 = - sI * (E2.' * Rj.' * Kj.');
    Si3 = - sI * (E3.' * Rj.' * Kj.');

    % d/df_j: (dKj/df) * Rj * s        -> rows: s * (Rj^T * dKj_df^T)
    dfj2 = sI * (Rj.' * dKj_df.');
    
    % d/df_i: Kj * Rj * Ri^T * (dKiInv/df) * [Ui;1]
    % rows: [Ui,1] * dKiInv_df^T * Ri * Rj^T * Kj^T
    dfi2 = ([Ui, ones(M,1)] * dKiInv_df.') * (Ri * Rj.' * Kj.');

    [Hii2, Hij2, Hjj2, gi2, gj2] = accumulate_blocks_huber(B11,B13,B22,B23, r2, alpha2, ...
        Si1,Si2,Si3, Sj1,Sj2,Sj3, dfi2, dfj2);

    Hblocks{bi,bi} = Hblocks{bi,bi} + Hii2;
    Hblocks{bj,bj} = Hblocks{bj,bj} + Hjj2;
    Hblocks{bi,bj} = Hblocks{bi,bj} + Hij2;
    Hblocks{bj,bi} = Hblocks{bj,bi} + Hij2.';
    gblocks{bi}    = gblocks{bi} + gi2;
    gblocks{bj}    = gblocks{bj} + gj2;
end

% Assemble sparse H, g
Nblk = 4*m;
H = spalloc(Nblk,Nblk, 16*m + 64*m*(m-1)/2);
g = zeros(Nblk,1);
for a=1:m
    ia = (a-1)*4 + (1:4);
    g(ia) = gblocks{a};
    for b=1:m
        ib = (b-1)*4 + (1:4);
        if any(Hblocks{a,b}(:))
            H(ia,ib) = H(ia,ib) + Hblocks{a,b};
        end
    end
end
end

function [Hii, Hij, Hjj, gi, gj] = accumulate_blocks_huber(A11,A13,A22,A23, r, alpha, ...
    Ti1,Ti2,Ti3, Tj1,Tj2,Tj3, dfi, dfj)
M = size(r,1);
Ji1 = two_by_three(A11,A13,A22,A23, Ti1); % M×2
Ji2 = two_by_three(A11,A13,A22,A23, Ti2);
Ji3 = two_by_three(A11,A13,A22,A23, Ti3);
Jif = two_by_three(A11,A13,A22,A23, dfi);

Jj1 = two_by_three(A11,A13,A22,A23, Tj1);
Jj2 = two_by_three(A11,A13,A22,A23, Tj2);
Jj3 = two_by_three(A11,A13,A22,A23, Tj3);
Jjf = two_by_three(A11,A13,A22,A23, dfj);

Ji = -cat(3, Ji1,Ji2,Ji3,Jif); % (M×2×4)
Jj = -cat(3, Jj1,Jj2,Jj3,Jjf);

Hii=zeros(4,4); Hjj=zeros(4,4); Hij=zeros(4,4);
gi=zeros(4,1);  gj=zeros(4,1);
for k=1:M
    ak = alpha(k);
    rk = r(k,:).';
    Jik = reshape(Ji(k,:,:),[2,4]);
    Jjk = reshape(Jj(k,:,:),[2,4]);

    Hii = Hii + ak*(Jik.'*Jik);
    Hjj = Hjj + ak*(Jjk.'*Jjk);
    Hij = Hij + ak*(Jik.'*Jjk);

    gi  = gi  + ak*(Jik.'*rk);
    gj  = gj  + ak*(Jjk.'*rk);
end
end

function J2 = two_by_three(A11,A13,A22,A23, T) % T: M×3
Tx=T(:,1); Ty=T(:,2); Tz=T(:,3);
J2 = [A11.*Tx + A13.*Tz,  A22.*Ty + A23.*Tz];
end

% ======================================================================
% ===========================  ENERGY  =================================
% ======================================================================
function E = total_energy(theta, cameras, pairs, active, imageSizes, sigma)
activeSet = false(1, numel(cameras)); activeSet(active)=true;
E = 0;
for t=1:numel(pairs)
    i=pairs(t).i; j=pairs(t).j;
    if ~(activeSet(i) && activeSet(j)), continue; end
    Ui=pairs(t).Ui; Uj=pairs(t).Uj; M=size(Ui,1);

    Ri=expm_hat(theta(i,:)); Rj=expm_hat(theta(j,:));
    [Ki, KiInv] = make_K_and_inv(cameras(i).f, imageSizes(i,:));
    [Kj, KjInv] = make_K_and_inv(cameras(j).f, imageSizes(j,:));

    % --- j -> i ---
    yj  = [Uj, ones(M,1)] * KjInv.';    % Mx3
    sJ  = yj * Rj;                      % Mx3
    XYZ = (sJ * Ri.') * Ki.';           % Mx3
    pix = [XYZ(:,1)./XYZ(:,3), XYZ(:,2)./XYZ(:,3)];
    r   = Ui - pix;
    E   = E + sum(huber_energy_only(r,sigma));
    
    % --- i -> j ---
    yi   = [Ui, ones(M,1)] * KiInv.';   % Mx3
    sI   = yi * Ri;                      % Mx3
    XYZp = (sI * Rj.') * Kj.';           % Mx3
    pix2 = [XYZp(:,1)./XYZp(:,3), XYZp(:,2)./XYZp(:,3)];
    r2   = Uj - pix2;
    E    = E + sum(huber_energy_only(r2,sigma));
end
end

% ======================================================================
% ====================  UPDATES AND UTILITIES  =========================
% ======================================================================
function [thetaNew, camerasNew] = apply_update(theta, cameras, d, active, imageSizes, seed)
thetaNew = theta; camerasNew = cameras;
for k=1:numel(active)
    i = active(k);
    di = d((k-1)*4 + (1:4));
    % rotation: keep seed fixed
    if i~=seed
        thetaNew(i,:) = theta(i,:) + di(1:3).';
        camerasNew(i).R = expm_hat(thetaNew(i,:));
    else
        thetaNew(i,:)   = [0 0 0];
        camerasNew(i).R = eye(3);
    end
    % focal
    camerasNew(i).f = max(1e-6, cameras(i).f + di(4));
    cx=imageSizes(i,2)/2; cy=imageSizes(i,1)/2; f=camerasNew(i).f;
    camerasNew(i).K = [f,0,cx; 0,f,cy; 0,0,1];
end
end

function [K, Kinv, dK_df, dKinv_df] = make_K_and_inv(f, sizes)
H = sizes(1); W = sizes(2);
cx = W/2; cy = H/2;
K  = [f, 0, cx; 0, f, cy; 0, 0, 1];
Kinv = [1/f, 0, -cx/f; 0, 1/f, -cy/f; 0, 0, 1];
dK_df    = [1,0,0; 0,1,0; 0,0,0];
dKinv_df = [-1/f^2, 0,  cx/f^2; 0, -1/f^2, cy/f^2; 0, 0, 0];
end

function [alpha, rho] = huber_alpha_and_energy(r, sigma)
nr = sqrt(sum(r.^2,2));
alpha = 2*ones(size(nr));
mask = nr >= sigma;
alpha(mask) = 2*sigma ./ max(nr(mask), 1e-12);
rho = nr.^2;
rho(mask) = 2*sigma*nr(mask) - sigma^2;
end

function rho = huber_energy_only(r, sigma)
nr = sqrt(sum(r.^2,2));
rho = nr.^2;
mask = nr >= sigma;
rho(mask) = 2*sigma*nr(mask) - sigma^2;
end

function R = expm_hat(w)
th = norm(w);
if th < 1e-12, R = eye(3); return; end
k = w(:)/th;
K = [  0   -k(3)  k(2);
      k(3)   0   -k(1);
     -k(2)  k(1)   0  ];
R = eye(3) + sin(th)*K + (1-cos(th))*(K*K);
end

% ======================================================================
% ===================  NORMALIZE CAMERA STRUCT  ========================
% ======================================================================
function cams = normalize_cameras_struct(camsIn, N)
% Accept either:
% (A) struct array (1×N) with fields R,f,K,initialized (target form), or
% (B) scalar struct with cell fields {R}, {K}, {f}, {initialized}.
if numel(camsIn)==N && isfield(camsIn, 'R') && ~iscell(camsIn(1).R)
    % Already struct array
    cams = camsIn;
    % Ensure defaults exist
    for i=1:N
        if ~isfield(cams(i),'initialized') || isempty(cams(i).initialized), cams(i).initialized = false; end
        if ~isfield(cams(i),'R') || isempty(cams(i).R), cams(i).R = eye(3); end
        if ~isfield(cams(i),'K') || isempty(cams(i).K)
            f = cams(i).f; if isempty(f), f = 1; end
            cams(i).K = [f,0,0;0,f,0;0,0,1];
        end
    end
    return;
end

% Otherwise convert scalar struct-of-cells -> struct array
assert(isstruct(camsIn) && iscell(camsIn.R) && iscell(camsIn.K) && iscell(camsIn.f) && iscell(camsIn.initialized), ...
    'initializeCameraMatrices must return either a struct array or a scalar struct with cell fields R,K,f,initialized.');

cams = repmat(struct('R',eye(3),'f',1,'K',eye(3),'initialized',false), 1, N);
for i=1:N
    cams(i).R = camsIn.R{i};
    cams(i).K = camsIn.K{i};
    cams(i).f = camsIn.f{i};
    cams(i).initialized = camsIn.initialized{i};
    if isempty(cams(i).R), cams(i).R = eye(3); end
    if isempty(cams(i).K), cams(i).K = [cams(i).f,0,0;0,cams(i).f,0;0,0,1]; end
    if isempty(cams(i).f), cams(i).f = 1; end
    if isempty(cams(i).initialized), cams(i).initialized = false; end
end
end

function print_step_deltas(theta_old, theta_new, cams_old, cams_new, active)
    rot_deg = zeros(numel(active),1);
    df       = zeros(numel(active),1);
    for k=1:numel(active)
        i = active(k);
        w = theta_new(i,:) - theta_old(i,:);
        rot_deg(k) = norm(w) * 180/pi;             % small-angle approx
        df(k)      = cams_new(i).f - cams_old(i).f;
    end
    fprintf('   Δrot(deg) median=%.4g  max=%.4g | Δf px median=%.4g  max=%.4g\n', ...
        median(rot_deg), max(rot_deg), median(df), max(df));
end


%--------------------------------------------------------------------------
% Camera parameters estimation functions
%--------------------------------------------------------------------------
function cameras = initializeCameraMatrices(input, imageSizes, initialTforms, n_images)
    % Estimate the focal lengths
    focalEstimates = arrayfun(@(tform) estimateFocals(tform), initialTforms);        
    focalEstimate = median(focalEstimates);
    
    % Populate the focal lengths
    focalLengths = zeros(1, length(initialTforms));
    
    % Logical indexing to check the condition
    useEstimate = focalEstimate > 0;
    
    % Assign focalEstimate to all indices if the condition is met
    if useEstimate
        focalLengths(:) = focalEstimate;
    else
        % Estimate initial focal length by computing the max of the image sizes 
        % and assign it to focalLengths. For typical rectilinear camera with ~60° 
        % field of view
        focalLengths(:) = max(imageSizes(:, 1:2), [], 2) * 0.8';
        fprintf(['Cannot estimate focal lengths, %s motion model is used! ', ...
                 'Therefore, using the max(h,w) x 0.8 values.\n'], input.transformationType);
    end

    % Print estimated focal length
    fprintf('Estimated focal length: %.2f pixels\n', focalLengths(1))

    % Initialize cx, and cy
    cx = imageSizes(:,2)' / 2;
    cy = imageSizes(:,1)' / 2;
    
    % Initialize cameras struct
    cameras = struct('R', repmat({eye(3)}, 1, n_images), ...
                     'f', num2cell(focalLengths), ...
                     'K', cell(1, n_images), ...
                     'initialized', num2cell(false(1, n_images)));
    
    % Precompute K matrices
    K_matrices = arrayfun(@(f, cx_val, cy_val) ...
                          [f, 0, cx_val; 0, f, cy_val; 0, 0, 1], ...
                          focalLengths, cx, cy, 'UniformOutput', false);
    
    % Assign K matrices to the struct
    [cameras.K] = K_matrices{:};
end

%--------------------------------------------------------------------------
% f = estimateFocals(h)
%--------------------------------------------------------------------------
%
%
%
%
function f = estimateFocals(h)
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
    v1 = -(h(1, 1) * h(1, 2) + h(2, 1) * h(2, 2)) / d1;
    v2 = (h(1, 1)^2 + h(2, 1)^2 - h(1, 2)^2 - h(2, 2)^2) / d2;
    
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
    d2 = h(1, 1)^2 + h(1, 2)^2 - h(2, 1)^2 - h(2, 2)^2;
    v1 = -h(1, 3) * h(2, 3) / d1;
    v2 = (h(2, 3)^2 - h(1, 3)^2) / d2;
    
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

%--------------------------------------------------------------------------
% Auxillary functions
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% [tforms] = getTforms(G, i, Tforms)
%--------------------------------------------------------------------------
%
% Computes and returns the projective transformations for all images in the
% connected component of image i in the tree with adjacency matrix G, given
% the set of keypoints and matching indices. All tforms are calculated with
% respect to image i.
function [tforms] = getTforms(G, i, Tforms)
    n = size(G, 1);
    visited = zeros(n, 1);
    tforms = repmat({eye(3)}, 1, n);
    tforms = updateTforms(G, i, visited, tforms, Tforms);
end

%--------------------------------------------------------------------------
% tforms = updateTforms(G, i, visited, tforms, Tforms)
%--------------------------------------------------------------------------
%
% Updates and returns the projective transformations for each image j that
% shares an edge with image i in the tree with adjacency matrix G, given
% the corresponding keypoints and matching indices. Recursively updates
% the tforms of the neighbors of each image j.
function tforms = updateTforms(G, i, visited, tforms, Tforms)
    n = size(G, 1);
    visited(i) = 1;
    for j = 1:n
        if G(i,j) > 0 && ~visited(j)            
            tform = Tforms{i,j};
            tform = tform * tforms{i};
            tforms{j} = tform ./ tform(3,3);
            tforms = updateTforms(G, j, visited, tforms, Tforms);
        end
    end
end


%--------------------------------------------------------------------------
% [tree] = getMST(G)
%--------------------------------------------------------------------------
%
% Returns the adjacency matrix of the maximum spanning tree of the
% undirected graph with weighted adjacency matrix G.
function [tree] = getMST(G)
n = size(G, 1);
ccs = (1:n)'; % list of component number of each vertex
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
            tree(i,j) = values(k);
            tree(j,i) = values(k);
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

%--------------------------------------------------------------------------
% [ordering] = getOrdering(indices, tree)
%--------------------------------------------------------------------------
%
% Given the adjacency matrix for a weighted forest (tree), finds an
% ordering on the specified indices (a single tree in the forest) that
% greedily maximizes the cumulative weight, i.e., starts with vertices with
% the highest edge weight, then expands outward along tree, adding the
% vertex sharing the highest edge weight in the fringe each time, until all
% vertices in the tree have been added. Returns a permutation of 1 to k
% corresponding to the indices of the ordering, where k is the number of
% vertices in the tree.
function [ordering] = getOrdering(indices, tree)
k = length(indices);
ordering = zeros(k, 1);
visited = zeros(k, 1);
subtree = tree(indices,indices);
edges = getEdges(subtree);
[~, index] = max(edges(3,:));
index_i = edges(1,index);
index_j = edges(2,index);
ordering(1) = index_i;
ordering(2) = index_j;
visited(index_i) = 1;
visited(index_j) = 1;
c = 2;

fringe = [];
for index = 1:k
    if subtree(index,index_j) > 0 && ~visited(index)
        fringe = [fringe, [index; index_j; subtree(index,index_j)]];
    end
end
while c < k
    for index = 1:k
        if subtree(index,index_i) > 0 && ~visited(index)
            fringe = [fringe, [index; index_i; subtree(index,index_i)]];
        end
    end
    [~, index] = max(fringe(3,:));
    index_i = fringe(1,index);
    fringe(:,index) = [];
    c = c + 1;
    ordering(c) = index_i;
    visited(index_i) = 1;
end
end

%--------------------------------------------------------------------------
% [edges] = getEdges(G)
%--------------------------------------------------------------------------
%
% Returns a list of weighted edges (i, j, w) of the undirected graph with
% adjacency matrix G.
function [edges] = getEdges(G)
n = size(G, 1);
edges = zeros(3, n * (n - 1) / 2);
c = 0;
for i = 1:n
    for j = i + 1:n
        if G(i,j) > 0
            c = c + 1;
            edges(:,c) = [i; j; G(i,j)];
        end
    end
end
edges = edges(:,1:c);
end