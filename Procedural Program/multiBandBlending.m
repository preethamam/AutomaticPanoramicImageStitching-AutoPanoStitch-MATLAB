function F = multiBandBlending(Ci, Wi, levels, onGPU, sigma)
% MULTIBANDBLENDING Multiband blend with Laplacian colors + Gaussian weights (GPU/CPU).
%   F = fuse_pyramids(Ci, Wi, levels, onGPU, sigma)
%   Ci: cell{K} of [h x w x C] single (C=1 or 3); raw colors per image/tile
%   Wi: cell{K} of [h x w]     single; per-pixel weights (un-normalized)
%   levels: integer >= 1 (recommended 2–5)
%   onGPU: logical. If true and inputs are CPU, they'll be moved to GPU.
%   sigma: Gaussian std for pyramid blurs (default 1.0)
%
%   Returns F: [h x w x C] single in [0,1].

% Weighted Laplacian fusion with single final normalization.
if nargin < 5 || isempty(sigma), sigma = 1.0; end
K = numel(Ci);  assert(K==numel(Wi) && K>=1);

% Ensure types/devices -----------------------------------------------------
for k=1:K
    if ~isa(Ci{k},'single'), Ci{k}=single(Ci{k}); end
    if ~isa(Wi{k},'single'), Wi{k}=single(Wi{k}); end
end
inputsOnGPU = isa(Ci{1},'gpuArray') || isa(Wi{1},'gpuArray');
if onGPU && ~inputsOnGPU
    for k=1:K, Ci{k}=gpuArray(Ci{k}); Wi{k}=gpuArray(Wi{k}); end
elseif ~onGPU && inputsOnGPU
    for k=1:K, Ci{k}=gather(Ci{k});   Wi{k}=gather(Wi{k});   end
end

% ---- PRE-NORMALIZE WEIGHTS (full-res) ----
Wsum = zeros(size(Wi{1}), 'like', Wi{1});
for k=1:K
    Wi{k} = max(0, Wi{k});      % clamp negatives
    Wsum  = Wsum + Wi{k};
end
epsW = 1e-8;
mask_nonzero = (Wsum > epsW);

for k=1:K
    Wn = zeros(size(Wi{k}), 'like', Wi{k});
    Wn(mask_nonzero) = Wi{k}(mask_nonzero) ./ Wsum(mask_nonzero);
    Wi{k} = Wn;                 % now Σ_k Wi{k} == 1 where covered, 0 elsewhere
end

% Internally use 3 channels
[h,w,origC] = size(Ci{1});
Ci = cellfun(@ensure3, Ci, 'uni', 0);

% Cap levels
maxLevels = floor(log2(min(h,w)));
levels = max(1, min(levels, maxLevels));

% Build pyramids
LC = cell(K,1);
GW = cell(K,1);
for k=1:K
    % Important: make sure weights are non-negative and zero outside support
    Wi{k} = max(0, Wi{k});
    LC{k} = laplacian_pyr(Ci{k}, levels, sigma);
    GW{k} = gaussian_pyr(Wi{k}, levels, sigma);  % scalar (h×w) per level
end

% Accumulate only a numerator pyramid
NumP = cell(levels,1);
for l=1:levels
    accNum = zeros(size(LC{1}{l}), 'like', LC{1}{l});
    for k=1:K
        accNum = accNum + bsxfun(@times, LC{k}{l}, GW{k}{l});
    end
    NumP{l} = accNum;
end

% Collapse: no denominator anymore
F3 = collapse_laplacian_pyr(NumP, sigma);

% Restore original channels and clamp
if origC==1, F = F3(:,:,1); else, F = F3; end
F = min(1, max(0, F));
end


% ---------- helpers (identical to yours except one new collapse) ----------
function GP = gaussian_pyr(I, levels, sigma)
GP = cell(levels,1);
GP{1} = I;
for l=2:levels
    J = imgaussfilt(GP{l-1}, sigma, 'Padding','replicate');
    GP{l} = imresize(J, 0.5, 'bilinear');
end
end

function LP = laplacian_pyr(I, levels, sigma)
LP = cell(levels,1);
G0 = I;
for l=1:levels-1
    G1 = imgaussfilt(G0, sigma, 'Padding','replicate');
    D  = imresize(G1, 0.5, 'bilinear');
    U  = imresize(D, size(G0(:,:,1)), 'bilinear');
    LP{l} = G0 - U;
    G0 = D;
end
LP{levels} = G0;
end

function I = collapse_laplacian_pyr(LP, ~)
L = numel(LP);
I = LP{L};
for l = L-1:-1:1
    U = imresize(I, size(LP{l}(:,:,1)), 'bilinear');
    I = U + LP{l};
end
end

function G = collapse_gaussian_pyr(GP)
L = numel(GP);
G = GP{L};
for l = L-1:-1:1
    U = imresize(G, size(GP{l}), 'bilinear');
    G = U + GP{l};             % standard Gaussian collapse
end
end

function X3 = ensure3(X)
if ndims(X)==2, X3 = cat(3,X,X,X);
elseif size(X,3)==1, X3 = repmat(X,[1 1 3]);
else, X3 = X;
end
end
