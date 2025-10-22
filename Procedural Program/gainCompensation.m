function gains = gainCompensation(images, cameras, mode, ref_idx, opts, ...
                                              H, W, u0, v0, th0, h0, ph0, srcW)
% Brown–Lowe (2007) gain compensation Eq. (29).
% Returns per-image RGB gains [N x 3].
%
% opts.f_pan (required)
% opts.overlap_stride (default 8)
% opts.min_overlap_samples (default 200)
% opts.sigma_N (default 10.0)   % intensity noise std (paper uses 10 for 8-bit)
% opts.sigma_g (default 0.1)    % gain prior std

N = numel(images);
gains = ones(N,3,'single');

% defaults
if ~isfield(opts,'overlap_stride'),      opts.overlap_stride      = 5;   end
if ~isfield(opts,'min_overlap_samples'), opts.min_overlap_samples = 50; end
if ~isfield(opts,'sigma_N'),             opts.sigma_N             = 10.0;end
if ~isfield(opts,'sigma_g'),             opts.sigma_g             = 0.1; end

stride = max(1, opts.overlap_stride);
minOv  = max(1, opts.min_overlap_samples);
sN2    = (opts.sigma_N)^2;
sg2    = (opts.sigma_g)^2;

% --- subsampled pano grid & world directions ---
[xp_s, yp_s] = meshgrid(single(1:stride:W), single(1:stride:H));
Hs = size(xp_s,1); Ws = size(xp_s,2);

[DWx, DWy, DWz] = pano_dirs_for_grid(xp_s, yp_s, mode, ref_idx, cameras, ...
                                     opts.f_pan, u0, v0, th0, h0, ph0);
DWs = cat(3, DWx, DWy, DWz);

% --- per-image projections & coverage ---
uL=cell(N,1); vL=cell(N,1); cov=cell(N,1);
for i=1:N
    [u_i, v_i, front_i, wang_i] = project_to_image(DWs, cameras(i));
    Wi = fast_sample_block(srcW{i}, u_i, v_i, Hs, Ws, false); Wi = Wi(:,1);
    Mi = isfinite(Wi) & (Wi>0) & front_i & isfinite(u_i) & isfinite(v_i);
    uL{i}=u_i; vL{i}=v_i; cov{i}=Mi;
end

% --- discover neighbors by co-coverage ---
edges = [];
for i=1:N-1
    Mi = cov{i}; if ~any(Mi), continue; end
    for j=i+1:N
        Mj = cov{j}; if ~any(Mj), continue; end
        if nnz(Mi & Mj) >= minOv, edges = [edges; i j]; end %#ok<AGROW>
    end
end
if isempty(edges), return; end

% --- Build A g = b per channel based on Eq. (29) ---
% A(i,i) += sum_j N_ij*(Ibar_ij^2/sN2 + 1/sg2)
% A(i,j) += - N_ij*(Ibar_ij*Ibar_ji/sN2)
% b(i)   += sum_j N_ij*(1/sg2)
A = zeros(N,N,3,'double');
b = zeros(N,1,'double');

for e=1:size(edges,1)
    i = edges(e,1); j = edges(e,2);

    Mij = cov{i} & cov{j};
    if ~any(Mij), continue; end

    % sample colors on overlap for i and j (subsampled grid)
    Ci = fast_sample_block(single(images{i}), uL{i}(Mij), vL{i}(Mij), Hs, Ws, false); % [K x 3]
    Cj = fast_sample_block(single(images{j}), uL{j}(Mij), vL{j}(Mij), Hs, Ws, false);
    good = all(isfinite(Ci),2) & all(isfinite(Cj),2);
    if ~any(good), continue; end
    Ci = Ci(good,:); Cj = Cj(good,:);
    Nij = size(Ci,1);                       % pixels in overlap (subsampled)

    % per-channel means over the overlap region
    Ibar_ij = mean(Ci,1);                   % 1x3
    Ibar_ji = mean(Cj,1);                   % 1x3

    % accumulate normal eqns per channel
    wN = double(Nij) / sN2;                 % scales data term
    wG = double(Nij) / sg2;                 % scales gain prior (paper uses Nij factor)

    for ch=1:3
        aii = wN * (Ibar_ij(ch)*Ibar_ij(ch)) + wG;
        ajj = wN * (Ibar_ji(ch)*Ibar_ji(ch)) + wG;
        aij = - wN * (Ibar_ij(ch)*Ibar_ji(ch));

        A(i,i,ch) = A(i,i,ch) + aii;
        A(j,j,ch) = A(j,j,ch) + ajj;
        A(i,j,ch) = A(i,j,ch) + aij;
        A(j,i,ch) = A(j,i,ch) + aij;

        % RHS from prior term: N_ij*(1/sg^2)
        % This adds equally to b(i) and b(j)
    end
    b(i) = b(i) + wG;
    b(j) = b(j) + wG;
end

% Solve per channel (no hard anchor; the prior biases g_i toward 1)
for ch=1:3
    % Small Tikhonov for numerical safety
    A(:,:,ch) = A(:,:,ch) + 1e-8*eye(N);
    x = A(:,:,ch) \ b;
    gains(:,ch) = single( max(0.25, min(4.0, x)) );
end
end


function [dwx, dwy, dwz] = pano_dirs_for_grid(xp, yp, mode, ref_idx, cameras, f_pan, u0, v0, th0, h0, ph0)
% Compute WORLD directions [h×w×3] for a pano grid (xp,yp).

switch lower(mode)
case 'planar'
    u = single(u0) + xp/single(f_pan);
    v = single(v0) + yp/single(f_pan);
    dx_ref = u; dy_ref = v; dz_ref = ones(size(u), 'like', u);
    Rref = single(cameras(ref_idx).R);   % world->ref
    Rt   = Rref.';                       % ref->world
    dwx =  Rt(1,1)*dx_ref + Rt(1,2)*dy_ref + Rt(1,3)*dz_ref;
    dwy =  Rt(2,1)*dx_ref + Rt(2,2)*dy_ref + Rt(2,3)*dz_ref;
    dwz =  Rt(3,1)*dx_ref + Rt(3,2)*dy_ref + Rt(3,3)*dz_ref;

case 'cylindrical'
    theta = single(th0) + xp/single(f_pan);
    h     = single(h0)  + yp/single(f_pan);
    dwx = -sin(theta);
    dwy = -h;
    dwz =  cos(theta);

case 'spherical'
    theta = single(th0) + xp/single(f_pan);
    phi   = single(ph0) + yp/single(f_pan);
    cphi  = cos(phi); sphi = sin(phi);
    dwx = -cphi.*sin(theta);
    dwy = -sphi;
    dwz =  cphi.*cos(theta);

otherwise
    error('Unknown mode "%s"', mode);
end
end

function [u,v,front,w_angle] = project_to_image(DWs, cam)
% PROJECT_TO_IMAGE  Project world rays into one camera.
%   DWs : [H x W x 3] world directions
%   cam : struct with fields R (w2c), K (3x3)
% Returns:
%   u,v : pixel coordinates (1-based)
%   front : logical mask (forward-facing)
%   w_angle : cosine of viewing angle (for weighting)

[h,w,~] = size(DWs);
DW = reshape(DWs, [], 3);         % [M x 3]
R = single(cam.R);
K = single(cam.K);

dir_c = DW * R.';                 % world -> camera
cx_w = dir_c(:,1);
cy_w = dir_c(:,2);
cz_w = dir_c(:,3);

front = cz_w > 1e-6;
u = K(1,1)*(cx_w./cz_w) + K(1,3);
v = K(2,2)*(cy_w./cz_w) + K(2,3);

fw = R(:,3);                      % camera forward vector
w_angle = max(0, DW * fw);
w_angle = w_angle .* single(front);
end

function S = fast_sample_block(Ic, u, v, H, W, onGPU)
% Vectorized bilinear sampling using interp2 with NaN extrapolation.
% Ic : [h x w x C] (single, CPU or gpuArray). C can be 1 or 3 (or any).
% u,v: [HW x 1] single (same device as Ic if onGPU==true)
% S  : [HW x C] (same device as u/v). NaNs for out-of-bounds.

[h, w, C] = size(Ic); %#ok<ASGLU> % h,w unused except for grid creation

% Build grids for interp2 (1..w columns, 1..h rows)
X = single(1:w);
Y = single(1:h);
if onGPU
    % Ensure grids live on GPU when sampling on GPU
    X = gpuArray(X);
    Y = gpuArray(Y);
end

% Sanitize coords; any non-finite -> will yield NaN from interp2
bad = ~isfinite(u) | ~isfinite(v);
if any(bad, 'all')
    % No need to force -1; extrapval=NaN handles oob and bad points
    % but keep them finite to avoid interp2 errors
    u(bad) = 1;  % arbitrary in-bounds
    v(bad) = 1;
end

% Allocate result like u (CPU or GPU), with C channels
S = zeros(numel(u), C, 'like', u);

% Interpolate each channel with NaN extrapolation
if C==1
    S(:,1) = interp2(X, Y, Ic, u, v, 'linear', NaN);
else
    for ch=1:C
        S(:,ch) = interp2(X, Y, Ic(:,:,ch), u, v, 'linear', NaN);
    end
end

end