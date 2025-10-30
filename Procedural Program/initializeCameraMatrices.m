%--------------------------------------------------------------------------
% Camera parameters estimation functions
%--------------------------------------------------------------------------
function cameras = initializeCameraMatrices(input, pairs, imageSizes, initialTforms, seed, num_images)
    % INITIALIZECAMERAMATRICES Build initial camera struct array (K,R,f,initialized).
    %   cameras = initializeCameraMatrices(input, pairs, imageSizes, initialTforms, seed, num_images)

    arguments
        input (1, 1) struct
        pairs (1, :) struct
        imageSizes (:, 3) double {mustBeFinite}
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
        imageSizes (:, 3) double {mustBeFinite}
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
                base = median(max(imageSizes(:,1:2),[],2));
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

                    if exist('f_used','var') && isscalar(f_used)
                        if f_used < 0.7*base || f_used > 6.0*base
                            f_used = 0.8*base;
                            fprintf('Clamped initial focal to %.1f px (bootstrap guard)\n', f_used);
                        end
                    end
                                   
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

