function matches = featureMatchingGlobal(input, allDescriptors, numImg)
    %FEATUREMATCHINGGLOBAL  AutoStitch-style global feature matching.
    %
    %   matches = featureMatchingGlobal(input, allDescriptors, numImg)
    %
    %   Implements a *global kNN ratio test* similar to Brown & Lowe (IJCV 2007):
    %     - All descriptors from all images are pooled together
    %     - Each descriptor queries its k nearest neighbors globally
    %     - Self-matches and same-image matches are removed
    %     - Lowe's ratio test is applied across *different images*
    %
    %   This avoids pairwise matching (O(N^2)) and provides:
    %     • Better global consistency
    %     • Fewer spurious pairwise matches
    %     • AutoStitch-style match discovery
    %
    %   INPUTS
    %   ------
    %   input.k              : number of nearest neighbors (kNN)
    %   input.Ratiothreshold : Lowe ratio threshold (e.g., 0.7–0.8)
    %   input.BFMatch        : (optional) force brute-force matcher
    %
    %   allDescriptors{i}    : [Ni x D] descriptor matrix for image i
    %                          single  -> float descriptors (SIFT, SURF)
    %                          uint8   -> binary descriptors (ORB, AKAZE)
    %
    %   numImg               : total number of images
    %
    %   OUTPUT
    %   ------
    %   matches{i,j} : [M x 2] index pairs (local feature indices)
    %                  representing matches between image i and j
    %

    arguments
        input struct
        allDescriptors cell
        numImg (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    % ---- Parameters ---------------------------------------------------------
    k = input.k; % number of neighbors per query
    ratioThr = input.Ratiothreshold; % Lowe ratio threshold
    trees = 4; % FLANN KD-tree count (float desc)
    checks = 32; % FLANN search checks
    useBF = isfield(input, 'BFMatch') && input.BFMatch; % force BF matcher?

    % ---- Count features per image ------------------------------------------
    if isempty(allDescriptors) || all(cellfun(@isempty, allDescriptors))
        matches = cell(numImg);
        return;
    end

    % ---- Detect descriptor type (binary vs float) ---------------------------
    firstNonEmpty = find(~cellfun(@isempty, allDescriptors), 1, 'first');
    isBinary = isa(allDescriptors{firstNonEmpty}, 'binaryFeatures');

    % ---- Count features per image (different for binary vs float) -----------
    if isBinary
        nFeatures = cellfun(@(x) ifelse(isempty(x), 0, x.NumFeatures), allDescriptors);
    else
        nFeatures = cellfun(@(x) size(x, 1), allDescriptors);
    end

    totalF = sum(nFeatures);
    matches = cell(numImg);
    if totalF == 0, return; end

    % ---- Concatenate all descriptors globally -------------------------------
    if isBinary
        % Extract .Features from each binaryFeatures object
        featuresCell = cellfun(@(x) ifelse(isempty(x), uint8([]), x.Features), ...
            allDescriptors, 'UniformOutput', false);
        allDesc = vertcat(featuresCell{:}); % [totalF x D] uint8
    else
        allDesc = vertcat(allDescriptors{:}); % [totalF x D] single/double
    end

    % ---- L2 normalize or keep unit8 -----------------------------------------
    if ~isBinary
        allDesc = single(allDesc); % ensure float
        allDesc = allDesc ./ sqrt(sum(allDesc .^ 2, 2) ...
            + eps('single')); % L2 normalize
    else
        allDesc = uint8(allDesc); % ensure uint8
    end

    % ---- Global → local index bookkeeping ----------------------------------
    imgIdx = repelem((1:numImg)', nFeatures); % image ID per row
    localIdx = zeros(totalF, 1, 'uint32'); % local feature ID

    ptr = 0;

    for i = 1:numImg
        localIdx(ptr + 1:ptr + nFeatures(i)) = uint32(1:nFeatures(i));
        ptr = ptr + nFeatures(i);
    end

    % ---- Query size ---------------------------------------------------------
    % We need at least two *cross-image* neighbors after pruning,
    % so k should be >= 2 (self + next-best)
    Kq = k;
    % Kq = max(k, 2 * numImg);
    % Kq = max(k, numImg + 3);

    % ---- Global kNN search (descriptor → descriptor) ------------------------
    % nnIdxAll, nnDistAll are [totalF x Kq]
    if ~isBinary
        % Float descriptors → FLANN (KD-tree)
        [nnIdxAll, nnDistAll] = flann_knn_win( ...
            allDesc, allDesc, Kq, 'flann', trees, checks);
    else
        % Binary descriptors → BF or FLANN-LSH
        if useBF
            [nnIdxAll, nnDistAll] = flann_knn_win( ...
                allDesc, allDesc, Kq, 'bf');
        else
            [nnIdxAll, nnDistAll] = flann_knn_win( ...
                allDesc, allDesc, Kq, 'flann');
        end

    end

    % ---- Per-feature filtering + ratio test --------------------------------  
    for q = 1:totalF
        qi = imgIdx(q);
    
        neighIdx  = nnIdxAll(q,:);
        neighDist = nnDistAll(q,:);
    
        % --- Remove self matches ---
        valid = neighIdx ~= uint32(q);
        neighIdx  = neighIdx(valid);
        neighDist = neighDist(valid);
    
        % --- Remove same-image neighbors ---
        neighImgs = imgIdx(double(neighIdx));
        valid = neighImgs ~= qi;
        neighIdx   = neighIdx(valid);
        neighDist  = neighDist(valid);
        neighImgs  = neighImgs(valid);
    
        if numel(neighDist) < 2
            continue;
        end
    
        % --- Ratio test (vectorized) ---
        ratios = neighDist(1:end-1) ./ max(neighDist(2:end), eps('single'));
        pass   = ratios <= ratioThr;
    
        if ~any(pass)
            continue;
        end
    
        neighIdx  = neighIdx(1:end-1);
        neighImgs = neighImgs(1:end-1);
        neighIdx  = neighIdx(pass);
        neighImgs = neighImgs(pass);
    
        % --- Keep only first match per target image (stable) ---
        [~, ia] = unique(neighImgs, 'stable');
        neighIdx  = neighIdx(ia);
        neighImgs = neighImgs(ia);
    
        li = localIdx(q);
        lj = localIdx(double(neighIdx));
    
        % --- Store matches ---
        for k = 1:numel(neighImgs)
            j = neighImgs(k);
            if qi < j
                matches{qi,j}(end+1,:) = [li lj(k)];
            else
                matches{j,qi}(end+1,:) = [lj(k) li];
            end
        end
    end
end

function out = ifelse(cond, valTrue, valFalse)
    %IFELSE Inline conditional: returns valTrue if cond is true, else valFalse.
    %
    %   OUT = IFELSE(COND, VALTRUE, VALFALSE) evaluates COND and returns
    %   VALTRUE if COND is true, otherwise returns VALFALSE.
    %
    %   This is a helper function that emulates the ternary operator
    %   (cond ? valTrue : valFalse) found in C/C++/Java/Python.
    %
    %   Inputs:
    %       cond     - Logical scalar condition
    %       valTrue  - Value to return if cond is true
    %       valFalse - Value to return if cond is false
    %
    %   Output:
    %       out      - Either valTrue or valFalse based on cond
    %
    %   Example:
    %       x = ifelse(n > 0, n, 0);          % clamp negative to zero
    %       s = ifelse(isempty(A), 0, sum(A)); % safe sum
    %
    %   See also IF, ELSE.
    arguments
        cond (1,1) {mustBeNumericOrLogical}
        valTrue
        valFalse
    end

    % Ensure the condition is logical (accepts numeric 0/1 as well)
    cond = logical(cond);
    if cond
        out = valTrue;
    else
        out = valFalse;
    end

end
