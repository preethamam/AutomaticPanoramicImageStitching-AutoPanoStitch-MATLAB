function matches = featureMatchingPairwise(input, allDescriptors, numImg)
    %FEATUREMATCHINGPAIRWISE Pairwise feature matching across images using upper-triangular schedule.
    %
    % Syntax
    %   matches = featureMatchingPairwise(input, allDescriptors, numImg)
    %
    % Description
    %   Builds a numImg-by-numImg cell array of pairwise feature matches. The
    %   function computes matches only for the upper-triangular (i<j) pairs using
    %   a parallel loop and leaves the diagonal and lower triangle empty. Matching
    %   is delegated to getMatches, which can use MATLAB's matchFeatures or a
    %   custom matcher depending on flags in input.
    %
    % Inputs
    %   input          - Struct of matching parameters. Expected fields include:
    %                    • useMATLABFeatureMatch (logical/0-1)
    %                    • Matchingmethod (e.g., 'Approximate'|'Exhaustive'|'NearestNeighborSymmetric')
    %                    • Matchingthreshold (scalar)
    %                    • Ratiothreshold (scalar)
    %                    • Approx* fields for custom matcher (if applicable)
    %   allDescriptors - 1-by-numImg cell array; each cell contains descriptors for one image
    %                    compatible with the chosen matching method.
    %   numImg         - Number of images (defines the size of the output cell matrix).
    %
    % Output
    %   matches        - numImg-by-numImg cell array. For i<j, matches{i,j} is a 2-by-K
    %                    array of matched indices [idxInImageI; idxInImageJ]. Other cells
    %                    remain empty.
    %
    % Notes
    %   - Uses parfor to parallelize pair matching over the upper-triangular index set.
    %   - The function does not mirror matches to the lower triangle; consumers should
    %     access only the upper triangle or symmetrize as needed.
    %
    % See also: matchFeatures, matchFeaturesScratch, ind2sub, triu, parfor

    arguments
        input struct
        allDescriptors cell
        numImg (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    % Initialize
    matches = cell(numImg);
    matSize = size(matches);

    % Use symmetry and run for upper triangular matrix
    IuppeIdx = nonzeros(triu(reshape(1:numel(matches), size(matches)), 1));

    % Initialize
    matches_ij = cell(1, length(IuppeIdx));

    % Match features
    parfor i = 1:length(IuppeIdx)
        % IND2SUB converts from a "linear" index into individual
        % subscripts
        [ii, jj] = ind2sub(matSize, IuppeIdx(i));
        matches_ij{i} = getMatches(input, allDescriptors{ii}, allDescriptors{jj});
    end

    % Populate A matrix
    matches(IuppeIdx) = matches_ij;
end

%--------------------------------------------------------------------------------------------------------
% Auxillary functions
%--------------------------------------------------------------------------------------------------------
% [matches_ij] = getMatches(features1,features2)
function matches = getMatches(input, features1, features2)
    %GETMATCHES Match descriptors between two images.
    %
    % Syntax
    %   matches = getMatches(input, features1, features2)
    %
    % Description
    %   Matches descriptors from features1 to features2 using either MATLAB's
    %   matchFeatures or a custom implementation (matchFeaturesScratch) based on
    %   input.useMATLABFeatureMatch. Returns index pairs as a 2-by-K array.
    %
    % Inputs
    %   input     - Struct of matching options. Common fields:
    %               • useMATLABFeatureMatch (logical)
    %               • Matchingmethod, Matchingthreshold, Ratiothreshold
    %               • Approx* fields used by matchFeaturesScratch (e.g., ApproxNumTables,
    %                 ApproxBitsPerKey, ApproxProbes, ApproxFloatNNMethod)
    %   features1 - Descriptors from image 1 (format compatible with selected matcher).
    %   features2 - Descriptors from image 2.
    %
    % Output
    %   matches   - 2-by-K array of matched descriptor indices: [idx1; idx2].
    %
    % Notes
    %   - Enforces unique matches (one-to-one) via Unique=true.
    %   - For the MATLAB matcher, MaxRatio applies Lowe's ratio test when applicable.
    %   - Indices are cast to double and transposed to produce a 2-by-K layout.

    arguments
        input struct
        features1  
        features2
    end

    if input.useMATLABFeatureMatch == 1
        matches = matchFeatures(features1, features2, 'Method', input.Matchingmethod, ...
            'MatchThreshold', input.Matchingthreshold, ...
            'MaxRatio', input.Ratiothreshold, Unique = true);
    else
        [matches, ~] = matchFeaturesScratch(features1, features2, ...
            'Method', input.Matchingmethod, ...
            'ApproxFloatNNMethod', input.ApproxFloatNNMethod, ...
            'MatchThreshold', input.Matchingthreshold, ... % SSD threshold
            'MaxRatio', input.Ratiothreshold, ...
            'Unique', true, ...
            'ApproxNumTables', input.ApproxNumTables, ...
            'ApproxBitsPerKey', input.ApproxBitsPerKey, ...
            'ApproxProbes', input.ApproxProbes, ...
            'ApproxKDBucketSize', 40);
    end

    matches = double(matches);
end
