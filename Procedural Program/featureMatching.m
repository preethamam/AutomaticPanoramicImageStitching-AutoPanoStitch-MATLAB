function matches = featureMatching(input, allDescriptors, numImg)

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

    if input.useMATLABFeatureMatch == 1
        matches = matchFeatures(features1, features2, 'Method', input.Matchingmethod, ...
            'MatchThreshold', input.Matchingthreshold, ...
            'MaxRatio', input.Ratiothreshold, Unique = true);
    else
        [matches, metric] = matchFeaturesScratch(features1, features2, ...
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

    matches = double(matches)';
end
