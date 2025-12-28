function [bundlerTformsAll, finalrefIdxsAll, panoIndices, concomps, panaromaCCs, connCompsNumber] = recognizePanoramas(input, ...
        numMatchesAll, matchesAll, keypointsAll, imageSizesAll, initialTformsAll, ...
        numMatchesG, concomps, ccBinSizes)
    %RECOGNIZEPANORAMAS Identify panorama groups and refine transforms by bundle adjustment.
    %
    % Syntax
    %   [bundlerTformsAll, finalrefIdxsAll, panoIndices, concomps, panaromaCCs, connCompsNumber] = ...
    %       recognizePanoramas(input, numMatches, matchesAll, keypointsAll, imageSizesAll, initialTformsAll)
    %
    % Description
    %   Given feature-match information across a set of images, this function:
    %     1) Builds a graph whose nodes are images and edges exist when there are
    %        matches (encoded in numMatches).
    %     2) Finds connected components (candidate panoramas).
    %     3) For each component with more than one image, runs Levenberg–Marquardt
    %        bundle adjustment (bundleAdjustmentLM) to refine transforms starting
    %        from initialTformsAll. Single-image components are detected and skipped.
    %
    % Inputs
    %   input            - Struct of algorithm parameters used by bundleAdjustmentLM.
    %                      Expected fields: maxIterLM, lambda, sigmaHuber, verboseLM.
    %   numMatchesAll    - N-by-N upper-triangular (or symmetric) numeric matrix with
    %                      the number of inlier matches between image i and j. Only
    %                      the upper triangle is interpreted when constructing the graph.
    %   matchesAll       - N-by-N cell array of match information for each image pair.
    %   keypointsAll     - N-by-1 cell array of keypoint sets per image.
    %   imageSizesAll    - N-by-2 array of image sizes as [height width] (rows, cols).
    %   initialTformsAll - N-by-N cell array of initial geometric transforms between
    %                      image pairs (e.g., projective2d/affine2d or compatible structs).
    %
    % Outputs
    %   bundlerTformsAll - C-by-1 cell array; for each multi-image component, the
    %                      refined transforms returned by bundleAdjustmentLM.
    %   finalrefIdxsAll  - C-by-1 numeric array of reference image indices selected by
    %                      bundle adjustment for each multi-image component.
    %   panoIndices      - C-by-1 cell array; each cell lists the absolute image indices
    %                      that belong to that panorama component.
    %   concomps         - 1-by-N vector assigning each image to a connected component label.
    %   panaromaCCs      - Indices of connected components that contain more than one image.
    %   connCompsNumber  - Total number of connected components (including singletons).
    %
    % Notes
    %   - The graph is built using graph(numMatches, 'upper'), treating numMatches as
    %     an upper-triangular adjacency.
    %   - Single-image components are reported via concomps but are skipped during
    %     bundle adjustment.
    %   - Implementation detail: arrays sized by the number of multi-image components
    %     should be indexed using the component list rather than the raw component
    %     labels to avoid gaps.
    %
    % See also: graph, conncomp, bundleAdjustmentLM

    arguments
        input (1, 1) struct
        numMatchesAll (:, :) {mustBeNumeric, mustBeFinite, mustBeNonnegative}
        matchesAll (:, :) cell
        keypointsAll cell
        imageSizesAll (:, 3) {mustBeNumeric, mustBeFinite, mustBeNonnegative}
        initialTformsAll (:, :) cell
        numMatchesG
        concomps
        ccBinSizes
    end

    % Cross-array consistency checks
    N = size(numMatchesAll, 1);

    % Image matches check
    if sum(sum(numMatchesAll)) == 0
        warning('No images matched.\n')
    end

    if size(numMatchesAll, 2) ~= N
        error('recognizePanoramas:SquareMatches', 'numMatches must be a square N-by-N matrix.');
    end

    if ~isequal(size(matchesAll), [N N])
        error('recognizePanoramas:SizeMismatch', 'matchesAll must be N-by-N cells consistent with numMatches.');
    end

    if ~isequal(size(initialTformsAll), [N N])
        error('recognizePanoramas:SizeMismatch', 'initialTformsAll must be N-by-N cells consistent with numMatches.');
    end

    if ~(isvector(keypointsAll) && numel(keypointsAll) == N)
        error('recognizePanoramas:KeypointsSize', 'keypointsAll must be a vector cell array with N elements.');
    end

    if size(imageSizesAll, 1) ~= N || size(imageSizesAll, 2) ~= 3
        error('recognizePanoramas:ImageSizesSize', 'imageSizesAll must be N-by-2 [height width pixels].');
    end

    % Validate required fields in input struct
    reqFields = ["maxIterLM", "lambda", "sigmaHuber", "verboseLM"];
    missing = reqFields(~isfield(input, reqFields));

    if ~isempty(missing)
        error('recognizePanoramas:MissingField', 'Missing required input fields: %s', strjoin(missing, ', '));
    end

    % Validate types/values
    if ~(isscalar(input.maxIterLM) && isnumeric(input.maxIterLM) && isfinite(input.maxIterLM) && input.maxIterLM > 0 && mod(input.maxIterLM, 1) == 0)
        error('recognizePanoramas:InvalidMaxIter', 'input.maxIterLM must be a positive integer scalar.');
    end

    if ~(isscalar(input.lambda) && isnumeric(input.lambda) && isfinite(input.lambda) && input.lambda > 0)
        error('recognizePanoramas:InvalidLambda', 'input.lambda must be a positive numeric scalar.');
    end

    if ~(isscalar(input.sigmaHuber) && isnumeric(input.sigmaHuber) && isfinite(input.sigmaHuber) && input.sigmaHuber > 0)
        error('recognizePanoramas:InvalidSigma', 'input.sigmaHuber must be a positive numeric scalar.');
    end

    if ~(islogical(input.verboseLM) && isscalar(input.verboseLM))
        error('recognizePanoramas:InvalidVerbose', 'input.verboseLM must be a logical scalar.');
    end

    % Find connected components of image matches
    panaromaCCsAll = find(ccBinSizes >= 1);
    panaromaCCs = find(ccBinSizes > 1);
    connCompsNumber = numel(panaromaCCsAll);

    fprintf('Found %i panorama images set(s).\n', connCompsNumber);

    % Show graphs
    if input.showAdjacencyGraph
        showAdjacencyGraphs(numMatchesAll, numMatchesG);
    end

    % Initialize
    bundlerTformsAll = cell(numel(panaromaCCs), 1);
    finalrefIdxsAll = zeros(numel(panaromaCCs), 1);
    panoIndices = cell(numel(panaromaCCs), 1);

    for i = 1:connCompsNumber
        idxs = find(concomps == i);

        % Get panorama matches
        numMatches = numMatchesAll(idxs, idxs);

        % Panorama image matches check
        if sum(sum(numMatches)) == 0
            warning('No panorama images matched')
        end

        if isscalar(idxs)
            % Skip bundle adjustment for single-image panorama
            warning('Skipping bundle adjustment as only one image found');
            continue;
        end

        % Store panorama image indices
        panoIndices{i} = idxs;

        % Get sub-matches keypoints, image sizes, matcehd and initial
        % transformation matrices
        keypoints = keypointsAll(idxs);
        imageSizes = imageSizesAll(idxs, :);
        matches = matchesAll(idxs, idxs);
        initialTforms = initialTformsAll(idxs, idxs);

        % Perform bundle adjustment
        BAtic = tic;
        [bundlerTforms, finalrefIdxs] = bundleAdjustmentRKf(input, numMatches, matches, keypoints, ...
            imageSizes, initialTforms, ...
            'MaxLMIters', input.maxIterLM, 'Lambda0', input.lambda, ...
            'SigmaHuber', input.sigmaHuber, 'Verbose', input.verboseLM, ...
            'OneDirection', input.residualOneDirection, ...
            'MaxMatches', input.MaxMatches);

        fprintf('Final alignment (Bundle adjustment): %f seconds\n', toc(BAtic));

        % Store panoramas R, K and f and reference indexes
        bundlerTformsAll{i} = bundlerTforms;
        finalrefIdxsAll(i) = finalrefIdxs;
    end

end

function showAdjacencyGraphs(numMatchesAll, numMatchesG)
    % showAdjacencyGraphs  Display adjacency matrix and corresponding graph.
    %
    %   showAdjacencyGraphs(numMatchesAll, numMatchesG) displays two figures:
    %     - an image of the adjacency/match-count matrix `numMatchesAll` using
    %       `imagesc`, and
    %     - a graph visualization created from the `numMatchesG` graph object.
    %
    % Inputs:
    %   numMatchesAll - N-by-N numeric matrix of match counts (nonnegative,
    %                   finite values expected)
    %   numMatchesG   - A `graph` or `digraph` object representing image
    %                   connectivity (used with `plot`).
    %
    % Example:
    %   G = graph(adjacencyMatrix>0);
    %   showAdjacencyGraphs(adjacencyMatrix, G);
    %
    % See also: imagesc, graph, digraph, plot

    arguments
        numMatchesAll (:, :) {mustBeNumeric, mustBeFinite, mustBeNonnegative}
        numMatchesG {mustBeA(numMatchesG, {'graph', 'digraph'})}
    end

    % Basic shape/consistency checks
    N = size(numMatchesAll, 1);

    if size(numMatchesAll, 2) ~= N
        error('showAdjacencyGraphs:SquareMatrix', 'numMatchesAll must be an N-by-N matrix.');
    end

    % Ensure graph has the same number of nodes as the adjacency matrix
    try
        nG = numnodes(numMatchesG);
    catch
        error('showAdjacencyGraphs:InvalidGraph', 'Unable to determine number of nodes in numMatchesG.');
    end

    if nG ~= N
        error('showAdjacencyGraphs:NodeMismatch', 'numMatchesG has %d nodes but numMatchesAll is %d-by-%d.', nG, N, N);
    end

    % Plot adjacency matrix (match counts)
    figure('Name', 'Adjacency matrix');
    imagesc(numMatchesAll);
    axis equal tight;
    colormap jet;
    colorbar;
    title('Match Counts (Adjacency Matrix View)');
    xlabel('Image j');
    ylabel('Image i');

    % Plot graph view
    figure('Name', 'Graph view');
    plot(numMatchesG);
    axis equal tight;
end
