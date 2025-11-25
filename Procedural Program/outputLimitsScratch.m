function [xLimitsOut, yLimitsOut] = outputLimitsScratch(tform, xLimitsIn, yLimitsIn, varargin)
    % OUTPUTLIMITSSCRATCH  Compute axis-aligned output limits after a 2-D transform.
    %
    %   [xLimitsOut, yLimitsOut] = outputLimitsScratch(tform, xLimitsIn, yLimitsIn)
    %   [xLimitsOut, yLimitsOut] = outputLimitsScratch(..., Name, Value)
    %
    % Description:
    %   Computes the axis-aligned bounding box (x- and y-limits) of the input
    %   rectangle defined by `xLimitsIn = [xmin xmax]` and `yLimitsIn = [ymin ymax]`
    %   after applying the 2-D forward transform `tform`. The rectangle edges are
    %   sampled (corners plus intermediate points controlled by
    %   'NumSamplesPerEdge') and the forward mapping is applied to determine the
    %   minimum and maximum output coordinates.
    %
    % Inputs:
    %   tform      - Transform to apply. Supported forms:
    %                  * 3x3 numeric matrix (homography; projective/affine)
    %                  * struct with field `.T` containing a 3x3 matrix
    %                  * struct with field `.forward` containing a function_handle
    %                  * function_handle accepting either (XY) -> UV or (x,y) -> (u,v)
    %   xLimitsIn  - 1x2 vector [xmin xmax]
    %   yLimitsIn  - 1x2 vector [ymin ymax]
    %
    % Name-Value options:
    %   'NumSamplesPerEdge' (default 2) - Number of samples per rectangle edge.
    %       A value of 2 samples corners only; increase (e.g., 32) for
    %       nonlinear warps to better capture curved edges.
    %   'IgnoreNaN' (default true)      - If true, drop NaN/Inf outputs before
    %       computing min/max.
    %
    % Outputs:
    %   xLimitsOut  - 1x2 vector [xmin_out xmax_out]
    %   yLimitsOut  - 1x2 vector [ymin_out ymax_out]
    %
    % Notes:
    %   - For function handles, this routine first attempts a vectorized call
    %%    of the form `UV = fwd(XY)` (N×2 -> N×2). If that call fails, it falls
    %   back to calling `fwd(x,y)`.
    %   - If all transformed points are NaN/Inf and 'IgnoreNaN' is true, outputs
    %     are returned as `[NaN NaN]`.
    %
    % Examples:
    %   % Using a homography/affine matrix:
    %   H = [1 0.1 0; 0.05 1 0; 1e-3 -2e-3 1];
    %   [xo, yo] = outputLimitsScratch(H, [1 640], [1 480]);
    %
    %   % Using a vectorized function handle (N×2 -> N×2):
    %   fwd = @(xy) [xy(:,1) + 0.02*xy(:,2), 0.98*xy(:,2)];
    %   [xo, yo] = outputLimitsScratch(fwd, [1 640], [1 480], 'NumSamplesPerEdge', 25);
    %
    % See also: imwarp, projective2d

    % Argument validation (declared immediately below the function help)
    arguments
        tform
        xLimitsIn (1, 2) double
        yLimitsIn (1, 2) double
    end

    arguments (Repeating)
        varargin
    end

    p = inputParser;
    p.addParameter('NumSamplesPerEdge', 2, @(n)isnumeric(n) && isscalar(n) && n >= 2);
    p.addParameter('IgnoreNaN', true, @(b)islogical(b) && isscalar(b));
    p.parse(varargin{:});
    Ns = max(2, round(p.Results.NumSamplesPerEdge));
    ignoreNaN = p.Results.IgnoreNaN;

    % --- build sampled rectangle boundary (clockwise) ---
    xmin = xLimitsIn(1); xmax = xLimitsIn(2);
    ymin = yLimitsIn(1); ymax = yLimitsIn(2);

    tx = linspace(xmin, xmax, Ns).';
    ty = linspace(ymin, ymax, Ns).';

    % edges: top, right, bottom, left (avoid duplicating corner points)
    top = [tx, repmat(ymin, Ns, 1)];
    right = [repmat(xmax, Ns - 1, 1), ty(2:end)];
    bottom = [flipud(tx(1:end - 1)), repmat(ymax, Ns - 1, 1)];
    left = [repmat(xmin, Ns - 2, 1), flipud(ty(2:end - 1))];

    XY = [top; right; bottom; left]; % M×2

    % --- resolve forward transformer ---
    fwd = resolveForward(tform);

    % --- apply transform (supports either (xy) or (x,y) style) ---
    try
        UV = fwd(XY);
    catch
        % Try calling as (x,y) -> (u,v) vectors
        [u, v] = fwd(XY(:, 1), XY(:, 2));
        UV = [u(:), v(:)];
    end

    % --- clean & bounds ---
    if ignoreNaN
        bad = any(~isfinite(UV), 2);
        UV = UV(~bad, :);
    end

    if isempty(UV)
        xLimitsOut = [NaN NaN];
        yLimitsOut = [NaN NaN];
        return;
    end

    xLimitsOut = [min(UV(:, 1)), max(UV(:, 1))];
    yLimitsOut = [min(UV(:, 2)), max(UV(:, 2))];
end

function fwd = resolveForward(tform)
    % RESOLVEFORWARD  Create forward mapping handle from various tform types.
    %
    %   fwd = resolveForward(tform)
    %
    % Inputs:
    %   tform - Transform specification. Supported forms:
    %           * numeric 3x3 matrix H
    %           * struct with field .T (3x3 matrix)
    %           * struct with field .forward (function handle)
    %           * function handle that maps either XY (N×2) -> N×2 or (x,y)->(u,v)
    %
    % Outputs:
    %   fwd - Function handle with signature `UV = fwd(XY)` returning N×2 numeric

    arguments
        tform
    end

    if isa(tform, 'function_handle')
        fwd = @(XY) callForward(tform, XY);
        return;
    end

    if isstruct(tform)

        if isfield(tform, 'forward') && isa(tform.forward, 'function_handle')
            fwd = @(XY) callForward(tform.forward, XY);
            return;
        elseif isfield(tform, 'T') && isnumeric(tform.T) && all(size(tform.T) == [3 3])
            H = double(tform.T);
            fwd = @(XY) applyH(H, XY);
            return;
        end

    end

    if isnumeric(tform) && all(size(tform) == [3 3])
        H = double(tform);
        fwd = @(XY) applyH(H, XY);
        return;
    end

    error('outputLimits:UnsupportedTform', ...
        ['Unsupported tform. Provide a 3x3 matrix, a struct with .T, ', ...
     'a struct with .forward, or a function handle.']);
end

function UV = callForward(fun, XY)
    % CALLFORWARD  Robustly call a forward mapping function.
    %
    %   UV = callForward(fun, XY)
    %
    % Inputs:
    %     fun - function handle mapping XY or (x,y) to outputs
    %     XY  - N×2 array of [x y] points
    %
    % Outputs:
    %     UV  - N×2 array of mapped [u v] points

    arguments
        fun function_handle
        XY (:,2) double
    end
    try
        UV = fun(XY);

        if ~isnumeric(UV) || size(UV, 2) ~= 2
            error('bad');
        end

    catch
        [u, v] = fun(XY(:, 1), XY(:, 2));
        UV = [u(:), v(:)];
    end

end

function UV = applyH(H, XY)
    % APPLYH  Apply a 3x3 homogeneous matrix to 2-D points (perspective divide).
    %
    %   UV = applyH(H, XY)
    %
    % Inputs:
    %     H  - 3×3 numeric homography matrix
    %     XY - N×2 array of [x y] points (column: x then y)
    %
    % Outputs:
    %     UV - N×2 array of mapped [u v] points after homogeneous divide

    arguments
        H (3,3) double
        XY (:,2) double
    end
    W = H * [XY.'; ones(1, size(XY, 1))];
    UV = [W(1, :) ./ W(3, :); W(2, :) ./ W(3, :)].';
end
