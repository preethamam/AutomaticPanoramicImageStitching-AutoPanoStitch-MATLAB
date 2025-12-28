function [x_out, y_out] = transformPointsForwardScratch(tform, x, y)
% TRANSFORMPOINTSFORWARD  Forward-map 2-D points through a 3×3 transform.
%
%   [x_out, y_out] = transformPointsForwardScratch(tform, x, y)
%
% Inputs:
%   tform - Transform to apply. Supported forms:
%            * numeric 3×3 matrix (homography/projective),
%            * 1x1 cell containing a 3×3 matrix,
%            * struct with a 3×3 field named 'H' or 'T'.
%   x, y  - Vectors or arrays of identical size containing point coordinates
%           (x = column coordinates, y = row coordinates). Numeric types
%           (integer or floating) are accepted and are converted to double.
%
% Outputs:
%   x_out, y_out - Mapped coordinates with the same shape as inputs. Points
%                  with near-zero homogeneous denominator are returned as NaN.
%
% Notes:
%   * Mapping is performed row-wise using [x y 1] * H'.
%   * Accepts several container forms for convenience (cell, struct, numeric).
%
% Example:
%   H = [1 0.02 0; -0.01 1 0; 1e-4 -2e-4 1];
%   [xo, yo] = transformPointsForwardScratch(H, [1 640], [1 480]);
%
% See also: transformPointsInverse, projective2d

    arguments
        tform
        x {mustBeNumeric}
        y {mustBeNumeric}
    end

    % --- unwrap common containers ---
    if iscell(tform)
        % accept tforms(i) style if tforms is a cell array
        tform = tform{1};
    end

    if isstruct(tform)

        if isfield(tform, 'H') && isequal(size(tform.H), [3 3])
            H = tform.H;
        elseif isfield(tform, 'T') && isequal(size(tform.T), [3 3])
            H = tform.T;
        else
            error('Struct tform must contain a 3x3 field ''H'' or ''T''.');
        end

    elseif isnumeric(tform) && isequal(size(tform), [3 3])
        H = tform;
    else
        error('TFORM must be a 3x3 numeric matrix (or 1x1 cell/struct holding one).');
    end

    % --- shape & compute ---
    assert(isequal(size(x), size(y)), 'X and Y must have the same size.');
    osz = size(x);
    x = double(x(:)); y = double(y(:));
    n = numel(x);

    XY1 = [x y ones(n, 1)] * H.'; % N×3
    w = XY1(:, 3);

    % Guard homogeneous divide
    epsw = 1e-12;
    bad = abs(w) < epsw;
    w(~bad) = 1 ./ w(~bad);

    xf = XY1(:, 1) .* w;
    yf = XY1(:, 2) .* w;

    xf(bad) = NaN; yf(bad) = NaN;

    x_out = reshape(xf, osz);
    y_out = reshape(yf, osz);
end
