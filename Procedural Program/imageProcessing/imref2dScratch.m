function R = imref2dScratch(imageSize, xWorldLimits, yWorldLimits)
%IMREF2DSCRATCH  Re-implementation of imref2d([height,width], xLimits, yLimits)
%
%   R = imref2dScratch(imageSize)
%   R = imref2dScratch(imageSize, xWorldLimits, yWorldLimits)
%
%   imageSize     - [M N] or [M N P], number of rows, columns, (and channels)
%   xWorldLimits  - Optional [xmin xmax] (default = [0.5 N+0.5])
%   yWorldLimits  - Optional [ymin ymax] (default = [0.5 M+0.5])
%
%   Output R is a struct mimicking MATLAB's imref2d object:
%       R.ImageSize
%       R.XWorldLimits
%       R.YWorldLimits
%       R.PixelExtentInWorldX
%       R.PixelExtentInWorldY
%       R.XIntrinsicLimits
%       R.YIntrinsicLimits
%       R.WorldToIntrinsicFcn
%       R.IntrinsicToWorldFcn
%
%   Example:
%       R = imref2dScratch([200 300], [-150 150], [-100 100]);
%       [x,y] = R.intrinsicToWorld(1,1);

    arguments
        imageSize (1,:) {mustBeNumeric}
        xWorldLimits (1,:) {mustBeNumeric}
        yWorldLimits (1,:){mustBeNumeric}
    end

    narginchk(1, 3);

    % --- Validate image size ---
    if numel(imageSize) < 2
        error('imageSize must be at least [height width].');
    end
    M = imageSize(1);
    N = imageSize(2);

    % --- Default world limits (center of pixel at integer coords) ---
    if nargin < 2 || isempty(xWorldLimits)
        xWorldLimits = [0.5, N + 0.5];
    end
    if nargin < 3 || isempty(yWorldLimits)
        yWorldLimits = [0.5, M + 0.5];
    end

    % --- Pixel spacing in world coordinates ---
    pixelExtentInWorldX = diff(xWorldLimits) / N;
    pixelExtentInWorldY = diff(yWorldLimits) / M;

    % --- Intrinsic coordinate limits ---
    xIntrinsicLimits = [0.5, N + 0.5];
    yIntrinsicLimits = [0.5, M + 0.5];

    % --- Coordinate transform functions ---
    intrinsicToWorld = @(xi, yi) deal( ...
        xWorldLimits(1) + (xi - 0.5) * pixelExtentInWorldX, ...
        yWorldLimits(1) + (yi - 0.5) * pixelExtentInWorldY);

    worldToIntrinsic = @(xw, yw) deal( ...
        (xw - xWorldLimits(1)) / pixelExtentInWorldX + 0.5, ...
        (yw - yWorldLimits(1)) / pixelExtentInWorldY + 0.5);

    % --- Construct struct mimicking imref2d object ---
    R = struct( ...
        'ImageSize',           imageSize, ...
        'XWorldLimits',        xWorldLimits, ...
        'YWorldLimits',        yWorldLimits, ...
        'PixelExtentInWorldX', pixelExtentInWorldX, ...
        'PixelExtentInWorldY', pixelExtentInWorldY, ...
        'XIntrinsicLimits',    xIntrinsicLimits, ...
        'YIntrinsicLimits',    yIntrinsicLimits, ...
        'IntrinsicToWorldFcn', intrinsicToWorld, ...
        'WorldToIntrinsicFcn', worldToIntrinsic);
end
