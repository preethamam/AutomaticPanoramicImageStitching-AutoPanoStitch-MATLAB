function [panorama, rgbAnnotation] = renderPanorama(input, images, imgSize, cameras, mode, refIdx, opts)
    % RENDERPANORAMA Render a panorama in the chosen projection with optional blending.
    %   [panorama, rgbAnnotation] = renderPanorama(images, cameras, mode, refIdx, opts)
    %   composes the set of input images onto a common panorama surface using the
    %   provided camera intrinsics/extrinsics. Supports multiple projection modes and
    %   blending strategies, with optional GPU acceleration and tiling for memory.
    %
    %   Inputs
    %   - images  : 1xN or Nx1 cell array of RGB images (uint8 or single).
    %   - imgSize : 1xN or Nx1 array of [height, width, channels] for each image.
    %   - cameras : 1xN or Nx1 struct array with fields R (3x3 world->cam) and K (3x3).
    %   - mode    : 'cylindrical' | 'spherical' | 'equirectangular' | 'planar' |
    %               'perspective' | 'stereographic'.
    %   - refIdx : Reference camera index (positive integer). Used to set planar frame
    %               and default panorama focal length.
    %   - opts    : Options struct (fields documented inline below) controlling focal
    %               length, resolution, tiling, GPU, blending, and annotations.
    %
    %   Outputs
    %   - panorama      : Rendered panorama image as uint8 [H x W x 3].
    %   - rgbAnnotation : Optional annotated panorama (only when enabled in opts),
    %                     else empty [].
    %
    %   Notes
    %   - For 'spherical' and 'equirectangular', the mapping is identical here.
    %   - 'stereographic' produces a fisheye-like projection centered on the reference view.
    %   - Use tiling and GPU options for large panoramas to control memory usage.
    %
    %   See also gainCompensation, multiBandBlending, gpuArray, gather

    arguments
        input (1, 1) struct
        images cell
        imgSize (:, 3) {mustBeNumeric, mustBeFinite, mustBePositive}
        cameras struct
        mode {mustBeTextScalar}
        refIdx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        opts (1, 1) struct = struct()
    end

    if nargin < 5, opts = struct(); end
    if ~isfield(opts, 'fPan'), opts.fPan = cameras(refIdx).K(1, 1); end % Seed focal length
    if ~isfield(opts, 'resScale'), opts.resScale = 1.0; end % Resolution scale factor
    if ~isfield(opts, 'anglePower'), opts.anglePower = 1; end % View-angle weight power
    if ~isfield(opts, 'cropBorder'), opts.cropBorder = true; end % Crop black border
    if ~isfield(opts, 'margin'), opts.margin = 0.01; end % Margin fraction on bounds
    if ~isfield(opts, 'useGPU'), opts.useGPU = true; end % Use GPU if available
    if ~isfield(opts, 'parfor'), opts.parfor = true; end % Use parfor for tiling
    if ~isfield(opts, 'tile'), opts.tile = []; end % [tileH tileW] or []
    if ~isfield(opts, 'maxMegapixel'), opts.maxMegapixel = 50; end % cap total pixels (e.g., 50 MP)
    if ~isfield(opts, 'robustPct'), opts.robustPct = [1 99]; end % percentile clip for planar bounds
    if ~isfield(opts, 'uvAbsCap'), opts.uvAbsCap = 8.0; end % |u|,|v| max in plane units (≈ FOV clamp)
    if ~isfield(opts, 'pixelPad'), opts.pixelPad = 24; end % extra border in **pixels** on panorama plane
    if ~isfield(opts, 'autoRef'), opts.autoRef = true; end % auto-pick best planar refIdx
    if ~isfield(opts, 'canvasColor'), opts.canvasColor = 'black'; end % 'black' | 'white'
    if ~isfield(opts, 'gainCompensation'), opts.gainCompensation = true; end % Gain compensation on/off
    if ~isfield(opts, 'sigmaN'), opts.sigmaN = 10.0; end % Noise stddev for gain comp.
    if ~isfield(opts, 'sigmag'), opts.sigmag = 0.1; end % Gain stddev for gain comp.
    if ~isfield(opts, 'blending'), opts.blending = 'multiband'; end % 'none' | 'linear' | 'multiband'
    if ~isfield(opts, 'overlapStride'), opts.overlapStride = 4; end % stride on panorama grid for overlap sampling
    if ~isfield(opts, 'pyrLevels'), opts.pyrLevels = 3; end % for multiband
    if ~isfield(opts, 'composeNonePolicy'), opts.composeNonePolicy = 'last'; end % pixel overwrite policy
    % 'last' | 'first' | 'maxangle'
    % "last" (default): later images overwrite earlier ones
    % "first": first valid source wins; later images don't overwrite filled pixels.
    % "maxangle": pick the single camera with the largest view-angle weight (still no blending; weight only decides which source wins).
    if ~isfield(opts, 'showPanoramaImgsNums'), opts.showPanoramaImgsNums = false; end % show image nums on pano
    if ~isfield(opts, 'showCropBoundingBox'), opts.showCropBoundingBox = false; end % show crop box on pano
    if ~isfield(opts, 'blendDevice'), opts.blendDevice = 'auto'; end % 'gpu' | 'cpu' | 'auto'
    if ~isfield(opts, 'tileMin'), opts.tileMin = [512 768]; end % lower bound for tiling
    if ~isfield(opts, 'gpuMemFrac'), opts.gpuMemFrac = 0.5; end % keep peak <55 % free memory

    % Get the number of images;
    numImages = numel(images);
    panorama = []; rgbAnnotation = [];

    % Check if the panorama has translation
    if cameras(1).noRotation == 1 || input.forcePlanarScan
        [panorama, rgbAnnotation] = pureNonRotationalPanoramas(images, cameras, numImages, opts);
        return
    end

    % ---- Auto-pick best planar refIdx by minimizing canvas area ----
    if (strcmpi(mode, 'planar') || strcmpi(mode, 'perspective') || strcmpi(mode, 'stereographic')) && opts.autoRef
        bestArea = inf; bestIdx = refIdx;

        for ii = 1:numImages
            Rrefi = cameras(ii).R;

            if strcmpi(mode, 'stereographic')
                [aMin, aMax, bMin, bMax] = stereographicBounds( ...
                    cameras, imgSize, Rrefi, opts.robustPct, opts.uvAbsCap);
                maxExtent = max([abs(aMin), abs(aMax), abs(bMin), abs(bMax)]);
                % apply same growth pattern
                maxExtent = maxExtent * (1 + 2 * opts.margin) + opts.pixelPad / opts.fPan;
                Wi = max(1, ceil(2 * opts.fPan * maxExtent * opts.resScale));
                Hi = Wi; % square
                areai = double(Wi) * double(Hi);
            else
                [uMin, uMax, vMin, vMax] = planarBounds( ...
                    cameras, imgSize, Rrefi, opts.robustPct, opts.uvAbsCap);

                % apply same growth as in planar case
                du = uMax - uMin; dv = vMax - vMin;
                uMin2 = uMin - opts.margin * du - opts.pixelPad / opts.fPan;
                uMax2 = uMax + opts.margin * du + opts.pixelPad / opts.fPan;
                vMin2 = vMin - opts.margin * dv - opts.pixelPad / opts.fPan;
                vMax2 = vMax + opts.margin * dv + opts.pixelPad / opts.fPan;

                Wi = max(1, ceil(opts.fPan * (uMax2 - uMin2) * opts.resScale));
                Hi = max(1, ceil(opts.fPan * (vMax2 - vMin2) * opts.resScale));
                areai = double(Wi) * double(Hi);
            end

            if areai < bestArea
                bestArea = areai; bestIdx = ii;
            end

        end

        refIdx = bestIdx; % use the best reference going forward
    end

    % -------- 1) Bounds on chosen surface (using R' = c2w) --------
    switch lower(mode)
        case 'cylindrical'
            [thetaMin, thetaMax, heightMin, heightMax] = cylindricalBounds(cameras, imgSize);
            dt = thetaMax - thetaMin; dh = heightMax - heightMin;
            thetaMin = thetaMin - opts.margin * dt; thetaMax = thetaMax + opts.margin * dt;
            heightMin = heightMin - opts.margin * dh; heightMax = heightMax + opts.margin * dh;

            W = max(1, ceil(opts.fPan * (thetaMax - thetaMin) * opts.resScale));
            H = max(1, ceil(opts.fPan * (heightMax - heightMin) * opts.resScale));

            th0 = thetaMin; h0 = heightMin;

        case 'spherical'
            [thetaMin, thetaMax, phiMin, phiMax] = sphericalBounds(cameras, imgSize);
            dt = thetaMax - thetaMin; dp = phiMax - phiMin;
            thetaMin = thetaMin - opts.margin * dt; thetaMax = thetaMax + opts.margin * dt;
            phiMin = phiMin - opts.margin * dp; phiMax = phiMax + opts.margin * dp;

            W = max(1, ceil(opts.fPan * (thetaMax - thetaMin) * opts.resScale));
            H = max(1, ceil(opts.fPan * (phiMax - phiMin) * opts.resScale));

            th0 = thetaMin; ph0 = phiMin;

        case {'planar', 'perspective'}
            % Tight bounds on the reference image plane (z_ref = +1)
            Rref = cameras(refIdx).R; % world->ref
            [uMin, uMax, vMin, vMax] = planarBounds( ...
                cameras, imgSize, Rref, opts.robustPct, opts.uvAbsCap);

            du = uMax - uMin; dv = vMax - vMin;
            uMin = uMin - opts.margin * du; uMax = uMax + opts.margin * du;
            vMin = vMin - opts.margin * dv; vMax = vMax + opts.margin * dv;

            % --- add small padding in pixel units on the panorama plane ---
            uMin = uMin - opts.pixelPad / opts.fPan;
            uMax = uMax + opts.pixelPad / opts.fPan;
            vMin = vMin - opts.pixelPad / opts.fPan;
            vMax = vMax + opts.pixelPad / opts.fPan;

            W = max(1, ceil(opts.fPan * (uMax - uMin) * opts.resScale));
            H = max(1, ceil(opts.fPan * (vMax - vMin) * opts.resScale));

            % --- global pixel cap (auto downscale if needed) ---
            maxPixel = round(opts.maxMegapixel * 1e6);
            HWest = double(H) * double(W);

            if HWest > maxPixel
                s = sqrt(maxPixel / HWest); % uniform scale on both axes
                opts.resScale = opts.resScale * s;
                W = max(1, ceil(opts.fPan * (uMax - uMin) * opts.resScale));
                H = max(1, ceil(opts.fPan * (vMax - vMin) * opts.resScale));
            end

            u0 = uMin; v0 = vMin; % lower-left in plane coords

        case {'equirectangular'} % alias of spherical
            [thetaMin, thetaMax, phiMin, phiMax] = sphericalBounds(cameras, imgSize);
            dt = thetaMax - thetaMin; dp = phiMax - phiMin;
            thetaMin = thetaMin - opts.margin * dt; thetaMax = thetaMax + opts.margin * dt;
            phiMin = phiMin - opts.margin * dp; phiMax = phiMax + opts.margin * dp;

            W = max(1, ceil(opts.fPan * (thetaMax - thetaMin) * opts.resScale));
            H = max(1, ceil(opts.fPan * (phiMax - phiMin) * opts.resScale));

            th0 = thetaMin; ph0 = phiMin;

        case 'stereographic'
            % Bounds on the stereographic plane (relative to the reference camera)
            Rref = cameras(refIdx).R; % world->ref
            [aMin, aMax, bMin, bMax] = stereographicBounds( ...
                cameras, imgSize, Rref, opts.robustPct, opts.uvAbsCap);

            % Force centered, square projection (like PTGui) ===
            maxExtent = max([abs(aMin), abs(aMax), abs(bMin), abs(bMax)]);
            aMin = -maxExtent;
            aMax = maxExtent;
            bMin = -maxExtent;
            bMax = maxExtent;

            da = aMax - aMin; db = bMax - bMin;
            aMin = aMin - opts.margin * da; aMax = aMax + opts.margin * da;
            bMin = bMin - opts.margin * db; bMax = bMax + opts.margin * db;

            % pixel padding in plane units
            aMin = aMin - opts.pixelPad / opts.fPan;
            aMax = aMax + opts.pixelPad / opts.fPan;
            bMin = bMin - opts.pixelPad / opts.fPan;
            bMax = bMax + opts.pixelPad / opts.fPan;

            W = max(1, ceil(opts.fPan * (aMax - aMin) * opts.resScale));
            H = max(1, ceil(opts.fPan * (bMax - bMin) * opts.resScale));

            % global pixel cap (same as planar)
            maxPixel = round(opts.maxMegapixel * 1e6);
            HWest = double(H) * double(W);

            if HWest > maxPixel
                s = sqrt(maxPixel / HWest);
                opts.resScale = opts.resScale * s;
                W = max(1, ceil(opts.fPan * (aMax - aMin) * opts.resScale));
                H = max(1, ceil(opts.fPan * (bMax - bMin) * opts.resScale));
            end

            u0 = aMin; v0 = bMin; % stereographic plane origin (same naming as planar)

        otherwise
            error('mode must be cylindrical, spherical, or planar/perspective');
    end

    % Choose device
    onGPU = opts.useGPU && (gpuDeviceCount > 0);

    % --- Estimate bytes we need ---
    % main float panorama (single) + logical covered + a working tile buffer margin
    bytesPano = double(H) * double(W) * 3 * 4; % single RGB
    bytesCover = double(H) * double(W) * 1; % logical ~1 byte
    bytesMargin = 128e6; % small slack (tune as you like)
    bytesNeeded = bytesPano + bytesCover + bytesMargin;

    % --- Allocate only after we know it fits ---
    try
        panorama = zeros(H, W, 3, 'single'); % ~12 bytes per pixel
        covered = false(H, W); % ~1 byte per pixel

        if onGPU
            panorama = gpuArray(panorama);
            covered = gpuArray(covered);
        end

    catch ME
        warning('renderPanorama:Alloc', ...
            'Skipping panorama (allocation failed): %s', ME.message);
        panorama = [];
        return
    end

    if ~canFit(bytesNeeded, onGPU)
        warning('renderPanorama:TooLarge', ...
            ['Skipping panorama: %dx%d requires ~%.2f GB; not enough %s memory.'], ...
            H, W, bytesNeeded / 1e9, tern(onGPU, 'GPU', 'CPU'));
        return
    end

    % ---------- Simple auto-tiler (mode-aware, streaming-safe) ----------
    if isempty(opts.tile)
        numImages = numel(images);
        % available memory
        if opts.useGPU && gpuDeviceCount > 0
            g = gpuDevice;
            avail = opts.gpuMemFrac * double(g.AvailableMemory); % bytes
        else
            % very rough CPU fallback: assume 6 GB usable (adjust if you like)
            avail = 6e9;
        end

        % bytes per pixel per image (single precision)
        % 'none' ≈ RGB only; 'linear' ≈ RGB + weight; 'multiband' ≈ add 1.33 pyr factor
        switch lower(opts.blending)
            case 'none', bpppi = 4 * (3); pyr = 1.0; % ~12 B/px/img
            case 'linear', bpppi = 4 * (3 + 1); pyr = 1.0; % ~16 B/px/img
            otherwise % 'multiband'
                bpppi = 4 * (3 + 1); pyr = 1.33; % ~21.3 B/px/img
        end

        % Use a safety factor (40% memory) and assume worst-case contributors = N
        bytesPerPix = numImages * bpppi * pyr; % bytes per pano-pixel for this tile
        pixelBudget = max(1, floor(0.4 * avail / bytesPerPix));
        side = floor(sqrt(pixelBudget));

        % clamp to sane bounds and image size
        lo = 512; hi = 2048; % good general-purpose limits
        side = max(lo, min([side, H, W, hi]));
        opts.tile = [side side];
    end

    % Warp weights for all images (for gain compensation and blending)
    srcW = warpWeights(images, numImages);

    if opts.gainCompensation

        switch lower(mode)
            case {'planar', 'perspective', 'stereographic'}
                tic;
                gains = gainCompensation(images, cameras, mode, refIdx, opts, ...
                    H, W, u0, v0, [], [], [], srcW);
                fprintf('Gain compensation planar/stereographic panorama rendering time : %f seconds\n', toc);

            case 'cylindrical'
                tic;
                gains = gainCompensation(images, cameras, mode, refIdx, opts, ...
                    H, W, [], [], th0, h0, [], srcW);
                fprintf('Gain compensation cylindrical panorama rendering time : %f seconds\n', toc);

            case {'spherical', 'equirectangular'}
                tic;
                gains = gainCompensation(images, cameras, mode, refIdx, opts, ...
                    H, W, [], [], th0, [], ph0, srcW);
                fprintf('Gain compensation spherical/equirectangular panorama rendering time : %f seconds\n', toc);

            otherwise
                error('Unsupported mode for gain compensation.');
        end

    else
        gains = ones(numImages, 3, 'single');
    end

    % Get signm and tile sizes
    if ~isfield(opts, 'pyrSigma'), opts.pyrSigma = 1.0; end
    tileH = opts.tile(1); tileW = opts.tile(2);

    % === ACCUMULATION & BLENDING (STREAMING, TILE-FIRST) =======================
    tic;
    % ---------------------- TILE LOOP (streaming) ------------------------------
    % You can parallelize **over tiles** (not over images) if you want:
    % parfor r = 1:tileH:H

    for r = 1:tileH:H
        rr = r:min(r + tileH - 1, H);

        for c = 1:tileW:W
            cc = c:min(c + tileW - 1, W);

            % --- build tile WORLD directions DWt ---
            [xpT, ypT] = meshgrid(single(cc - 1), single(rr - 1));

            switch lower(mode)
                case 'cylindrical'
                    theta = single(th0) + xpT / single(opts.fPan);
                    hline = single(h0) + ypT / single(opts.fPan);
                    dwx = sin(theta); dwy = hline; dwz = cos(theta);

                case {'spherical', 'equirectangular'}
                    theta = single(th0) + xpT / single(opts.fPan);
                    phi = single(ph0) + ypT / single(opts.fPan);
                    cphi = cos(phi); sphi = sin(phi);
                    dwx = cphi .* sin(theta); dwy = sphi; dwz = cphi .* cos(theta);

                case {'planar', 'perspective'}
                    u = single(u0) + xpT / single(opts.fPan);
                    v = single(v0) + ypT / single(opts.fPan);
                    dzRef = ones(size(u), 'like', u);
                    Rref = single(cameras(refIdx).R);
                    dwx = Rref(1, 1) .* u + Rref(2, 1) .* v + Rref(3, 1) .* dzRef;
                    dwy = Rref(1, 2) .* u + Rref(2, 2) .* v + Rref(3, 2) .* dzRef;
                    dwz = Rref(1, 3) .* u + Rref(2, 3) .* v + Rref(3, 3) .* dzRef;

                case 'stereographic'
                    a = single(u0) + xpT / single(opts.fPan);
                    b = single(v0) + ypT / single(opts.fPan);
                    r2 = a .* a + b .* b; denom = (1 + r2);
                    dxr = 2 * a ./ denom; dyr = 2 * b ./ denom; dzr = (1 - r2) ./ denom;
                    Rref = single(cameras(refIdx).R);
                    dwx = Rref(1, 1) .* dxr + Rref(2, 1) .* dyr + Rref(3, 1) .* dzr;
                    dwy = Rref(1, 2) .* dxr + Rref(2, 2) .* dyr + Rref(3, 2) .* dzr;
                    dwz = Rref(1, 3) .* dxr + Rref(2, 3) .* dyr + Rref(3, 3) .* dzr;
                otherwise
                    error('mode');
            end

            DWt = [dwx(:), dwy(:), dwz(:)];
            nrm = sqrt(sum(DWt .^ 2, 2));
            nrm = max(nrm, 1e-8); % guard
            DWt = DWt ./ nrm; % unit world rays

            if onGPU, DWt = gpuArray(DWt); end

            % --- fuse this tile immediately ---
            [FTile, WsumTile] = fuseTile(rr, cc, DWt, images, cameras, onGPU, srcW, gains, opts);

            % --- write back & coverage ---
            if ~onGPU && isa(FTile, 'gpuArray'), FTile = gather(FTile); end
            panorama(rr, cc, :) = FTile;

            if ~onGPU && isa(WsumTile, 'gpuArray'), WsumTile = gather(WsumTile); end
            covered(rr, cc) = covered(rr, cc) | (WsumTile > 0);

            % CLEANUP after each tile (add this)
            clear DWt FTile WsumTile xpT ypT dwx dwy dwz
        end

    end

    maskVoid = ~covered; % [H x W] logical

    fprintf('Streaming %s blending panorama rendering time : %f seconds\n', opts.blending, toc);
    % === END ACCUMULATION & BLENDING (STREAMING) ===============================

    % Paint the empty pixels to the requested canvas color
    if strcmpi(opts.canvasColor, 'white')
        % expand mask to 3 channels and set to 1.0 (before uint8 conversion)
        panorama(repmat(maskVoid, [1 1 3])) = 1;
    else
        % black canvas is already zero; nothing needed
        % (optional) ensure zeros where void:
        panorama(repmat(maskVoid, [1 1 3])) = 0;
    end

    % Gather & convert
    if onGPU, panorama = gather(panorama); end
    panorama = uint8(max(0, min(255, round(255 * panorama))));

    % -------- 4) Optional crop --------
    cropRectangle = [1 size(panorama, 1) 1 size(panorama, 2)]; % default (no shift)

    if opts.cropBorder
        [panorama, cropRectangle, ~] = cropNonzeroBbox(panorama, opts.canvasColor);
    end

    r1 = cropRectangle(1); c1 = cropRectangle(3); % top and left of the crop
    dx = c1 - 1; dy = r1 - 1; % shift to apply to x/y

    % --- Draw debugging annotations -----------------------------------------
    if opts.showPanoramaImgsNums && opts.showCropBoundingBox

        switch lower(mode)
            case {'planar', 'perspective', 'stereographic'}
                [xBoxes, yBoxes, centers] = allWarpedBoxes( ...
                    images, cameras, mode, refIdx, opts, u0, v0, [], [], []);
            case 'cylindrical'
                [xBoxes, yBoxes, centers] = allWarpedBoxes( ...
                    images, cameras, mode, refIdx, opts, [], [], th0, h0, []);
            case {'spherical', 'equirectangular'}
                [xBoxes, yBoxes, centers] = allWarpedBoxes( ...
                    images, cameras, mode, refIdx, opts, [], [], th0, [], ph0);
        end

        % Shift by crop origin (panorama was cropped after coords were computed)
        xBoxes = cellfun(@(x) x - dx, xBoxes, 'uni', 0);
        yBoxes = cellfun(@(y) y - dy, yBoxes, 'uni', 0);
        centers = centers - [dx dy];

        % Validate, pack, and draw
        isValid = cellfun(@(x, y) numel(x) == numel(y) && numel(x) >= 3 && ...
            all(isfinite(x)) && all(isfinite(y)), ...
            xBoxes, yBoxes);

        xBoxesV = xBoxes(isValid);
        yBoxesV = yBoxes(isValid);
        centersV = centers(isValid, :);

        polyCells = cellfun(@(x, y) reshape([x(:) y(:)]', 1, []), ...
            xBoxesV, yBoxesV, 'uni', 0);

        % Generate N random colors (each row is [R G B])
        colorMat = brightColors(numImages);

        rgbAnnotation = insertShape(panorama, 'Polygon', polyCells, ...
            'Color', colorMat, 'LineWidth', 2);
        labels = arrayfun(@num2str, 1:numel(xBoxesV), 'uni', 0);
        rgbAnnotation = insertText(rgbAnnotation, centersV, labels, ...
            'FontSize', 24, 'BoxColor', 'red', 'TextColor', 'white');
    end

    % ========== GPU MEMORY CLEANUP (add before final 'end') ==========
    if onGPU
        % Clear any remaining GPU variables
        clear DWt FTile WsumTile covered

        % Wait for all GPU operations to complete
        wait(gpuDevice);
    end

end

function colorMat = brightColors(N)
    % BRIGHTCOLORS  Generate N vivid, bright RGB colors (uint8).
    %
    %   colorMat = brightColors(N)
    %   returns an N-by-3 uint8 matrix of saturated, high-intensity colors.
    %
    % Inputs:
    %   N - positive integer number of colors to generate
    %
    % Outputs:
    %   colorMat - N-by-3 uint8 matrix of RGB colors

    arguments
        N (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    % Random base colors
    lineRGB = rand(N, 3);

    % Normalize each row so the maximum channel becomes 1
    maxChannel = max(lineRGB, [], 2);
    maxChannel(maxChannel == 0) = eps; % safety guard

    normalized = lineRGB ./ maxChannel;

    % Scale to RGB 0–255
    colorMat = uint8(255 * normalized);
end

function [panorama, rgbAnnotation] = pureNonRotationalPanoramas(images, cameras, numImages, opts)
    % PURENONROTATIONALPANORAMAS  Render a panorama for pure non-rotational cameras.
    %
    %   [panorama, rgbAnnotation] = pureNonRotationalPanoramas(images, cameras, numImages, opts)
    %
    % Inputs:
    %     images    - cell array of source images
    %     cameras   - struct array of camera data (include H2refined fields)
    %     numImages - scalar number of images
    %     opts      - options struct
    %
    % Outputs:
    %     panorama      - rendered panorama image (uint8)
    %     rgbAnnotation - optional annotated panorama or []

    arguments
        images cell
        cameras struct
        numImages (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        opts (1, 1) struct
    end

    % Initialize the empty panorama and rgbAnnotation
    panorama = []; rgbAnnotation = [];

    % Get homography tforms
    tforms = {cameras(:).H2refined};

    % Set the panorama view limits
    xlim = zeros(numImages, 2);
    ylim = zeros(numImages, 2);

    parfor i = 1:numImages
        h = size(images{i}, 1);
        w = size(images{i}, 2);
        [xlim(i, :), ylim(i, :)] = outputLimitsScratch(tforms{i}, [1, w], [1, h]);
    end

    % Find the minimum and maximum output limits
    xMin = min(xlim(:, 1));
    xMax = max(xlim(:, 2));
    yMin = min(ylim(:, 1));
    yMax = max(ylim(:, 2));

    % Width and height of panorama
    width = round(xMax - xMin);
    height = round(yMax - yMin);

    % Create a 2-D spatial reference object defining the size of the panorama.
    xLimits = [xMin, xMax];
    yLimits = [yMin, yMax];

    if opts.useMATLABImwarp == 1
        panoramaoutputView = imref2d([height, width], xLimits, yLimits);
    else
        panoramaoutputView = imref2dScratch([height, width], xLimits, yLimits);
    end

    % Warp weights for all images (for gain compensation and blending)
    srcWeights = warpWeights(images, numImages);

    % Warp the pure non-rotational images
    [Iw, Ww, xBoxes, yBoxes, centers] = pureNonRotationalImagesToCanvas(images, tforms, ...
        panoramaoutputView, srcWeights, opts);

    % Gain compensation
    tic;

    if opts.gainCompensation
        gains = gainCompensationH(Iw, Ww, opts);
    else
        gains = ones(numImages, 3, 'single');
    end

    fprintf('Gain compensation planar (non-rotational) panorama rendering time : %f seconds\n', toc);

    % Apply gain compensation
    for k = 1:numel(Iw)

        if ~isempty(Iw{k})
            Iw{k} = bsxfun(@times, Iw{k}, reshape(gains(k, :), 1, 1, 3));
        end

    end

    tic;

    switch lower(opts.blending)
        case 'none'
            % Winner-take-all compositing
            Wstack = cat(3, Ww{:});
            [~, idx] = max(Wstack, [], 3);
            Istack = cat(4, Iw{:});
            [H, W, C, numImages] = size(Istack);

            % Create one-hot masks and blend
            masks = reshape(idx == reshape(1:numImages, 1, 1, numImages), H, W, 1, numImages);
            panorama = sum(Istack .* masks, 4);
        case 'linear'
            panorama = linearBlending(Iw, Ww);
        case 'multiband'

            try
                useGPUBlend = parallel.gpu.GPUDevice.isAvailable;
            catch
                useGPUBlend = false;
            end

            panorama = multiBandBlending(Iw, Ww, opts.pyrLevels, useGPUBlend, opts.pyrSigma);
        otherwise
            error('Wrong blening mode.')
    end

    fprintf('Streaming %s blending panorama rendering time : %f seconds\n', opts.blending, toc);

    % Get background mask
    Wstack = cat(3, Ww{:}); % H x W x N
    maskVoid = ~any(Wstack > 0, 3); % H x W

    % Paint the empty pixels to the requested canvas color
    if strcmpi(opts.canvasColor, 'white')
        % expand mask to 3 channels and set to 1.0 (before uint8 conversion)
        panorama(repmat(maskVoid, [1 1 3])) = 1;
    else
        % black canvas is already zero; nothing needed
        % (optional) ensure zeros where void:
        panorama(repmat(maskVoid, [1 1 3])) = 0;
    end

    % Gather & convert
    if isgpuarray(panorama), panorama = gather(panorama); end
    panorama = uint8(max(0, min(255, round(255 * panorama))));

    % --- Draw debugging annotations -----------------------------------------
    if opts.showPanoramaImgsNums && opts.showCropBoundingBox
        % Validate, pack, and draw
        isValid = cellfun(@(x, y) numel(x) == numel(y) && numel(x) >= 3 && ...
            all(isfinite(x)) && all(isfinite(y)), ...
            xBoxes, yBoxes);

        xBoxesV = xBoxes(isValid);
        yBoxesV = yBoxes(isValid);
        centersV = centers(isValid, :);

        polyCells = cellfun(@(x, y) reshape([x(:) y(:)]', 1, []), ...
            xBoxesV, yBoxesV, 'uni', 0);

        % Generate N random colors (each row is [R G B])
        colorMat = brightColors(numImages);

        % Annotation of bounding boxes
        rgbAnnotation = insertShape(panorama, 'Polygon', polyCells, ...
            'Color', colorMat, 'LineWidth', 2);
        labels = arrayfun(@num2str, 1:numel(xBoxesV), 'uni', 0);

        % Annotation of image numbers
        rgbAnnotation = insertText(rgbAnnotation, centersV, labels, ...
            'FontSize', 24, 'BoxColor', 'red', 'TextColor', 'white');
    end

    % Clear GPU arrays if used
    clear Iw Ww Wstack
end

function [Iw, Ww, xBoxes, yBoxes, centers] = pureNonRotationalImagesToCanvas(images, tforms, outputView, srcWeights, opts)
    % pureNonRotationalImagesToCanvas  Warp all images to a common canvas.
    %
    %   [Iw, Ww, xBoxes, yBoxes, centers] = pureNonRotationalImagesToCanvas(images, tforms, outputView, srcWeights, opts)
    %
    % Inputs:
    %   images     - cell array of source images
    %   tforms     - cell array of transforms (3x3 matrices or projective types)
    %   outputView - imref2d-like struct/object describing canvas size and limits
    %   srcWeights - (optional) cell array of per-image weight maps
    %   opts       - options struct with fields controlling imwarp fallback and GPU usage
    %
    % Outputs:
    %   Iw      - cell array of warped images (single)
    %   Ww      - cell array of warped weight maps (single)
    %   xBoxes  - cell array of projected polygon X coordinates per image
    %   yBoxes  - cell array of projected polygon Y coordinates per image
    %   centers - Nx2 matrix of centroid positions on the canvas

    arguments
        images cell
        tforms cell
        outputView
        srcWeights cell
        opts (1, 1) struct
    end

    if nargin < 4
        srcWeights = [];
    end

    numImages = numel(images);
    Iw = cell(numImages, 1);
    Ww = cell(numImages, 1);

    % ---- choose CPU/GPU here ----
    try
        useGPU = parallel.gpu.GPUDevice.isAvailable;
    catch
        useGPU = false;
    end

    % Canvas size (same for all)
    outSize = outputView.ImageSize;
    Hc = outSize(1);
    Wc = outSize(2);
    xBoxes = cell(numImages, 1); yBoxes = cell(numImages, 1); centers = zeros(numImages, 2);

    for k = 1:numImages
        % Get image
        Ik = images{k};

        % Get warped image corners
        [imRows, imCols, ~] = size(Ik);

        imCorners = [1, 1; 1, imCols; imRows, imCols; imRows, 1; 1, 1];
        [x, y] = transformPointsForwardScratch(tforms{k}, imCorners(:, 2), imCorners(:, 1));
        xBoxes{k} = x - outputView.XWorldLimits(1);
        yBoxes{k} = y - outputView.YWorldLimits(1);
        centers(k, :) = [mean(xBoxes{k}), mean(yBoxes{k})];

        % ---- normalize to single [0,1] or single arbitrary ----
        if isa(Ik, 'uint8')
            Ik = single(Ik) / 255;
        else
            Ik = single(Ik);
        end

        % Optional: move to GPU
        if useGPU && ~isa(Ik, 'gpuArray')
            Ik = gpuArray(Ik);
        end

        % ---- warp color with NaN outside ----
        % Get tforms{k} is a plain 3x3 H,
        % assume numeric 3x3 homography
        T = tforms{k};

        if opts.useMATLABImwarp == 1
            Iwk = imwarp(Ik, projtform2d(T), ...
                'OutputView', outputView, ...
                'Interp', 'linear', ...
                'FillValues', 0);
        else
            Iwk = imageWarp(Ik, T, outputView);
        end

        % Ensure size matches canvas (robustness vs rounding)
        if size(Iwk, 1) ~= Hc || size(Iwk, 2) ~= Wc
            Iwk = Iwk(1:Hc, 1:Wc, :);
        end

        % ---- build / warp weight map ----
        if ~isempty(srcWeights)
            % User-provided source-domain weight
            Ws = srcWeights{k};
            Ws = single(Ws);

            if useGPU && ~isa(Ws, 'gpuArray')
                Ws = gpuArray(Ws);
            end

            if opts.useMATLABImwarp == 1
                Wk = imwarp(Ws, projtform2d(T), ...
                    'OutputView', outputView, ...
                    'Interp', 'linear', ...
                    'FillValues', 0);
            else
                Wk = imageWarp(Ws, T, outputView);
            end

        else
            % Simple coverage mask: warp ones
            [h, w, ~] = size(images{k});
            onesSrc = ones(h, w, 'like', Ik(:, :, 1)); % HxW, same type as Ik

            if useGPU && ~isa(onesSrc, 'gpuArray')
                onesSrc = gpuArray(onesSrc);
            end

            if opts.useMATLABImwarp == 1
                Wk = imwarp(onesSrc, projtform2d(T), ...
                    'OutputView', outputView, ...
                    'Interp', 'linear', ...
                    'FillValues', 0);
            else
                Wk = imageWarp(onesSrc, T, outputView);
            end

        end

        % Clamp to [0,1] and force single
        Wk = single(max(0, min(1, Wk)));

        % ---- store results (gather back to CPU to keep rest of pipeline simple) ----
        Iw{k} = Iwk;
        Ww{k} = Wk;
    end

end

% Per-blending-mode tile worker -------------------------------------------
function [FTile, WsumTile] = fuseTile(rr, cc, DWt, images, cameras, onGPU, srcW, gains, opts)
    % FUSETILE Fuse a panorama tile using the selected blending strategy.
    %   [FTile, WsumTile] = fuseTile(rr, cc, DWt, images, cameras, onGPU, srcW, gains, opts)
    %   computes the blended RGB tile and a coverage indicator for the region
    %   specified by rr (rows) and cc (cols) using the world direction vectors DWt.
    %
    %   FTile is [ht x wt x 3] single; WsumTile indicates coverage.
    %
    % Inputs:
    %   rr      - vector of row indices for this tile (1-based)
    %   cc      - vector of column indices for this tile (1-based)
    %   DWt     - T×3 array of unit world direction vectors for each pano pixel
    %   images  - cell array of source images
    %   cameras - struct array of camera intrinsics/extrinsics
    %   onGPU   - logical flag indicating GPU usage
    %   srcW    - cell array of per-image source weight maps
    %   gains   - N×3 array of per-image RGB gains
    %   opts    - options struct controlling blending and policies
    %
    % Outputs:
    %   FTile    - ht×wt×3 single RGB tile (blended)
    %   WsumTile - ht×wt single coverage / accumulated weight map

    arguments
        rr (:, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        cc (:, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        DWt (:, 3) {mustBeNumeric}
        images cell
        cameras struct
        onGPU (1, 1) logical
        srcW cell
        gains (:, 3) {mustBeNumeric}
        opts (1, 1) struct
    end

    N = numel(images);
    ht = numel(rr); wt = numel(cc);

    switch lower(opts.blending)
        case 'none'
            % For 'none', we implement policies streaming:
            %  - 'last'    : later images overwrite earlier ones
            %  - 'first'   : first valid wins
            %  - 'maxangle': select by largest angle weight
            outTile = zeros(ht, numel(cc), 3, 'single');
            fillTile = false(ht, numel(cc));

            if strcmpi(opts.composeNonePolicy, 'maxangle')
                bestW = zeros(ht, numel(cc), 'single');
            end

            for iImg = 1:N
                [Si, Mi, Wang] = sampleOneTile(images{iImg}, cameras(iImg), ...
                    DWt, ht, wt, onGPU, ...
                    opts.anglePower, srcW{iImg}, gains(iImg, :));
                Simg = reshape(Si, [ht wt 3]);
                Mimg = reshape(Mi, [ht wt]);

                assert(isequal(size(Simg), [ht wt 3]));
                assert(isequal(size(Mimg), [ht wt]));

                switch lower(opts.composeNonePolicy)
                    case 'last'
                        mask2d = Mimg;
                        m3 = repmat(mask2d, [1 1 3]); % [ht x wt x 3]
                        outTile(m3) = Simg(m3); % write all 3 channels
                        fillTile = fillTile | mask2d;

                    case 'first'
                        mask2d = Mimg & ~fillTile;
                        m3 = repmat(mask2d, [1 1 3]);
                        outTile(m3) = Simg(m3);
                        fillTile(mask2d) = true;

                    case 'maxangle'
                        Wimg = reshape(Wang, [ht wt]);
                        assert(isequal(size(Wimg), [ht wt]));
                        upd2d = Mimg & (Wimg > bestW);
                        upd3 = repmat(upd2d, [1 1 3]);
                        outTile(upd3) = Simg(upd3);
                        bestW(upd2d) = Wimg(upd2d);
                        fillTile(upd2d) = true;
                    otherwise
                        error('composeNonePolicy must be ''last'',''first'' or ''maxangle''.');
                end

            end

            FTile = outTile;
            WsumTile = single(fillTile); % only used for coverage

        case 'linear'
            ht = numel(rr); wt = numel(cc);

            if onGPU
                accumTile = gpuArray.zeros(ht, wt, 3, 'single');
                wsumTile = gpuArray.zeros(ht, wt, 1, 'single');
            else
                accumTile = zeros(ht, wt, 3, 'single');
                wsumTile = zeros(ht, wt, 1, 'single');
            end

            anyValid = false(ht, wt);
            bestW = zeros(ht, wt, 'single');
            bestRGB = zeros(ht, wt, 3, 'single');

            for iImg = 1:N
                [Si, Mi, Wang, Wf] = sampleOneTile(images{iImg}, cameras(iImg), ...
                    DWt, ht, wt, onGPU, ...
                    opts.anglePower, srcW{iImg}, gains(iImg, :));
                if ~any(Mi), continue; end

                % safe feather (clamp to [eps,1])
                Wf(~isfinite(Wf)) = 0;
                Wf = max(Wf, single(1e-4));

                wVec = Wang .* Wf; wVec(~Mi) = 0;

                Simg = reshape(Si, [ht wt 3]);
                Wimg = reshape(wVec, [ht wt]);
                Mimg = reshape(Mi, [ht wt]);

                accumTile = accumTile + bsxfun(@times, Simg, Wimg);
                wsumTile = wsumTile + Wimg;

                anyValid = anyValid | Mimg; % << NEW

                better = Mimg & (Wimg > bestW);

                if any(better, 'all')
                    bestW(better) = Wimg(better);
                    b3 = repmat(better, [1 1 3]);
                    bestRGB(b3) = Simg(b3);
                end

            end

            FTile = zeros(ht, wt, 3, 'single');
            z = (wsumTile > 1e-12);

            if any(z, 'all')
                wsafe = wsumTile; wsafe(~z) = 1;
                FTile = bsxfun(@rdivide, accumTile, wsafe);
            end

            needFallback = ~z & anyValid; % << per-pixel fallback

            if any(needFallback, 'all')
                nf3 = repmat(needFallback, [1 1 3]);
                FTile(nf3) = bestRGB(nf3);
            end

            FTile = gather(FTile);
            WsumTile = gather(wsumTile);

        case 'multiband'
            ht = numel(rr); wt = numel(cc);
            Ci = {}; Wc = {};
            anyValid = false(ht, wt);

            for iImg = 1:N
                [Si, Mi, Wang, Wf] = sampleOneTile(images{iImg}, cameras(iImg), ...
                    DWt, ht, wt, onGPU, ...
                    opts.anglePower, srcW{iImg}, gains(iImg, :));
                if ~any(Mi), continue; end

                % same safe weight as linear
                wVec = Wang .* Wf; wVec(~Mi) = 0;

                Simg = reshape(Si, [ht wt 3]);
                Wimg = reshape(wVec, [ht wt]);

                Ci{end + 1} = Simg; %#ok<AGROW>
                Wc{end + 1} = Wimg; %#ok<AGROW>
                anyValid = anyValid | (Wimg > 0);
            end

            if isempty(Ci)
                FTile = zeros(ht, wt, 3, 'single');
                WsumTile = zeros(ht, wt, 'single');
                return;
            end

            % Normalize masks to sum~1 (avoid division by 0 outside contributors)
            sumW = zeros(ht, wt, 'like', Wc{1});
            for k = 1:numel(Wc), sumW = sumW + Wc{k}; end
            sumWnz = sumW > 1e-8;
            invSum = zeros(ht, wt, 'like', sumW);
            invSum(sumWnz) = 1 ./ sumW(sumWnz);

            for k = 1:numel(Wc)
                Wc{k} = Wc{k} .* invSum; % now \sum_k Wc{k} ≈ 1 where there are contributors
            end

            % Device choice (as you had)
            useGPUBlend = onGPU;
            if strcmpi(opts.blendDevice, 'cpu'), useGPUBlend = false; end

            if strcmpi(opts.blendDevice, 'auto') && onGPU
                g = gpuDevice; Ktile = numel(Ci);
                needBytes = 4 * 1.33 * Ktile * ht * wt * 4;
                if needBytes > 0.45 * g.AvailableMemory, useGPUBlend = false; end
            end

            if ~useGPUBlend

                for k = 1:numel(Ci)
                    if isa(Ci{k}, 'gpuArray'), Ci{k} = gather(Ci{k}); end
                    if isa(Wc{k}, 'gpuArray'), Wc{k} = gather(Wc{k}); end
                end

            end

            FTile = multiBandBlending(Ci, Wc, opts.pyrLevels, useGPUBlend, opts.pyrSigma);

            % coverage: any place that had at least one contributor is "covered"
            WsumTile = single(anyValid);

            % Accumulate WsumTile
            for k = 1:numel(Wc), WsumTile = WsumTile + Wc{k}; end

        otherwise
            error('Unknown opts.blending.');
    end

    % At the END of the function, before return:
    if onGPU
        % Ensure outputs are on CPU
        if isa(FTile, 'gpuArray'), FTile = gather(FTile); end
        if isa(WsumTile, 'gpuArray'), WsumTile = gather(WsumTile); end

        % Clear intermediate GPU variables
        clear accumTile wsumTile bestW bestRGB Ci Wc
    end

end

% Per-image per-tile sampler (streaming) -----------------------------------
function [Svec, Mvec, Wangvec, Wfvec] = sampleOneTile( ...
        I, cam, DWt, ht, wt, onGPU, anglePow, srcWi, gaini)
    % SAMPLEONETILE Sample one image onto a panorama tile with weights and mask.
    %   [Svec, Mvec, Wangvec, Wfvec] = sampleOneTile(I, cam, DWt, ht, wt, onGPU, anglePow, srcWi, gaini)
    %   projects world directions DWt into image I using camera cam, then samples
    %   colors and weights. Returns flattened vectors for colors Svec, validity
    %   mask Mvec, angle-based weights Wangvec, and feather weights Wfvec.
    %
    % Inputs:
    %   I        - HxWx3 source image (uint8 or single)
    %   cam      - camera struct containing fields R and K
    %   DWt      - [T x 3] world direction vectors for tile pixels
    %   ht, wt   - tile height and width (scalars)
    %   onGPU    - logical flag: run on GPU
    %   anglePow - scalar angle power for view-angle weighting
    %   srcWi    - source weight map (HxW) for this image
    %   gaini    - 1x3 RGB multiplicative gain for this image
    %
    % Outputs:
    %   Svec    - [T x 3] sampled RGB colors (flattened)
    %   Mvec    - [T x 1] logical valid mask (finite samples & front-facing)
    %   Wangvec - [T x 1] angle-based (cosine) weights
    %   Wfvec   - [T x 1] feather weights sampled from srcWi

    arguments
        I (:, :, :) {mustBeNumeric}
        cam (1, 1) struct
        DWt (:, 3) {mustBeNumeric}
        ht (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        wt (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        onGPU (1, 1) logical
        anglePow (1, 1) {mustBeNumeric, mustBeFinite}
        srcWi (:, :) {mustBeNumeric}
        gaini (1, 3) {mustBeNumeric}
    end

    if ~isfield(cam, 'R') || ~isfield(cam, 'K')
        error('sampleOneTile:CamFields', 'cam must contain fields R and K.');
    end

    % Convert & gain
    if isa(I, 'uint8'), Ic = single(I) / 255; else, Ic = single(I); end
    Ic = bsxfun(@times, Ic, reshape(gaini, 1, 1, 3));
    if onGPU, Ic = gpuArray(Ic); end

    % Project WORLD dirs to this camera
    R = single(cam.R); K = single(cam.K);
    fx = K(1, 1); fy = K(2, 2); cx = K(1, 3); cy = K(2, 3);

    dirc = DWt * R.'; % [T x 3] in *camera* frame
    cxW = dirc(:, 1); cyW = dirc(:, 2); czW = dirc(:, 3);

    epsZ = single(1e-6);
    front = czW > epsZ;
    cz = max(czW, epsZ);

    u = fx * (cxW ./ cz) + cx;
    v = fy * (cyW ./ cz) + cy;

    % View-angle weight (cosine falloff)
    % For w2c R, camera +Z in WORLD coords is fw_world = R' * [0;0;1] = (R(3,:)).'
    % ---- in sampleOneTile() BEFORE computing Wangvec ----
    fw = R(3, :).'; % camera forward in WORLD coords
    Wangvec = max(0, DWt * fw) .^ single(anglePow); % [T x 1]
    Wangvec = Wangvec .* single(front); % zero out back-facing

    % Samples
    Svec = sampleBlock(Ic, u, v, ht, wt, onGPU); % [T x 3]
    Wf = sampleBlock(srcWi, u, v, ht, wt, onGPU); % [T x 1]
    Wfvec = Wf(:, 1); % feather only

    % Validity mask (independent of weights)
    Mvec = all(isfinite(Svec), 2) & Wangvec > 0; % discard back-facing

    % ---- ADD THIS BLOCK (exactly like the OOM path's behavior) ----
    bad = ~Mvec; % invalid or out-of-bounds

    if any(bad)
        Svec(bad, :) = 0; % kill NaNs in color
        Wfvec(bad) = 0; % kill NaNs in feather
        Wangvec(bad) = 0; % ensure zero weight everywhere bad
    end

end

function [xBoxes, yBoxes, centers] = allWarpedBoxes(images, cameras, mode, refIdx, opts, u0, v0, th0, h0, ph0)
    % ALLWARPEDBOXES Compute projected image boxes and centers on the panorama.
    %   [xBoxes, yBoxes, centers] = allWarpedBoxes(images, cameras, mode, refIdx, opts, ...)
    %   returns polygon coordinates and centers for each input image in panorama
    %   coordinates for annotation.
    %
    % Inputs:
    %   images  - cell array of source images
    %   cameras - struct array of camera params
    %   mode    - projection mode string
    %   refIdx  - reference camera index
    %   opts    - options struct
    %   u0,v0,th0,h0,ph0 - optional origin parameters depending on mode
    %
    % Outputs:
    %   xBoxes  - cell array of polygon X coordinates (per-image)
    %   yBoxes  - cell array of polygon Y coordinates (per-image)
    %   centers - Nx2 matrix of polygon centroids

    arguments
        images cell
        cameras struct
        mode {mustBeTextScalar}
        refIdx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        opts (1, 1) struct
        u0
        v0
        th0
        h0
        ph0
    end

    N = numel(images);
    xBoxes = cell(N, 1); yBoxes = cell(N, 1); centers = zeros(N, 2);

    for i = 1:N
        [xPoly, yPoly, ctr] = warpedBBoxes(size(images{i}), cameras(i), mode, ...
            refIdx, cameras, opts, u0, v0, th0, h0, ph0, opts.fPan);
        xBoxes{i} = xPoly; yBoxes{i} = yPoly; centers(i, :) = ctr;
    end

end

function [xPoly, yPoly, centroid] = warpedBBoxes(imageSize, cam, mode, ...
        refIdx, cameras, opts, u0, v0, th0, h0, ph0, fPan)
    % WARPEDBBOXES Project the image boundary into panorama coordinates.
    %   [xPoly, yPoly, centroid] = warpedBBoxes(imageSize, cam, mode, refIdx, cameras, opts, ...)
    %   computes the 4-corner polygon and its centroid on the panorama surface.
    %
    % Inputs:
    %   imageSize - [H W C] size of the source image
    %   cam       - camera struct for this image
    %   mode      - projection mode string
    %   refIdx    - index of reference camera (for planar/stereographic)
    %   cameras   - struct array of all cameras
    %   opts      - options struct
    %   u0,v0,th0,h0,ph0 - optional origin parameters
    %   fPan      - panorama focal length (scalar)
    %
    % Outputs:
    %   xPoly    - 1x5 vector of X polygon coordinates (closed loop)
    %   yPoly    - 1x5 vector of Y polygon coordinates (closed loop)
    %   centroid - 1x2 centroid coordinate [x y]

    arguments
        imageSize (:, :) {mustBeNumeric}
        cam (1, 1) struct
        mode {mustBeTextScalar}
        refIdx (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        cameras struct
        opts (1, 1) struct
        u0
        v0
        th0
        h0
        ph0
        fPan (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    Hc = imageSize(1); Wc = imageSize(2);
    corners = [1, 1; 1, Wc; Hc, Wc; Hc, 1]; % [row, col]
    xy1 = [corners(:, 2)'; corners(:, 1)'; ones(1, 4)]; % [u;v;1] (col = x, row = y)

    rayC = cam.K \ xy1;
    rayW = cam.R' * rayC; % to world

    % Map world ray to panorama coords (u,v) or (theta,h) or (theta,phi)
    switch lower(mode)
        case 'planar'
            Rref = cameras(refIdx).R;
            rayR = Rref * rayW; zr = rayR(3, :);
            ur = rayR(1, :) ./ zr; vr = rayR(2, :) ./ zr; % plane coords
            x = (ur - u0) * fPan; y = (vr - v0) * fPan;

        case 'cylindrical'
            xw = rayW(1, :); yw = rayW(2, :); zw = rayW(3, :);
            theta = atan2(xw, zw); h = yw ./ hypot(xw, zw);
            x = (theta - th0) * fPan; y = (h - h0) * fPan;

        case 'spherical'
            xw = rayW(1, :); yw = rayW(2, :); zw = rayW(3, :);
            theta = atan2(xw, zw); phi = atan2(yw, hypot(xw, zw));
            x = (theta - th0) * fPan; y = (phi - ph0) * fPan;

        case 'equirectangular' % alias of 'spherical'
            xw = rayW(1, :); yw = rayW(2, :); zw = rayW(3, :);
            theta = atan2(xw, zw); phi = atan2(yw, hypot(xw, zw));
            x = (theta - th0) * fPan; y = (phi - ph0) * fPan;

        case 'stereographic'
            % Project to the reference camera frame first
            Rref = cameras(refIdx).R;
            rayR = Rref * rayW; % to ref camera frame
            % normalize direction
            nr = sqrt(sum(rayR .^ 2, 1));
            xr = rayR(1, :) ./ nr; yr = rayR(2, :) ./ nr; zr = rayR(3, :) ./ nr;

            % Stereographic forward map (plane tangent at +Z; project from -Z):
            % a = x / (1 + z),  b = y / (1 + z)
            denom = 1 + zr;
            % guard degenerate (z ~ -1)
            denom = max(denom, 1e-6);
            a = xr ./ denom;
            b = yr ./ denom;

            x = (a - u0) * fPan;
            y = (b - v0) * fPan;
        otherwise , error('mode');
    end

    xPoly = [x, x(1)]; yPoly = [y, y(1)];
    centroid = [mean(x), mean(y)];
end

function srcW = warpWeights(images, N)
    % WARPWEIGHTS Build simple per-image feathering weights.
    %   srcW = warpWeights(images, N) returns a cell array of [h x w] single
    %   weight maps with a gentle center emphasis, one per input image.
    %
    % Inputs:
    %   images - cell array of source images
    %   N      - number of images
    %
    % Outputs:
    %   srcW   - cell array of [h x w] single weight maps

    arguments
        images cell
        N (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
    end

    srcW = cell(N, 1);

    for i = 1:N
        [h, w, ~] = size(images{i});
        wx = ones(1, w, 'single');
        wx(1:ceil(w / 2)) = linspace(0, 1, ceil(w / 2));
        wx(floor(w / 2) + 1:w) = linspace(1, 0, w - floor(w / 2));
        wy = ones(h, 1, 'single');
        wy(1:ceil(h / 2)) = linspace(0, 1, ceil(h / 2));
        wy(floor(h / 2) + 1:h) = linspace(1, 0, h - floor(h / 2));
        srcW{i} = wy * wx; % [h x w], single
    end

end

function tf = canFit(bytesNeeded, onGPU)
    % CANFIT Quick check for available CPU/GPU memory.
    % tf = canFit(bytesNeeded, onGPU)
    % - bytesNeeded: required bytes.
    % - onGPU: true=GPU, false=CPU (default false).
    % Returns true if allocation is likely to fit (best-effort).
    %
    % Inputs:
    %   bytesNeeded - required number of bytes (scalar)
    %   onGPU       - logical flag; true to check GPU memory
    %
    % Outputs:
    %   tf - logical true if memory likely sufficient
    arguments
        bytesNeeded (1, 1) {mustBeNumeric, mustBeFinite, mustBeNonnegative}
        onGPU (1, 1) logical = false
    end

    if onGPU

        try
            g = gpuDevice;
            % leave a safety headroom (50%)
            tf = bytesNeeded < 0.5 * double(g.AvailableMemory);
        catch
            tf = false; % no GPU available / error
        end

    else
        % CPU side: use MaxPossibleArrayBytes as a proxy; keep a headroom
        try
            m = memory; % Windows/desktop MATLAB only
            tf = bytesNeeded < 0.5 * double(m.MaxPossibleArrayBytes);
        catch
            % Fallback if 'memory' unsupported (e.g., Linux w/o swap info):
            % be conservative (require < 2 GB).
            tf = bytesNeeded < 2e9;
        end

    end

end

function s = tern(cond, a, b)
    % TERN Ternary-like conditional selection.
    % s = tern(cond, a, b)
    % - cond: logical scalar or array. If true selects from a; otherwise from b.
    % - a: value(s) returned where cond is true. Must be size-compatible with cond/b.
    % - b: value(s) returned where cond is false. Must be size-compatible with cond/a.
    % Returns s: selected value(s); size and type follow standard MATLAB assignment rules.
    % Notes:
    % - If cond is a scalar, returns a when cond is true, otherwise b.
    % - If cond is an array, selection is element-wise; scalars in a or b are expanded as needed.
    % Examples:
    % - s = tern(true, 1, 2)              % returns 1
    % - s = tern([true false], 10, [1 2]) % returns [10 2]
    %
    % Inputs:
    %   cond - logical scalar or array
    %   a    - value(s) chosen where cond is true
    %   b    - value(s) chosen where cond is false
    %
    % Outputs:
    %   s    - selected value(s) (type/shape depends on inputs)
    arguments
        cond {mustBeNumericOrLogical}
        a
        b
    end

    if cond
        s = a;
    else
        s = b;
    end

end

% ============================================================
function S = sampleBlock(Ic, u, v, H, W, onGPU)
    % SAMPLEBLOCK Vectorized bilinear sampling using interp2 with NaN extrapolation.
    %   S = sampleBlock(Ic, u, v, H, W, onGPU) samples the C-channel image Ic at
    %   query points (u,v) and returns S as a [numel(u) x C] array. Out-of-bounds yield NaN.
    %
    % Inputs:
    %   Ic    - HxWxC image (single or gpuArray)
    %   u,v   - column and row query coordinates (vectors of length T)
    %   H,W   - tile height and width (scalars; present to satisfy signature)
    %   onGPU - logical flag indicating whether sampling should use GPU arrays
    %
    % Outputs:
    %   S     - T x C sampled values; NaN indicates out-of-bounds

    arguments
        Ic (:, :, :) {mustBeNumeric}
        u (:, 1) {mustBeNumeric}
        v (:, 1) {mustBeNumeric}
        H (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        W (1, 1) {mustBeNumeric, mustBeFinite, mustBePositive}
        onGPU (1, 1) logical
    end

    % Touch H/W to avoid unused-arg warnings given signature constraints
    H = H; %#ok<NASGU>
    W = W; %#ok<NASGU>

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
        u(bad) = 1; % arbitrary in-bounds
        v(bad) = 1;
    end

    % Allocate result like u (CPU or GPU), with C channels
    S = zeros(numel(u), C, 'like', u);

    % Interpolate each channel with NaN extrapolation
    if C == 1
        S(:, 1) = interp2(X, Y, Ic, u, v, 'linear', NaN);
    else

        for ch = 1:C
            S(:, ch) = interp2(X, Y, Ic(:, :, ch), u, v, 'linear', NaN);
        end

    end

end

% ============================================================
function [panoCropped, rect, didCrop] = cropNonzeroBbox(panorama, canvasColor)
    % CROPNONZEROBBOX Crop panorama to non-canvas content with small padding.
    %   [panoCropped, rect, didCrop] = cropNonzeroBbox(panorama, canvasColor)
    %   returns the cropped panorama, the crop rectangle [r1 r2 c1 c2], and a
    %   boolean didCrop.
    %
    % Inputs:
    %   panorama    - HxWx3 RGB image (uint8)
    %   canvasColor - text scalar 'white' or 'black' indicating background color
    %
    % Outputs:
    %   panoCropped - cropped H'xW'x3 panorama (or original if nothing to crop)
    %   rect        - [r1 r2 c1 c2] crop rectangle (1-based inclusive)
    %   didCrop     - logical true if cropping was performed

    arguments
        panorama (:, :, :) {mustBeNumeric}
        canvasColor {mustBeTextScalar}
    end

    % rect = [r1 r2 c1 c2] (1-based, inclusive), didCrop = true if cropped

    G = rgb2gray(panorama);

    if strcmpi(canvasColor, 'white')
        fg = (G < 255); % foreground is anything not pure white
    else
        fg = (G > 0); % foreground is anything not pure black
    end

    [r, c] = find(fg);

    if ~isempty(r)
        pad = 6; H = size(panorama, 1); W = size(panorama, 2);
        r1 = max(1, min(r) - pad); r2 = min(H, max(r) + pad);
        c1 = max(1, min(c) - pad); c2 = min(W, max(c) + pad);
        panoCropped = panorama(r1:r2, c1:c2, :);
        rect = [r1 r2 c1 c2];
        didCrop = true;
    else
        panoCropped = panorama;
        rect = [1 size(panorama, 1) 1 size(panorama, 2)];
        didCrop = false;
    end

end

% ============================================================
function [thetaMin, thetaMax, heightMin, heightMax] = cylindricalBounds(cams, imgSize)
    % CYLINDRICALBOUNDS Compute cylindrical bounds (theta,h) over all cameras.
    %   [thetaMin, thetaMax, heightMin, heightMax] = cylindricalBounds(cams, imgSize)
    %   samples each camera frame and accumulates global min/max in cylindrical coords.
    %
    % Inputs:
    %   cams    - struct array of camera parameters (fields K,R,...)
    %   imgSize - Nx3 array of [H W C] per-image sizes
    %
    % Outputs:
    %   thetaMin, thetaMax - angular bounds (radians)
    %   heightMin, heightMax - height bounds (normalized plane units)

    arguments
        cams struct
        imgSize (:, 3) {mustBeNumeric}
    end

    thetaMin = inf; thetaMax = -inf; heightMin = inf; heightMax = -inf;
    nx = 48; ny = 32;

    parfor i = 1:numel(cams)
        H = imgSize(i, 1); W = imgSize(i, 2);
        xs = linspace(1, W, nx); ys = linspace(1, H, ny);
        [U, V] = meshgrid(xs, ys); u = U(:)'; v = V(:)';
        xy1 = [u; v; ones(1, numel(u), 'like', u)];
        rayC = cams(i).K \ xy1;
        rayW = cams(i).R' * rayC; % to world
        x = rayW(1, :); y = rayW(2, :); z = rayW(3, :);
        theta = atan2(x, z);
        h = y ./ hypot(x, z);
        thetaMin = min(thetaMin, min(theta)); thetaMax = max(thetaMax, max(theta));
        heightMin = min(heightMin, min(h)); heightMax = max(heightMax, max(h));
    end

end

function [thetaMin, thetaMax, phiMin, phiMax] = sphericalBounds(cams, imgSize)
    % SPHERICALBOUNDS Compute spherical bounds (theta,phi) over all cameras.
    %   [thetaMin, thetaMax, phiMin, phiMax] = sphericalBounds(cams, imgSize)
    %   samples each camera frame and accumulates global min/max in spherical coords.
    %
    % Inputs:
    %   cams    - struct array of camera parameters
    %   imgSize - Nx3 array of per-image sizes
    %
    % Outputs:
    %   thetaMin, thetaMax - azimuth angular bounds (radians)
    %   phiMin, phiMax     - elevation angular bounds (radians)

    arguments
        cams struct
        imgSize (:, 3) {mustBeNumeric}
    end

    thetaMin = inf; thetaMax = -inf; phiMin = inf; phiMax = -inf;
    nx = 48; ny = 32;

    parfor i = 1:numel(cams)
        H = imgSize(i, 1); W = imgSize(i, 2);
        xs = linspace(1, W, nx); ys = linspace(1, H, ny);
        [U, V] = meshgrid(xs, ys); u = U(:)'; v = V(:)';
        xy1 = [u; v; ones(1, numel(u), 'like', u)];
        rayC = cams(i).K \ xy1;
        rayW = cams(i).R' * rayC;
        x = rayW(1, :); y = rayW(2, :); z = rayW(3, :);
        theta = atan2(x, z);
        phi = atan2(y, hypot(x, z));
        thetaMin = min(thetaMin, min(theta)); thetaMax = max(thetaMax, max(theta));
        phiMin = min(phiMin, min(phi)); phiMax = max(phiMax, max(phi));
    end

end

function [uMin, uMax, vMin, vMax] = planarBounds( ...
        cams, imgSize, Rref, robustPct, uvAbsCap)
    % PLANARBOUNDS Robust bounds on the reference image plane z_ref = +1.
    %   [uMin, uMax, vMin, vMax] = planarBounds(cams, imgSize, Rref, robustPct, uvAbsCap)
    %   computes planar coordinate bounds with percentile clipping and hard caps.
    %
    % Inputs:
    %   cams      - struct array of cameras
    %   imgSize   - Nx3 array of per-image sizes
    %   Rref      - 3x3 reference rotation (world->ref)
    %   robustPct - [low high] percentiles for clipping
    %   uvAbsCap  - hard cap for u/v values (scalar)
    %
    % Outputs:
    %   uMin,uMax - planar u bounds
    %   vMin,vMax - planar v bounds

    arguments
        cams struct
        imgSize (:, 3) {mustBeNumeric}
        Rref (3, 3) {mustBeNumeric, mustBeFinite}
        robustPct (1, 2) {mustBeNumeric}
        uvAbsCap (1, 1) {mustBeNumeric}
    end

    uMin = inf; uMax = -inf;
    vMin = inf; vMax = -inf;

    nx = 48; ny = 32; % coarse interior
    ne = 512; % dense borders
    zEps = 1e-4; % slightly bigger than before for stability

    parfor i = 1:numel(cams)
        H = imgSize(i, 1); W = imgSize(i, 2);

        % interior samples
        xs = linspace(1, W, nx); ys = linspace(1, H, ny);
        [U, V] = meshgrid(xs, ys); u = U(:)'; v = V(:)';

        % border samples
        xb = linspace(1, W, ne); yb = linspace(1, H, ne);
        ub = [xb, xb, ones(1, ne), W * ones(1, ne)];
        vb = [ones(1, ne), H * ones(1, ne), yb, yb];

        u = [u, ub]; v = [v, vb];

        xy1 = [u; v; ones(1, numel(u), 'like', u)];
        rayC = cams(i).K \ xy1;
        rayW = cams(i).R' * rayC; % to world
        rayR = Rref * rayW; % to ref camera frame

        zr = rayR(3, :);
        m = zr > zEps; % strictly front-facing and away from grazing

        if ~any(m), continue; end

        ur = rayR(1, m) ./ zr(m);
        vr = rayR(2, m) ./ zr(m);

        % Hard cap to avoid extreme blow-ups
        if isfinite(uvAbsCap) && uvAbsCap > 0
            ur = max(-uvAbsCap, min(uvAbsCap, ur));
            vr = max(-uvAbsCap, min(uvAbsCap, vr));
        end

        % Per-camera percentile clip (robust bounds)
        lowPct = robustPct(1); hiPct = robustPct(2);
        uLow = prctile(ur, lowPct); uHi = prctile(ur, hiPct);
        vLow = prctile(vr, lowPct); vHi = prctile(vr, hiPct);

        % Update global bounds
        uMin = min(uMin, uLow); uMax = max(uMax, uHi);
        vMin = min(vMin, vLow); vMax = max(vMax, vHi);
    end

    % Safety: if something went wrong, fall back to a sane box
    if ~isfinite(uMin) || ~isfinite(uMax) || uMin >= uMax
        uMin = -1; uMax = 1;
    end

    if ~isfinite(vMin) || ~isfinite(vMax) || vMin >= vMax
        vMin = -1; vMax = 1;
    end

end

function [aMin, aMax, bMin, bMax] = stereographicBounds( ...
        cams, imgSize, Rref, robustPct, absCap)
    % STEREOGRAPHICBOUNDS Robust bounds on the stereographic plane.
    %   [aMin, aMax, bMin, bMax] = stereographicBounds(cams, imgSize, Rref, robustPct, absCap)
    %   computes bounds after stereographic mapping with percentile clipping and caps.
    %
    % Inputs:
    %   cams      - struct array of cameras
    %   imgSize   - Nx3 array of per-image sizes
    %   Rref      - 3x3 reference rotation (world->ref)
    %   robustPct - percentile clip bounds [low high]
    %   absCap    - absolute cap for stereographic coords
    %
    % Outputs:
    %   aMin,aMax - stereographic a bounds
    %   bMin,bMax - stereographic b bounds

    arguments
        cams struct
        imgSize (:, 3) {mustBeNumeric}
        Rref (3, 3) {mustBeNumeric, mustBeFinite}
        robustPct (1, 2) {mustBeNumeric}
        absCap (1, 1) {mustBeNumeric}
    end

    aMin = inf; aMax = -inf;
    bMin = inf; bMax = -inf;

    nx = 48; ny = 32; % interior grid
    ne = 512; % dense borders

    parfor i = 1:numel(cams)
        H = imgSize(i, 1); W = imgSize(i, 2);

        % interior samples
        xs = linspace(1, W, nx); ys = linspace(1, H, ny);
        [U, V] = meshgrid(xs, ys); u = U(:)'; v = V(:)';

        % border samples
        xb = linspace(1, W, ne); yb = linspace(1, H, ne);
        ub = [xb, xb, ones(1, ne), W * ones(1, ne)];
        vb = [ones(1, ne), H * ones(1, ne), yb, yb];

        u = [u, ub]; v = [v, vb];

        xy1 = [u; v; ones(1, numel(u), 'like', u)];
        rayC = cams(i).K \ xy1; % camera frame (not unit)
        rayW = cams(i).R' * rayC; % to world
        rayR = Rref * rayW; % to ref frame

        % normalize to unit directions
        nr = sqrt(sum(rayR .^ 2, 1));
        xr = rayR(1, :) ./ nr; yr = rayR(2, :) ./ nr; zr = rayR(3, :) ./ nr;

        % stereographic forward map (plane tangent at +Z; project from -Z)
        denom = 1 + zr;
        % avoid explosion near zr ~ -1 (those go to infinity anyway)
        valid = denom > 1e-6;
        if ~any(valid), continue; end

        a = xr(valid) ./ denom(valid);
        b = yr(valid) ./ denom(valid);

        % optional hard cap (like planar uvAbsCap)
        if isfinite(absCap) && absCap > 0
            a = max(-absCap, min(absCap, a));
            b = max(-absCap, min(absCap, b));
        end

        % per-camera percentile clip then update global
        lowPct = robustPct(1); hiPct = robustPct(2);
        aLow = prctile(a, lowPct); aHi = prctile(a, hiPct);
        bLow = prctile(b, lowPct); bHi = prctile(b, hiPct);

        aMin = min(aMin, aLow); aMax = max(aMax, aHi);
        bMin = min(bMin, bLow); bMax = max(bMax, bHi);
    end

    % safety fallback
    if ~isfinite(aMin) || ~isfinite(aMax) || aMin >= aMax
        aMin = -1; aMax = 1;
    end

    if ~isfinite(bMin) || ~isfinite(bMax) || bMin >= bMax
        bMin = -1; bMax = 1;
    end

end
