function panoStore = displayPanorama(input, finalPanoramaTforms, ...
        finalrefIdxs, imagesAll, imageSizesAll, panoIndices)
    %DISPLAYPANORAMA Render panoramas for selected projections and optionally display them.
    %
    % Syntax
    %   panoStore = displayPanorama(input, finalPanoramaTforms, finalrefIdxs, imagesAll, panoIndices)
    %
    % Description
    %   For each panorama component, this function renders panoramas in one or more
    %   projection models specified by input.panorama2DisplaynSave using renderPanorama.
    %   The rendered RGB panorama and an optional annotated RGB image are stored in
    %   the returned panoStore struct. When input.displayPanoramas is true, the
    %   panoramas are displayed in figures. The function also constructs an options
    %   struct for rendering and blending from fields in input.
    %
    % Inputs
    %   input               - Struct of parameters and display options. Expected fields include:
    %                         • panorama2DisplaynSave: string/cell array of projections to render,
    %                           e.g., ["planar","cylindrical",...].
    %                         • canvas_color, gainCompensation, sigmaN, sigmag, bands,
    %                           blending, MBBsigma, showPanoramaImgsNums, showCropBoundingBox,
    %                           displayPanoramas (logical).
    %   finalPanoramaTforms - C-by-1 cell array; for each component i, finalPanoramaTforms{i}
    %                         is an array of camera structs with fields including K (intrinsics)
    %                         and the required pose/transform data for rendering.
    %   finalrefIdxs        - C-by-1 numeric array with the reference image index for each component.
    %   imagesAll           - N-by-1 cell array of source images.
    %   imageSizesAll       - N-by-3 numeric array of image sizes [height, width, channels].
    %   panoIndices         - C-by-1 cell array; panoIndices{i} lists the absolute image indices
    %                         belonging to component i.
    %
    % Output
    %   panoStore           - 1-by-C struct array. For each component i, fields may include
    %                         'planar', 'cylindrical', 'spherical', 'equirectangular',
    %                         'stereographic' depending on requested projections. Each field is
    %                         a 1x2 cell array: {1} RGB panorama, {2} annotated RGB (may be empty).
    %
    % Side effects
    %   - Closes all open figures at the start of each component iteration (close all).
    %   - When input.displayPanoramas is true, opens figure windows and displays panoramas.
    %
    % Notes
    %   - The focal length for planar projection is taken as f_pan = cameras(refIdx).K(1,1).
    %   - The function assumes renderPanorama supports the given projections and options.
    %   - Projection-specific figure titles are used when displaying results.
    %
    % See also: renderPanorama, cropNsavePanorama, imshow, figure

    arguments
        input (1, 1) struct
        finalPanoramaTforms cell
        finalrefIdxs (:, 1) double {mustBeInteger, mustBeInteger}
        imagesAll cell
        imageSizesAll (:, 3) double {mustBeNumeric, mustBeFinite, mustBePositive}
        panoIndices cell
    end

    % Basic runtime validations
    C = numel(finalPanoramaTforms);

    if numel(finalrefIdxs) ~= C
        error('displayPanorama:InvalidRefIdxsLength', 'finalrefIdxs must have one entry per component (%d).', C);
    end

    if numel(panoIndices) ~= C
        error('displayPanorama:InvalidPanoIndicesLength', 'panoIndices must have one cell per component (%d).', C);
    end

    Nimgs = numel(imagesAll);

    for iiChk = 1:C
        idxs = panoIndices{iiChk};

        if ~isnumeric(idxs) || ~isvector(idxs)
            if isempty(idxs), continue, end
            error('displayPanorama:InvalidPanoIndexType', 'panoIndices{%d} must be a numeric vector of image indices.', iiChk);
        end

        if ~isempty(idxs) && (min(idxs) < 1 || max(idxs) > Nimgs)
            error('displayPanorama:ImageIndexOutOfBounds', 'panoIndices{%d} has indices outside 1..%d.', iiChk, Nimgs);
        end

    end

    % Initialize
    panoStore = struct();

    for ii = 1:length(finalPanoramaTforms)
        
        if isempty(finalPanoramaTforms{ii}), continue, end

        % Close all figures
        close all;
        cameras = finalPanoramaTforms{ii};
        refIdx = finalrefIdxs(ii);
        images = imagesAll(panoIndices{ii});
        imageSizes = imageSizesAll(panoIndices{ii}, :);
        
        % Example call after your BA
        opts = struct('f_pan', cameras(refIdx).K(1, 1), 'res_scale', 1.0, ...
            'angle_power', 2, 'crop_border', true, ...
            'canvas_color', input.canvas_color, ...
            'gain_compensation', input.gainCompensation, ...
            'sigma_N', input.sigmaN, ...
            'sigma_g', input.sigmag, ...
            'pyr_levels', input.bands, ...
            'blending', input.blending, ...
            'pyr_sigma', input.MBBsigma, ...
            'showPanoramaImgsNums', input.showPanoramaImgsNums, ...
            'showCropBoundingBox', input.showCropBoundingBox);

        % Render all projections panorama
        for jj = 1:numel(input.panorama2DisplaynSave)
            projection = input.panorama2DisplaynSave(jj);
            [panorama, annoRGB] = renderPanorama(images, imageSizes, cameras, projection, refIdx, opts);
            
            if isempty(panorama)
                warning('displayPanorama:Skipped', ...
                    'Panorama skipped due to insufficient memory for %s frame.', projection);
                % continue / move on to next CC
                return
            end

            % Store panoramas
            if strcmp(projection, "planar")
                panoStore(ii).planar = {panorama, annoRGB};
            elseif strcmp(projection, "cylindrical")
                panoStore(ii).cylindrical = {panorama, annoRGB};
            elseif strcmp(projection, "spherical")
                panoStore(ii).spherical = {panorama, annoRGB};
            elseif strcmp(projection, "equirectangular")
                panoStore(ii).equirectangular = {panorama, annoRGB};
            elseif strcmp(projection, "stereographic")
                panoStore(ii).stereographic = {panorama, annoRGB};
            end

        end

        if input.displayPanoramas
            % Full planar panorama
            if isfield(panoStore, "planar")
                figure('Name', 'Planar panorama');
                imshow(panoStore(ii).planar{1})
            end

            % ---------------------------------------------------------------
            % Full cylindrical panorama
            if isfield(panoStore, "cylindrical")
                figure('Name', 'Cyllindrical panorama');
                imshow(panoStore(ii).cylindrical{1})
            end

            % ---------------------------------------------------------------
            % Full Spherical panorama
            if isfield(panoStore, "spherical")
                figure('Name', 'Spherical panorama');
                imshow(panoStore(ii).spherical{1})
            end

            % ---------------------------------------------------------------
            % Full equirectangular panorama
            if isfield(panoStore, "equirectangular")
                figure('Name', 'Equirectangular panorama');
                imshow(panoStore(ii).equirectangular{1})
            end

            % ---------------------------------------------------------------
            % Full stereographic panorama
            if isfield(panoStore, "stereographic")
                figure('Name', 'Stereographic panorama');
                imshow(panoStore(ii).stereographic{1})
            end

        end

        % Pause to visuzalize the panorama
        pause(1)

    end

end
