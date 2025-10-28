function panoStore = displayPanorama(input, finalPanoramaTforms, ...
                                finalrefIdxs, imagesAll, panoIndices)

    % Initialize
    panoStore = struct();

    for ii = 1:length(finalPanoramaTforms)

        % Close all figures
        close all;
        cameras = finalPanoramaTforms{ii};
        refIdx = finalrefIdxs(ii);

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
            [panorama, annoRGB] = renderPanorama(imagesAll(panoIndices{ii}), cameras, projection, refIdx, opts);

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
        pause(0)

    end

end
