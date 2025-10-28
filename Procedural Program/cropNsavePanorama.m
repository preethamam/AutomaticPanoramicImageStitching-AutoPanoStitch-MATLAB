function panoStore = cropNsavePanorama(input, panoStore, myImg, datasetName)

    for ii = 1:numel(panoStore)
        % Panorama cropper
        if input.cropPanorama == 1

            for jj = 1:numel(fieldnames(panoStore))
                tic;

                if isfield(panoStore, "planar")
                    projName = "planar";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).planar{1});
                    panoStore(ii).planar{3} = croppedPanorama;
                elseif isfield(panoStore, "cylindrical")
                    projName = "cylindrical";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).cylindrical{1});
                    panoStore(ii).cylindrical{3} = croppedPanorama;
                elseif isfield(panoStore, "spherical")
                    projName = "spherical";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).spherical{1});
                    panoStore(ii).spherical{3} = croppedPanorama;
                elseif isfield(panoStore, "equirectangular")
                    projName = "equirectangular";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).equirectangular{1});
                    panoStore(ii).equirectangular{3} = croppedPanorama;
                elseif isfield(panoStore, "stereographic")
                    projName = "stereographic";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).stereographic{1});
                    panoStore(ii).stereographic{3} = croppedPanorama;
                end

                fprintf('Cropped %s panorama in : %f seconds\n', projName, toc);
            end

        end

        % Image write
        if input.imageWrite
            % RGB panorama writer
            if isfield(panoStore, "planar")
                imwrite(panoStore(ii).planar{1}, ['planar' '_' input.transformationType '_' ...
                                                      num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            elseif isfield(panoStore, "cylindrical")
                imwrite(panoStore(ii).cylindrical{1}, ['cylindrical' '_' input.transformationType '_' ...
                                                           num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            elseif isfield(panoStore, "spherical")
                imwrite(panoStore(ii).spherical{1}, ['spherical' '_' input.transformationType '_' ...
                                                         num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            elseif isfield(panoStore, "equirectangular")
                imwrite(panoStore(ii).equirectangular{1}, ['equirectangular' '_' input.transformationType '_' ...
                                                               num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            elseif isfield(panoStore, "stereographic")
                imwrite(panoStore(ii).stereographic{1}, ['stereographic' '_' input.transformationType '_' ...
                                                             num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            end

            %-------------------------------------------------------------------------------
            if input.cropPanorama == 1
                % RGB cropped panorama writer
                if isfield(panoStore, "planar")
                    imwrite(panoStore(ii).planar{3}, ['planar_cropped' '_' input.transformationType '_' ...
                                                          num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                elseif isfield(panoStore, "cylindrical")
                    imwrite(panoStore(ii).cylindrical{3}, ['cylindrical_cropped' '_' input.transformationType '_' ...
                                                               num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                elseif isfield(panoStore, "spherical")
                    imwrite(panoStore(ii).spherical{3}, ['spherical_cropped' '_' input.transformationType '_' ...
                                                             num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                elseif isfield(panoStore, "equirectangular")
                    imwrite(panoStore(ii).equirectangular{3}, ['equirectangular_cropped' '_' input.transformationType '_' ...
                                                                   num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                elseif isfield(panoStore, "stereographic")
                    imwrite(panoStore(ii).stereographic{3}, ['stereographic_cropped' '_' input.transformationType '_' ...
                                                                 num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                end

            end

            %-------------------------------------------------------------------------------
            if input.showPanoramaImgsNums && input.showCropBoundingBox
                % RGB annotations writer
                if isfield(panoStore, "planar") && ~isempty(panoStore(ii).planar{2})
                    imwrite(panoStore(ii).planar{2}, ['planar_annotated' '_' input.transformationType '_' ...
                                                          num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                elseif isfield(panoStore, "cylindrical") && ~isempty(panoStore(ii).cylindrical{2})
                    imwrite(panoStore(ii).cylindrical{2}, ['cylindrical_annotated' '_' input.transformationType '_' ...
                                                               num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                elseif isfield(panoStore, "spherical") && ~isempty(panoStore(ii).spherical{2})
                    imwrite(panoStore(ii).spherical{2}, ['spherical_annotated' '_' input.transformationType '_' ...
                                                             num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                elseif isfield(panoStore, "equirectangular") && ~isempty(panoStore(ii).equirectangular{2})
                    imwrite(panoStore(ii).equirectangular{2}, ['equirectangular_annotated' '_' input.transformationType '_' ...
                                                                   num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                elseif isfield(panoStore, "stereographic") && ~isempty(panoStore(ii).stereographic{2})
                    imwrite(panoStore(ii).stereographic{2}, ['stereographic_annotated' '_' input.transformationType '_' ...
                                                                 num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                end

            end

        end

    end

end
