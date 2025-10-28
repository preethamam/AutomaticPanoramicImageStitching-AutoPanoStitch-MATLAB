function panoStore = cropNsavePanorama(input, panoStore, myImg, datasetName)
    %CROPNSAVEPANORAMA Optionally crop panoramas and write images to disk.
    %
    % Syntax
    %   panoStore = cropNsavePanorama(input, panoStore, myImg, datasetName)
    %
    % Description
    %   Iterates through the panorama store and, if enabled, crops each panorama
    %   using panoramaCropper. It then writes the base, cropped, and annotated
    %   panoramas (when available and enabled) to disk with filenames that encode
    %   projection type, transformation type, dataset index, panorama index, and
    %   dataset name.
    %
    % Inputs
    %   input        - Struct of control flags and parameters:
    %                  • cropPanorama (logical/0-1): if true, compute cropped panoramas.
    %                  • imageWrite (logical): if true, write images to disk.
    %                  • transformationType (char/string): tag used in output filenames.
    %                  • showPanoramaImgsNums (logical): used to control annotated output.
    %                  • showCropBoundingBox (logical): used to control annotated output.
    %   panoStore    - 1-by-M struct array of panoramas. Each element may contain any of
    %                  the fields: 'planar', 'cylindrical', 'spherical', 'equirectangular',
    %                  'stereographic'. Each such field is a cell array with slots:
    %                  {1} base RGB panorama, {2} annotated RGB (optional), {3} cropped RGB (set here).
    %   myImg        - Scalar dataset index used in output filenames.
    %   datasetName  - Cell array of dataset names; datasetName{myImg} is used in filenames.
    %
    % Output
    %   panoStore    - The input panoStore updated in-place to include cropped panoramas
    %                  in cell slot {3} for any available projection fields when
    %                  input.cropPanorama is true.
    %
    % Side effects
    %   - Writes PNG files to the current working directory when input.imageWrite is true.
    %     Filenames follow the patterns:
    %       '<proj>_<transformationType>_<myImg>_<ii>_<datasetName>.png'
    %       '<proj>_cropped_<transformationType>_<myImg>_<ii>_<datasetName>.png'
    %       '<proj>_annotated_<transformationType>_<myImg>_<ii>_<datasetName>.png'
    %     where <proj> ∈ {planar, cylindrical, spherical, equirectangular, stereographic}
    %     and ii is the panorama index within panoStore.
    %
    % Notes
    %   - Single projection per struct element is assumed; the first matching field among
    %     the supported projection names is processed.
    %   - Cropping times are printed using fprintf.
    %   - This function checks for projection fields using isfield on the struct array;
    %     ensure consistency of fields across elements to avoid unexpected behavior.
    %
    % See also: panoramaCropper, imwrite

    arguments
        input (1, 1) struct
        panoStore (1, :) struct
        myImg (1, 1) double {mustBeInteger, mustBePositive}
        datasetName cell
    end

    % Basic runtime validations
    if myImg > numel(datasetName) || myImg < 1
        error('cropNsavePanorama:InvalidDatasetIndex', 'myImg (%d) must index into datasetName (numel=%d).', myImg, numel(datasetName));
    end

    if ~(ischar(datasetName{myImg}) || isstring(datasetName{myImg}))
        error('cropNsavePanorama:InvalidDatasetNameType', 'datasetName{myImg} must be char or string.');
    end

    if isfield(input, 'imageWrite') && input.imageWrite && ~isfield(input, 'transformationType')
        error('cropNsavePanorama:MissingTransformationType', 'input.transformationType is required when input.imageWrite is true.');
    end

    for ii = 1:numel(panoStore)
        % Panorama cropper
        if input.cropPanorama == 1

            panoProjs = fieldnames(panoStore);

            for jj = 1:numel(panoProjs)
                tic;

                if strcmp(panoProjs{jj}, "planar")
                    projName = "planar";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).planar{1});
                    panoStore(ii).planar{3} = croppedPanorama;
                end

                if strcmp(panoProjs{jj}, "cylindrical")
                    projName = "cylindrical";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).cylindrical{1});
                    panoStore(ii).cylindrical{3} = croppedPanorama;
                end

                if strcmp(panoProjs{jj}, "spherical")
                    projName = "spherical";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).spherical{1});
                    panoStore(ii).spherical{3} = croppedPanorama;
                end

                if strcmp(panoProjs{jj}, "equirectangular")
                    projName = "equirectangular";
                    croppedPanorama = panoramaCropper(input, panoStore(ii).equirectangular{1});
                    panoStore(ii).equirectangular{3} = croppedPanorama;
                end

                if strcmp(panoProjs{jj}, "stereographic")
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
            if isfield(panoStore, "planar") && ~isempty(panoStore(ii).planar)
                imwrite(panoStore(ii).planar{1}, fullfile(input.imageSaveFolder, ['planar' '_' input.transformationType '_' ...
                                                                                      num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
            end

            if isfield(panoStore, "cylindrical") && ~isempty(panoStore(ii).cylindrical)
                imwrite(panoStore(ii).cylindrical{1}, fullfile(input.imageSaveFolder, ['cylindrical' '_' input.transformationType '_' ...
                                                                                           num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
            end

            if isfield(panoStore, "spherical") && ~isempty(panoStore(ii).spherical)
                imwrite(panoStore(ii).spherical{1}, fullfile(input.imageSaveFolder, ['spherical' '_' input.transformationType '_' ...
                                                                                         num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
            end

            if isfield(panoStore, "equirectangular") && ~isempty(panoStore(ii).equirectangular)
                imwrite(panoStore(ii).equirectangular{1}, fullfile(input.imageSaveFolder, ['equirectangular' '_' input.transformationType '_' ...
                                                                                               num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
            end

            if isfield(panoStore, "stereographic") && ~isempty(panoStore(ii).stereographic)
                imwrite(panoStore(ii).stereographic{1}, fullfile(input.imageSaveFolder, ['stereographic' '_' input.transformationType '_' ...
                                                                                             num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
            end

            %-------------------------------------------------------------------------------
            if input.cropPanorama == true
                % RGB cropped panorama writer
                if isfield(panoStore, "planar") && ~isempty(panoStore(ii).planar)
                    imwrite(panoStore(ii).planar{3}, fullfile(input.imageSaveFolder, ['planar_cropped' '_' input.transformationType '_' ...
                                                                                          num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

                if isfield(panoStore, "cylindrical") && ~isempty(panoStore(ii).cylindrical)
                    imwrite(panoStore(ii).cylindrical{3}, fullfile(input.imageSaveFolder, ['cylindrical_cropped' '_' input.transformationType '_' ...
                                                                                               num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

                if isfield(panoStore, "spherical") && ~isempty(panoStore(ii).spherical)
                    imwrite(panoStore(ii).spherical{3}, fullfile(input.imageSaveFolder, ['spherical_cropped' '_' input.transformationType '_' ...
                                                                                             num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

                if isfield(panoStore, "equirectangular") && ~isempty(panoStore(ii).equirectangular)
                    imwrite(panoStore(ii).equirectangular{3}, fullfile(input.imageSaveFolder, ['equirectangular_cropped' '_' input.transformationType '_' ...
                                                                                                   num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

                if isfield(panoStore, "stereographic") && ~isempty(panoStore(ii).stereographic)
                    imwrite(panoStore(ii).stereographic{3}, fullfile(input.imageSaveFolder, ['stereographic_cropped' '_' input.transformationType '_' ...
                                                                                                 num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

            end

            %-------------------------------------------------------------------------------
            if input.showPanoramaImgsNums && input.showCropBoundingBox
                % RGB annotations writer
                if isfield(panoStore, "planar") && ~isempty(panoStore(ii).planar)
                    imwrite(panoStore(ii).planar{2}, fullfile(input.imageSaveFolder, ['planar_annotated' '_' input.transformationType '_' ...
                                                                                          num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

                if isfield(panoStore, "cylindrical") && ~isempty(panoStore(ii).cylindrical)
                    imwrite(panoStore(ii).cylindrical{2}, fullfile(input.imageSaveFolder, ['cylindrical_annotated' '_' input.transformationType '_' ...
                                                                                               num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

                if isfield(panoStore, "spherical") && ~isempty(panoStore(ii).spherical)
                    imwrite(panoStore(ii).spherical{2}, fullfile(input.imageSaveFolder, ['spherical_annotated' '_' input.transformationType '_' ...
                                                                                             num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

                if isfield(panoStore, "equirectangular") && ~isempty(panoStore(ii).equirectangular)
                    imwrite(panoStore(ii).equirectangular{2}, fullfile(input.imageSaveFolder, ['equirectangular_annotated' '_' input.transformationType '_' ...
                                                                                                   num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

                if isfield(panoStore, "stereographic") && ~isempty(panoStore(ii).stereographic)
                    imwrite(panoStore(ii).stereographic{2}, fullfile(input.imageSaveFolder, ['stereographic_annotated' '_' input.transformationType '_' ...
                                                                                                 num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png']))
                end

            end

        end

    end

end
