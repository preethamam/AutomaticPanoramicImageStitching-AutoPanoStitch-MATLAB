function [allPanoramas, croppedPanoramas] = displayPanorama(input, finalPanoramaTforms, ...
                                                finalrefIdxs, images, myImg, datasetName)

    %%***********************************************************************%
    %*                   Automatic panorama stitching                       *%
    %*                        Display panorama                              *%
    %*                                                                      *%
    %* Code author: Preetham Manjunatha                                     *%
    %* Github link: https://github.com/preethamam                           *%
    %* Date: 10/21/2025                                                     *%
    %************************************************************************%
    
    % Initialize
    allPanoramas = cell(1,length(finalPanoramaTforms));
    croppedPanoramas = cell(1,length(finalPanoramaTforms));
    annoRGBPanoramas = cell(1,length(finalPanoramaTforms));
    
    for ii = 1:length(finalPanoramaTforms)

        % Close all figures
        close all;

        cameras = finalPanoramaTforms{ii};
        refIdx = finalrefIdxs(ii);

        % Example call after your BA
        opts = struct('f_pan', cameras(refIdx).K(1,1), 'res_scale',1.0, ...
                      'angle_power',2, 'crop_border',true, ...
                      'canvas_color', input.canvas_color, ...
                      'sigma_N', input.sigmaN,...
                      'sigma_g', input.sigmag, ...
                      'pyr_levels', input.bands, ...
                      'blending', input.blending, ...
                      'pyr_sigma', input.MBBsigma, ...
                      'showPanoramaImgsNums', input.showPanoramaImgsNums, ...
                      'showCropBoundingBox', input.showCropBoundingBox);
        
        % Render all projections panorama
        [panoramaPlanar, annoRGBPlanar] = renderPanorama(images, cameras, 'planar', refIdx, opts);
        [panoramaCylindrical, annoRGBCylindrical] = renderPanorama(images, cameras, 'cylindrical', refIdx, opts);
        [panoramaSpherical, annoRGBSpherical] = renderPanorama(images, cameras, 'spherical',   refIdx, opts);

            
        % Panorama cropper
        croppedPlanar = panoramaCropper(input, panoramaPlanar);   
        croppedCylindrical = panoramaCropper(input, panoramaCylindrical);   
        croppedSpherical = panoramaCropper(input, panoramaSpherical);   
        
        % Store panoramas
        allPanoramas{ii,1}     = panoramaPlanar;
        allPanoramas{ii,2}     = panoramaCylindrical;
        allPanoramas{ii,3}     = panoramaSpherical;

        croppedPanoramas{ii,1} = croppedPlanar;
        croppedPanoramas{ii,2} = croppedCylindrical;
        croppedPanoramas{ii,3} = croppedSpherical;

        annoRGBPanoramas{ii,1} = annoRGBPlanar;
        annoRGBPanoramas{ii,2} = annoRGBCylindrical;
        annoRGBPanoramas{ii,3} = annoRGBSpherical;
            
        if input.displayPanoramas
            % Full planar panorama
            if ~isempty(annoRGBPlanar)
                figure('Name','Planar Annotations panorama');
                imshow(annoRGBPlanar)
            end

            % Plot the bounding boxes
            figure('Name','Planar panorama');
            imshow(panoramaPlanar)
            
            if input.showCropBoundingBox
                figure('Name','Planar cropped panorama');
                imshow(croppedPlanar)
            end
            
            % ---------------------------------------------------------------
            % Full cylindrical panorama
            if ~isempty(annoRGBCylindrical)
                figure('Name','Cyllindrical Annotations panorama');
                imshow(annoRGBCylindrical)
            end

            % Plot the bounding boxes
            figure('Name','Cyllindrical panorama');
            imshow(panoramaCylindrical)
            
            if input.showCropBoundingBox
                figure('Name','Cyllindrical cropped panorama');
                imshow(croppedCylindrical)
            end

            % ---------------------------------------------------------------
            % Full Spherical panorama
            if ~isempty(annoRGBCylindrical)
                figure('Name','Spherical Annotations panorama');
                imshow(annoRGBSpherical)
            end

            % Plot the bounding boxes
            figure('Name','Spherical panorama');
            imshow(panoramaSpherical)
            
            if input.showCropBoundingBox
                figure('Name','Spherical cropped panorama');
                imshow(croppedSpherical)
            end
        end
    
        % Image write
        if input.imageWrite
            imwrite(panoramaPlanar, [ 'planar' '_' input.transformationType '_' ...
            num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
    
            imwrite(panoramaCylindrical, [ 'cylindrical' '_' input.transformationType '_' ...
            num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
    
            imwrite(croppedSpherical, [ 'spherical' '_' input.transformationType '_' ...
            num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
        end
    end
end