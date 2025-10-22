function [allPanoramas, annoRGBPanoramas] = displayPanorama(input, finalPanoramaTforms, ...
                                                finalrefIdxs, images)

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
        renderPlanartic = tic;  
        [panoramaPlanar, annoRGBPlanar] = renderPanorama(images, cameras, 'planar', refIdx, opts);
        fprintf('Planar panorama rendering time : %f seconds\n', toc(renderPlanartic));

        renderCylindricaltic = tic;  
        [panoramaCylindrical, annoRGBCylindrical] = renderPanorama(images, cameras, 'cylindrical', refIdx, opts);
        fprintf('Cylindrical panorama rendering time : %f seconds\n', toc(renderCylindricaltic));
        
        renderSphericaltic = tic;  
        [panoramaSpherical, annoRGBSpherical] = renderPanorama(images, cameras, 'spherical',   refIdx, opts);
        fprintf('Spherical panorama rendering time : %f seconds\n', toc(renderSphericaltic));

        
        % Store panoramas
        allPanoramas{ii,1}     = panoramaPlanar;
        allPanoramas{ii,2}     = panoramaCylindrical;
        allPanoramas{ii,3}     = panoramaSpherical;

        annoRGBPanoramas{ii,1} = annoRGBPlanar;
        annoRGBPanoramas{ii,2} = annoRGBCylindrical;
        annoRGBPanoramas{ii,3} = annoRGBSpherical;
            
        if input.displayPanoramas           
            % Full planar panorama
            figure('Name','Planar panorama');
            imshow(panoramaPlanar)                     
            
            % ---------------------------------------------------------------
            % Full cylindrical panorama
           
            % Plot the bounding boxes
            figure('Name','Cyllindrical panorama');
            imshow(panoramaCylindrical)
         
            % ---------------------------------------------------------------
            % Full Spherical panorama         
            figure('Name','Spherical panorama');
            imshow(panoramaSpherical)                     
        end    
    end
end