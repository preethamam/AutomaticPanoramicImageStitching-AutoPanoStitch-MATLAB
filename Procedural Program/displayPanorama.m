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
    allPanoramas = cell(length(finalPanoramaTforms),5);
    annoRGBPanoramas = cell(length(finalPanoramaTforms),5);
    
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

        renderEquirectangulartic = tic;  
        [panoramaEquirectangular, annoRGBEquirectangular] = renderPanorama(images, cameras, 'equirectangular', refIdx, opts);
        fprintf('Equirectangular panorama rendering time : %f seconds\n', toc(renderEquirectangulartic));
        
        renderStereographictic = tic;  
        [panoramaStereographic, annoRGBStereographic] = renderPanorama(images, cameras, 'stereographic', refIdx, opts);
        fprintf('Stereographic panorama rendering time : %f seconds\n', toc(renderStereographictic));

        
        % Store panoramas
        allPanoramas{ii,1}     = panoramaPlanar;
        allPanoramas{ii,2}     = panoramaCylindrical;
        allPanoramas{ii,3}     = panoramaSpherical;
        allPanoramas{ii,4}     = panoramaEquirectangular;
        allPanoramas{ii,5}     = panoramaStereographic;

        annoRGBPanoramas{ii,1} = annoRGBPlanar;
        annoRGBPanoramas{ii,2} = annoRGBCylindrical;
        annoRGBPanoramas{ii,3} = annoRGBSpherical;
        annoRGBPanoramas{ii,4} = annoRGBEquirectangular;
        annoRGBPanoramas{ii,5} = annoRGBStereographic;
            
        if input.displayPanoramas           
            % Full planar panorama
            figure('Name','Planar panorama');
            imshow(panoramaPlanar)                     
            
            % ---------------------------------------------------------------
            % Full cylindrical panorama          
            figure('Name','Cyllindrical panorama');
            imshow(panoramaCylindrical)
         
            % ---------------------------------------------------------------
            % Full Spherical panorama         
            figure('Name','Spherical panorama');
            imshow(panoramaSpherical) 

            % ---------------------------------------------------------------
            % Full equirectangular panorama          
            figure('Name','Equirectangular panorama');
            imshow(panoramaEquirectangular)
         
            % ---------------------------------------------------------------
            % Full stereographic panorama         
            figure('Name','Stereographic panorama');
            imshow(panoramaStereographic) 
        end    
    end
end