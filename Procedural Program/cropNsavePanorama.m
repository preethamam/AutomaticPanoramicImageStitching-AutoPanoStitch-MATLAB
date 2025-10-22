function croppedPanoramas = cropNsavePanorama(input, allPanoramas, annotatedPanoramas, ...
                                              myImg, datasetName)
    
    croppedPanoramas = cell(length(allPanoramas), 3);
    for ii = 1:size(allPanoramas,1)        
        % Panorama cropper
        if input.cropPanorama == 1
            croppedPanoramas{ii,1} = panoramaCropper(input, allPanoramas{ii,1});   
            croppedPanoramas{ii,2} = panoramaCropper(input, allPanoramas{ii,2});   
            croppedPanoramas{ii,3} = panoramaCropper(input, allPanoramas{ii,3});   
        end
        
        % Image write
        if input.imageWrite
            imwrite(allPanoramas{ii,1}, [ 'planar' '_' input.transformationType '_' ...
            num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            
            imwrite(allPanoramas{ii,2}, [ 'cylindrical' '_' input.transformationType '_' ...
            num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            
            imwrite(allPanoramas{ii,3}, [ 'spherical' '_' input.transformationType '_' ...
            num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            
            %-------------------------------------------------------------------------------
            if input.cropPanorama == 1
                imwrite(croppedPanoramas{ii,1}, [ 'planar_cropped' '_' input.transformationType '_' ...
                num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                
                imwrite(croppedPanoramas{ii,2}, [ 'cylindrical_cropped' '_' input.transformationType '_' ...
                num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                
                imwrite(croppedPanoramas{ii,3}, [ 'spherical_cropped' '_' input.transformationType '_' ...
                num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            end

            %-------------------------------------------------------------------------------
            if ~isempty(annotatedPanoramas{ii,1})
                imwrite(annotatedPanoramas{ii,1}, [ 'planar_annotated' '_' input.transformationType '_' ...
                num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                
                imwrite(annotatedPanoramas{ii,2}, [ 'cylindrical_annotated' '_' input.transformationType '_' ...
                num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
                
                imwrite(annotatedPanoramas{ii,3}, [ 'spherical_annotated' '_' input.transformationType '_' ...
                num2str(myImg) '_' num2str(ii) '_' char(datasetName{myImg}) '.png'])
            end
        end
    end
end