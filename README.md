# AutoPanoStitch

[![View Automatic panorama stitcher (AutoPanoStitch) on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/105850-automatic-panorama-stitcher-autopanostitch) [![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=preethamam/AutomaticPanoramicImageStitching-AutoPanoStitch-MATLAB)

AutoPanoStitch is an automatic panorama stitching package for MATLAB that detects and matches local features (e.g., SIFT, SURF, ORB), estimates transformations via RANSAC/MLESAC, and refines camera parameters with bundle adjustment. It supports planar, cylindrical, spherical, equirectangular (360x180), and stereographic projections, can identify multiple panoramas in a set, and uses gain compensation and multiband blending for seamless composites. The pipeline is parallel-ready and highly configurable to balance speed and accuracy, with a simple entry script (main.m) and sample datasets to help you get started quickly.

# Stitched images 1:

| Type           | Images                            |
| -------------- | --------------------------------- |
| Stitched image | ![pano_full](assets/church_01.jpeg) |
| Crop box       | ![pano_bbox](assets/church_02.jpeg) |
| Cropped image  | ![pano_crop](assets/church_03.jpeg) |

# Stitched images 2:

| Type           | Images                                  |
| -------------- | --------------------------------------- |
| Stitched image | ![result_26](assets/grand_canyon_01.jpeg) |
| Cropped image  | ![cropped](assets/grand_canyon_02.jpeg)   |

# Requirements

* MATLAB
* Computer Vision Toolbox
* Image Processing Toolbox
* Parallel Computing Toolbox

# Run command

Please use the `main.m` to run the program. Change the `folderPath      = '../../../Data/Generic';` to your desired folder path. Also, change the `folderName      = '';` to a valid name. You can download the whole Generic folder datasets in [AutoPanoStitch Stitching Datasets Compilation](https://1drv.ms/f/s!AlFYM4jwmzqrtaBpxVMpJegvN9QVZw?e=UIaYug).

Change the hyper parameters accordingly if needed. But it is not required though.

```
%% Inputs 2
%--------------------------------------------------------------------------
% Parallel workers
input.numCores = str2double(getenv('NUMBER_OF_PROCESSORS')); % Number of cores for parallel processing
input.poolType = 'numcores'; % 'numcores' | 'Threads'

%% Inputs 3
% Feature extraction (SIFT recommended as you get large number of consistent features)
input.detector = 'SIFT'; % 'SIFT' | 'vl_SIFT' | 'HARRIS' | 'FAST' | 'SURF' | 'BRISK' | 'ORB' | 'KAZE'
input.Sigma = 1.6; % Sigma of the Gaussian (1.4142135623)
input.NumLayersInOctave = 4; % Number of layers in each octave -- SIFT only
input.ContrastThreshold = 0.00133; % Contrast threshold for selecting the strongest features,
% specified as a non-negative scalar in the range [0,1].
% The threshold is used to filter out weak features in
% low-contrast regions of the image. -- SIFT only
input.EdgeThreshold = 6; % Edge threshold, specified as a non-negative scalar greater than or equal to 1.
% The threshold is used to filter out unstable edge-like features  -- SIFT only

% Features matching
input.useMATLABFeatureMatch = 0; % Use MATLAB default matchFeatures function: 0-off | 1-on (very fast)
input.Matchingmethod = 'Approximate'; % 'Exhaustive' (default) | 'Approximate'
input.ApproxFloatNNMethod = 'subset_pdist2'; % Nearset neighbor finding methods: 'pca_2nn' 'subset_pdist2'; 'kdtree'
                                                                            % Speed: fast | slow | super slow
                                                                            % Accuracy: ordinary | very accurate | very accurate
input.Matchingthreshold = 3.5; % 10.0 or 1.0 (default) | percent value in the range (0, 100] | depends on

% binary and non-binary features. Default: 3.5. Increase this to >= 10 for binary features
input.Ratiothreshold = 0.6; % ratio in the range (0,1]
input.ApproxNumTables = 8;
input.ApproxBitsPerKey = 24; % for 256-bit ORB; 32 is also fine
input.ApproxProbes = 8;

% Image matching (RANSAC/MLESAC)            MLESAC - recommended
input.useMATLABImageMatching = 0; % Use MATLAB default estgeotform2d function: 0-off | 1-on
input.imageMatchingMethod = 'ransac'; % 'ransac' | 'mlesac'. RANSAC or MLESAC. Both gives consistent matches.
% MLESAC - recommended. As it has some tight bounds and validation checks.

% RANSAC execution time for projective case is ~1.35 times higher than MLESAC.
input.maxIter = 500; % RANSAC/MLESAC maximum iterations
input.maxDistance = 2.5; % Maximum distance (pixels) increase this to get more matches. Default: 1.5
% For large image RANSAC/MLESAC requires maxDistance 1-3 pixels
% more than the default value of 1.5 pixels.
input.inliersConfidence = 99.9; % Inlier confidence [0, 100]
input.transformationType = 'projective'; % Motion model: 'projective'

% Bundle adjustment (BA)
input.maxIterLM = 30;
input.lambda = 1e-3;
input.sigmaHuber = 2.0;
input.verboseLM = false;
input.focalEstimateMethod = 'wConstraint'; % 'shumSzeliski' (sometimes unstable)
                                           %  'wConstraint' (stable)
                                           %  'shumSzeliskiOneH' (sometimes unstable)

% Gain compensation
input.gainCompensation = 1; % 1 - on | 2 - off
input.sigmaN = 10; % Standard deviations of the normalised intensity error
input.sigmag = 0.1; % Standard deviations of the gain

% Blending
input.blending = 'multiband'; % 'multiband' | 'linear' | 'none'
input.bands = 2; % bands (2 - 6 is enough)
input.MBBsigma = 1; % Multi-band Gaussian sigma

% Rendering panorama
input.resizeImage = 1; % Resize input images
input.resizeStitchedImage = 0; % Resize stitched image
input.panorama2DisplaynSave = ... % "planar" | "cylindrical" | "spherical" | "equirectangular" | "stereographic" | Use:[]
"spherical"; %      ["planar", "cylindrical", "spherical" ...
% "equirectangular", "stereographic"];

% Post-processing
input.canvas_color = 'black'; % Panorama canvas color 'black' | 'white'
input.blackRange = 0; % Minimum dark pixel value to crop panaroma
input.whiteRange = 250; % Minimum bright pixel value to crop panaroma
input.showKeypointsPlot = 0; % Display keypoints plot (parfor suppresses this flag, so no use)
input.displayPanoramas = true; % Display panoramas in figure
input.showPanoramaImgsNums = false; % Display the panorama images with numbers after tranform 0 or 1
input.showCropBoundingBox = false; % Display cropping bounding box 0 | 1
input.cropPanorama = false; % Write panorama image to disk 0 | 1
input.imageWrite = false; % Write panorama image to disk 0 | 1
```

# Note

This program produces `planar`, `cylindrical`, `spherical`, `equirectangular (360x180)` and `stereographic` inputs for feature extraction, matching and image matching should be selected. Generally, SO(3) formulation works well on `planar`, `cylindrical` or `spherical` projections with `projective` transformation should work well in most of the cases. However, some panoramas specifically looks good in `affine`/`rigid`/`similarity`/`translation` transformations, e.g. flatbed scanner or whiteboard (`affine` works well too) images. However, using these motion models cannot be directly used with SO(3) formulation and requires a separate formulation accordingly.

Currently, planar, cyclindrical, spherical, equirectangular (360x180) and stereographic (planet) projections stitching is supported in this version and can recognize multiple panoramas. This work is in progress, further improvements such as the inclusion of other projections and motion models, runtime speed optimization and Graphical User Interface (GUI) will be made in the future.

# Image stitching/panorama datasets

Creating image stitching datasets takes a lot of time and effort. During my Ph.D. days, I tried to compile datasets that were comprehensive to have `planar`, `cylindrical`, `spherical`, and full view `360 x 180-degree` panoramas. These datasets posed a real challenge to the automatic stitching method. If all these datasets are stitched well, it definitely shows the robustness of your stitching method.

All these datasets are public! Some of them were from my Ph.D. studies (especially on cracks) and most of them were downloaded from the internet. I do not remember the individual names of the dataset providers. But I acknowledge their work and I am thankful to all of them! I hope you appreciate their efforts in making these datasets public to advance the research!

Below are some samples from the datasets. There are 100+ `panorama` or `image stitching/registration` datasets in total. You can download them in [AutoPanoStitch Stitching Datasets Compilation](https://1drv.ms/f/s!AlFYM4jwmzqrtaBpxVMpJegvN9QVZw?e=UIaYug). Please note that this dataset compilation is more aligned towards the qualitative analysis of the image stitching problem. For quantitative analaysis, I recommend using [Quantitative Image Stitching Datasets](https://github.com/visionxiang/Image-Stitching-Dataset). If I come across any interesting and challenging datasets, I will expand this compilation.

| Type         | Images                                                               |
| ------------ | -------------------------------------------------------------------- |
| CMU          | ![dataset_samples_CMU0](assets/dataset_samples_CMU0.png)               |
| Grand Canyon | ![dataset_samples_grandcanyon](assets/dataset_samples_grandcanyon.png) |
| Shanghai     | ![dataset_samples_shanghai](assets/dataset_samples_shanghai.png)       |
| UCSB         | ![dataset_samples_ucsb4](assets/dataset_samples_ucsb4.png)             |
| Yellowstone  | ![dataset_samples_yellowstone](assets/dataset_samples_yellowstone.png) |
| Rio          | ![dataset_samples_rio](assets/dataset_samples_rio.png)                 |

## Known issues

Long, chain-like 1D and 2D sequences (typically more than 10–12 images) may fail to stitch reliably because global bundle adjustment can diverge or settle in poor minima as drift accumulates; moreover, the current bundle adjustment implementation is not optimized and can be slow, with runtime increasing noticeably as the number of images grows. In practice, sequences with fewer than 10–12 images generally stitch well. As a workaround, consider stitching in smaller batches and merging results, ensuring strong image overlap, or running without bundle adjustment for quick previews.

## Citation

### Original work

1. Brown, Matthew, and David G. Lowe. "Automatic panoramic image stitching using invariant features." International journal of computer vision 74 (2007): 59-73.
2. Brown, Matthew, and David G. Lowe. "Recognising panoramas." In ICCV, vol. 3, p. 1218. 2003.

```bibtex
@article{brown2007automatic,
  title={Automatic panoramic image stitching using invariant features},
  author={Brown, Matthew and Lowe, David G},
  journal={International journal of computer vision},
  volume={74},
  pages={59--73},
  year={2007},
  publisher={Springer}
}

@inproceedings{brown2003recognising,
  title={Recognising panoramas.},
  author={Brown, Matthew and Lowe, David G and others},
  booktitle={ICCV},
  volume={3},
  pages={1218},
  year={2003}
}
```

### Cracks change detection datasets

Image stitching datasets for cracks are made available to the public. If you use the dataset related to the cracks in this compilation in your research, please use the following BibTeX entry to cite:

```bibtex
@PhdThesis{preetham2021vision,
author = {{Aghalaya Manjunatha}, Preetham},
title = {Vision-Based and Data-Driven Analytical and Experimental Studies into Condition Assessment and Change Detection of Evolving Civil, Mechanical and Aerospace Infrastructures},
school =  {University of Southern California},
year = 2021,
type = {Dissertations & Theses},
address = {3550 Trousdale Parkway Los Angeles, CA 90089},
month = {December},
note = {Condition assessment, Crack localization, Crack change detection, Synthetic crack generation, Sewer pipe condition assessment, Mechanical systems defect detection and quantification}
}
```

# Licensing

The original implementation of the automatic panaroma stitching by Dr. Matthew Brown was written in C++ and patent licensed under the University of British Columbia. This is being programmed and made available to public for academic and research purposes only. Please cite the relevant citations as provided in the main file.

# Acknowledgements

I express my sincere gratitude to Dr. Matthew Brown for his invaluable time in discussion and who provided clarifications on many questions. In addition, I am thankful to all the authors who made their image stitching datasets public.

# Feedback

Please rate and provide feedback for the further improvements.
