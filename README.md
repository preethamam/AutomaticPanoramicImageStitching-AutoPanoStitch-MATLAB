# AutoPanoStitch
[![View Automatic panorama stitcher (AutoPanoStitch) on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/105850-automatic-panorama-stitcher-autopanostitch) [![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=preethamam/AutomaticPanoramicImageStitching-AutoPanoStitch-MATLAB)

Automatic Panorama Stitching software in MATLAB. Spherical, cyclindrical and planar projections stitching is supported in this version and can recognize multiple panoramas.

# Stitched images 1:
| Type | Images |
| --- | --- |
| Stitched image | ![pano_full](assets/church_01.jpeg) |
| Crop box | ![pano_bbox](assets/church_02.jpeg) |
| Cropped image | ![pano_crop](assets/church_03.jpeg) |

# Stitched images 2:
| Type | Images |
| --- | --- |
| Stitched image | ![result_26](assets/grand_canyon_01.jpeg) |
| Cropped image | ![cropped](assets/grand_canyon_02.jpeg) |

# Requirements
MATLAB <br />
Computer Vision Toolbox <br />
Image Processing Toolbox <br />
Parallel Computing Toolbox <br />

# Run command
Please use the `Main_AutoPanoStitch.m` to run the program. Change the `folderPath      = '../../../Data/Generic';` to your desired folder path. Also, change the `folderName      = '';` to a valid name. You can download the whole Generic folder datasets in [AutoPanoStitch Stitching Datasets Compilation](https://1drv.ms/f/s!AlFYM4jwmzqrtaBpxVMpJegvN9QVZw?e=UIaYug).

Change the hyper parameters accordingly if needed. But it is not required though.
```
%% Inputs 3
% Warping
input.warpType = 'cylindrical';   % 'planar' | 'cylindrical' | 'spherical'

% Lens distortion coefficients [k1, k2, k3, p1, p2]
input.distCoeffs = [0, 0, 0, 0, 0];

% Feature extraction (SIFT recommended as you get large number of consistent features)
input.detector = 'SIFT';                % 'SIFT' | 'vl_SIFT' | 'HARRIS' | 'FAST' | 'SURF' | 'BRISK' | 'ORB' | 'KAZE'
input.Sigma = 1.6;                      % Sigma of the Gaussian (1.4142135623)
input.NumLayersInOctave = 4;            % Number of layers in each octave -- SIFT only
input.ContrastThreshold = 0.00133;      % Contrast threshold for selecting the strongest features, 
                                        % specified as a non-negative scalar in the range [0,1]. 
                                        % The threshold is used to filter out weak features in 
                                        % low-contrast regions of the image. -- SIFT only
input.EdgeThreshold = 6;                % Edge threshold, specified as a non-negative scalar greater than or equal to 1. 
                                        % The threshold is used to filter out unstable edge-like features  -- SIFT only

% Features matching 
input.useMATLABFeatureMatch = 0;        % Use MATLAB default matchFeatures function: 0-off | 1-on
input.Matchingmethod = 'Approximate';   % 'Exhaustive' (default) | 'Approximate'
input.Matchingthreshold = 3.5;          % 10.0 or 1.0 (default) | percent value in the range (0, 100] | depends on 
                                        % binary and non-binary features. Default: 3.5. Increase this to >= 10 for binary features
input.Ratiothreshold = 0.6;             % ratio in the range (0,1]
input.ApproxNumTables = 8;
input.ApproxBitsPerKey = 24;            % for 256-bit ORB; 32 is also fine
input.ApproxProbes =  8;

% Image matching (RANSAC/MLESAC)            MLESAC - recommended
input.useMATLABImageMatching = 0;           % Use MATLAB default estgeotform2d function: 0-off | 1-on
input.imageMatchingMethod = 'ransac';       % 'ransac' | 'mlesac'. RANSAC or MLESAC. Both gives consistent matches. 
                                            % MLESAC - recommended. As it has some tight bounds and validation checks.
                                            
                                            % RANSAC execution time for projective case is ~1.35 times higher than MLESAC.
input.maxIter = 500;                        % RANSAC/MLESAC maximum iterations
input.maxDistance = 2.5;                    % Maximum distance (pixels) increase this to get more matches. Default: 1.5
                                            % For large image RANSAC/MLESAC requires maxDistance 1-3 pixels 
                                            % more than the default value of 1.5 pixels.
input.inliersConfidence = 99.9;             % Inlier confidence [0, 100]
input.transformationType = 'projective';    % Motion model: 'projective' | 'affine' | 'similarity' | 'rigid' | 'translation'

% Bundle adjustment (BA)
input.maxIterLM = 100;
input.lambda = 1e-3;
input.sigma_theta = pi/16;
input.rotmTolerance = 1e-9;             % Small epsilon value for numerical stability of rotation matrix

% Gain compensation
input.sigmaN = 10;                  % Standard deviations of the normalised intensity error
input.sigmag = 0.1;                 % Standard deviations of the gain
input.nearestFeaturesNum = 5;       % Nearest images minimum number of features to filter
                                    % distant image matches (filter gain overlap images to reduce time complexity)

% Blending
input.blending = 'multiband';       % 'multiband' | 'linear' | 'none'
input.bands = 2;                    % bands (2 - 6 is enough)
input.MBBsigma = 1;                 % Multi-band Gaussian sigma

% Rendering panorama
input.resizeImage = 1;              % Resize input images
input.resizeStitchedImage = 0;      % Resize stitched image

% Post-processing
input.canvas_color = 'black';       % Panorama canvas color 'black' | 'white'
input.blackRange = 0;               % Minimum dark pixel value to crop panaroma
input.whiteRange = 250;             % Minimum bright pixel value to crop panaroma
input.showKeypointsPlot  = 0;       % Display keypoints plot (parfor suppresses this flag, so no use)
input.displayPanoramas = 1;         % Display panoramas in figure
input.showPanoramaImgsNums = 0;     % Display the panorama images with numbers after tranform 0 or 1
input.showCropBoundingBox = 0;      % Display cropping bounding box 0 | 1
input.imageWrite = 0;               % Write panorama image to disk 0 | 1
```

# Note
Depending on how your images are captured, panaroma being a `spherical`, `cylindrical` or `planar` should be selected judicially using the `input.warpType` and `input.Transformationtype`. Generally, `spherical` or `cylindrical` projections with `affine` or `rigid` transformation should work well in most of the cases. However, some panoramas specifically looks good in `projective` transformation, e.g. flatbed scanner or whiteboard (`affine` works well too) images.

Currently, spherical, cyclindrical and planar projections stitching is supported in this version and can recognize multiple panoramas. This work is in progress, further improvements such as the inclusion of a full view `360 x 180-degree` panoramas stitching (everything visible from a point), automatic panorama straightening, runtime speed optimization and Graphical User Interface (GUI) are under development. Your patience will be appreciated.

# Image stitching/panorama datasets
Creating image stitching datasets takes a lot of time and effort. During my Ph.D. days, I tried to compile datasets that were comprehensive to have `spherical`, `cylindrical`, `planar` and full view `360 x 180-degree` panoramas. These datasets posed a real challenge to the automatic stitching method. If all these datasets are stitched well, it definitely shows the robustness of your stitching method.

All these datasets are public! Some of them were from my Ph.D. studies (especially on cracks) and most of them were downloaded from the internet. I do not remember the individual names of the dataset providers. But I acknowledge their work and I am thankful to all of them! I hope you appreciate their efforts in making these datasets public to advance the research!

Below are some samples from the datasets. There are 100+ `panorama` or `image stitching/registration` datasets in total. You can download them in [AutoPanoStitch Stitching Datasets Compilation](https://1drv.ms/f/s!AlFYM4jwmzqrtaBpxVMpJegvN9QVZw?e=UIaYug). Please note that this dataset compilation is more aligned towards the qualitative analysis of the image stitching problem. For quantitative analaysis, I recommend using [Quantitative Image Stitching Datasets](https://github.com/visionxiang/Image-Stitching-Dataset). If I come across any interesting and challenging datasets, I will expand this compilation.
| Type | Images |
| --- | --- |
| CMU | ![dataset_samples_CMU0](assets/dataset_samples_CMU0.png) |
| Grand Canyon | ![dataset_samples_grandcanyon](assets/dataset_samples_grandcanyon.png) |
| Shanghai | ![dataset_samples_shanghai](assets/dataset_samples_shanghai.png) |
| UCSB | ![dataset_samples_ucsb4](assets/dataset_samples_ucsb4.png) |
| Yellowstone | ![dataset_samples_yellowstone](assets/dataset_samples_yellowstone.png) |
| Rio | ![dataset_samples_rio](assets/dataset_samples_rio.png) |

## Citation
### Original work
[1]. Brown, Matthew, and David G. Lowe. "Automatic panoramic image stitching using invariant features." International journal of computer vision 74 (2007): 59-73. <br />
[2]. Brown, Matthew, and David G. Lowe. "Recognising panoramas." In ICCV, vol. 3, p. 1218. 2003.

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
