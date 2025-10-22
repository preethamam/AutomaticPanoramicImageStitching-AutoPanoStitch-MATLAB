%% Inputs
%--------------------------------------------------------------------------
% I/O

% Folder path
if ismac
    % Code to run on Mac platform
    folderPath      = '../../../../../../../Team Work/Team CrackSTITCH/Datasets/Generic';
elseif isunix
    % Code to run on Linux platform
    folderPath      = '../../../../../../../Team Work/Team CrackSTITCH/Datasets/Generic';
elseif ispc
    % Code to run on Windows platform
    folderPath      = '..\..\..\..\..\..\Team Work\Team CrackSTITCH\Datasets\Generic';
else
    disp('Platform not supported')
end

% Folder name that consists of the images set
folderName      = '';

%% Inputs 2
%--------------------------------------------------------------------------
% Parallel workers
input.numCores = str2double(getenv('NUMBER_OF_PROCESSORS'));    % Number of cores for parallel processing
input.poolType = 'numcores';     % 'numcores' | 'Threads'

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