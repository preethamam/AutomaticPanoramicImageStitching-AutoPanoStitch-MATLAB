// OpenCV 4.12.0 based
// Windows
// vc16
// mex -v flann_knn.cpp ...
//     -I"C:\opencv\build\include" ...
//     -L"C:\opencv\build\x64\vc16\lib" ...
//     -lopencv_world4120
// 
// vc17
// mex -v flann_knn.cpp ...
//     -I"C:\opencv\build\include" ...
//     -L"C:\opencv\build\x64\vc17\lib" ...
//     -lopencv_world4120
// 
// 
// Linux
// 
// mex -v flann_knn.cpp \
//     CXXFLAGS="\$CXXFLAGS -std=c++14" \
//     LDFLAGS="\$LDFLAGS -Wl,-rpath,'\$ORIGIN'" \
//     -lopencv_world
// 
// macOS
// Using opencv_world (Homebrew OpenCV)
// mex -v flann_knn.cpp \
//     CXXFLAGS="\$CXXFLAGS -std=c++14 -stdlib=libc++" \
//     LDFLAGS="\$LDFLAGS -Wl,-rpath,@loader_path" \
//     -lopencv_world
// 
// If OpenCV is in a non-standard location (Homebrew):
// Intel macs:
// -I/usr/local/include/opencv4
// -L/usr/local/lib
// 
// Apple Silicon (M1/M2/M3):
// -I/opt/homebrew/include/opencv4
// -L/opt/homebrew/lib
// 
// Example: Apple Silicon
// mex -v flann_knn.cpp \
//     CXXFLAGS="\$CXXFLAGS -std=c++14 -stdlib=libc++" \
//     LDFLAGS="\$LDFLAGS -Wl,-rpath,@loader_path" \
//     -I/opt/homebrew/include/opencv4 \
//     -L/opt/homebrew/lib \
//     -lopencv_world

// MATLAB usage
// Non-binary descriptors
// [nnIdxAll, nnDistAll] = flann_knn_win( ...
//         allDesc, allDesc, Kq, 'flann', trees, checks);
// 
// Binary descriptors → BF or FLANN-LSH
// [nnIdxAll, nnDistAll] = flann_knn_win( ...
//     allDesc, allDesc, Kq, 'bf');
// 
// [nnIdxAll, nnDistAll] = flann_knn_win( ...
//     allDesc, allDesc, Kq, 'flann');

// Windows
// Make sure the dll or dependency file is in the same folder level 
// of mex file
// flann_knn_win.mexw64 
// opencv_world4120.dll or opencv_worldxxxx.dll
// 
// Unix/Linux
// flann_knn_unix.mexa64
// libopencv_world.so.4.12.0
// libopencv_world.so.4.12
// libopencv_world.so
// 
// macOS 
// flann_knn.mexmaci64
// You need one of:
// libopencv_world.4.12.0.dylib
// libopencv_world.4.12.dylib
// libopencv_world.dylib

#include "mex.h"
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/features2d.hpp>
#include <string>

static std::string getString(const mxArray* arr)
{
    char* cstr = mxArrayToString(arr);
    std::string s(cstr ? cstr : "");
    if (cstr) mxFree(cstr);
    return s;
}

static void checkReal2D(const mxArray* a, const char* name)
{
    if (mxIsComplex(a)) mexErrMsgIdAndTxt("flann_knn:type", "%s must be real", name);
    if (mxGetNumberOfDimensions(a) != 2) mexErrMsgIdAndTxt("flann_knn:type", "%s must be 2D", name);
}

// Convert MATLAB column-major to OpenCV row-major (transpose)
template<typename T>
static cv::Mat matlabToRowMajor(const mxArray* arr, int cvType)
{
    const mwSize rows = mxGetM(arr);  // MATLAB rows = num features
    const mwSize cols = mxGetN(arr);  // MATLAB cols = descriptor dim
    const T* src = (const T*)mxGetData(arr);
    
    cv::Mat dst((int)rows, (int)cols, cvType);
    
    for (mwSize r = 0; r < rows; ++r) {
        T* dstRow = dst.ptr<T>((int)r);
        for (mwSize c = 0; c < cols; ++c) {
            // MATLAB: column-major, element (r,c) at src[r + c*rows]
            dstRow[c] = src[r + c * rows];
        }
    }
    return dst;
}

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{
    // Backward-compatible signatures:
    //  (A) [idx, dist] = flann_knn(train, k [, method, trees, checks])
    //  (B) [idx, dist] = flann_knn(train, query, k [, method, trees, checks])

    if (nrhs < 2)
        mexErrMsgIdAndTxt("flann_knn:args",
            "Usage: [idx, dist] = flann_knn(train, k [, method, trees, checks])\n"
            "   or: [idx, dist] = flann_knn(train, query, k [, method, trees, checks])");

    checkReal2D(prhs[0], "train");

    const bool trainFloat  = mxIsSingle(prhs[0]);
    const bool trainBinary = mxIsUint8(prhs[0]);
    if (!trainFloat && !trainBinary)
        mexErrMsgIdAndTxt("flann_knn:type",
            "Descriptors must be single (float) or uint8 (binary)");

    // Determine if we have query matrix
    bool hasQuery = false;
    int kArg = 1;      // index of k argument in prhs
    if (nrhs >= 3 && (mxIsSingle(prhs[1]) || mxIsUint8(prhs[1])))
    {
        // treat prhs[1] as query if it's same type as train
        checkReal2D(prhs[1], "query");
        const bool queryFloat  = mxIsSingle(prhs[1]);
        const bool queryBinary = mxIsUint8(prhs[1]);
        if ((trainFloat && queryFloat) || (trainBinary && queryBinary))
        {
            hasQuery = true;
            kArg = 2;
        }
    }

    if (!mxIsDouble(prhs[kArg]) || mxGetNumberOfElements(prhs[kArg]) != 1)
        mexErrMsgIdAndTxt("flann_knn:type", "k must be a scalar double");
    const int k = (int)mxGetScalar(prhs[kArg]);
    if (k <= 0) mexErrMsgIdAndTxt("flann_knn:k", "k must be > 0");

    // Method / params
    std::string method = "flann";
    int methodArg = kArg + 1;
    if (nrhs >= methodArg + 1)
        method = getString(prhs[methodArg]);

    const int trees  = (nrhs >= methodArg + 2) ? (int)mxGetScalar(prhs[methodArg + 1]) : 4;
    const int checks = (nrhs >= methodArg + 3) ? (int)mxGetScalar(prhs[methodArg + 2]) : 32;

    // Train sizes
    const mwSize Ft = mxGetM(prhs[0]);  // num features
    const mwSize D  = mxGetN(prhs[0]);  // descriptor dim

    // Query sizes
    const mxArray* qArr = hasQuery ? prhs[1] : prhs[0];
    const mwSize Fq = mxGetM(qArr);
    const mwSize Dq = mxGetN(qArr);
    if (Dq != D)
        mexErrMsgIdAndTxt("flann_knn:dim", "query must have same descriptor dimension as train");

    // Convert from MATLAB column-major to OpenCV row-major
    cv::Mat train, query;
    if (trainFloat)
    {
        train = matlabToRowMajor<float>(prhs[0], CV_32F);
        query = matlabToRowMajor<float>(qArr, CV_32F);
    }
    else
    {
        train = matlabToRowMajor<uint8_t>(prhs[0], CV_8U);
        query = matlabToRowMajor<uint8_t>(qArr, CV_8U);
    }

    // Outputs: [Fq x k]
    plhs[0] = mxCreateNumericMatrix(Fq, k, mxUINT32_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(Fq, k, mxSINGLE_CLASS, mxREAL);
    uint32_t* outIdx = (uint32_t*)mxGetData(plhs[0]);
    float*    outDst = (float*)mxGetData(plhs[1]);

    // BF for binary
    if (method == "bf")
    {
        if (!trainBinary)
            mexErrMsgIdAndTxt("flann_knn:bf", "BFMatcher only supports uint8 (binary) descriptors");

        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(query, train, matches, k);

        for (mwSize i = 0; i < Fq; ++i)
        {
            const size_t nMatches = matches[(size_t)i].size();
            for (int j = 0; j < k; ++j)
            {
                if ((size_t)j < nMatches) {
                    outIdx[i + (mwSize)j*Fq] = (uint32_t)(matches[(size_t)i][(size_t)j].trainIdx + 1);
                    outDst[i + (mwSize)j*Fq] = (float)matches[(size_t)i][(size_t)j].distance;
                } else {
                    outIdx[i + (mwSize)j*Fq] = 0;
                    outDst[i + (mwSize)j*Fq] = std::numeric_limits<float>::infinity();
                }
            }
        }
        return;
    }

    // FLANN index built once on TRAIN, then query against it
    cv::Mat indices((int)Fq, k, CV_32S);
    cv::Mat dists;

    if (trainFloat)
    {
        dists.create((int)Fq, k, CV_32F);
        cv::flann::Index index(train, cv::flann::KDTreeIndexParams(trees), cvflann::FLANN_DIST_L2);
        index.knnSearch(query, indices, dists, k, cv::flann::SearchParams(checks));
    }
    else
    {
        dists.create((int)Fq, k, CV_32S);
        cv::flann::Index index(train, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
        index.knnSearch(query, indices, dists, k, cv::flann::SearchParams(checks));
    }

    // Copy back (to MATLAB column-major output)
    for (mwSize i = 0; i < Fq; ++i)
    {
        const int* idxRow = indices.ptr<int>((int)i);
        for (int j = 0; j < k; ++j)
        {
            outIdx[i + (mwSize)j*Fq] = (uint32_t)(idxRow[j] + 1);
            if (trainFloat) outDst[i + (mwSize)j*Fq] = dists.at<float>((int)i, j);
            else            outDst[i + (mwSize)j*Fq] = (float)dists.at<int>((int)i, j);
        }
    }
}