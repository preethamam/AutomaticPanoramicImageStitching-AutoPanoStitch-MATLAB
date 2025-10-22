// Compile (MSVC): mex -O CXXFLAGS="\$CXXFLAGS -O3 -march=native" nearest2_hamming_candidates_mex.cpp
// Compile (GCC/Clang/MinGW): mex -O CXXFLAGS="\$CXXFLAGS -O3 -march=native -std=c++11" nearest2_hamming_candidates_mex.cpp


#include "mex.h"
#include <algorithm>
#include <cstdint>
#include <vector>

static inline void make_popcount_lut(uint8_t lut[256]) {
    for (int v = 0; v < 256; ++v) {
        uint8_t c = 0, x = (uint8_t)v;
        while (x) { c += (x & 1); x >>= 1; }
        lut[v] = c;
    }
}

// [ci, d1, d2] = nearest2_hamming_candidates_mex(Arow, Bcand)
// Arow: 1 x nbytes uint8
// Bcand: C x nbytes uint8
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 2) mexErrMsgIdAndTxt("cand:args", "Two inputs required.");
    if (!mxIsUint8(prhs[0]) || !mxIsUint8(prhs[1]))
        mexErrMsgIdAndTxt("cand:type", "Inputs must be uint8.");

    const mwSize* asz = mxGetDimensions(prhs[0]);
    const mwSize* bsz = mxGetDimensions(prhs[1]);

    if (asz[0] != 1) mexErrMsgIdAndTxt("cand:shape", "Arow must be 1 x nbytes.");
    const mwSize nbytes = asz[1];
    const mwSize C = bsz[0];
    if (bsz[1] != nbytes) mexErrMsgIdAndTxt("cand:width", "Arow and Bcand width mismatch.");

    const uint8_t* A = (const uint8_t*)mxGetData(prhs[0]);
    const uint8_t* B = (const uint8_t*)mxGetData(prhs[1]);

    plhs[0] = mxCreateNumericMatrix(1,1,mxUINT32_CLASS,mxREAL); // ci (1-based index within candidates)
    plhs[1] = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL); // d1
    plhs[2] = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL); // d2
    uint32_t* ci = (uint32_t*)mxGetData(plhs[0]);
    float* d1 = (float*)mxGetData(plhs[1]);
    float* d2 = (float*)mxGetData(plhs[2]);

    uint8_t LUT[256]; make_popcount_lut(LUT);

    if (C == 0) { *ci = 0; *d1 = mxGetNaN(); *d2 = mxGetNaN(); return; }
    if (C == 1) { 
        // only one candidate
        uint16_t h = 0; 
        for (mwSize b=0; b<nbytes; ++b) h += (uint16_t)LUT[ A[b] ^ B[b] ];
        *ci = 1; *d1 = (float)h; *d2 = (float)(nbytes*8); 
        return; 
    }

    // find best & second (distinct)
    uint16_t best = 0xFFFF, second = 0xFFFF;
    mwSize ibest = (mwSize)(-1), isecond = (mwSize)(-1);

    for (mwSize j = 0; j < C; ++j) {
        const uint8_t* bj = B + j*nbytes;
        uint16_t h = 0;
        for (mwSize b = 0; b < nbytes; ++b) {
            h += (uint16_t)LUT[ A[b] ^ bj[b] ];
        }
        if (h < best) {
            second = best;  isecond = ibest;
            best   = h;     ibest   = j;
        } else if ( (h <= second) && (j != ibest) ) {
            second = h; isecond = j;
        }
    }

    if (isecond == (mwSize)(-1)) { second = (uint16_t)(nbytes*8); isecond = ibest; }

    *ci = (uint32_t)(ibest + 1);
    *d1 = (float)best;
    *d2 = (float)second;
}
