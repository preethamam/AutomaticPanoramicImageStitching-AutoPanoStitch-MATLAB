// Compile (MSVC): mex -O CXXFLAGS="\$CXXFLAGS -O3 -march=native" nearest2_hamming_exhaustive_mex.cpp
// Compile (GCC/Clang/MinGW): mex -O CXXFLAGS="\$CXXFLAGS -O3 -march=native -std=c++11" nearest2_hamming_exhaustive_mex.cpp

#include "mex.h"
#include <cstdint>
#include <algorithm>

static inline void make_popcount_lut(uint8_t lut[256]){
    for (int v=0; v<256; ++v){
        uint8_t c=0, x=(uint8_t)v;
        while (x){ c += (x & 1); x >>= 1; }
        lut[v]=c;
    }
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
    if (nrhs!=2) mexErrMsgIdAndTxt("hamm2nn:nrhs","Need Abytes,Bbytes");
    const mxArray* A = prhs[0];
    const mxArray* B = prhs[1];
    if (!mxIsUint8(A) || !mxIsUint8(B)) mexErrMsgIdAndTxt("hamm2nn:type","Inputs must be uint8.");
    if (mxGetNumberOfDimensions(A)!=2 || mxGetNumberOfDimensions(B)!=2) mexErrMsgIdAndTxt("hamm2nn:dim","2D only.");

    const mwSize* sa = mxGetDimensions(A);
    const mwSize* sb = mxGetDimensions(B);
    const mwSize N1 = sa[0];
    const mwSize nb = sa[1];
    const mwSize N2 = sb[0];
    if (sb[1] != nb) mexErrMsgIdAndTxt("hamm2nn:cols","Byte width mismatch.");

    const uint8_t* Ap = (const uint8_t*)mxGetData(A);  // size N1 x nb (column-major)
    const uint8_t* Bp = (const uint8_t*)mxGetData(B);  // size N2 x nb (column-major)

    plhs[0] = mxCreateNumericMatrix(N1,1,mxUINT32_CLASS,mxREAL); // idx2
    plhs[1] = mxCreateNumericMatrix(N1,1,mxSINGLE_CLASS,mxREAL); // d1
    plhs[2] = mxCreateNumericMatrix(N1,1,mxSINGLE_CLASS,mxREAL); // d2
    uint32_t* idx2 = (uint32_t*)mxGetData(plhs[0]);
    float*    d1   = (float*)mxGetData(plhs[1]);
    float*    d2   = (float*)mxGetData(plhs[2]);

    uint8_t LUT[256]; make_popcount_lut(LUT);

    if (N2==0){
        for (mwSize i=0;i<N1;++i){ idx2[i]=0; d1[i]=mxGetNaN(); d2[i]=mxGetNaN(); }
        return;
    }

    // Column-major addressing:
    //  A(i,b) is at Ap[i + b*N1]
    //  B(j,b) is at Bp[j + b*N2]
    for (mwSize i = 0; i < N1; ++i) {

        uint16_t best = 0xFFFF, second = 0xFFFF;
        mwSize ibest = (mwSize)(-1), isecond = (mwSize)(-1);

        for (mwSize j = 0; j < N2; ++j) {
            uint16_t h = 0;
            for (mwSize b = 0; b < nb; ++b) {
                const uint8_t ai = Ap[i + b*N1];
                const uint8_t bj = Bp[j + b*N2];
                h += (uint16_t)LUT[ ai ^ bj ];
            }

            if (h < best) {
                second = best;  isecond = ibest;
                best   = h;     ibest   = j;
            } else if ((h <= second) && (j != ibest)) {
                second = h;     isecond = j;
            }
        }

        if (N2 == 1 || isecond == (mwSize)(-1)) {
            second = (uint16_t)(nb * 8);
            isecond = ibest;
        }

        idx2[i] = (uint32_t)(ibest + 1);  // 1-based
        d1[i]   = (float)best;
        d2[i]   = (float)second;
    }
}
