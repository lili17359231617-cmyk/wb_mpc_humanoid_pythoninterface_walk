#include <math.h>
#include <stdio.h>

typedef struct Array {
    void* data;
    unsigned long size;
    int sparse;
    const unsigned long* idx;
    unsigned long nnz;
} Array;

struct LangCAtomicFun {
    void* libModel;
    int (*forward)(void* libModel,
                   int atomicIndex,
                   int q,
                   int p,
                   const Array tx[],
                   Array* ty);
    int (*reverse)(void* libModel,
                   int atomicIndex,
                   int p,
                   const Array tx[],
                   Array* px,
                   const Array py[]);
};

void foot_r_contact_orientation_wrt_plane_forward_zero(double const *const * in,
                                                       double*const * out,
                                                       struct LangCAtomicFun atomicFun) {
   //independent variables
   const double* x = in[0];

   //dependent variables
   double* y = out[0];

   // auxiliary variables
   double v[33];

   v[0] = sin(x[3]);
   v[1] = cos(x[4]);
   v[2] = v[0] * v[1];
   v[3] = cos(x[12]);
   v[4] = sin(x[4]);
   v[5] = v[0] * v[4];
   v[6] = cos(x[5]);
   v[7] = cos(x[3]);
   v[8] = sin(x[5]);
   v[9] = v[5] * v[6] - v[7] * v[8];
   v[10] = sin(x[12]);
   v[11] = 0 - v[10];
   v[12] = v[2] * v[3] + v[9] * v[11];
   v[9] = v[2] * v[10] + v[9] * v[3];
   v[2] = 0.984743944795031 * v[12] + 0.174009664069329 * v[9];
   v[13] = sin(x[14]);
   v[14] = 0 - v[13];
   v[15] = sin(x[13]);
   v[16] = 0.984743944795031 * v[15];
   v[5] = v[5] * v[8] + v[7] * v[6];
   v[17] = cos(x[13]);
   v[18] = -0.174009664069329 * v[15];
   v[19] = v[9] * v[16] + v[5] * v[17] + v[12] * v[18];
   v[20] = cos(x[14]);
   v[21] = 0 - sin(x[17]);
   v[22] = v[2] * v[20] + v[19] * v[13];
   v[23] = cos(x[15]);
   v[24] = sin(x[15]);
   v[25] = 0 - v[24];
   v[26] = 0.984743944795031 * v[23] + 0.174009664069329 * v[25];
   v[27] = 0.984743944795031 * v[17];
   v[15] = 0 - v[15];
   v[28] = -0.174009664069329 * v[17];
   v[5] = v[9] * v[27] + v[5] * v[15] + v[12] * v[28];
   v[25] = -0.174009664069329 * v[23] + 0.984743944795031 * v[25];
   v[9] = sin(x[16]);
   v[12] = 0.984743944795031 * v[24] + 0.174009664069329 * v[23];
   v[24] = -0.174009664069329 * v[24] + 0.984743944795031 * v[23];
   v[23] = cos(x[16]);
   v[29] = cos(x[17]);
   v[5] = (v[2] * v[14] + v[19] * v[20]) * v[21] + ((v[22] * v[26] + v[5] * v[25]) * v[9] + (v[22] * v[12] + v[5] * v[24]) * v[23]) * v[29];
   v[22] = 0 - v[4];
   v[19] = v[1] * v[6];
   v[2] = v[22] * v[3] + v[19] * v[11];
   v[19] = v[22] * v[10] + v[19] * v[3];
   v[22] = 0.984743944795031 * v[2] + 0.174009664069329 * v[19];
   v[30] = v[1] * v[8];
   v[31] = v[19] * v[16] + v[30] * v[17] + v[2] * v[18];
   v[32] = v[22] * v[20] + v[31] * v[13];
   v[30] = v[19] * v[27] + v[30] * v[15] + v[2] * v[28];
   v[30] = (v[22] * v[14] + v[31] * v[20]) * v[21] + ((v[32] * v[26] + v[30] * v[25]) * v[9] + (v[32] * v[12] + v[30] * v[24]) * v[23]) * v[29];
   v[32] = v[5] * x[60] - v[30] * x[59];
   v[1] = v[7] * v[1];
   v[7] = v[7] * v[4];
   v[4] = v[7] * v[6] + v[0] * v[8];
   v[11] = v[1] * v[3] + v[4] * v[11];
   v[4] = v[1] * v[10] + v[4] * v[3];
   v[1] = 0.984743944795031 * v[11] + 0.174009664069329 * v[4];
   v[7] = v[7] * v[8] - v[0] * v[6];
   v[18] = v[4] * v[16] + v[7] * v[17] + v[11] * v[18];
   v[13] = v[1] * v[20] + v[18] * v[13];
   v[7] = v[4] * v[27] + v[7] * v[15] + v[11] * v[28];
   v[7] = (v[1] * v[14] + v[18] * v[20]) * v[21] + ((v[13] * v[26] + v[7] * v[25]) * v[9] + (v[13] * v[12] + v[7] * v[24]) * v[23]) * v[29];
   v[13] = v[30] * x[58] - v[7] * x[60];
   v[18] = v[7] * x[59] - v[5] * x[58];
   v[7] = 1 + v[7] * x[58] + v[5] * x[59] + v[30] * x[60];
   v[7] = sqrt(v[13] * v[13] + v[32] * v[32] + v[18] * v[18] + v[7] * v[7]);
   y[0] = 0 - v[32] / v[7];
   y[1] = 0 - v[13] / v[7];
   y[2] = 0 - v[18] / v[7];
}

