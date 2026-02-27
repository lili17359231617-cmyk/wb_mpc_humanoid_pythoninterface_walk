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

void dynamics_jump_map_forward_zero(double const *const * in,
                                    double*const * out,
                                    struct LangCAtomicFun atomicFun) {
   //independent variables
   const double* x = in[0];

   //dependent variables
   double* y = out[0];

   // auxiliary variables

   // dependent variables without operations
   y[0] = x[1];
   y[1] = x[2];
   y[2] = x[3];
   y[3] = x[4];
   y[4] = x[5];
   y[5] = x[6];
   y[6] = x[7];
   y[7] = x[8];
   y[8] = x[9];
   y[9] = x[10];
   y[10] = x[11];
   y[11] = x[12];
   y[12] = x[13];
   y[13] = x[14];
   y[14] = x[15];
   y[15] = x[16];
   y[16] = x[17];
   y[17] = x[18];
   y[18] = x[19];
   y[19] = x[20];
   y[20] = x[21];
   y[21] = x[22];
   y[22] = x[23];
   y[23] = x[24];
   y[24] = x[25];
   y[25] = x[26];
   y[26] = x[27];
   y[27] = x[28];
   y[28] = x[29];
   y[29] = x[30];
   y[30] = x[31];
   y[31] = x[32];
   y[32] = x[33];
   y[33] = x[34];
   y[34] = x[35];
   y[35] = x[36];
   y[36] = x[37];
   y[37] = x[38];
   y[38] = x[39];
   y[39] = x[40];
   y[40] = x[41];
   y[41] = x[42];
   y[42] = x[43];
   y[43] = x[44];
   y[44] = x[45];
   y[45] = x[46];
   y[46] = x[47];
   y[47] = x[48];
   y[48] = x[49];
   y[49] = x[50];
   y[50] = x[51];
   y[51] = x[52];
   y[52] = x[53];
   y[53] = x[54];
   y[54] = x[55];
   y[55] = x[56];
   y[56] = x[57];
   y[57] = x[58];
}

