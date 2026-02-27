void foot_r_contact_orientationError_jacobian_sparsity(unsigned long const** row,
                                                       unsigned long const** col,
                                                       unsigned long* nnz) {
   static unsigned long const rows[27] = {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2};
   static unsigned long const cols[27] = {3,4,5,12,13,14,15,16,17,3,4,5,12,13,14,15,16,17,3,4,5,12,13,14,15,16,17};
   *row = rows;
   *col = cols;
   *nnz = 27;
}
