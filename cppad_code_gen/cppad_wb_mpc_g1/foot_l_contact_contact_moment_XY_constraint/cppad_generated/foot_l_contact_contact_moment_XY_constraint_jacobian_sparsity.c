void foot_l_contact_contact_moment_XY_constraint_jacobian_sparsity(unsigned long const** row,
                                                                   unsigned long const** col,
                                                                   unsigned long* nnz) {
   static unsigned long const rows[60] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3};
   static unsigned long const cols[60] = {4,5,6,7,8,9,10,11,12,59,60,61,62,63,64,4,5,6,7,8,9,10,11,12,59,60,61,62,63,64,4,5,6,7,8,9,10,11,12,59,60,61,62,63,64,4,5,6,7,8,9,10,11,12,59,60,61,62,63,64};
   *row = rows;
   *col = cols;
   *nnz = 60;
}
