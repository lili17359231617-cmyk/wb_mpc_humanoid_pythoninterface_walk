void foot_r_contact_contact_moment_XY_constraint_jacobian_sparsity(unsigned long const** row,
                                                                   unsigned long const** col,
                                                                   unsigned long* nnz) {
   static unsigned long const rows[60] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3};
   static unsigned long const cols[60] = {4,5,6,13,14,15,16,17,18,65,66,67,68,69,70,4,5,6,13,14,15,16,17,18,65,66,67,68,69,70,4,5,6,13,14,15,16,17,18,65,66,67,68,69,70,4,5,6,13,14,15,16,17,18,65,66,67,68,69,70};
   *row = rows;
   *col = cols;
   *nnz = 60;
}
