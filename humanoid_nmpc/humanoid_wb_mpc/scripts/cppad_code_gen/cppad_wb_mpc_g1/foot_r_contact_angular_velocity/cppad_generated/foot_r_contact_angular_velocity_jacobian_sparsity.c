void foot_r_contact_angular_velocity_jacobian_sparsity(unsigned long const** row,
                                                       unsigned long const** col,
                                                       unsigned long* nnz) {
   static unsigned long const rows[53] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
   static unsigned long const cols[53] = {3,4,5,12,13,14,15,16,17,32,33,34,41,42,43,44,45,46,3,4,5,12,13,14,15,16,17,32,33,34,41,42,43,44,45,46,4,5,12,13,14,15,16,17,32,33,34,41,42,43,44,45,46};
   *row = rows;
   *col = cols;
   *nnz = 53;
}
