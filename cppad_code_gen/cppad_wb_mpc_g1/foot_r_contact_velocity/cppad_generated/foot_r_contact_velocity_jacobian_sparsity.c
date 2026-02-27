void foot_r_contact_velocity_jacobian_sparsity(unsigned long const** row,
                                               unsigned long const** col,
                                               unsigned long* nnz) {
   static unsigned long const rows[63] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
   static unsigned long const cols[63] = {3,4,5,12,13,14,15,16,17,29,30,31,32,33,34,41,42,43,44,45,46,3,4,5,12,13,14,15,16,17,29,30,31,32,33,34,41,42,43,44,45,46,3,4,5,12,13,14,15,16,17,29,30,31,32,33,34,41,42,43,44,45,46};
   *row = rows;
   *col = cols;
   *nnz = 63;
}
