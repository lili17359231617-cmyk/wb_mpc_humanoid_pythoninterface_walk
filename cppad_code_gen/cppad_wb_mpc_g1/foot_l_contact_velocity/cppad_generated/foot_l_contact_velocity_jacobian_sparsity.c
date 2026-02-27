void foot_l_contact_velocity_jacobian_sparsity(unsigned long const** row,
                                               unsigned long const** col,
                                               unsigned long* nnz) {
   static unsigned long const rows[63] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
   static unsigned long const cols[63] = {3,4,5,6,7,8,9,10,11,29,30,31,32,33,34,35,36,37,38,39,40,3,4,5,6,7,8,9,10,11,29,30,31,32,33,34,35,36,37,38,39,40,3,4,5,6,7,8,9,10,11,29,30,31,32,33,34,35,36,37,38,39,40};
   *row = rows;
   *col = cols;
   *nnz = 63;
}
