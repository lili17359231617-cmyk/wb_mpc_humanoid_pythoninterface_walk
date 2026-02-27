void foot_l_contact_position_jacobian_sparsity(unsigned long const** row,
                                               unsigned long const** col,
                                               unsigned long* nnz) {
   static unsigned long const rows[29] = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2};
   static unsigned long const cols[29] = {0,3,4,5,6,7,8,9,10,11,1,3,4,5,6,7,8,9,10,11,2,4,5,6,7,8,9,10,11};
   *row = rows;
   *col = cols;
   *nnz = 29;
}
