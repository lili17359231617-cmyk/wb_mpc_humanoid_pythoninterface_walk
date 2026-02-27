void foot_l_contact_orientation_wrt_plane_jacobian_sparsity(unsigned long const** row,
                                                            unsigned long const** col,
                                                            unsigned long* nnz) {
   static unsigned long const rows[27] = {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2};
   static unsigned long const cols[27] = {3,4,5,6,7,8,9,10,11,3,4,5,6,7,8,9,10,11,3,4,5,6,7,8,9,10,11};
   *row = rows;
   *col = cols;
   *nnz = 27;
}
