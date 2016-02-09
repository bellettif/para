#include <emmintrin.h> 
const char* dgemm_desc = "Optimized version for n %4 == 0.";

#define BLOCK_SIZE 128
#define nb 1000

#define min( i, j ) ( (i)<(j) ? (i): (j) )
void copyA(int k, double *A, int lda, double *copy)
{
  int j;
  for(j=0; j<k; j++)
  {  /* loop over columns of A */
    double *a_ptr = A+j*lda;

    *copy = *a_ptr;
    *(copy+1) = *(a_ptr+1);
    *(copy+2) = *(a_ptr+2);
    *(copy+3) = *(a_ptr+3);

    copy += 4;
  }
}

void copyB(int k, double *B, int ldb, double *copy)
{
  int i;
  double *b_i0 = B, *b_i1 = B+ldb, *b_i2 = B+2*ldb, *b_i3 = B+3*ldb;

  // B is packed in row-major form
  for(i=0; i<k; i++)
  {
    *copy++ = *b_i0++;
    *copy++ = *b_i1++;
    *copy++ = *b_i2++;
    *copy++ = *b_i3++;
  }
}

void update4X4(int k, double *A, int lda,  double *B, int ldb, double *C, int ldc)
{
  int p;
  __m128d c_00_c_10, c_01_c_11, c_02_c_12, c_03_c_13,
          c_20_c_30, c_21_c_31, c_22_c_32, c_23_c_33,
          a_0p_a_1p, a_2p_a_3p,
          b_p0, b_p1, b_p2, b_p3;

  c_00_c_10 = _mm_loadu_pd(C);
  c_01_c_11 = _mm_loadu_pd(C+ldc);
  c_02_c_12 = _mm_loadu_pd(C+ldc*2);
  c_03_c_13 = _mm_loadu_pd(C+ldc*3);
  c_20_c_30 = _mm_loadu_pd(C+2);
  c_21_c_31 = _mm_loadu_pd(C+2+ldc);
  c_22_c_32 = _mm_loadu_pd(C+2+ldc*2);
  c_23_c_33 = _mm_loadu_pd(C+2+ldc*3);

  for(p=0; p<k; p++)
  {
    a_0p_a_1p = _mm_load_pd(A);
    a_2p_a_3p = _mm_load_pd(A+2);
    A += 4;

    b_p0 = _mm_load1_pd(B);
    b_p1 = _mm_load1_pd(B+1);
    b_p2 = _mm_load1_pd(B+2);
    b_p3 = _mm_load1_pd(B+3);

    B += 4;

    /* First row and second rows */
    c_00_c_10 = _mm_add_pd(c_00_c_10, _mm_mul_pd(a_0p_a_1p, b_p0));
    c_01_c_11 = _mm_add_pd(c_01_c_11, _mm_mul_pd(a_0p_a_1p, b_p1));
    c_02_c_12 = _mm_add_pd(c_02_c_12, _mm_mul_pd(a_0p_a_1p, b_p2));
    c_03_c_13 = _mm_add_pd(c_03_c_13, _mm_mul_pd(a_0p_a_1p, b_p3));

    /* Third and fourth rows */
    c_20_c_30 = _mm_add_pd(c_20_c_30, _mm_mul_pd(a_2p_a_3p, b_p0));
    c_21_c_31 = _mm_add_pd(c_21_c_31, _mm_mul_pd(a_2p_a_3p, b_p1));
    c_22_c_32 = _mm_add_pd(c_22_c_32, _mm_mul_pd(a_2p_a_3p, b_p2));
    c_23_c_33 = _mm_add_pd(c_23_c_33, _mm_mul_pd(a_2p_a_3p, b_p3));
  }
  _mm_storeu_pd(C, c_00_c_10);
  _mm_storeu_pd(C+ldc, c_01_c_11);
  _mm_storeu_pd(C+ldc*2, c_02_c_12);
  _mm_storeu_pd(C+ldc*3, c_03_c_13);
  _mm_storeu_pd(C+2, c_20_c_30);
  _mm_storeu_pd(C+2+ldc, c_21_c_31);
  _mm_storeu_pd(C+2+ldc*2, c_22_c_32);
  _mm_storeu_pd(C+2+ldc*3, c_23_c_33);
}

void do_block(int lda, int m, int n, int k, double *A, double *B, double *C, int need_to_packB)
{
  int i, j;
  double packedA[m*k], packedB[BLOCK_SIZE*nb];

  for (j=0; j < n / 4; ++j){        /* Loop over the columns of C, unrolled by 4 */
    if (need_to_packB)
      copyB(k, B + 4 * j * lda, lda, &packedB[4 * j * k]);
    for (i=0; i < m / 4; ++i){        /* Loop over the rows of C */
      if (j == 0)
        copyA(k, A + 4 * i, lda, &packedA[4 * i * k]);
      update4X4(k, &packedA[4 * i * k], 4, &packedB[4 * j * k], k, C + 4 * i + 4 * j * lda, lda);
    }
  }

}


static void update4X4_no_copy(int k, int lda, double* A, double* B, double* C)
{
  // This routine updates a 4 X 4 submatric of C
  // Using SIMD to compute C(0, 0) to C(3,3)
  int p;
  __m128d c_00_10, c_01_11, c_02_12, c_03_13,
          c_20_30, c_21_31, c_22_32, c_23_33,
          a_0p_1p, a_2p_3p,
          b_p0, b_p1, b_p2, b_p3;

  double *b_p0_ptr, *b_p1_ptr, *b_p2_ptr, *b_p3_ptr;
  // Initialize the four columns of B
  b_p0_ptr = B;
  b_p1_ptr = B+lda;
  b_p2_ptr = B+2*lda;
  b_p3_ptr = B+3*lda;

  c_00_10 = _mm_loadu_pd(C);
  c_01_11 = _mm_loadu_pd(C+lda);
  c_02_12 = _mm_loadu_pd(C+lda*2);
  c_03_13 = _mm_loadu_pd(C+lda*3);
  c_20_30 = _mm_loadu_pd(C+2);
  c_21_31 = _mm_loadu_pd(C+2+lda);
  c_22_32 = _mm_loadu_pd(C+2+lda*2);
  c_23_33 = _mm_loadu_pd(C+2+lda*3);

  for(p=0; p<k; p++)
  {
    // Load the first column of A
    a_0p_1p = _mm_load_pd(A+p*lda);
    a_2p_3p = _mm_load_pd(A+p*lda+2);

    // Load the first row ob B
    b_p0 = _mm_load1_pd(b_p0_ptr++);
    b_p1 = _mm_load1_pd(b_p1_ptr++);
    b_p2 = _mm_load1_pd(b_p2_ptr++);
    b_p3 = _mm_load1_pd(b_p3_ptr++);

    // First and second row of C, as outer product of A and B
    c_00_10 = _mm_add_pd(c_00_10, _mm_mul_pd(a_0p_1p, b_p0));
    c_01_11 = _mm_add_pd(c_01_11, _mm_mul_pd(a_0p_1p, b_p1));
    c_02_12 = _mm_add_pd(c_02_12, _mm_mul_pd(a_0p_1p, b_p2));
    c_03_13 = _mm_add_pd(c_03_13, _mm_mul_pd(a_0p_1p, b_p3));

    // Third and fourth row of C
    c_20_30 = _mm_add_pd(c_20_30, _mm_mul_pd(a_2p_3p, b_p0));
    c_21_31 = _mm_add_pd(c_21_31, _mm_mul_pd(a_2p_3p, b_p1));
    c_22_32 = _mm_add_pd(c_22_32, _mm_mul_pd(a_2p_3p, b_p2));
    c_23_33 = _mm_add_pd(c_23_33, _mm_mul_pd(a_2p_3p, b_p3));
  }

  _mm_storeu_pd(C, c_00_10);
  _mm_storeu_pd(C+lda, c_01_11);
  _mm_storeu_pd(C+2*lda, c_02_12);
  _mm_storeu_pd(C+3*lda, c_03_13);
  _mm_storeu_pd(C+2, c_20_30);
  _mm_storeu_pd(C+2+lda, c_21_31);
  _mm_storeu_pd(C+2+lda*2, c_22_32);
  _mm_storeu_pd(C+2+lda*3, c_23_33);

}


static void do_block_no_copy (int lda, int m, int n, int k, double* A, double* B, double* C)
{
  // This routine serves as an inner loop for computing block of C
  int i, j;
  for(j=0; j < n / 4; ++j)
  {
    for(i=0; i < m / 4; ++i)
      // Update the 4X4 submatrix of block C in one routine
      update4X4_no_copy(k, lda, A + 4 * i, B + 4 * j * lda, C + 4 * i + 4 * j * lda);
  }
}

void square_dgemm(int n, double* A, double* B, double* C)
{
  int i, j, pb, ib;

  /* This time, we compute a BLOCK_SIZE x n block of C by a call to the do_block */

  for (j=0; j<n; j+=BLOCK_SIZE){
    pb = min(n-j, BLOCK_SIZE ); // Row stride
    for (i=0; i<n; i+=BLOCK_SIZE){
      ib = min(n-i, BLOCK_SIZE); // Column stride
      do_block(n, ib, n, pb, A+i+j*n, B+j, C+i, i==0);
      //do_block_no_copy(n, ib, n, pb, A+i+j*n, B+j, C+i);
    }
  }

  // Finish the job is the size of the block is not a multiple of 4

  int l;
  double c_i_j;
  for (j = 0; j < n - (n % 4); ++j){
    for (i = n - (n % 4); i < n; ++ i){
      c_i_j = 0;
      for(l = 0; l < n; ++l){
        c_i_j += (*(A + i + l * n)) * (*(B + l + j * n));
      }
      *(C + i + j*n) = c_i_j;
    }
  }

  for (j = n - (n % 4); j < n; ++j){
    for (i = 0; i < n; ++ i){
      c_i_j = 0;
      for(l = 0; l < n; ++l){
        c_i_j += (*(A + i + l * n)) * (*(B + l + j * n));
      }
      *(C + i+ j*n) = c_i_j;
    }
  }


}
