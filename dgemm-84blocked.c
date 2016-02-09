#include <emmintrin.h> 
#include <immintrin.h> //AVX intrinsics
#include <stdio.h>
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

// This subroutine copy an 8 by k block of A
void copyA8(int k, double *A, int lda, double *copy)
{
  int j;
  for(j = 0; j<k; j++)
  {
    double *a_ptr = A+j*lda;
    *copy = *a_ptr;
    *(copy+1) = *(a_ptr+1);
    *(copy+2) = *(a_ptr+2);
    *(copy+3) = *(a_ptr+3);
    *(copy+4) = *(a_ptr+4);
    *(copy+5) = *(a_ptr+5);
    *(copy+6) = *(a_ptr+6);
    *(copy+7) = *(a_ptr+7);

    copy += 8;
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

void update4X4(int k, double *A, int lda, double *B, int ldb, double *C, int ldc)
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

// Test another subroutine that updates an 8 by 4 block of C at a time
void update8X4(int k, double *A, int lda, double* B, int ldb, double *C, int ldc)
{
  int p;
  __m256d c_00_30, c_01_31, c_02_32, c_03_33, c_40_70, c_41_71, c_42_72, c_43_73,
          a_0p_3p, a_4p_7p,
          b_p0, b_p1, b_p2, b_p3;

  c_00_30 = _mm256_loadu_pd(C);
  c_01_31 = _mm256_loadu_pd(C+ldc);
  c_02_32 = _mm256_loadu_pd(C+ldc*2);
  c_03_33 = _mm256_loadu_pd(C+ldc*3);
  c_40_70 = _mm256_loadu_pd(C+4);
  c_41_71 = _mm256_loadu_pd(C+4+ldc);
  c_42_72 = _mm256_loadu_pd(C+4+ldc*2);
  c_43_73 = _mm256_loadu_pd(C+4+ldc*3);


  for(p=0; p<k; p++)
  {
    // Load a
    a_0p_3p = _mm256_loadu_pd(A);
    a_4p_7p = _mm256_loadu_pd(A+4);
    A += lda;
    //printf("Succesfully loaded a col of A.\n");
    // Load b
    b_p0 = _mm256_broadcast_sd(B);     
    b_p1 = _mm256_broadcast_sd(B+1); 
    b_p2 = _mm256_broadcast_sd(B+2); 
    b_p3 = _mm256_broadcast_sd(B+3); 

    B += 4; 

    // First four rows of C updated once
    c_00_30 = _mm256_add_pd(c_00_30, _mm256_mul_pd(a_0p_3p, b_p0));
    c_01_31 = _mm256_add_pd(c_01_31, _mm256_mul_pd(a_0p_3p, b_p1));
    c_02_32 = _mm256_add_pd(c_02_32, _mm256_mul_pd(a_0p_3p, b_p2));
    c_03_33 = _mm256_add_pd(c_03_33, _mm256_mul_pd(a_0p_3p, b_p3));

    // Last four rows of C updated once
    c_40_70 = _mm256_add_pd(c_40_70, _mm256_mul_pd(a_4p_7p, b_p0));
    c_41_71 = _mm256_add_pd(c_41_71, _mm256_mul_pd(a_4p_7p, b_p1));
    c_42_72 = _mm256_add_pd(c_42_72, _mm256_mul_pd(a_4p_7p, b_p2));
    c_43_73 = _mm256_add_pd(c_43_73, _mm256_mul_pd(a_4p_7p, b_p3));
  }

  _mm256_storeu_pd(C, c_00_30);
  _mm256_storeu_pd(C+ldc, c_01_31);
  _mm256_storeu_pd(C+ldc*2, c_02_32);
  _mm256_storeu_pd(C+ldc*3, c_03_33);
  _mm256_storeu_pd(C+4, c_40_70);
  _mm256_storeu_pd(C+4+ldc, c_41_71);
  _mm256_storeu_pd(C+4+ldc*2, c_42_72);
  _mm256_storeu_pd(C+4+ldc*3, c_43_73);
}

void do_block(int lda, int m, int n, int k, double *A, double *B, double *C, int need_to_packB)
{
  int i, j;
  double packedA[m*k], packedB[BLOCK_SIZE*nb]; 

  for (j=0; j<n; j+=4){        /* Loop over the columns of C, unrolled by 4 */
    if (need_to_packB)
      copyB(k, B+j*lda, lda, &packedB[j*k]);
    for (i=0; i<m; i+=8){        /* Loop over the rows of C */
      if (j == 0) copyA8(k, A+i, lda, &packedA[i*k]);
      // for(int p = 0; p < 8*k; p++)
      // {
      //   printf("%f ", packedA[p]);
      //   if (p % k == k - 1) printf("\n");
      // }
      update8X4(k, &packedA[i*k], 8, &packedB[j*k], k, C+i+j*lda, lda);
    }
  }
}


void square_dgemm(int n, double* A, double* B, double* C)
{
  int i, j, pb, ib;

  /* This time, we compute a BLOCK_SIZE x n block of C by a call to the do_block */

  for (j=0; j<n; j+=BLOCK_SIZE){
    pb = min( n-j, BLOCK_SIZE );
    for (i=0; i<n; i+=BLOCK_SIZE){
      ib = min(n-i, BLOCK_SIZE);
      do_block(n, ib, n, pb, A+i+j*n, B+j, C+i, i==0);
    }
  }
}
