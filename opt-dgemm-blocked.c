#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#include <stdlib.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
inline static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C, double* transpose_buffer)
{
  int i, j, k;
  register double cij;
  double *b_ptr;
  /* For each row i of A */
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      transpose_buffer[k] = A[i + k * lda];
    }
    /* For each column j of B */
    for (j = 0; j < N; ++j) {
      /* Compute C(i,j) */
      cij = C[i + j * lda];
      b_ptr = B + j * lda;
      for (k = 0; k < K; ++k) {
        cij += transpose_buffer[k] * b_ptr[k];
      }
      C[i + j * lda] = cij;
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  int i, j, k, M, N, K;
  double* transpose_buffer = (double *) malloc(BLOCK_SIZE * sizeof(double));
  /* For each block-row of A */ 
  for (i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	M = min (BLOCK_SIZE, lda-i);
	N = min (BLOCK_SIZE, lda-j);
	K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda, transpose_buffer);
      }
  free(transpose_buffer);
}
