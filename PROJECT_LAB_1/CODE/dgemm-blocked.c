/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

    COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
*/

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
/*cache size = 25 MB
 size of double = 8 bytes
 total no of doubles = 3200
 block size = sqrt(3200/3) for 3 matrix A, B and C = 32 */
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* A, double* B, double* C)
{
    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
    {
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
        {
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE, lda-i);
                int N = min (BLOCK_SIZE, lda-j);
                int K = min (BLOCK_SIZE, lda-k);

                /* Perform individual block dgemm */
                double* A1 = A + i + k*lda;
                double* B1 = B + k + j*lda;
                double* C1 = C + i + j*lda;

                /* For each row i of A1 */
                for (int i1 = 0; i1 < M; ++i1)
                {
                    /* For each column j of B1 */
                    for (int j1 = 0; j1 < N; ++j1)
                    {
                        /* code motion */
                        int jlda = j1*lda;

                        /* Compute C1(i,j) */
                        double cij = C1[i1+jlda];
                        double cij2 = 0;
                        double cij3 = 0;
                        /* Declare k1 outside so that both the loops can use the it. */
                        int k1;
                        for (k1 = 0; k1 < K-6; k1 += 6)
                        {
                            cij += A1[i1+k1*lda] * B1[k1+jlda]
                                        + A1[i1+(k1+1)*lda] * B1[k1+1+jlda];
                            cij2 += A1[i1+(k1+2)*lda] * B1[k1+2+jlda]
                                        + A1[i1+(k1+3)*lda] * B1[k1+3+jlda];
                            cij3 += A1[i1+(k1+4)*lda] * B1[k1+4+jlda]
                                        + A1[i1+(k1+5)*lda] * B1[k1+5+jlda];
                        }

                        for (; k1 < K; ++k1)
                        {
                            cij += A1[i1+k1*lda] * B1[k1+jlda];
                        }

                        C1[i1+jlda] = cij + cij2 + cij3;
                    }
                }
            }
        }
    }
}