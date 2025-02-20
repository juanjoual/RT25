#include "SparseMatrix.h"


void SparseMatrix::malloc_cpu() {
    rows = (int *) malloc(n_nz*sizeof(int));
    cols = (int *) malloc(n_nz*sizeof(int));
    vals = (double *) malloc(n_nz*sizeof(double));
    n_rows = 0;
    n_cols = 0;

}

void SparseMatrix::free_cpu() {
        free(rows);
        free(cols);
        free(vals);
}