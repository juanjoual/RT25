#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "includes.h"

struct SparseMatrix {
    int n_nz;
    int n_rows;
    int n_cols;
    int *rows;
    int *cols;
    double *vals;
    int *rows_t;
    int *cols_t;
    double *vals_t;

    //methods
    void malloc_cpu();
    void free_cpu();

};

#endif // SPARSEMATRIX_H