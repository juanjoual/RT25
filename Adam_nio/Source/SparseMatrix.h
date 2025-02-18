#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H


#include <stdio.h> 
#include <stdlib.h> 
#include <dirent.h> 
#include <signal.h>
#include <string.h>


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

    void malloc_cpu();
    void free_cpu();

};

#endif // SPARSEMATRIX_H