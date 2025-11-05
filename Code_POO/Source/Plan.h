#ifndef PLAN_H
#define PLAN_H
#include "Region.h"
#include "SparseMatrix.h"

#define BEAM_MAP_X 120
#define BEAM_MAP_Y 120

double get_time_ms();
double get_time_s();
int compare_strings(const void *a, const void *b);
int read_files(const char *path, const char *pattern, char **files);


struct Plan {
    
    char *name;
    int n_beams;
    int n_beamlets;
    int *n_beamlets_beam;
    int n_voxels;
    int n_regions;
    int n_plans;
    double dose_grid_scaling;
    Region* regions;
    char *voxel_regions;
    SparseMatrix spm;
    double *fluence;
    double *smoothed_fluence;
    double *doses;
    char *files[100];
    int *beam_maps;
    sparse_matrix_t m;
    sparse_matrix_t m_t;
    struct matrix_descr descr;


    void check_line(int result);
    char *get_str(char *line, size_t len);
    int get_int(char *line, char **end);
    float get_float(char *line, char **end);
    void parse_config(const char *path);
    void parse_voxel_regions(const char *path);
    void load(const char *path);
    void load_spm(const char *path);
    void load_fluence(const char *path, const char *prefix);
    void load_coords(const char *path);
    void init_fluence(float value);
    void print();
    void compute_dose();

    void stats();
    void print_table(int pid);
    void load(const char *plan_path, const char *fluence_path, const char *fluence_prefix);
    void smooth_cpu();
   

};


#endif // PLAN_H