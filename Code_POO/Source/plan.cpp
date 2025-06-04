#include "Plan.h"

double get_time_ms() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec * 1000 + ts.tv_nsec / 1000000.0;
    } else {
        return 0;
    }
}

double get_time_s() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec + ts.tv_nsec / 1e9;
    } else {
        return 0;
    }
}

int compare_strings(const void *va, const void *vb) {

    char **a = (char **) va;
    char **b = (char **) vb;
    return strcmp(*a, *b);
}

int read_files(const char *path, const char *pattern, char **files) {

    int n_files = 0;
    DIR *d = opendir(path);
    struct dirent *dir;
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (strstr(dir->d_name, pattern) != NULL) {
                files[n_files] = (char *) malloc(1000 * sizeof(char));
                strcpy(files[n_files], path);
                if (path[strlen(path) - 1] != '/') {
                    strcat(files[n_files], (char *) "/");
                }
                strcat(files[n_files], dir->d_name);
                n_files++;
            }
        }
        closedir(d);
    }

    qsort(files, n_files, sizeof(char *), compare_strings);
    return n_files;
}

void Plan::check_line(int result) {
        if (result < 0) {
            fprintf(stderr, "ERROR in %s (%s:%d): Unable to read line.\n", 
                    __func__, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
}


char* Plan::get_str(char *line, size_t len) {
    char *temp = (char *) malloc(len*sizeof(char));
    snprintf(temp, len, "%s", line);
    temp[strcspn(temp, "\r\n")] = 0; // Remove newline
    return temp;
}

int Plan::get_int(char *line, char **end) {
    return strtoll(line, end, 10);
}

float Plan::get_float(char *line, char **end) {
    return strtof(line, end);
}

void Plan::parse_config(const char *path) {
    int n_files = read_files(path, "m_", files);

    FILE *f = fopen(files[0], "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n", 
                __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("Loading %s...\n", files[0]);

    char *line = NULL;
    char *end = NULL;
    size_t len = 0;

    check_line(getline(&line, &len, f));
    name = get_str(line, len);

    check_line(getline(&line, &len, f));
    n_beams = get_int(line, &end);

    n_beamlets = 0;
    n_beamlets_beam = (int *)malloc(n_beams * sizeof(int));
    for (int i = 0; i < n_beams; i++) {
        check_line(getline(&line, &len, f));
        int index = get_int(line, &end);
        int beamlets = get_int(end, &line);
        n_beamlets_beam[index - 1] = beamlets;
        n_beamlets += beamlets;
    }

    check_line(getline(&line, &len, f));
    n_voxels = get_int(line, &end);

    check_line(getline(&line, &len, f));
    dose_grid_scaling = get_float(line, &end);

    check_line(getline(&line, &len, f));
    n_regions = get_int(line, &end);

    regions = (Region *) malloc(n_plans*n_regions*sizeof(Region));
    for (int i = 0; i < n_regions; i++) {
        check_line(getline(&line, &len, f));
        get_int(line, &end);
        char *name = get_str(end + 1, len);
        regions[i].name = name;
        regions[i].n_voxels = 0;
    }
    for (int k = 1; k < n_plans; k++) {
        for (int i = 0; i < n_regions; i++) {
            regions[k*n_regions + i].name = regions[i].name;
            regions[k*n_regions + i].n_voxels = 0;
        }
    }

    line = NULL;
    len = 0;
    while (getline(&line, &len, f) != -1) {
        fprintf(stderr, "[WARNING] Line not processed: %s", line);
    }

    fclose(f);
    free(files[0]);
}

void Plan::parse_voxel_regions(const char *path) {
    int n_files = read_files(path, "v_", files);

    FILE *f = fopen(files[0], "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n",
                __func__, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("Loading %s...\n", files[0]);

    voxel_regions = (char *) malloc(n_voxels*n_regions*sizeof(char));
    char line[1024];
    int num = 0;
    int offset = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%d", &num)) {
            for (int i = 0; i < n_regions; i++) {
                voxel_regions[offset + i*n_voxels] = num & 1;
                num >>= 1;
            }
            offset++;
        } else {
            fprintf(stderr, "ERROR in %s (%s:%d): Unable to read voxel regions.\n",
                    __func__, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < n_regions; i++) {
        for (int j = 0; j < n_voxels; j++) {
            if (voxel_regions[i*n_voxels + j]) {
                for (int k = 0; k < n_plans; k++) {
                    regions[k*n_regions + i].n_voxels += 1;
                }
            }
        }
    }

    fclose(f);
    free(files[0]);
    
}

void Plan::load_spm(const char *path) {
    int n_files = read_files(path, "d_", files);

    FILE **fp = (FILE **) malloc(n_files*sizeof(FILE *));
    for (int i = 0; i < n_files; i++) {
        fp[i] = fopen(files[i], "r");
        if (fp[i] == NULL) {
            fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n",
                    __func__, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        int n_nz = 0;
        int count = fscanf(fp[i], "%d", &n_nz);
        spm.n_nz += n_nz;
    }

    spm.malloc_cpu();

    int idx = 0;
    int offset = 0;
    for (int i = 0; i < n_files; i++) {
        printf("Loading %s... ", files[i]);

        int n_read = 0;
        while (true) {
            int row, col;
            double val;
            int count = fscanf(fp[i], "%d %d %lf", &row, &col, &val);
            if(count == EOF || !count) {
                break;
            }

            int new_col = offset + col;
            spm.rows[idx] = row;
            spm.cols[idx] = new_col;
            spm.vals[idx] = val;
            //printf("%d %d %lf\n", row, new_col, val);
            idx++;
            n_read++;

            if (row > spm.n_rows) {
                spm.n_rows = row;
            }
            if (new_col > spm.n_cols) {
                spm.n_cols = new_col;
            }
        }

        printf("%d values read.\n", n_read);
        offset = spm.n_cols + 1;
        fclose(fp[i]);
        free(files[i]);
    }

    spm.n_rows++;
    // Sometimes there's missing voxels,
    // but we want the dimensions to match for SpMM
    if (spm.n_rows < n_voxels) {
        spm.n_rows = n_voxels;
    }
    spm.n_cols++;
    free(fp);

    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_create_coo(&m, SPARSE_INDEX_BASE_ZERO, spm.n_rows, spm.n_cols, spm.n_nz, spm.rows, spm.cols, spm.vals);
    mkl_sparse_convert_csr(m, SPARSE_OPERATION_TRANSPOSE, &m_t);
    mkl_sparse_convert_csr(m, SPARSE_OPERATION_NON_TRANSPOSE, &m);
    mkl_sparse_set_mv_hint(m, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1e6);
    mkl_sparse_set_mv_hint(m_t, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1e6);
    mkl_sparse_optimize(m);
    mkl_sparse_optimize(m_t);

}

void Plan::load_fluence(const char *path, const char *prefix) {
    int n_files = read_files(path, prefix, files);

    FILE **fp = (FILE **) malloc(n_files*sizeof(FILE *));
    int idx = 0;
    for (int i = 0; i < n_files; i++) {
        printf("Loading %s... ", files[i]);
        fp[i] = fopen(files[i], "r");
        if (fp[i] == NULL) {
            fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n",
                    __func__, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        int n_read = 0;
        while (true) {
            int count = fscanf(fp[i], "%lf", &(fluence[idx]));
            if(count == EOF || !count) {
                break;
            }

            idx++;
            n_read++;
        }

        printf("%d values read.\n", n_read);
        fclose(fp[i]);
        free(files[i]);
    }

    free(fp);

    for (int i = 1; i < n_plans; i++) {
        for (int j = 0; j < n_beamlets; j++) {
            fluence[i*n_beamlets + j] = fluence[j];
        }
    }

}

void Plan::load_coords(const char *path) {
    int n_files = read_files(path, "xcoords_", files);

    beam_maps = (int *) malloc(n_beams*BEAM_MAP_Y*BEAM_MAP_X*sizeof(int));

    for (int i = 0; i < n_beams*BEAM_MAP_Y*BEAM_MAP_X; i++) {
        beam_maps[i] = -1;
    }

    int idx = 0;
    FILE **fp = (FILE **) malloc(n_files*sizeof(FILE *));
    for (int i = 0; i < n_files; i++) {
        fp[i] = fopen(files[i], "r");
        if (fp[i] == NULL) {
            fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n",
                    __func__, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        printf("Loading %s... ", files[i]);

        char ignored[1024];
        int n_read = 0;
        while (true) {
            int col, row;
            int count = fscanf(fp[i], "%d %d", &col, &row);
            if(count == EOF) {
                break;
            } else if (count == 0) {
                if (fgets(ignored, sizeof(ignored), fp[i]) == NULL) {
                    fprintf(stderr, "WARNING: fgets returned NULL while reading %s.\n", files[i]);
                }
            } else if (count == 1) {
                // Header values, ignored
                continue;
            } else if (count == 2) {
                beam_maps[i*BEAM_MAP_Y*BEAM_MAP_X + col*BEAM_MAP_Y + row] = idx;
                n_read++;
                idx++;
            } else {
                fprintf(stderr, "ERROR in %s (%s:%d): While reading coordinate file.\n",
                        __func__, __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }
        }
        printf("%d coordinates read.\n", n_read);
        fclose(fp[i]);
        free(files[i]);
    }

    free(fp);
}

void Plan::init_fluence(float value) {
    for (int i = 0; i < n_plans; i++) {
        for (int j = 0; j < n_beamlets; j++) {
            fluence[i*n_beamlets + j] = value;
            
        }
    }
}

void Plan::print() {
    printf("Name: %s\n", name);
    printf("Number of beams: %d\n", n_beams);
    for (int i = 0; i < n_beams; i++) {
        printf("  Beam %d: %d beamlets\n", i + 1, n_beamlets_beam[i]);
    }
    printf("Total: %d beamlets\n", n_beamlets);
    printf("Number of voxels: %d\n", n_voxels);
    printf("Dose Grid Scaling: %e\n", dose_grid_scaling);
    printf("Number of regions: %d\n", n_regions);
    for (int i = 0; i < n_regions; i++) {
        printf("  Region %2d: %-16s %8d voxels\n", i, regions[i].name, regions[i].n_voxels);
    }
    printf("Dose matrix: %d x %d with %d nonzeros.\n", spm.n_rows, spm.n_cols, spm.n_nz);
    printf("Number of plans: %d.\n", n_plans);
}

void Plan::compute_dose() {
    memset(doses, 0, n_plans*n_voxels*sizeof(*doses));

    double alpha = 1., beta = 0.0;

    double start_time = get_time_s();
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, m, descr, SPARSE_LAYOUT_COLUMN_MAJOR, fluence, n_plans, n_beamlets, beta, doses, n_voxels);
    double elapsed = get_time_s() - start_time;
    //printf("Compute dose mm: %.4f seconds.\n", elapsed);

    #pragma omp parallel for
    for (int i = 0; i < n_voxels*n_plans; i++) {
        doses[i] *= dose_grid_scaling;
        
    }

   
 
}


void Plan::stats() {
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < n_plans; k++) {
        for (int i = 0; i < n_regions; i++) {
            Region *r = &regions[k*n_regions + i];
            
            r->min = 1e10;
            r->max = 0;
            r->avg = 0;
            r->sum_alpha = 0;
            r->v_sum_alpha = 0;
            for (int j = 0; j < n_voxels; j++) {
                if (voxel_regions[i*n_voxels + j]) {
                    double dose = doses[k*n_voxels + j];
                    if (r->min > dose) {
                        r->min = dose;
                    }
                    if (r->max < dose) {
                        r->max = dose;
                    }
                    r->avg += dose;

                    if (dose > 0) {
                        r->sum_alpha += pow((double) dose, 
                                            (double) r->alpha);
                        if (r->is_ptv) {
                            r->v_sum_alpha += pow((double) dose, 
                                                    (double) -r->alpha);
                        }
                    }
                }
            }
            r->avg /= r->n_voxels;
            r->eud = pow(r->sum_alpha/r->n_voxels, 
                            1.0/r->alpha);
            if (r->is_ptv) {
                r->v_eud = pow(r->v_sum_alpha/r->n_voxels, 
                                1.0/-r->alpha);
            }
            if (r->is_optimized) {
                int n = r->penalty;
                int pd = r->pr_eud;
                double eud = r->eud;
                int v_pd = pd + 1; // Hardcoded virtual PTV prescribed dose
                double v_eud = r->v_eud;
                if (r->is_ptv) {
                    r->f = 1/(1 + pow(pd/eud, n));
                    r->dF_dEUD =  (n*r->f/eud) * pow(pd/eud, n);
                    // Virtual EUD to control PTV over-dosage
                    r->v_f = 1/(1 + pow(v_eud/v_pd, n));
                    r->v_dF_dEUD = -(n*r->v_f/v_eud) * pow(v_eud/v_pd, n);
                } else {
                    r->f = 1/(1 + pow(eud/pd, n));
                    r->dF_dEUD = -(n*r->f/eud) * pow(eud/pd, n);
                }
            }
        }
    }
}


void Plan::print_table(int pid) {
    printf("%2d    Region               Min       Avg       Max       EUD     v_EUD\n", pid); 
    for (int i = 0; i < n_regions; i++) {
        Region r = regions[pid*n_regions + i];
        if (true || r.is_optimized) { // Vamos a imprimir todas para probar TROTS
            printf("%-20s %9.4lf %9.4lf %9.4lf %9.4lf %9.4lf\n", r.name, r.min, r.avg, r.max, r.eud, r.v_eud);
        }
    }
}

void Plan::load(const char *plan_path, const char *fluence_path, const char *fluence_prefix) {
    parse_config(plan_path);
    parse_voxel_regions(plan_path);
    load_spm(plan_path);

    fluence = (double *) malloc(n_plans*n_beamlets*sizeof(double));
    smoothed_fluence = (double *) malloc(n_plans*n_beamlets*sizeof(double));
    doses = (double *) malloc(n_plans*n_voxels*sizeof(double));

    load_coords(plan_path);

    //load_fluence(fluence_path, fluence_prefix);
    init_fluence(10);
    print();
}

void Plan::smooth_cpu() {
    int n_neighbors = 8;
    int sum_weights = 1000;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < n_plans; k++) {
        for (int i = 0; i < n_beams; i++) {
            int *neighbors = (int *) malloc(n_neighbors*sizeof(int));
            int poff = k*n_beamlets;
            for (int y = 0; y < BEAM_MAP_Y; y++) {
                for (int x = 0; x < BEAM_MAP_X; x++) {
                    int offset = i*BEAM_MAP_Y*BEAM_MAP_X;
                    int idx = beam_maps[offset + BEAM_MAP_X*y + x];
                    float center_weight = sum_weights - n_neighbors;
                    if (idx >= 0 && idx < n_beamlets) {
                        smoothed_fluence[poff + idx] = 0;
                        neighbors[0] = beam_maps[offset + BEAM_MAP_Y*(y-1) + (x-1)];
                        neighbors[1] = beam_maps[offset + BEAM_MAP_Y*(y-1) + (x  )];
                        neighbors[2] = beam_maps[offset + BEAM_MAP_Y*(y-1) + (x+1)];
                        neighbors[3] = beam_maps[offset + BEAM_MAP_Y*(y  ) + (x-1)];
                        neighbors[4] = beam_maps[offset + BEAM_MAP_Y*(y  ) + (x+1)];
                        neighbors[5] = beam_maps[offset + BEAM_MAP_Y*(y+1) + (x-1)];
                        neighbors[6] = beam_maps[offset + BEAM_MAP_Y*(y+1) + (x  )];
                        neighbors[7] = beam_maps[offset + BEAM_MAP_Y*(y+1) + (x+1)];

                        //if (neighbors[3] < 0 || neighbors[4] < 0) {
                        //    // This is a border beamlet, ignore other rows
                        //    neighbors[0] = -1;
                        //    neighbors[1] = -1;
                        //    neighbors[2] = -1;
                        //    neighbors[5] = -1;
                        //    neighbors[6] = -1;
                        //    neighbors[7] = -1;

                        //}
                        for (int j = 0; j < n_neighbors; j++) {
                            if (neighbors[j] >= 0 && neighbors[j] < n_beamlets) {
                                smoothed_fluence[poff + idx] += fluence[poff + neighbors[j]];
                            } else {
                                center_weight += 0.8;
                            }
                        }
                        smoothed_fluence[poff + idx] += center_weight*fluence[poff + idx];
                        smoothed_fluence[poff + idx] /= sum_weights;
                    }
                }
            }
            free(neighbors);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n_beamlets*n_plans; i++) {
        fluence[i] = smoothed_fluence[i];
        
        
    }



}
