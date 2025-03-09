#include "Optimize_SGD.h"
#include "Region.h"

static volatile int running = 1;

void interrupt_handler(int signal) {
    running = 0;
}


void Optimize_sgd::voxels_eud(Plan plan, int rid, int pid, double *voxels) {
    Region *r = &plan.regions[pid*plan.n_regions + rid];

    #pragma omp parallel for
    for (int i = 0; i < plan.n_voxels; i++) {
        double dose = plan.doses[pid*plan.n_voxels + i];
        if (plan.voxel_regions[rid*plan.n_voxels + i]) {
            double dEUD_dd = r->eud*pow(dose, r->alpha - 1)/r->sum_alpha;
            voxels[i] = r->dF_dEUD * dEUD_dd;
            if (r->is_ptv) {
                dEUD_dd = r->v_eud*pow(dose, -r->alpha - 1)/r->v_sum_alpha;
                voxels[plan.n_voxels + i] = r->v_dF_dEUD * dEUD_dd;
            }
        } else {
            voxels[i] = 0;
            if (r->is_ptv) {
                voxels[plan.n_voxels + i] = 0;
            }
        }
    }
}

double Optimize_sgd::penalty(Plan plan, unsigned int pid) {
    double penalty = 0;

    for (int i = 0; i < plan.n_regions; i++) {
        Region region = plan.regions[pid*plan.n_regions + i];
        if (region.is_optimized) {
            if (region.pr_min > 0 &&
                region.min < region.pr_min) {
                penalty += region.pr_min - region.min;
            }
            if (region.pr_max > 0 && 
                region.max > region.pr_max) {
                penalty += region.max - region.pr_max;
            }
            if (region.pr_avg_min > 0 && 
                region.avg < region.pr_avg_min) {
                penalty += region.pr_avg_min - region.avg;
            }
            if (region.pr_avg_max > 0 && 
                region.avg > region.pr_avg_max) {
                penalty += region.avg - region.pr_avg_max;
            }
        }
    }
    return penalty;
}

double Optimize_sgd::objective(Plan plan, unsigned int pid) {
    double objective = 1;

    for (int i = 0; i < plan.n_regions; i++) {
        Region region = plan.regions[pid*plan.n_regions + i];
        if (region.is_optimized) {
            objective *= region.f;
            if (region.is_ptv) {
                objective *= region.v_f;
            }
        }
    }
    return objective;
}

void Optimize_sgd::vector_stats(const char *name, double *vector, int n_values) {
    double min = 1e10, max = 0, avg = 0;
    for (int i = 0; i < n_values; i++) {
        if (vector[i] < min) {
            min = vector[i];
        }
        if (vector[i] > max) {
            max = vector[i];
        }
        avg += vector[i];
    }
    avg /= n_values;

    printf("%s: %f %f %f\n", name, min, max, avg);
}

void Optimize_sgd::reduce_gradient(double *voxels, int n_voxels, int n_gradients, int n_plans) {
    for (int k = 0; k < n_plans; k++) {
        unsigned int poff = k*n_voxels*n_gradients;
        for (int i = 0; i < n_voxels; i++) {
            double influence = 0;
            for (int j = 0; j < n_gradients; j++) {
                influence += voxels[poff + j*n_voxels + i];
            }
            voxels[k*n_voxels + i] = influence;
        }
    }
}

void Optimize_sgd::apply_gradient(double *gradient, double *momentum, int n_beamlets, float step, double *fluence, int n_plans) {

    int beta = 0.9;

    //#pragma omp parallel for
    for (int i = 0; i < n_plans*n_beamlets; i++) {
        if (isnan(gradient[i])) {
            printf("%d %f %f %f %f\n", i, fluence[i], gradient[i], momentum[i], fluence[i] + step*momentum[i]);
            exit(EXIT_FAILURE);
        }

        momentum[i] = beta*momentum[i] + (1-beta)*gradient[i];
        fluence[i] += step*momentum[i];

        if (fluence[i] < 0) {
            fluence[i] = 0;
        }
        if (fluence[i] > 0.3) {
            fluence[i] = 0.3;
        }
    }
}

int Optimize_sgd::descend(Plan plan, double *voxels, double *gradient, double *momentum, float step) {
    memset(voxels, 0, plan.n_plans*plan.n_voxels*sizeof(*voxels));
    memset(gradient, 0, plan.n_plans*plan.n_beamlets*sizeof(*gradient));

    int n_gradients = 0;
    int offset = 0;

    for (int k = 0; k < plan.n_plans; k++) {
        for (int i = 0; i < plan.n_regions; i++) {
            Region *region = &plan.regions[i];
            if (region->is_optimized) {
                voxels_eud(plan, i, k, &voxels[offset]);
                offset += plan.n_voxels;
                n_gradients++;
                if (region->is_ptv) {
                    offset += plan.n_voxels;
                    n_gradients++;
                }
            }
        }
    }

    n_gradients /= plan.n_plans;
    reduce_gradient(voxels, plan.n_voxels, n_gradients, plan.n_plans);
    
    double alpha = 1.0, beta = 0.0;
    double start_time = get_time_s();
    //mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, plan.m_t, plan.descr, voxels, beta, gradient);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, plan.m_t, plan.descr,SPARSE_LAYOUT_COLUMN_MAJOR, 
                    voxels, plan.n_plans, plan.n_voxels, beta, gradient, plan.n_beamlets);
    double elapsed = get_time_s() - start_time;
    //printf("Descend mm: %.4f seconds.\n", elapsed);
    
    apply_gradient(gradient, momentum, plan.n_beamlets, step, plan.fluence, plan.n_plans);

    return n_gradients;
}

void Optimize_sgd::optimize(Plan plan) {

    int gradients_per_region = 3; // Warning, hardcoded!
    // *2 because we need two for PTVs, but we're wasting space on organs.
    double *voxels = (double *) malloc(plan.n_plans*plan.n_voxels*plan.n_regions*2*sizeof(*voxels)); 
    double *gradient = (double *) malloc(plan.n_plans*plan.n_beamlets*sizeof(*gradient));
    double *momentum = (double *) malloc(plan.n_plans*gradients_per_region*plan.n_regions*plan.n_beamlets*sizeof(*momentum));
    memset(momentum, 0, plan.n_plans*gradients_per_region*plan.n_regions*plan.n_beamlets*sizeof(*momentum));

    int rid_sll = 0;
    int rid_slr = 0;
    for (int i = 0; i < plan.n_regions; i++) {
        if (strcmp(plan.regions[i].name, "slinianka L") == 0) {
            rid_sll = i;
        } else if (strcmp(plan.regions[i].name, "slinianka P") == 0) {
            rid_slr = i;
        }
    }

    plan.compute_dose();
    plan.stats();
    printf("Initial solution:\n");
    for (int k = 0; k < plan.n_plans; k++) {
        unsigned int poff = k*plan.n_regions;
        double pen = penalty(plan, k);
        double obj = plan.regions[poff + rid_sll].avg + plan.regions[poff + rid_slr].avg;
        double f = objective(plan, k);
        printf("%2d   penalty: %9.6f\n", k, pen);
        printf("%2d objective: %9.6f\n", k, obj); 
        printf("%2d         f: %9.8f\n", k, f);
        plan.print_table(k);
    }
    double step = 5e-7;
    double decay = 1e-7;
    double min_step = 1e-1;
    double start_time = get_time_s();
    double current_time;

    int it = 0;
    while (running && get_time_s() - start_time < 600000) {
        descend(plan, voxels, gradient, momentum, step);
        plan.smooth_cpu();
        plan.compute_dose();
        plan.stats();
        //break;

        if (it % 100 == 0) {
            current_time = get_time_s();

            printf("\n[%.3f] Iteration %d %e\n", current_time - start_time, it, step);

            for (int k = 0; k < plan.n_plans; k++) {
                unsigned int poff = k*plan.n_regions;

                double pen = penalty(plan, k);
                double obj = plan.regions[poff + rid_sll].avg + plan.regions[poff + rid_slr].avg;
                double f = objective(plan, k);
                printf("%2d   penalty: %9.6f\n", k, pen);
                printf("%2d objective: %9.6f\n", k, obj); 
                printf("%2d         f: %9.8f\n", k, f);
                plan.print_table(k);
            }
        }
        //if (step > min_step) 
        //    step = step/(1 + decay*it);
        it++;
        if (it == 12000)
            break;
    }

    double elapsed = get_time_s() - start_time;
    printf("\nRan %d iterations in %.4f seconds (%.4f ms/it) \n", it, elapsed, elapsed*1000/it/plan.n_plans);
    for (int k = 0; k < plan.n_plans; k++) {
        unsigned int poff = k*plan.n_regions;

        double pen = penalty(plan, k);
        double obj = plan.regions[poff + rid_sll].avg + plan.regions[poff + rid_slr].avg;
        double f = objective(plan, k);
        printf("%2d   penalty: %9.6f\n", k, pen);
        printf("%2d objective: %9.6f\n", k, obj); 
        printf("%2d         f: %9.8f\n", k, f);
        plan.print_table(k);
    }

    free(voxels);
    free(gradient);
    free(momentum);
}