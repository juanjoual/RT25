#include "Optimizer.h"

volatile int running = 1;

void interrupt_handler(int signal) {
    running = 0;
}

void Optimizer::voxels_eud(Plan *plan, int rid, int pid, double *voxels) {
    Region *r = &plan->regions[pid*plan->n_regions + rid];

    #pragma omp parallel for
    for (int i = 0; i < plan->n_voxels; i++) {
        double dose = plan->doses[pid*plan->n_voxels + i];
        
        if (plan->voxel_regions[rid*plan->n_voxels + i]) {
            double dEUD_dd = r->eud*pow(dose, r->alpha - 1)/r->sum_alpha;
            voxels[i] = r->dF_dEUD * dEUD_dd;
            if (r->is_ptv) {
                dEUD_dd = r->v_eud*pow(dose, -r->alpha - 1)/r->v_sum_alpha;
                voxels[plan->n_voxels + i] = r->v_dF_dEUD * dEUD_dd;
            }
        } else {
            voxels[i] = 0;
            if (r->is_ptv) {
                voxels[plan->n_voxels + i] = 0;
            }
        }

    }

}

double Optimizer::penalty(Plan *plan, unsigned int pid) {
    double penalty = 0;

    for (int i = 0; i < plan->n_regions; i++) {
        Region region = plan->regions[pid*plan->n_regions + i];
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

double Optimizer::objective(Plan *plan, unsigned int pid) {
    double objective = 1;

    for (int i = 0; i < plan->n_regions; i++) {
        Region region = plan->regions[pid*plan->n_regions + i];
        if (region.is_optimized) {
            objective *= region.f;
            if (region.is_ptv) {
                objective *= region.v_f;
            }
        }
    }
    
    return objective;
}

void Optimizer::vector_stats(const char *name, double *vector, int n_values) {
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

void Optimizer::reduce_gradient(double *voxels, int n_voxels, int n_gradients, int n_plans) {
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

void Optimizer::adam(double *gradient, double *momentum, double *variance, int n_beamlets, float step, double *fluence, int n_plans, int t, double beta1, double beta2, double epsilon) {
      
    //#pragma omp parallel for
    for (int i = 0; i < n_plans*n_beamlets; i++) {
        if (isnan(gradient[i])) {
            printf("i[%d], fluence[%d] = %f, gradient[%d] = %f, momentum[%d] = %f, variance[%d]= %f \n", i, i, fluence[i], i, gradient[i], i, momentum[i], i, variance[i]);
            exit(EXIT_FAILURE);
        }

        // Actualizar momento del primer orden (m_t)
        momentum[i] = beta1*momentum[i] + (1-beta1)*gradient[i]; 
        // Actualizar momento del segundo orden (v_t)
        variance[i] = beta2*variance[i] + (1-beta2)*gradient[i]*gradient[i] + epsilon;

        double m_hat = momentum[i]/(1 - pow(beta1, t));
        double v_hat = variance[i]/(1 - pow(beta2, t));
     
        // Actualización de la fluencia
        fluence[i] += step * m_hat/(sqrt(v_hat) + epsilon);


        if (fluence[i] < 0) {
            fluence[i] = 0;
        }
        // There's no max fluence on TROTS??
        //if (fluence[i] > 0.3) {
        //    fluence[i] = 0.3;
        //}
    }
}

int Optimizer::descend(Plan *plan, double *voxels, double *gradient, double *momentum, float step) {

    memset(voxels, 0, plan->n_plans*plan->n_voxels*sizeof(*voxels));
    memset(gradient, 0, plan->n_plans*plan->n_beamlets*sizeof(*gradient));

    int n_gradients = 0;
    int offset = 0;

    for (int k = 0; k < plan->n_plans; k++) {
        for (int i = 0; i < plan->n_regions; i++) {
            Region *region = &plan->regions[i];
            if (region->is_optimized) {
                voxels_eud(plan, i, k, &voxels[offset]);
                offset += plan->n_voxels;
                n_gradients++;
                if (region->is_ptv) {
                    offset += plan->n_voxels;
                    n_gradients++;
                }
            }
        }
    }

    n_gradients /= plan->n_plans;
    reduce_gradient(voxels, plan->n_voxels, n_gradients, plan->n_plans);
    
    double alpha = 1.0, beta = 0.0;
    double start_time = get_time_s();
    //mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, plan.m_t, plan.descr, voxels, beta, gradient);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, plan->m_t, plan->descr,SPARSE_LAYOUT_COLUMN_MAJOR, 
                    voxels, plan->n_plans, plan->n_voxels, beta, gradient, plan->n_beamlets);
    double elapsed = get_time_s() - start_time;
    //printf("Descend mm: %.4f seconds.\n", elapsed);

    return n_gradients;
}

void Optimizer::optimize(Plan *plan) {

    int gradients_per_region = 3; // Warning, hardcoded!
    // *2 because we need two for PTVs, but we're wasting space on organs.
     
    beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
 
    double *voxels = (double *) malloc(plan->n_plans*plan->n_voxels*plan->n_regions*2*sizeof(*voxels)); 
    double *gradient = (double *) malloc(plan->n_plans*plan->n_beamlets*sizeof(*gradient));
    double *momentum = (double *) malloc(plan->n_plans*gradients_per_region*plan->n_regions*plan->n_beamlets*sizeof(*momentum));
    double *variance = (double *) malloc(plan->n_plans*gradients_per_region*plan->n_regions*plan->n_beamlets*sizeof(*variance));
    memset(momentum, 0, plan->n_plans*gradients_per_region*plan->n_regions*plan->n_beamlets*sizeof(*momentum));
    memset(variance, 0, plan->n_plans*gradients_per_region*plan->n_regions*plan->n_beamlets*sizeof(*variance));

    int rid_sll = 3;
    int rid_slr = 4;

    plan->compute_dose();
    plan->stats();
    printf("Initial solution:\n");
    for (int k = 0; k < plan->n_plans; k++) {
        unsigned int poff = k*plan->n_regions;
        double pen = penalty(plan, k);
        double obj = plan->regions[poff + rid_sll].avg + plan->regions[poff + rid_slr].avg;
        double obj2 = objective(plan, k);
        printf("%2d penalty: %9.6f\n", k, pen);
        printf("%2d    obj: %9.6f\n", k, obj); 
        printf("%2d   obj2: %9.24f\n", k, obj2);
        plan->print_table(k);
    }

    step = 1e1;
    decay = 1e-4;
    min_step = 1e-1;
    start_time = get_time_s();
    
    int it = 0;
    while (running && get_time_s() - start_time < 600000) {
        int t = it+1;

        descend(plan, voxels, gradient, momentum, step);
        adam(gradient, momentum, variance, plan->n_beamlets, step, plan->fluence, plan->n_plans, t, beta1, beta2, epsilon);

        //plan.smooth_cpu();
        plan->compute_dose();
        plan->stats();
        //break;

        // Verifica si la penalidad ha bajado a 2.58
        double total_penalty = 0;
        for (int k = 0; k < plan->n_plans; k++) {
            double pen = penalty(plan, k);
            total_penalty += pen;
        }

 

        if (it % 100 == 0) {
            current_time = get_time_s();

            printf("\n[%.3f] Iteration %d %e\n", current_time - start_time, it, step);

            for (int k = 0; k < plan->n_plans; k++) {
                unsigned int poff = k*plan->n_regions;

                double pen = penalty(plan, k);
                double obj = plan->regions[poff + rid_sll].avg + plan->regions[poff + rid_slr].avg;
                double obj2 = objective(plan, k);
                printf("%2d penalty: %9.6f\n", k, pen);
                printf("%2d    obj: %9.6f\n", k, obj); 
                printf("%2d   obj2: %9.24f\n", k, obj2);
                plan->print_table(k);


            }
        }
        // if (step > min_step) 
        //    step = step/(1 + decay*t); // Decaimiento inverso
        it++;
        //if (it == 2000)
        //    break;
    }

    double elapsed = get_time_s() - start_time;
    printf("\nRan %d iterations in %.4f seconds (%.4f ms/it) \n", it, elapsed, elapsed*1000/it/plan->n_plans);
    for (int k = 0; k < plan->n_plans; k++) {
        unsigned int poff = k*plan->n_regions;

        double pen = penalty(plan, k);
        double obj = plan->regions[poff + rid_sll].avg + plan->regions[poff + rid_slr].avg;
        double obj2 = objective(plan, k);
        printf("%2d penalty: %9.6f\n", k, pen);
        printf("%2d    obj: %9.6f\n", k, obj); 
        printf("%2d   obj2: %9.24f\n", k, obj2);
        plan->print_table(k);
     
    }

    free(voxels);
    free(gradient);
    free(momentum);
    free(variance);
}