#ifndef OPTIMIZER_GRADIENT_H
#define OPTIMIZER_GRADIENT_H

#include "Plan.h"
#include "Region.h"
#include "SparseMatrix.h"

extern volatile int running;

void interrupt_handler(int signal);

struct Optimizer_Gradient {
    Plan plan;
    double *voxels;
    double *gradient;
    double *momentum;
    double *fluence;
    float step;
    double beta;
    double decay;
    double min_step;
    double start_time;
    double *objective_values;
    double current_time;

    void voxels_eud(Plan *plan, int rid, int pid, double *voxels);
    double penalty(Plan *plan, unsigned int pid);
    double objective(Plan *plan, unsigned int pid);
    void vector_stats(const char *name, double *vector, int n_values);
    void reduce_gradient(double *voxels, int n_voxels, int n_gradients, int n_plans);
    void apply_gradient(double *gradient, double *momentum, int n_beamlets, float step, double *fluence, int n_plans, int beta);
    int descend(Plan *plan, double *voxels, double *gradient, double *momentum, float step);
    void optimize(Plan *plan);

};



#endif // OPTIMIZER_GRADIENT_H