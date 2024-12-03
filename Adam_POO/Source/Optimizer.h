#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Plan.h"
#include "Region.h"


static volatile int running = 1;

void interrupt_handler(int signal);

struct Optimizer {
    Plan plan;
    double *voxels;
    double *gradient;
    double *momentum;
    double *variance;
    double *fluence;
    float step;
    int t;
    double beta1;
    double beta2;
    double epsilon;
    double decay;
    double *objective_values;

    void voxels_eud(Plan *plan, int rid, int pid, double *voxels);
    double penalty(Plan *plan, unsigned int pid);
    double objective(Plan *plan, unsigned int pid);
    void vector_stats(const char *name, double *vector, int n_values);
    void reduce_gradient(double *voxels, int n_voxels, int n_gradients, int n_plans);
    void adam(double *gradient, double *momentum, double *variance, int n_beamlets, float step, double *fluence, int n_plans, int t, double beta1, double beta2, double epsilon);
    int descend(Plan *plan, double *voxels, double *gradient, double *momentum, double *variance, float step, int t, double beta1, double beta2, double epsilon);
    void optimize(Plan *plan);

};



#endif // OPTIMIZER_H