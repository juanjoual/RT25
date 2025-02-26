#ifndef Optimizer_SGD_H
#define Optimizer_SGD_H


#include "Plan.h"
#include "Utils.h"
#include "includes.h"

void interrupt_handler(int signal);


struct Optimize_sgd {
    Plan plan;
   

    // methods
   
    void voxels_eud(Plan plan, int rid, int pid, double *voxels);
    double penalty(Plan plan, unsigned int pid);
    double objective(Plan plan, unsigned int pid);
    void vector_stats(const char *name, double *vector, int n_values);
    void reduce_gradient(double *voxels, int n_voxels, int n_gradients, int n_plans);
    void apply_gradient(double *gradient, double *momentum, int n_beamlets, float step, double *fluence, int n_plans);
    int descend(Plan plan, double *voxels, double *gradient, double *momentum, float step);
    void optimize(Plan plan);





};

#endif //Optimizer_SGD_H