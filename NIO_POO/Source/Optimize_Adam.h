#ifndef Optimizer_ADAM_H
#define Optimizer_ADAM_H


#include "Plan.h"
#include "Utils.h"
#include "includes.h"

void interrupt_handler(int signal);


struct Optimize_adam {
    Plan plan;
   
    
    // methods
    
    void voxels_eud(Plan plan, int rid, int pid, double *voxels);
    double penalty(Plan plan, unsigned int pid);
    double objective(Plan plan, unsigned int pid);
    void vector_stats(const char *name, double *vector, int n_values);
    void reduce_gradient(double *voxels, int n_voxels, int n_gradients, int n_plans);
    void adam(double *gradient, double *momentum, double *variance, int n_beamlets, float step, double *fluence, int n_plans, int t);
    int descend(Plan plan, double *voxels, double *gradient, double *momentum, double *variance, float step, int t);
    void optimize(Plan plan);





};

#endif //Optimizer_ADAM_H