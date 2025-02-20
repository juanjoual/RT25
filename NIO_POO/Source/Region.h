#ifndef Region_H
#define Region_H

#include "includes.h"
struct Region {

  char *name;
    int n_voxels;
    double min;
    double max;
    double avg;

    double f; // Objective function evaluation
    double eud;
    double dF_dEUD;
    double sum_alpha;
    // Virtual EUD to control PTV overdosage
    // Hardcoded to eud + 1 for now
    double v_f;
    double v_eud;
    double v_dF_dEUD;
    double v_sum_alpha;

    bool is_optimized;
    bool is_ptv;
    double pr_min;
    double pr_max;
    double pr_avg_min;
    double pr_avg_max;
    double *grad_avg;
    
    double pr_eud;
    int penalty;
    int alpha;

    //methods
    void set_targets(bool t_ptv, double t_min, double t_avg_min, double t_avg_max, double t_max, double t_eud, int t_alpha, int t_penalty);

};

#endif // Region_H