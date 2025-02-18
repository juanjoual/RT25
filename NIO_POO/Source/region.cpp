#include "Region.h"


void Region::set_targets(bool t_ptv, double t_min, double t_avg_min, double t_avg_max, double t_max, double t_eud, int t_alpha, int t_penalty) {
    
    if (t_eud < 0 && t_min < 0 && t_max < 0 && t_avg_min < 0 && t_avg_max < 0) {
        is_optimized = false;
    } else {
        is_optimized = true;
        is_ptv = t_ptv;
        pr_min = t_min;
        pr_max = t_max;
        pr_avg_min = t_avg_min;
        pr_avg_max = t_avg_max;
        pr_eud = t_eud;
        alpha = t_alpha;
        penalty = t_penalty;
        f = 0;
        v_f = 0;
        eud = 0;
        v_eud = 0;
        dF_dEUD = 0;
        v_dF_dEUD = 0;
        sum_alpha = 0;
        v_sum_alpha = 0;
    }
}