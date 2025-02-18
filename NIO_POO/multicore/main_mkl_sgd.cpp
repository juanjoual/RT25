#include "SparseMatrix.h" 
#include "Plan.h"
#include "Region.h"
#include "Optimizer_Gradient.h"



int main(int argc, char **argv) {

    signal(SIGINT, interrupt_handler);

    const char* plan_path = argv[1];
    const char* out_path = argv[2];
    const char* fluence_path;
    const char* fluence_prefix;

    if (argc > 4) {
        fluence_path = argv[3];
        fluence_prefix = argv[4];
    }

    int n_plans = 1; // Hardcoded to 1 plan for TROTS

    Plan plan = {};
    Optimizer_Gradient optimizer_gradient;
    plan.n_plans = n_plans;
    plan.load(plan_path, fluence_path, fluence_prefix);

    size_t len = strlen(plan_path);
    char plan_n = plan_path[len - 1];

    if (plan_n == '3') {
        for (int k = 0; k < plan.n_plans; k++) {
            plan.regions[k*plan.n_regions +  0].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
            plan.regions[k*plan.n_regions +  1].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
            plan.regions[k*plan.n_regions +  2].set_targets(false,    -1,    -1,    -1,    60,    60,  10,   5);
            plan.regions[k*plan.n_regions +  3].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
            plan.regions[k*plan.n_regions +  4].set_targets(false,    -1,    -1,    -1,    50,    50,  20,   5);
            plan.regions[k*plan.n_regions +  5].set_targets(false,    -1,    -1,    26,    -1,    26,   1,   5);
            plan.regions[k*plan.n_regions +  6].set_targets(false,    -1,    -1,    26,    -1,    26,   1,   5);
            plan.regions[k*plan.n_regions +  7].set_targets(false,    -1,    -1,    -1,    70,    70,  10,   5);
            plan.regions[k*plan.n_regions +  8].set_targets(false,    -1,    -1,    -1, 74.25, 74.25,  40,   5);
            plan.regions[k*plan.n_regions +  9].set_targets( true, 60.75, 66.15, 68.85, 74.25, 67.50, -40,  50);
            plan.regions[k*plan.n_regions + 10].set_targets( true, 54.00, 58.80, 61.20, 66.00, 60.00, -50, 100);
            plan.regions[k*plan.n_regions + 11].set_targets( true, 48.60, 52.92, 55.08, 59.40, 54.00, -40, 100);
        }
    } else if (plan_n == '4') {
        for (int k = 0; k < plan.n_plans; k++) {
            plan.regions[k*plan.n_regions +  0].set_targets(false,    -1,    -1,    -1,    70,    70,  10,   5);
            plan.regions[k*plan.n_regions +  1].set_targets(false,    -1,    -1,    26,    -1,    26,   1,   5);
            plan.regions[k*plan.n_regions +  2].set_targets(false,    -1,    -1,    26,    -1,    26,   1,   5);
            plan.regions[k*plan.n_regions +  3].set_targets(false,    -1,    -1,    -1,    50,    50,  10,   5);
            plan.regions[k*plan.n_regions +  4].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
            plan.regions[k*plan.n_regions +  5].set_targets( true, 59.40, 64.67, 67.32, 72.60, 66.00, -50,  50);
            plan.regions[k*plan.n_regions +  6].set_targets( true, 53.46, 58.21, 60.59, 65.34, 59.40, -50,  50);
            plan.regions[k*plan.n_regions +  7].set_targets(false,    -1,    -1,    -1,    60,    60,  10,   5);
            plan.regions[k*plan.n_regions +  8].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
            plan.regions[k*plan.n_regions +  9].set_targets(false,    -1,    -1,    -1, 74.25, 74.25,  10,   5);
            plan.regions[k*plan.n_regions + 10].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        }
    } else if (plan_n == '5') {
        for (int k = 0; k < plan.n_plans; k++) {
            plan.regions[k*plan.n_regions +  0].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
            plan.regions[k*plan.n_regions +  1].set_targets(false,    -1,    -1,    -1, 74.25, 74.25,  10,  5);
            plan.regions[k*plan.n_regions +  2].set_targets(false,    -1,    -1,    -1,    70,    70,  10,   5);
            plan.regions[k*plan.n_regions +  3].set_targets(false,    -1,    -1,    26,    -1,    26,   1,   5);
            plan.regions[k*plan.n_regions +  4].set_targets(false,    -1,    -1,    26,    -1,    26,   1,   5);
            plan.regions[k*plan.n_regions +  5].set_targets(false,    -1,    -1,    -1,    50,    50,  10,   5);
            plan.regions[k*plan.n_regions +  6].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
            plan.regions[k*plan.n_regions +  7].set_targets(false,    -1,    -1,    -1,    60,    60,  10,   5);
            plan.regions[k*plan.n_regions +  8].set_targets( true, 48.60, 52.92, 55.08, 59.40, 54.00, -50,   50);
            plan.regions[k*plan.n_regions +  9].set_targets( true, 54.00, 58.80, 61.20, 66.00, 60.00, -50,   50);
            plan.regions[k*plan.n_regions + 10].set_targets( true, 59.40, 64.67, 67.32, 72.60, 66.00, -50,   50);
            plan.regions[k*plan.n_regions + 11].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        }
    }
    
    optimizer_gradient.optimize(&plan);
    
    FILE *f = fopen(out_path, "w");
    for (int i = 0; i < plan.n_beamlets; i++) {
        fprintf(f, "%.10e\n", plan.fluence[i]);
    }
    fclose(f);
    printf("Last fluence written to %s\n", out_path);
}
