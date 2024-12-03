#include "SparseMatrix.h" 
#include "Plan.h"
#include "Region.h"
#include "Optimizer.h"


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
    Optimizer optimizer;
    plan.n_plans = n_plans;
    plan.load(plan_path, fluence_path, fluence_prefix);

    plan.regions[ 0].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Patient
    plan.regions[ 1].set_targets(false,    -1,    -1,    -1, 38.00, 38.00,  10,   5); // Spinal Cord
    plan.regions[ 2].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Parotid (R)
    plan.regions[ 3].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Parotid (L)
    plan.regions[ 4].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // SMG (R)
    plan.regions[ 5].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // SMG (L)
    plan.regions[ 6].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // MCS
    plan.regions[ 7].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // MCM
    plan.regions[ 8].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // MCI
    plan.regions[ 9].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // MCP
    plan.regions[10].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Oesophagus
    plan.regions[11].set_targets(false,    -1,    -1,    -1, 38.00, 38.00,  10,   5); // Brainstem
    plan.regions[12].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Oral Cavity
    plan.regions[13].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Larynx
    plan.regions[14].set_targets( true, 45.00, 46.00, 47.00, 48.00, 46.00, -50, 100); // PTV 0-46Gy
    plan.regions[15].set_targets(false,    -1,    -1,    -1, 36.80, 36.80,  10,   5); // PTV Shell 15mm
    plan.regions[16].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // PTV Shell 30mm
    plan.regions[17].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // PTV Shell 40mm
    plan.regions[18].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // PTV Shell 5mm
    plan.regions[19].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // PTV Shell 0mm
    plan.regions[20].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // Ext. Ring 20mm

    optimizer.optimize(&plan);

    FILE *f = fopen(out_path, "w");
    for (int i = 0; i < plan.n_beamlets; i++) {
        fprintf(f, "%.10e\n", plan.fluence[i]);
    }
    fclose(f);
    printf("Last fluence written to %s\n", out_path);
}
