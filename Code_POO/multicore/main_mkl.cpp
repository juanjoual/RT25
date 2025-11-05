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

    // Head-and-Neck
    plan.regions[ 0].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Patient
    plan.regions[ 1].set_targets(false,    -1,    -1,    -1, 38.00, 38.00,  5,   30); // Spinal Cord
    plan.regions[ 2].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Parotid (R)
    plan.regions[ 3].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Parotid (L)
    plan.regions[ 4].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // SMG (R)
    plan.regions[ 5].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // SMG (L)
    plan.regions[ 6].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // MCS
    plan.regions[ 7].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // MCM
    plan.regions[ 8].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // MCI
    plan.regions[ 9].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // MCP
    plan.regions[10].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Oesophagus
    plan.regions[11].set_targets(false,    -1,    -1,    -1, 38.00, 38.00,  20,   60); // Brainstem
    plan.regions[12].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Oral Cavity
    plan.regions[13].set_targets(false,    -1,    -1,    -1, 48.30, 48.30,  10,   5); // Larynx
    plan.regions[14].set_targets(true,  46.00, 46.00, 47.00, 48.00, 48.00, -10, 10); // PTV 0-46Gy
    plan.regions[15].set_targets(false,    -1,    -1,    -1, 30.00, 30.00,  5,   20); // Cochlea (L)
    plan.regions[16].set_targets(false,    -1,    -1,    -1, 30.00, 30.00,  10,   5); // Cochlea (R)
    plan.regions[17].set_targets(false,    -1,    -1,    -1, 36.80, 36.80,  10,   5); // PTV Shell 15mm
    plan.regions[18].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // PTV Shell 30mm
    plan.regions[19].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // PTV Shell 40mm
    plan.regions[20].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // PTV Shell 5mm
    plan.regions[21].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // PTV Shell 0mm
    plan.regions[22].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5); // Ext. Ring 20mm

        

    // // Prostate CK
    // plan.regions[0].set_targets( true, 31.00, 40.00, 64.00, 64.00, 64.00, -2, 3); // PTV 3 mm
    // plan.regions[1].set_targets(false,    -1,    -1,    -1, 39.80, 39.80,  25,  25); // Bladder
    // plan.regions[2].set_targets(false,    -1,    -1,    -1, 37.00, 37.00,  20,  20); // Rectum
    // plan.regions[3].set_targets(false,    -1,    -1,    -1, 40.00, 40.00,  10,  10); // Urethra
    // plan.regions[5].set_targets(false,    -1,    -1,    -1, 4.00,  4.00,  5,   20); // Penis/Scrotum
    // plan.regions[6].set_targets(false,    -1,    -1,    -1, 24.00, 24.00,  5,   10); // Femoral Head (L)
    // plan.regions[6].set_targets(false,    -1,    -1,    -1, 24.00, 24.00,  5,   10); // Femoral Head (R)
    // plan.regions[7].set_targets(false,    -1,    -1,    -1, 66.00, 66.00, 50, 80); //PZ
    // plan.regions[8].set_targets(false,    -1,    -1,    -1, 15.00, 15.00,  10,   5); // From 30 mm to External -20 mm
    // plan.regions[9].set_targets(false,    -1,    -1,    -1, 18.00, 19.00,  50,   50); // PTV Ring 20 mm - 30 mm
    // plan.regions[10].set_targets(false,    -1,    -1,    -1, 15.00, 15.00,  50,   50); // External Ring 20 mm
    // plan.regions[11].set_targets(false,    -1,    -1,    -1, 45.70, 45.70,  25,   40); // PTV 7 mm


 
    // Liver
    // plan.regions[0].set_targets(true, 60., 60., 62., 68., 72., 20, 6); // PTV 
    // plan.regions[0].set_targets(false, -1., -1., -1, 68., 72., 10, 5); // PTV 
    // plan.regions[1].set_targets(false, -1, -1, -1, 5.0, 21.0, 10, 5);  // Stomach    
    // plan.regions[2].set_targets(false, -1, -1, -1, 5.0, 21.0, 10, 5);  // Oesophagus
    // plan.regions[3].set_targets(false, -1, -1, -1, 27.00, 30.00, 10, 5);  // Heart
    // plan.regions[4].set_targets(false, -1, -1, -1, 19.00, 21.00, 10, 5);  // Duodenum
    // plan.regions[5].set_targets(false, -1, -1, -1, 3.00,  5.00, 10, 5);  // Kidney (R)
    // plan.regions[6].set_targets(false, -1, -1, -1, 3.00,  5.00, 10, 5);  // Kidney (L)
    // plan.regions[7].set_targets(false, -1, -1, -1, 16.00, 18.00, 10, 5);  // Spinal Cord
    // plan.regions[8].set_targets(false, -1, -1, -1, 6.00, 10.00, 15, 5);  // Liver minus CTV
    // plan.regions[9].set_targets(false, -1, -1, -1, 18.00, 21.00, 10, 5);  // Pancreas
    // plan.regions[10].set_targets(false, -1, -1, -1, 70.00, 76.00, 10, 5);  // Patient
    // plan.regions[11].set_targets(true, 60., 60, 62, 68., 72., -5, 5);     // Isocenter
    // plan.regions[12].set_targets(false, -1, -1, -1, 17.00, 19.00, 10, 5); // External Ring
    // plan.regions[13].set_targets(false, -1, -1, -1, -1, -1, 10, 5);  // PTV40%  
    // plan.regions[14].set_targets(false, -1, -1, -1, -1, -1, 10, 5); //PTV80%  
    // plan.regions[15].set_targets(false, -1, -1, -1, -1, -1, 10, 5);   //Ring1
   
    optimizer.optimize(&plan);

    FILE *f = fopen(out_path, "w");
    for (int i = 0; i < plan.n_beamlets; i++) {
        fprintf(f, "%.10e\n", plan.fluence[i]);
    }

    fclose(f);
    printf("Last fluence written to %s\n", out_path);
}
