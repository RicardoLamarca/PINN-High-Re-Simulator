# main.jl
# ==============================================================================
# Main entry point for the CFD-PINN hybrid solver.
# This script orchestrates data generation, model setup, and training.
# ==============================================================================

using Revise # Allows for code updates without restarting the REPL
include("src/simulator.jl")
include("src/network.jl")
include("src/physics.jl")

function main()
    # --- Step 1: Data Generation ---
    # Define the Reynolds numbers for training. 
    # High Re (1e6) requires the robust stability implemented in simulator.jl
    re_list = [1e5, 1e6]
    data_path = "data"
    
    println(">>> Starting Numerical Simulation...")
    # This generates the CSV and VTK files with the wake behind the sphere
    generate_training_data(re_list, folder_name=data_path)
    println(">>> Simulation data ready in /$data_path")

    # --- Step 2: Network Initialization ---
    println(">>> Initializing Neural Network...")
    # We fetch the Lux model and initial parameters defined in network.jl
    model, ps, st = get_pinn_model(2, 3) # 2 inputs (x,y), 3 outputs (u,v,p)

    # --- Step 3: PINN Training ---
    # We pass the data folder to physics.jl so the additional_loss 
    # can penalize deviations from the simulated wake.
    println(">>> Starting PINN Training...")
    # Setting a modest max_iters for the initial run to verify convergence
    results, discretization = train_pinn(data_path, max_iters=2000)

    # --- Step 4: Results and Inference ---
    println(">>> Training Complete!")
    
    # You can now use 'results.u' to get the optimized weights
    # and use the 'discretization.phi' to predict flow at any (x, y) point.
    # To check the error, you could compare predictions against the CSV again.
end

# Execute the pipeline
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
