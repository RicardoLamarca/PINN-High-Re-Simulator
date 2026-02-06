# src/network.jl
using Lux, Random

"""
    get_pinn_model(input_dims::Int, output_dims::Int)

Defines a Multi-Layer Perceptron (MLP) for the fluid simulation.
Returns a Lux.Chain and the initial parameters.
"""
function get_pinn_model(input_dims=2, output_dims=3)
    # 2 inputs: (x, y)
    # 3 outputs: (u, v, p)
    
    # We use tanh activation because it is smooth and twice-differentiable,
    # which is required for calculating the second-order derivatives in 
    # the Navier-Stokes equations (e.g., Viscosity/Laplacian).
    
    hidden_layer_size = 32
    
    model = Lux.Chain(
        # First layer: Projecting coordinates into higher dimensional space
        Lux.Dense(input_dims, hidden_layer_size, Lux.tanh),
        
        # Hidden layers: Deepening the network to capture turbulence and wake patterns
        Lux.Dense(hidden_layer_size, hidden_layer_size, Lux.tanh),
        Lux.Dense(hidden_layer_size, hidden_layer_size, Lux.tanh),
        Lux.Dense(hidden_layer_size, hidden_layer_size, Lux.tanh),
        
        # Output layer: Final mapping to velocity (u, v) and pressure (p)
        Lux.Dense(hidden_layer_size, output_dims)
    )
    
    # Initialize parameters and states with a fixed seed for reproducibility
    rng = Random.default_rng()
    Random.seed!(rng, 1234)
    
    ps, st = Lux.setup(rng, model)
    
    return model, ps, st
end

# Example of how the data flows:
# Input [x, y] -> Dense -> Tanh -> Dense -> Tanh -> Output [u, v, p]
