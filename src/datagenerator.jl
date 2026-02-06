# src/simulator.jl
using LinearAlgebra, CSV, DataFrames, WriteVTK

function generate_training_data(Re_list; folder_name="data")
    base_path = joinpath(pwd(), folder_name)
    !isdir(base_path) && mkpath(base_path)

    Nx, Ny = 120, 60 
    Lx, Ly = 2.0, 1.0 
    dx, dy = Lx/(Nx-1), Ly/(Ny-1)
    dt = 0.0001 # Paso de tiempo muy pequeño para estabilidad a Re alto

    # Obstáculo
    cx, cy, r = 0.5, 0.5, 0.1

    for Re in Re_list
        println("\n--- Simulando flujo real Re: $Re ---")
        u = ones(Nx, Ny)
        v = zeros(Nx, Ny)
        p = zeros(Nx, Ny)
        u_star = copy(u)
        v_star = copy(v)
        nu = 1.0 / Re

        for iter in 1:2500
            # 1. Paso de Momento (con Upwind para evitar NaNs)
            for j in 2:Ny-1, i in 2:Nx-1
                dist = sqrt((i*dx - cx)^2 + (j*dy - cy)^2)
                if dist < r
                    u_star[i,j] = 0.0; v_star[i,j] = 0.0
                    continue
                end

                # Advección Upwind: solo mira "atrás" en el flujo
                u_adv = u[i,j] * (u[i,j] - u[i-1,j])/dx
                v_adv = v[i,j] * (u[i,j] - u[i,j-1])/dy
                u_diff = nu * ((u[i+1,j]-2u[i,j]+u[i-1,j])/dx^2 + (u[i,j+1]-2u[i,j]+u[i,j-1])/dy^2)

                u_star[i,j] = u[i,j] + dt * (-u_adv + u_diff)
            end

            # 2. Condiciones de Contorno
            u_star[1, :] .= 1.0      # Entrada constante
            u_star[Nx, :] .= u[Nx-1, :] # Neumann (Salida libre)
            u_star[:, [1, Ny]] .= 0.0 # Paredes no-slip

            # 3. Poisson para Presión (Esto genera la interacción con la esfera)
            for _ in 1:50
                for j in 2:Ny-1, i in 2:Nx-1
                    rhs = (1.0/dt) * ((u_star[i+1,j]-u_star[i-1,j])/(2dx) + (v_star[i,j+1]-v_star[i,j-1])/(2dy))
                    p[i,j] = ((p[i+1,j]+p[i-1,j])*dy^2 + (p[i,j+1]+p[i,j-1])*dx^2 - rhs*dx^2*dy^2)/(2*(dx^2+dy^2))
                end
                p[Nx, :] .= 0.0 # Referencia en salida
            end

            # 4. Corrección Final
            for j in 2:Ny-1, i in 2:Nx-1
                if sqrt((i*dx - cx)^2 + (j*dy - cy)^2) >= r
                    u[i,j] = u_star[i,j] - dt * (p[i+1,j]-p[i-1,j])/(2dx)
                    v[i,j] = v_star[i,j] - dt * (p[i,j+1]-p[i,j-1])/(2dy)
                end
            end
        end
        save_to_vtk(u, v, p, Re, Nx, Ny, dx, dy, base_path)
    end
end
generate_training_data([1e5, 1e4])
