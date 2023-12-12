using Random, DelimitedFiles, Statistics, StatsPlots, Plots

@time begin
Random.seed!(325420)


const m = 108*(1.66/16)*1e-27
const kb = 1/11603
const sigma = 2.644 #Å
const epsilon = 0.345 #eV
const rc = 4.5
const rp = 4.2
const sx = 16.641600
#const T_in = 20

#### Relative position function #####

function relative_positions(pos)
    n, _ = size(pos)
    dx, dy, dz = [zeros(n, n) for _ in 1:3]
    rij = zeros(n, n)

    for i in 1:n
        for j in 1+i:n
            dx[i, j] = pos[i, 1] - pos[j, 1]
            dy[i, j] = pos[i, 2] - pos[j, 2]
            dz[i, j] = pos[i, 3] - pos[j, 3]
            rij[i, j] = sqrt(dx[i, j]^2 + dy[i, j]^2 + dz[i, j]^2)
            dx[j, i] = -dx[i, j]
            dy[j, i] = -dy[i, j]
            dz[j, i] = -dz[i, j]
            rij[j, i] = rij[i, j]
        end
    end
    return dx, dy, dz, rij
end


function neighbors_list(positions)
    rc = 4.5
    # Calculate the relative positions (pairwise distances) between atoms
    pos = positions
    n, _ = size(pos)
    relpos = relative_positions(pos)[4]
    
    # Initialize a list to store the neighbors for each atom
    neighbors = [Int[] for _ in 1:n]
    numnbrs = zeros(Int, n)
    
    # For each atom, identify the neighbors within the cutoff radius
    for i in 1:n
        for j in i+1:n
            if relpos[i, j] < rc
                numnbrs[i] += 1
                numnbrs[j] += 1
                push!(neighbors[i], j)
                push!(neighbors[j], i)
            end
        end
    end
    return numnbrs, neighbors
end
#= 
Potential energy with polynomial junction
=#

function potential_energy(positions, neighbors)
    n, _ = size(positions)
    sigma = 2.644 #Å
    epsilon = 0.345 #eV
    local rc = 4.5
    local rp = 4.2
    # Define the cutoff distances
    local apoly, bpoly, cpoly,  dpoly, epoly, fpoly, gpoly, hpoly = [176215059.1975983, -284023613.00387365, 196149015.1653697, -75238858.28765386, 17311961.891894903, -2389452.1702240175, 183178.53861204657, -6016.872195722496]
    rik = relative_positions(positions)[4]
    Epot = 0.0
    nghbrs_count, nghbrs_indices = neighbors

    for k in 1:n
        for j in 1:nghbrs_count[k]

            i = nghbrs_indices[k][j]
            drik = rik[i, k]

            # LJ = 4 * epsilon * ((sigma / drik) ^ 12 - (sigma / drik) ^ 6)
            # Epot += LJ

            if drik < rp
                # Lennard-Jones potential
                LJ = 4 * epsilon * ((sigma / drik) ^ 12 - (sigma / drik) ^ 6)
                Epot += LJ
            elseif rp <= drik <= rc
                # Polynomial junction
                polynomial = apoly + bpoly * drik + cpoly * drik^2 + dpoly * drik^3 +
                             epoly * drik^4 + fpoly * drik^5 + gpoly * drik^6 + hpoly * drik^7
                Epot += polynomial
            end
        end
    end

    Epot = 0.5 * Epot
    return Epot
end



function kinetic_energy(velocities)
    E_kin = 0
    for i in axes(velocities, 1)
        a = 0.5 * m * ( velocities[i, 1] ^ 2 + velocities[i, 2] ^ 2 + velocities[i, 3] ^ 2 )
        E_kin += a
    end
    return E_kin
end


function avg_temp(velocities)
    E = kinetic_energy(velocities)
    avg_T = E*(2/3)/(n*kb)
    return avg_T
end


function assgn_mom_sub_velocities_pythonseed(T_initial, random_file)
    Cc = sqrt( ( 3 * kb * T_initial ) / m )
    velocities = zeros(n, 3)
    random_numbers = readdlm(random_file, ',')
    one = ones(n, 3)
    
    velocities = Cc * (2 * random_numbers - one)
    
    avg_velocity = transpose(mean(velocities, dims = 1))
    
    # Subtract the average velocity from each velocity component
    for i in 1:n
        for j in 1:3
            velocities[i, j] -= avg_velocity[j]
        end
    end
    

    # Rescale velocities to maintain the temperature after subtracting the average
    T_actual = avg_temp(velocities)  # Ensure avg_temp function is correctly implemented
    rescaling_factor = sqrt(T_initial / T_actual)
    velocities .*= rescaling_factor
    return velocities
end

function assgn_mom_sub_velocities(T_initial)
    Cc = sqrt( ( 3 * kb * T_initial ) / m )
    velocities = zeros(n, 3)

    for i in 1:n
        for j in 1:3
            velocities[i, j] = Cc * (2 * rand() - 1)
        end
    end
    
    avg_velocity = transpose(mean(velocities, dims = 1))
    
    # Subtract the average velocity from each velocity component
    for i in 1:n
        for j in 1:3
            velocities[i, j] -= avg_velocity[j]
        end
    end
    

    # Rescale velocities to maintain the temperature after subtracting the average
    T_actual = avg_temp(velocities)  # Ensure avg_temp function is correctly implemented
    rescaling_factor = sqrt(T_initial / T_actual)
    velocities .*= rescaling_factor
    return velocities
end

function calc_forces(neighbors, positions, return_max_force = false)
    dx, dy, dz, rik = relative_positions(positions)  # Assuming this function is defined elsewhere
    f = zeros(n, 3)
    nghbrs_count, nghbrs_indices = neighbors
    max_force_magnitude = 0  # Initialize the variable to store the maximum force magnitude

    for k in 1:n
        for j in 1:nghbrs_count[k]
            i = nghbrs_indices[k][j]
            drik = rik[k, i]
            c1 = 24 * epsilon * (sigma ^ 6 / drik ^ 8) * ((2 * sigma ^ 6) / drik ^ 6 - 1)
            f[k, 1] += c1 * dx[k, i]
            f[k, 2] += c1 * dy[k, i]
            f[k, 3] += c1 * dz[k, i]

            force_magnitude = sqrt(f[k, 1] ^ 2 + f[k, 2] ^ 2 + f[k, 3] ^ 2)
            if force_magnitude > max_force_magnitude
                max_force_magnitude = force_magnitude  # Update maximum force magnitude
            end
        end
    end

    if return_max_force
        return f, max_force_magnitude
    else
        return f
    end
end

function update_position!(pos, velocities, forces, dt)
    local n1 = size(pos,1)
    for i in 1:n1
        for j in 1:3
            pos[i, j] += velocities[i, j] * dt + 0.5 * (1/m) * forces[i, j] * dt ^ 2
        end
    end
end

function update_velocity!(velocities, forces, new_forces, dt)
    local n1 = size(velocities,1)
    for i in 1:n1
        for j in 1:3
            velocities[i, j] += 0.5 * dt * (1/m) * (forces[i, j] + new_forces[i, j])
        end
    end
end

function steepest_descent!(positions, forces, nsteep, Csteep)
    f = forces
    pos = positions
    fmax = 0
    n = size(positions, 1)  # Number of particles

    for step in 1:nsteep

        for i in 1:n
            for k in 1:3
                mvmnt = Csteep * f[i, k]
                pos[i, k] = pos[i, k] + mvmnt
            end
        end

        _nghbrs = neighbors_list(pos)
        f, fmax = calc_forces(_nghbrs, pos, true)
    end

    return pos, fmax
end

function create_animation2(filtered_positions, n_timesteps)
    anim = Animation()

    for t in 1:n_timesteps

        # Extract positions for this timestep
        positions_at_t = filtered_positions[:, :, t]

        # Create a scatter plot of the positions
        # Assuming the second column is x, the third column is y
        scatter(positions_at_t[:, 2], positions_at_t[:, 3], title="Timestep $t", 
                xlabel="X", ylabel="Y", zlabel="Z", 
                xlims=(-3, 20), ylims=(-3, 20), zlims=(-0.5, 0.5)) 
        frame(anim)
    end

    # Save the animation
    gif(anim, "atom_animation.gif", fps = 15)
end

function create_animation(filtered_positions, n_timesteps)
    anim = Animation()

    for t in 1:n_timesteps
        positions_at_t = filtered_positions[:, :, t]

        # Debugging: print some position data
        #println("Timestep $t: ", positions_at_t)

        scatter(positions_at_t[:, 1], positions_at_t[:, 2], title="Timestep $t", 
                xlabel="X", ylabel="Y", xlim = (-5, 20), ylim = (-5, 20), legend = false)
        frame(anim)
    end

    gif(anim, "atom_animation.gif", fps = 15)
end



#=
Faster version of the main simulation loop using functions for the Verlet algorithm instead of hard coded loops in the function
=#
function Verlet(positions, neighbors, velocity, timestep, simulation_time)
    dt = timestep
    local thermalisation_time = 3e-12
    local sim_time = simulation_time
    local nmd = Int((thermalisation_time + sim_time)/timestep)
    temperatures = zeros(nmd)  # Array to store temperature at each timestep
    ev_Energy = zeros(nmd)
    pos = positions
    nghbrs = neighbors
    velocities = velocity
    forces = calc_forces(nghbrs, pos)
    pos, _ = steepest_descent!(pos, forces, nsteep, Csteep)

    tol = 1
    z_zero_indices = abs.(positions[:, 3]) .< tol  # Initial filter for z=0

    anim_timestep = nmd ÷ anim_frequency
    # Define a 3D array to store filtered positions for animation
    z0_positions_filtered = Array{Float64, 3}(undef, sum(z_zero_indices), 2, anim_timestep)
    
    #z0_positions = Array{Float64, 3}(undef, n, 3, anim_timestep)

    Ekin = 0
    E_potential = 0


    for ind in 1:nmd
        update_position!(pos, velocities, forces, dt)
        nghbrs = neighbors_list(pos)
        new_forces = calc_forces(nghbrs, pos)
        update_velocity!(velocities, forces, new_forces, dt)
        
        forces = new_forces
        nghbrs = neighbors_list(pos)
        E_potential = potential_energy(pos, nghbrs)
        Ekin = kinetic_energy(velocities)

        # # Store positions within z-range for this timestep
        # if ind % anim_frequency == 0
        #     x = ind ÷ anim_frequency
        #     for i in 1:n
        #         z0_positions[i, :, x] = pos[i, :]
        #     end
        # end

        if ind % anim_frequency == 0
            x = ind ÷ anim_frequency
            filtered_pos = pos[z_zero_indices, 1:2]  # Filter and keep only x, y
            if size(filtered_pos, 1) != 0
                z0_positions_filtered[:, :, x] = filtered_pos
            end
        end
        temperatures[ind] = avg_temp(velocities)
        ev_Energy[ind] = E_potential + Ekin
    end
    return temperatures, ev_Energy, z0_positions_filtered
end


filenames = ["fcc256.txt", "fcc864.txt"]
positions = readdlm(filenames[1], ' ', Float64)
const n, _ = size(positions)
# Initial temperature
const Tin = 10
# Ideal time step
const dt = 8e-15

# Simulation time
const sim_t = 1e-11
const anim_frequency = 30
# More steps (nsteep) than this is unnecessary
const nsteep = 3000
const Csteep = 0.001

const nmd = Int((3e-12 + sim_t)/dt)



neighbors = neighbors_list(positions)
filename2 = "random_numbers.csv"
velocities = assgn_mom_sub_velocities_pythonseed(Tin, filename2) 

# Identify rows where z-coordinate (3rd column) is 0
z_zero_indices = positions[:, 3] .== 0

# Extract rows corresponding to the (001) plane
xy_plane = positions[z_zero_indices, 1:2]




T_array, Etot_array, z0_positions = Verlet(positions, neighbors, velocities, dt, sim_t)

avg_E = mean(Etot_array)
avg_T = mean(T_array)

println(avg_E, ", ", avg_T)


## Regular plots

# Calculate the timesteps
timesteps = (1:nmd) .* dt

# Filter the arrays based on the condition dt * ind > 3e-12
filtered_indices = timesteps .> 3e-12
filtered_timesteps = timesteps[filtered_indices]
filtered_T_array = T_array[filtered_indices]
filtered_Etot_array = Etot_array[filtered_indices]




# Plot Temperature vs. Timestep after dt * ind > 3e-12
plot(filtered_timesteps, filtered_T_array, title = "Temperature vs Time", xlabel = "Timestep", ylabel = "Temperature", legend = false, ylim = (0, 15))

# Save the temperature plot if needed
savefig("temperature_vs_time_filtered.png")

# Plot Total Energy vs. Timestep after dt * ind > 3e-12
plot(filtered_timesteps, filtered_Etot_array, title = "Total Energy vs Time", xlabel = "Timestep", ylabel = "Total Energy", color = :red, legend = false)

# Save the total energy plot if needed
savefig("total_energy_vs_time_filtered.png")

create_animation(z0_positions, nmd ÷ anim_frequency)






end