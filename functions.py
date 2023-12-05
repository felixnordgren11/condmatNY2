import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as manimation
from matplotlib.animation import *
##Do not change:
random.seed(-325420)

filenames = ['fcc256.txt']
positions = [np.genfromtxt(filenames[i]) for i in range(len(filenames))]
filenumber = 0

m = 108*(1.66/16)*1e-27
kb = 1/11603
rc = 4.5
rp = 4.2
sigma = 2.644 #Å
epsilon = 0.345 #eV

#Time step
dt = 13E-15
#Number of simulation steps

#initial temperature
T = 450

apoly = (1 / ((rc - rp) ** 7 * rp ** 12)) * 4 * epsilon * rc ** 4 * sigma ** 6 * \
         (2 * rp ** 6 * (-42 * rc ** 3 + 182 * rc ** 2 * rp - 273 * rc * rp ** 2 + 143 * rp ** 3) + \
         (455 * rc ** 3 - 1729 * rc ** 2 * rp + 2223 * rc * rp ** 2 - 969 * rp ** 3) * sigma ** 6)

bpoly = (1 / ((rc - rp) ** 7 * rp ** 13)) * 16 * epsilon * rc ** 3 * sigma ** 6 * \
         (rp ** 6 * (54 * rc ** 4 - 154 * rc ** 3 * rp + 351 * rc * rp ** 3 - 286 * rp ** 4) + \
         (-315 * rc ** 4 + 749 * rc ** 3 * rp + 171 * rc ** 2 * rp ** 2 - 1539 * rc * rp ** 3 + 969 * rp ** 4) * sigma ** 6)

cpoly = (1 / ((rc - rp) ** 7 * rp ** 14)) * 12 * epsilon * rc ** 2 * sigma ** 6 * \
         (rp ** 6 * (-63 * rc ** 5 - 7 * rc ** 4 * rp + 665 * rc ** 3 * rp ** 2 - 975 * rc ** 2 * rp ** 3 - 52 * rc * rp ** 4 + 572 * rp ** 5) + \
         2 * (195 * rc ** 5 + 91 * rc ** 4 * rp - 1781 * rc ** 3 * rp ** 2 + 1995 * rc ** 2 * rp ** 3 + 399 * rc * rp ** 4 - 969 * rp ** 5) * sigma ** 6)

dpoly = (1 / ((rc - rp) ** 7 * rp ** 15)) * 16 * epsilon * sigma ** 6 * \
         (rc * rp ** 6 * (14 * rc ** 6 + 126 * rc ** 5 * rp - 420 * rc ** 4 * rp ** 2 - 90 * rc ** 3 * rp ** 3 + 1105 * rc ** 2 * rp ** 4 - 624 * rc * rp ** 5 - 286 * rp ** 6) +  +
         rc * (-91 * rc ** 6 - 819 * rc ** 5 * rp + 2145 * rc ** 4 * rp ** 2 + 1125 * rc ** 3 * rp ** 3 - 5035 * rc ** 2 * rp ** 4 + 1881 * rc * rp ** 5 + 969 * rp ** 6) * sigma ** 6)

epoly = (1 / ((rc - rp) ** 7 * rp ** 15)) * 4 * epsilon * sigma ** 6 * \
         (2 * rp ** 6 * (-112 * rc ** 6 - 63 * rc ** 5 * rp + 1305 * rc ** 4 * rp ** 2 - 1625 * rc ** 3 * rp ** 3 - 585 * rc ** 2 * rp ** 4 + 1287 * rc * rp ** 5 + 143 * rp ** 6) + \
         (1456 * rc ** 6 + 1404 * rc ** 5 * rp - 14580 * rc ** 4 * rp ** 2 + 13015 * rc ** 3 * rp ** 3 + 7695 * rc ** 2 * rp ** 4 - 8721 * rc * rp ** 5 - 969 * rp ** 6) * sigma ** 6)

fpoly = (1 / ((rc - rp) ** 7 * rp ** 15)) * 48 * epsilon * sigma ** 6 * \
         (-rp ** 6 * (-28 * rc ** 5 + 63 * rc ** 4 * rp + 65 * rc ** 3 * rp ** 2 - 247 * rc ** 2 * rp ** 3 + 117 * rc * rp ** 4 + 65 * rp ** 5) + \
         (-182 * rc ** 5 + 312 * rc ** 4 * rp + 475 * rc ** 3 * rp ** 2 - 1140 * rc ** 2 * rp ** 3 + 342 * rc * rp ** 4 + 228 * rp ** 5) * sigma ** 6)

gpoly = (1 / ((rc - rp) ** 7 * rp ** 15)) * 4 * epsilon * sigma ** 6 * \
         (rp ** 6 * (-224 * rc ** 4 + 819 * rc ** 3 * rp - 741 * rc ** 2 * rp ** 2 - 429 * rc * rp ** 3 + 715 * rp ** 4) + \
         2 * (728 * rc ** 4 - 2223 * rc ** 3 * rp + 1425 * rc ** 2 * rp ** 2 + 1292 * rc * rp ** 3 - 1292 * rp ** 4) * sigma ** 6)

hpoly = (1 / ((rc - rp) ** 7 * rp ** 15)) * 16 * epsilon * sigma ** 6 * \
         (rp ** 6 * (14 * rc ** 3 - 63 * rc ** 2 * rp + 99 * rc * rp ** 2 - 55 * rp ** 3) + \
         (-91 * rc ** 3 + 351 * rc ** 2 * rp - 459 * rc * rp ** 2 + 204 * rp ** 3) * sigma ** 6)



n, _ = positions[filenumber].shape


def relative_pos(positions):
    global n
    '''
    Calculates relative positions
    '''
    pos = positions
    dx = np.zeros((n,n))
    dy = np.zeros((n,n))
    dz = np.zeros((n,n))
    n1 = int(np.size(pos, 0))
    rij = np.zeros((n1,n1))
    for i in range(n1):
        for j in range(i, n1):
            dx[i][j] = pos[i][0] - pos[j][0]
            dy[i][j] = pos[i][1] - pos[j][1]
            dz[i][j] = pos[i][2] - pos[j][2]
            rij[i][j] = np.sqrt((dx[i][j])**2 + (dy[i][j])**2 + (dz[i][j])**2)
            dx[j][i] = -dx[i][j]
            dy[j][i] = -dy[i][j]
            dz[j][i] = -dz[i][j]
            rij[j][i] = rij[i][j]
    return [dx, dy, dz, rij]

def plot_atoms(positions, timestep, ax):
    ax.clear()
    ax.scatter(positions[timestep][:, 0], positions[timestep][:, 1])  # Assuming 2D positions
    ax.set_xlim([xmin, xmax])  # Set limits based on your data
    ax.set_ylim([ymin, ymax])
    # Add more formatting as needed
        
        
        
####################################################
# ENERGY CALCULATIONS
####################################################


# def potential_energy(positions, neighbors):
#     global epsilon, sigma, n, rc
#     dx, dy, dz, rik = relative_pos(positions)
#     Epot = 0
#     nghbrs_count, nghbrs_indices = neighbors
    
#     for k in range(n):
#         for j in range(int(nghbrs_count[k])):
#             i = nghbrs_indices[k][j]
#             drik = rik[i][k]
#             LJ = 4*epsilon*((sigma/drik)**12-(sigma/drik)**6)
#             Epot += LJ
#     Epot = 0.5*Epot
#     return Epot

def potential_energy(positions, neighbors):
    global epsilon, sigma, n, apoly, bpoly, cpoly, dpoly, epoly, fpoly, gpoly, hpoly
    # Define the cutoff distances
    rp = 4.2  # r'
    rc = 4.5  # rc

    # Polynomial coefficients calculated as shown previously
      # Use the calculated values

    dx, dy, dz, rik = relative_pos(positions)
    Epot = 0
    nghbrs_count, nghbrs_indices = neighbors

    for k in range(n):
        for j in range(int(nghbrs_count[k])):
            i = nghbrs_indices[k][j]
            drik = rik[i][k]

            if drik < rp:
                # Lennard-Jones potential
                LJ = 4 * epsilon * ((sigma / drik) ** 12 - (sigma / drik) ** 6)
                Epot += LJ
            elif rp <= drik <= rc:
                # Polynomial junction
                polynomial = (apoly + bpoly * drik + cpoly * drik ** 2 + dpoly * drik ** 3 +
                              epoly * drik ** 4 + fpoly * drik ** 5 + gpoly * drik ** 6 + hpoly * drik ** 7)
                Epot += polynomial
            # No potential for distances greater than rc

    Epot = 0.5 * Epot
    return Epot


    
    
    
    ##
    r_ij = relative_pos(positions_file)[3]
    rows, cols = r_ij.shape
    s = sigma
    e = epsilon
    V = 0
    for i in range(rows):
        for j in range(i+1, cols):
            V += lenny_j(r_ij[i, j], s, e)
    return V



def kinetic_energy(velocities):
    global m
    E_kin = 0
    for i in range(n):
        a = 0.5 * m * ( velocities[i, 0]**2 + velocities[i, 1]**2 + velocities[i, 2]**2)
        E_kin += a
    return E_kin

def avg_temp(velocities):
    global n, kb
    E = kinetic_energy(velocities)
    avg_T = E*(2/3)/(n*kb)
    return avg_T

    E_kin = kinetic_energy(velocities)
    E_pot = potential_energy(positions, neighbors)
    E_tot = E_pot + E_kin
    return E_pot, E_kin, E_tot
####################################################
# NEIGHBORS CALCULATIONS
####################################################


def neighbors_list(positions, rc = rc):
    '''
    Creates a list with the amount of neighbors for each atom in the specified cutoff radius as well as a list of the indices for each neighbor
    '''
    # Calculate the relative positions (pairwise distances) between atoms
    pos = positions
    relpos = relative_pos(pos)[3]
    
    # Initialize a list to store the neighbors for each atom
    neighbors = [[] for _ in range(n)]
    numnbrs = np.zeros(n)
    
    # For each atom, identify the neighbors within the cutoff radius
    for i in range(n):
        for j in range(i+1, n):
            if relpos[i][j] < rc:
                numnbrs[i] += 1
                numnbrs[j] += 1
                neighbors[i].append(j)
                neighbors[j].append(i)
    return [numnbrs, neighbors]

def neighbors_list_verlet(positions, verlet_list, last_positions, rc, buffer):
    """
    Generates a list of neighbors for each particle based on the Verlet list.

    :param positions: Current positions of particles
    :param verlet_list: Current Verlet list
    :param last_positions: Positions of particles at the last Verlet list update
    :param rc: Interaction cutoff radius
    :param buffer: Additional buffer distance for the Verlet list
    :return: List of neighbors for each particle
    """
    n = len(positions)
    neighbors = [[] for _ in range(n)]
    numnbrs = np.zeros(n)

    # Check if the Verlet list needs to be updated
    max_displacement_squared = np.max(np.sum((positions - last_positions)**2, axis=1))
    if max_displacement_squared > (buffer / 2) ** 2:
        verlet_list, _ = update_verlet_list(positions, rc, buffer)
    
    # Use the Verlet list to find neighbors within the cutoff radius
    for i in range(n):
        for j in verlet_list[i]:
            if np.linalg.norm(positions[i] - positions[j]) < rc:
                numnbrs[i] += 1
                neighbors[i].append(j)

    return [numnbrs, neighbors], verlet_list


def update_verlet_list(positions, rc, buffer):
    """
    Update the Verlet list for all particles.

    :param positions: Positions of all particles
    :param rc: Cutoff radius for interactions
    :param buffer: Additional buffer distance for the Verlet list
    :return: Verlet list and maximum squared displacement since last update
    """
    n = len(positions)
    verlet_list = [[] for _ in range(n)]
    max_squared_displacement = 0

    # Extended cutoff radius for the Verlet list
    extended_rc = rc + buffer

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < extended_rc:
                verlet_list[i].append(j)
                verlet_list[j].append(i)

    return verlet_list, max_squared_displacement

        
        
###########################################################
#VELOCITY ASSIGNING AND RESCALING
###########################################################

def assgn_mom_sub_velocities(T_initial):
    global n, kb, T, m
    Cc = np.sqrt((3*kb*T_initial)/m)
    velocities = np.zeros((n,3))
    
    # Assign new randomized velocities
    for i in range(n):
        for j in range(3):
            velocities[i, j] = Cc * (2*random.random() - 1)
    
    avg_velocity = np.mean(velocities, axis = 0)
    
    # Subtract the average velocity from each velocity component
    velocities -= avg_velocity
    

    # Rescale velocities to maintain the temperature after subtracting the average
    actual_temp = avg_temp(velocities)  # Ensure avg_temp function is correctly implemented
    rescaling_factor = np.sqrt(T_initial / actual_temp)
    velocities *= rescaling_factor
    

    return velocities

###########################################################
#FORCE CALCULATION
###########################################################

def calc_forces(neighbors, pos, return_max_force=False):
    global epsilon, sigma, n
    
    dx, dy, dz, rik = relative_pos(pos)  # Assuming this function is defined elsewhere
    f = np.zeros((n, 3))
    nghbrs_count, nghbrs_indices = neighbors
    
    max_force_magnitude = 0  # Initialize the variable to store the maximum force magnitude

    for k in range(n):
        for j in range(int(nghbrs_count[k])):
            i = nghbrs_indices[k][j]
            drik = rik[k][i]
            c1 = 24 * epsilon * (sigma**6 / drik**8) * ((2 * sigma**6) / drik**6 - 1)
            f[k, 0] += c1 * dx[k][i]
            f[k, 1] += c1 * dy[k][i]
            f[k, 2] += c1 * dz[k][i]

            force_magnitude = np.sqrt(f[k, 0]**2 + f[k, 1]**2 + f[k, 2]**2)
            if force_magnitude > max_force_magnitude:
                max_force_magnitude = force_magnitude  # Update maximum force magnitude

    if return_max_force:
        return f, max_force_magnitude
    else:
        return f


def steepest_descent(positions, forces, nsteep, Csteep, ret_fmax_epot = False):
    global n
    f = forces
    pos = positions
    _nsteep = nsteep
    _Csteep = Csteep
    Epot = 0
    fmax = 0
    fmax_array = np.zeros((_nsteep, 1))
    for _ in range(_nsteep):
        _nghbrs = neighbors_list(pos)
        f, fmax = calc_forces(_nghbrs, pos, return_max_force = True)
        for i in range(n):
            pos[i][0] = pos[i][0] + _Csteep * f[i, 0]
            pos[i][1] = pos[i][1] + _Csteep * f[i, 1]
            pos[i][2] = pos[i][2] + _Csteep * f[i, 2]
        Epot = potential_energy(pos, _nghbrs)
        fmax_array[_] = fmax
        print(fmax, Epot)
    return pos, fmax_array