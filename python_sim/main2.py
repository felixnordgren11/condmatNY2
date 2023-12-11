import numpy as np
import matplotlib.pyplot as plt
import random
from functions import *
from time import time
# import matplotlib.animation as manimation
# from matplotlib.animation import *
# from functions import *
##Do not change:
random.seed(-325420)

time_beg = time()

filenames = ['fcc256.txt']
positions = [np.genfromtxt(filenames[i]) for i in range(len(filenames))]
filenumber = 0

m = 108*(1.66/16)*1e-27
kb = 1/11603
sigma = 2.644 #Å
epsilon = 0.345 #eV
rc = 4.5
rp = 4.2

#Initial temperature
T = 400
#Time step
dt = 8E-15


Csteep = 0.01
nsteep = 100

# Number of iterations of Verlet algorithm
nmd = 1000




        
###################################
#Main code


pos = positions[filenumber]
velocities = assgn_mom_sub_velocities(T)
rel_pos = relative_pos(positions[filenumber])

nghbrs = neighbors_list(pos)
forces = calc_forces(nghbrs, pos)

###
'''
Steepest descent 
'''
# pos, fmax_array = steepest_descent(pos, forces, nsteep, Csteep)
# nghbrs = neighbors_list(pos)
# forces = calc_forces(nghbrs, pos)



ev_Energy = np.zeros((nmd,3))
temperatures = np.zeros(nmd)  # Array to store temperature at each timestep
ev_Energy = np.zeros((nmd, 3))  # Array to store energies

# plt.subplot(1, 1, 1)
# plt.plot(np.arange(nsteep), fmax_array, label='f_max')
# plt.xlabel('nsteep')
# plt.ylabel('Fmax')
# plt.title('fmax vs nsteep')
# plt.legend()
# plt.show()

atom_number = [i for i in range(n)]
rowf, colf = forces.shape
new_forces = np.zeros((rowf,colf))



for ind in range(nmd):
    
        
    for i in range(n):
        for j in range(3):
            pos[i][j] = pos[i][j] + velocities[i][j] * dt + 0.5 * (1/m) * forces[i][j] * (dt ** 2)
    
    nghbrs = neighbors_list(pos)
    new_forces = calc_forces(nghbrs, pos)
    for i in range(n):
        for j in range(3):
            velocities[i][j] = velocities[i][j] + 0.5 * dt * (1/m) * (forces[i][j] + new_forces[i][j])
    
    forces = new_forces
    
    Epot = potential_energy(pos, nghbrs)
    Ekin = kinetic_energy(velocities)
    Etot = Epot + Ekin
    temperatures[ind] = avg_temp(velocities)
    ev_Energy[ind] = Etot
    #print(f'{ind}: Avg temp: {avg_temp(velocities)}, Etot: {Etot}')

time_end = time()

tot_time = time_end - time_beg
print(f'Time elapsed: {tot_time}')

timesteps = np.arange(nmd)
plt.figure(figsize=(18, 6))

# Subplot 1: Temperature vs Time
plt.subplot(1, 2, 1)
plt.plot(np.arange(nmd), temperatures, label='Temperature')
plt.xlabel('Timestep')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend()

# Subplot 2: Total Energy vs Time
plt.subplot(1, 2, 2)
plt.plot(np.arange(nmd), ev_Energy[:, 2], label='Total Energy', color='red')
plt.xlabel('Timestep')
plt.ylabel('Total Energy')
plt.title('Total Energy vs Time')
plt.legend()

# Flatten the velocities array to create a histogram
# flat_velocities = velocities.flatten()

# # Subplot 3: Histogram of Velocities
# plt.subplot(1, 3, 3)
# plt.hist(flat_velocities, bins=20, color='blue', edgecolor='black')
# plt.xlabel('Velocity')
# plt.ylabel('Frequency')
# plt.title('Histogram of Velocities')

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()



    
    
    



    
