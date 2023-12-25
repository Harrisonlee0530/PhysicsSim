import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import combinations
from IPython.display import HTML
# import matplotlib
# from scipy.optimize import curve_fit
# from scipy.integrate import odeint, solve_ivp
# from .ode_solver import ODESolver

def generate_random_array(n, m, l):
    """Generates a random array of minimum n, maximum m and size l"""
    return np.random.randint(n, m, l)


# pos: position, vel_t: translational velocity, vel_r: rotational velocity
# dama
dama_pos = np.array([])
dama_vel_t = np.array([])
dama_vel_r = np.array([])
# ken
ken_pos = np.array([])
ken_vel_t = np.array([])
ken_vel_r = np.array([])


# initial value of position, velocity
dama_pos = np.append(dama_pos, generate_random_array(-10, 10, 2))
dama_vel_t = np.append(dama_vel_t, generate_random_array(-1, 1, 2))
dama_vel_r = np.append(dama_vel_r, generate_random_array(-1, 1, 2))
ken_pos = np.append(ken_pos, generate_random_array(-10, 10, 2))
ken_vel_t = np.append(ken_vel_t, generate_random_array(-1, 1, 2))
ken_vel_r = np.append(ken_vel_r, generate_random_array(-1, 1, 2))

print(dama_pos)
print(ken_pos)


def dot(A, B):
    """ Calculates and returns the dot product of vector A and B """
    return A[0] * B[0] + A[1] * B[1]

# a list of coordinates to outline the shape of the kendama

plt.scatter(dama_pos[0], dama_pos[1])
plt.scatter(ken_pos[0], ken_pos[1])
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()