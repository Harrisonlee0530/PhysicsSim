# generate L, C and R values here
# YOUR CODE HERE
import numpy as np

# random number generator for L, C
aL = np.random.randint(10, 100)
bL = np.random.randint(-9, -4)
aC = np.random.randint(10, 100)
bC = np.random.randint(-9, -4)

# L, C
L = aL * (10**bL)
C = aC * (10**bC)
# Resistance
R2 = 2 * np.sqrt(L / C)  # critically damped
R1 = 0.1 * R2  # underdamped
R3 = 5 * R2  # overdamped


# Generate I vs t figures here.
# YOUR CODE HERE
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def dIdt_arr(t, arr, R, freq):
    """
    function for the RLC series circuit
    d^2 q / dt^2 + R/L dq/dt + q/(LC) = V(t)/L
    I = dq/dt
    dI/dt + I R/L + q/(LC) = V(t)/L
    """
    q = arr[0]
    dq = arr[1]
    V = 0.1 * np.cos(freq * t)  # input voltage
    return np.array([dq, V / L - dq * R / L - q / (L * C)])


def solution(R, freq):
    """
    Function to use solve_ivp with different frequency and resistance
    """
    return solve_ivp(dIdt_arr, (t[0], t[-1]),
                     initial_condition,
                     t_eval=t,
                     method='LSODA',
                     args=(R, freq))


omega = 1 / np.sqrt(L * C)  # frequency
t = np.linspace(0, 50 * np.sqrt(L * C), 2000)  # array of time
initial_condition = np.array([0, 0])

# solutions
underdamped_sol = solution(R1, omega)
critical_sol = solution(R2, omega)
overdamped_sol = solution(R3, omega)

# current
y1 = underdamped_sol['y'][1]
y2 = critical_sol['y'][1]
y3 = overdamped_sol['y'][1]
# time
t1 = underdamped_sol['t']
t2 = critical_sol['t']
t3 = overdamped_sol['t']

plt.figure(figsize=(10, 16))
# plot three vertically stacked plot
plt.subplot(3, 1, 1)
plt.ylabel("Current (A)")
plt.xlabel("time (s)")
plt.title("Underdamped: L = {L:.9f}H, C = {C:.9f}F, R1 = {R1:.6f}Ω".format(
    L=L, C=C, R1=R1))
plt.plot(t1, y1)

plt.subplot(3, 1, 2)
plt.ylabel("Current (A)")
plt.xlabel("time (s)")
plt.title(
    "Critically Damped: L = {L:.9f}H, C = {C:.9f}F, R2 = {R2:.6f}Ω".format(
        L=L, C=C, R2=R2))
plt.plot(t2, y2)

plt.subplot(3, 1, 3)
plt.ylabel("Current (A)")
plt.xlabel("time (s)")
plt.title("Overdamped: L = {L:.9f}H, C = {C:.9f}F, R3 = {R3:.6f}Ω".format(
    L=L, C=C, R3=R3))
plt.plot(t3, y3)
plt.show()


# Generate resonance plots here
# YOUR CODE HERE

frequencies = np.linspace(0.1 * omega / (2 * np.pi), 2 * omega / (2 * np.pi),
                          100)
current1 = []
current2 = []
current3 = []

for i in range(len(frequencies)):
    sol1 = solution(R1, frequencies[i] * 2 * np.pi)
    sol2 = solution(R2, frequencies[i] * 2 * np.pi)
    sol3 = solution(R3, frequencies[i] * 2 * np.pi)
    current1.append((max(sol1['y'][-1]) - min(sol1['y'][-1])) / 2)
    current2.append((max(sol2['y'][-1]) - min(sol2['y'][-1])) / 2)
    current3.append((max(sol3['y'][-1]) - min(sol3['y'][-1])) / 2)

plt.figure(figsize=(10, 16))

# plot three vertically stacked plot
plt.subplot(3, 1, 1)
plt.ylabel("Current Amplitude (A)")
plt.xlabel("Frequency (Hz)")
plt.title("Underdamped: L = {L:.9f}H, C = {C:.9f}F, R1 = {R1:.6f}Ω".format(
    L=L, C=C, R1=R1))
plt.plot(frequencies, current1)

plt.subplot(3, 1, 2)
plt.ylabel("Current Amplitude (A)")
plt.xlabel("Frequency (Hz)")
plt.title(
    "Critically Damped: L = {L:.9f}H, C = {C:.9f}F, R2 = {R2:.6f}Ω".format(
        L=L, C=C, R2=R2))
plt.plot(frequencies, current2)

plt.subplot(3, 1, 3)
plt.ylabel("Current Amplitude (A)")
plt.xlabel("Frequency (Hz)")
plt.title("Overdamped: L = {L:.9f}H, C = {C:.9f}F, R3 = {R3:.6f}Ω".format(
    L=L, C=C, R3=R3))
plt.plot(frequencies, current3)
plt.show()