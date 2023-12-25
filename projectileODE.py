# Main solution in this cell
# Please do any project extension work in the indicated cell below
# YOUR CODE HERE
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np


def dvdt_arr(t, v_arr):
    """
    ODE of the projectile with drag: m dv/dt = -mg (j) -b v0 (v)

    This returns dy/dt as an array, takes v_arr and t as input,
    v_arr includes v_x , v_y , xx, yy (x and y components of v and position)

    x_component: dv/dt = -b/m*v*v_x
    y_component: dv/dt = -g - (b/m*v*v_y)
    dx/dt = vx
    dy/dt = vy
    returns [ax, ay, vx, vy]
    """

    vx = v_arr[0]  # x component of velocity
    vy = v_arr[1]  # y component of velocity
    xx = v_arr[2]  # x position
    yy = v_arr[3]  # y position
    v = np.sqrt(vx**2 + vy**2)
    return np.array([-b / m * v * vx, -g - (b / m * v * vy), vx, vy])


def highest(t, v_arr):
    """
    returns the vertical velocity of the projectile
    used in events in solve_ivp
    so solve_ivp returns solutions when the vertical velocity is 0
    (located at the hiest point)
    """
    return v_arr[1]


def impact(t, v_arr):
    """
    returns the y position of the projectile, used in events in solve_ivp,
    so solve_ivp returns solutions when the projectile is at y=0
    (launch point, impact point)
    """
    return v_arr[3]


# constants
b = 0.006  # kg/m
g = 9.81  # m/s^2
m = 0.45  # kg
v0 = 18  # m/s
theta = 40  # degrees
theta = np.deg2rad(theta)  # radians

# projectile without drag
# v = [horizontal velocity, vertical velocity, x position, y position]
# same as the initial condition of the ODE for the trajectory with drag)
v = np.array([v0 * np.cos(theta), v0 * np.sin(theta), 0, 0])
t_tof = v[1] / (0.5 * g)  # time of flight
t = np.arange(0, t_tof + t_tof / 1000, t_tof / 1000)
projectile_without_drag = [
    v[0] * t, v[1] * t - 0.5 * g * t**2
]  # list containing 2 arrays: [x position, y position] of the projectile

# projectile with drag ODE solution
sol = integrate.solve_ivp(dvdt_arr, (t[0], t[-1]),
                          v,
                          t_eval=t,
                          method='LSODA',
                          events=[highest, impact])

# solution of the DE
t_sol = sol['t']  # stores time corresponding to v_sol
# stores acceleration, velocity in x, y direction in v_sol = [ax, ay, vx, vy]
# after differentiating [vx, vy, xx, xy] at time t_sol
v_sol = sol['y']


# turn array into 20 elements
idx = np.where(v_sol[3] >= 0)
idx = idx[0]
x_pos = []
y_pos = []
for i in np.linspace(0, max(idx), 20).astype(int):
    x_pos.append(v_sol[2][i])
    y_pos.append(v_sol[3][i])

# parameters of the projectile at the highest point returned by events
maximum_time = sol['t_events'][0]  # time
maximum_arr = sol['y_events'][0][0]  # array of vx, vy, x, y

# parameters of the projectile at the point of impact returned by events
impact_time = sol['t_events'][1][1]  # time
impact_arr = sol['y_events'][1][1]  # array of vx, vy, x, y

# plt.figure(figsize=((10, 8)))

# plotting the trajectories of the projectile
plt.plot(projectile_without_drag[0],
         projectile_without_drag[1],
         label="Analytic Trajectory w/o drag")
plt.plot(x_pos, y_pos, 'r.', label="Numerical Trajectory w/ drag")

# plot projectile at the maximum height
plt.plot(maximum_arr[2],
         maximum_arr[3],
         'ko',
         markersize=3,
         label="Max height of projectile w/ drag")

# plot projectile at impact point
plt.plot(impact_arr[2],
         impact_arr[3],
         'go',
         markersize=3,
         label="Impact point of the projectile w/ drag")

# labels of the plot
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.title("Trajectories of 2D projectile motion with and without drag")
plt.legend(loc='upper right', prop={'size': 10})

# axis settings
plt.xlim([0, max(projectile_without_drag[0]) * 1.05])
plt.ylim([0, max(projectile_without_drag[1]) * 1.05])

# add grey dotted grid to the plot
plt.grid(ls=":", c="grey")

# show plot
plt.show()

# print output
print("What is the distance d to impact?\t\t\t: {ans:.3f} m".
      format(ans=impact_arr[2]))
print(
    "What is the maximum height h_max reached?\t\t: {ans:.3f} m".
    format(ans=maximum_arr[3]))
print("What is the time of flight t_tof? \t\t\t: {ans:.3f} s".
      format(ans=impact_time))
print(
    "What is the velocity V (vector) at the impact point? \t: " +
    "({ans1:.3f}, {ans2:.3f}) m/s, magnitude = {ans3:.3f}m/s"
    .format(ans1=impact_arr[0],
            ans2=impact_arr[1],
            ans3=(impact_arr[0]**2 + impact_arr[1]**2)**0.5))
print(
    "By how much distance does drag reduce the range?\t: {ans:.3f} m"
    .format(ans=max(projectile_without_drag[0]) - impact_arr[2]))

# library and function to make animation (matplotlib.animation) FuncAnimation()
from matplotlib import animation
# create subplots
fig, ax = plt.subplots()

# data and lines to update for the animation
xdata, ydata = [], []
xdata2, ydata2 = [], []
ln, = ax.plot([], [], 'b-', markersize=2, label='projectile path without drag')
ln2, = ax.plot([], [], 'r-', markersize=2, label='projectile path with drag')

def init():
    """
    function used to draw a clear frame
    """
    ax.set_xlim([0 - 0.01, max(projectile_without_drag[0]) * 1.05])
    ax.set_ylim([0 - 0.01, max(projectile_without_drag[1]) * 1.05])
    return ln,

def update(frame):
    """
    Updates the data and lines to animate while running animation.FuncAnimation(...)
    """
    xdata.append(projectile_without_drag[0][frame])
    ydata.append(projectile_without_drag[1][frame])
    xdata2.append(v_sol[2][frame])
    ydata2.append(v_sol[3][frame])
    ln.set_data(xdata, ydata)
    ln2.set_data(xdata2, ydata2)
    return ln, ln2,

# animates the function
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=t_tof, init_func=init, blit=True, repeat=False)

ani.save('Projectile.gif', writer=animation.PillowWriter(fps=600))

# labels and titles of the plot
plt.title("Animation of the trajectories of the projectile w/ and w/o drag")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")

# legend
plt.legend()

# show plot
# plt.show()