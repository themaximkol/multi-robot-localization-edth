"""
Project: Simulation of a swarm of robots with relative localization
Author: Shushuai Li, MAVLab, TUDelft
Reference: arxiv link

This file: main file to show animation or figure plot of the relative position and yaw
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataCreate import dataCreate
from relativeEKF import EKFonSimData
import transform

# Simulation settings
show_animation = True  # True: animation; False: figure
np.random.seed(19910620)  # seed the random number generator for reproducibility

# border = {"xmin": -4, "xmax": 4, "ymin": -4, "ymax": 4, "zmin": 0, "zmax": 4}
border = {"xmin": -20, "xmax": 20, "ymin": -20, "ymax": 20, "zmin": 0, "zmax": 20}
numRob = 5  # number of robots
dt = 0.01  # time interval [s]
simTime = 70.0  # simulation time [s]
maxVel = 1  # maximum velocity [m/s]

devInput = np.array([[0.25, 0.25, 0.01]]).T  # input deviation in simulation, Vx[m/s], Vy[m/s], yawRate[rad/s]
devObser = 0.1  # observation deviation of distance[m]
ekfStride = 1  # update interval of EKF is simStride*0.01[s]

# Variables being updated in simulation
xTrue = np.random.uniform(-3, 3, (3, numRob))  # random initial groundTruth of state [x, y, yaw]' of numRob robots
relativeState = np.zeros((3, numRob, numRob))  # [x_ij, y_ij, yaw_ij]' of the second robot in the first robot's view
data = dataCreate(numRob, border, maxVel, dt, devInput, devObser)
relativeEKF = EKFonSimData(10, 0.1, 0.25, 0.4, 0.1, numRob)

# ---------- JUMP EVENT CONFIG ----------
enable_jump_event = True  # set False to disable
min_jump_time = 15.0  # seconds, earliest possible jump
max_jump_time = 40.0  # seconds, latest possible jump
jump_distance_min = 3.0  # meters
jump_distance_max = 4.0  # meters
jump_target_robot = 1  # index of robot that will jump (0..numRob-1)

# choose random jump step if enabled
jumped = False
if enable_jump_event:
    min_jump_step = int(min_jump_time / dt)
    max_jump_step = int(max_jump_time / dt)
    max_jump_step = max(min_jump_step + 1, max_jump_step)  # safety
    jump_step = np.random.randint(min_jump_step, max_jump_step)
else:
    jump_step = None


# --------------------------------------


def random_jump_vector():
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    # radius = np.random.uniform(jump_distance_min, jump_distance_max)
    radius = 15
    dx = radius * np.cos(angle)
    dy = radius * np.sin(angle)
    return dx, dy


def animate(step):
    # global xTrue, relativeState, jumped, jump_step

    global xTrue, relativeState, jumped, jump_step
    global pointsTrue_main, pointTrue_jump, pointsEsti_main, pointEsti_jump

    u = data.calcInput_FlyIn1m(step)
    # u = data.calcInput_PotentialField(step, xTrue)
    # u = data.calcInput_Formation01(step, relativeState)
    xTrue, zNois, uNois = data.update(xTrue, u)

    # ----- JUMP EVENT DURING ANIMATION -----
    if enable_jump_event and (not jumped) and (jump_step is not None) and (step >= jump_step):
        dx, dy = random_jump_vector()
        xTrue[0, jump_target_robot] += dx
        xTrue[1, jump_target_robot] += dy

        # keep inside borders
        xTrue[0, jump_target_robot] = np.clip(xTrue[0, jump_target_robot], border["xmin"], border["xmax"])
        xTrue[1, jump_target_robot] = np.clip(xTrue[1, jump_target_robot], border["ymin"], border["ymax"])

        jumped = True
        print(
            f"JUMP EVENT (animation): robot {jump_target_robot} moved by ({dx:.2f}, {dy:.2f}) at step {step}, t={step * dt:.2f}s")
    # --------------------------------------

    if step % ekfStride == 0:
        relativeState = relativeEKF.EKF(uNois, zNois, relativeState, ekfStride)

    xEsti = transform.calcAbsPosUseRelaPosWRTRob0(xTrue[:, 0], relativeState, xTrue, numRob)
    # pointsTrue.set_data(xTrue[0, :], xTrue[1, :])  # plot groundTruth points
    # pointsEsti.set_data(xEsti[0, :], xEsti[1, :])  # plot estimated points
    # pointsTrueHead.set_data(xTrue[0, :] + 0.07 * np.cos(xTrue[2, :]),
    #                         xTrue[1, :] + 0.07 * np.sin(xTrue[2, :]))  # heading
    # pointsEstiHead.set_data(xEsti[0, :] + 0.07 * np.cos(xEsti[2, :]),
    #                         xEsti[1, :] + 0.07 * np.sin(xEsti[2, :]))  # heading

    xEsti = transform.calcAbsPosUseRelaPosWRTRob0(xTrue[:, 0], relativeState, xTrue, numRob)

    # --- split ground-truth & estimated into "jumped" and "others" ---
    xs_true = xTrue[0, :]
    ys_true = xTrue[1, :]
    xs_esti = xEsti[0, :]
    ys_esti = xEsti[1, :]

    if jumped:
        mask = np.ones(numRob, dtype=bool)
        mask[jump_target_robot] = False

        # ground truth
        pointsTrue_main.set_data(xs_true[mask], ys_true[mask])
        pointTrue_jump.set_data(xs_true[~mask], ys_true[~mask])

        # estimated
        pointsEsti_main.set_data(xs_esti[mask], ys_esti[mask])
        pointEsti_jump.set_data(xs_esti[~mask], ys_esti[~mask])
    else:
        # before jump: everyone the same color
        pointsTrue_main.set_data(xs_true, ys_true)
        pointTrue_jump.set_data([], [])
        pointsEsti_main.set_data(xs_esti, ys_esti)
        pointEsti_jump.set_data([], [])

    # headings stay the same
    pointsTrueHead.set_data(xTrue[0, :] + 0.07 * np.cos(xTrue[2, :]),
                            xTrue[1, :] + 0.07 * np.sin(xTrue[2, :]))
    pointsEstiHead.set_data(xEsti[0, :] + 0.07 * np.cos(xEsti[2, :]),
                            xEsti[1, :] + 0.07 * np.sin(xEsti[2, :]))

    circle.center = (xTrue[0, 0], xTrue[1, 0])
    circle.radius = zNois[0, 1]  # plot a circle to show the distance between robot 0 and robot 1
    time_text.set_text("t={:.2f}s".format(step * dt))
    # return pointsTrue, pointsEsti, circle, pointsTrueHead, pointsEstiHead, time_text
    return (pointsTrue_main, pointTrue_jump,
            pointsEsti_main, pointEsti_jump,
            circle, pointsTrueHead, pointsEstiHead, time_text)


if show_animation:
    # Set up an animation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set(xlim=(border["xmin"], border["xmax"]), ylim=(border["ymin"], border["ymax"]))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    title = ax.set_title('Simulated swarm')
    # pointsTrue, = ax.plot([], [], linestyle="", marker="o", color="b", label="GroundTruth")
    # pointsEsti, = ax.plot([], [], linestyle="", marker="o", color="r", label="Relative EKF")
    # pointsTrueHead, = ax.plot([], [], linestyle="", marker=".", color="g")
    # pointsEstiHead, = ax.plot([], [], linestyle="", marker=".", color="g")

    # Ground-truth: blue for normal robots, red for jumped robot
    pointsTrue_main, = ax.plot([], [], linestyle="", marker="o", color="b", label="GroundTruth")
    pointTrue_jump, = ax.plot([], [], linestyle="", marker="o", color="r", label="Jumped robot")

    # Estimated: blue for others, red for jumped robot (optional but nice)
    pointsEsti_main, = ax.plot([], [], linestyle="", marker="o", color="c", label="Relative EKF")
    pointEsti_jump, = ax.plot([], [], linestyle="", marker="o", color="m", label="Relative EKF jumped")

    pointsTrueHead, = ax.plot([], [], linestyle="", marker=".", color="g")
    pointsEstiHead, = ax.plot([], [], linestyle="", marker=".", color="g")

    ax.legend()
    circle = plt.Circle((0, 0), 0.2, color='black', fill=False)
    ax.add_patch(circle)
    time_text = ax.text(0.01, 0.97, '', transform=ax.transAxes)
    time_text.set_text('')
    ani = animation.FuncAnimation(fig, animate, frames=None, interval=10, blit=True)
    ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    # ani.save('particle_box.mp4')
    plt.show()
# else:
#     # Simulation using while-loop; figure of X-Y-Yaw errors
#     xEsti = relativeState[:, 0, :]  # relative states in robot0's body-frame
#     xTrueRL = transform.calcRelaState(xTrue, numRob)  # groundTruth relative states
#     dataForPlot = np.array([xEsti[0, 1], xTrueRL[0, 1], xEsti[1, 1], xTrueRL[1, 1],
#                             xEsti[2, 1], xTrueRL[2, 1]])  # x, xGT, y, yGT, yaw, yawGT
#
#     step = 0
#     global jumped, jump_step  # reuse same jump config in non-animation mode
#
#     while simTime >= dt * step:
#         step += 1
#         u = data.calcInput_FlyIn1m(step)
#         # u = data.calcInput_PotentialField(step, xTrue)
#         # u = data.calcInput_Formation01(step, relativeState)
#         # u = data.calcInput_FlyIn1mRob1NoVel(step)
#         xTrue, zNois, uNois = data.update(xTrue, u)
#
#         # ----- JUMP EVENT IN NON-ANIMATION MODE -----
#         if enable_jump_event and (not jumped) and (jump_step is not None) and (step >= jump_step):
#             dx, dy = random_jump_vector()
#             xTrue[0, jump_target_robot] += dx
#             xTrue[1, jump_target_robot] += dy
#
#             xTrue[0, jump_target_robot] = np.clip(xTrue[0, jump_target_robot], border["xmin"], border["xmax"])
#             xTrue[1, jump_target_robot] = np.clip(xTrue[1, jump_target_robot], border["ymin"], border["ymax"])
#
#             jumped = True
#             print(
#                 f"JUMP EVENT (no-animation): robot {jump_target_robot} moved by ({dx:.2f}, {dy:.2f}) at step {step}, t={step * dt:.2f}s")
#         # -------------------------------------------
#
#         if step % ekfStride == 0:
#             relativeState = relativeEKF.EKF(uNois, zNois, relativeState, ekfStride)
#
#         xEsti = relativeState[:, 0, :]
#         xTrueRL = transform.calcRelaState(xTrue, numRob)
#         dataForPlot = np.vstack([dataForPlot, np.array(
#             [xEsti[0, 1], xTrueRL[0, 1],
#              xEsti[1, 1], xTrueRL[1, 1],
#              xEsti[2, 1], xTrueRL[2, 1]])])
#
#     dataForPlotArray = dataForPlot.T
#     timePlot = np.arange(0, len(dataForPlotArray[0])) / 100
#     f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
#     plt.margins(x=0)
#     ax1.plot(timePlot, dataForPlotArray[0, :])
#     ax1.plot(timePlot, dataForPlotArray[1, :])
#     ax1.set_ylabel(r"$x_{ij}$ (m)", fontsize=12)
#     ax1.grid(True)
#     ax2.plot(timePlot, dataForPlotArray[2, :])
#     ax2.plot(timePlot, dataForPlotArray[3, :])
#     ax2.set_ylabel(r"$y_{ij}$ (m)", fontsize=12)
#     ax2.grid(True)
#     ax3.plot(timePlot, dataForPlotArray[4, :], label='Relative EKF')
#     ax3.plot(timePlot, dataForPlotArray[5, :], label='Ground-truth')
#     ax3.set_ylabel(r"$\mathrm{\psi_{ij}}$ (rad)", fontsize=12)
#     ax3.set_xlabel("Time (s)", fontsize=12)
#     ax3.grid(True)
#     ax3.legend(loc='upper center', bbox_to_anchor=(0.8, 0.6), shadow=True, ncol=1, fontsize=12)
#     # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
#     f.subplots_adjust(hspace=0)
#     plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#     plt.show()
