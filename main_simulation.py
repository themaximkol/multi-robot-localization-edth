"""
Project: Simulation of a swarm of robots with relative localization
Author: Shushuai Li, MAVLab, TUDelft
Reference: arxiv link

This file: main file to show animation or figure plot of the relative position and yaw
"""
import numpy as np
import matplotlib.pyplot as plt

from dataCreate import dataCreate
from relativeEKF import EKFonSimData
import transform

from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2 as cv

# ---------------------------------------------------------
# Settings & icon
# ---------------------------------------------------------
show_animation = True  # True: animation; False: figure
np.random.seed(19910620)  # seed the random number generator for reproducibility

icon = cv.imread("asset/droneicon.webp", cv.IMREAD_UNCHANGED)  # BGRA
if icon is None:
    raise FileNotFoundError("Could not load icon at 'asset/drone.webp'")

icon_h, icon_w = 40, 40  # size on screen
icon = cv.resize(icon, (icon_w, icon_h))


def overlay_icon(frame, icon, x, y):
    """
    Draws the icon onto frame at pixel top-left (x, y).
    frame: BGR
    icon: BGRA
    x: column index
    y: row index
    """
    ih, iw = icon.shape[:2]

    # Check bounds (avoid negative indices / out-of-frame slicing)
    H, W = frame.shape[:2]
    if x < 0 or y < 0 or x + iw > W or y + ih > H:
        return

    roi = frame[y:y + ih, x:x + iw]
    if roi.shape[0] != ih or roi.shape[1] != iw:
        return  # skip if out of bounds or shape mismatch

    icon_rgb = icon[:, :, :3]
    alpha = icon[:, :, 3] / 255.0  # [0,1]

    for c in range(3):
        roi[:, :, c] = icon_rgb[:, :, c] * alpha + roi[:, :, c] * (1 - alpha)

    frame[y:y + ih, x:x + iw] = roi


def mpl_to_cv_coords(ax, x, y, fig, frame):
    """
    Converts Matplotlib data coords (x,y) into pixel coords of the OpenCV frame.
    Uses exact axes → figure → renderer transforms.
    """
    # Data → Display coords (pixel coords in figure)
    disp = ax.transData.transform((x, y))

    # Display coords are relative to the figure, NOT the image buffer.
    # But the Agg buffer is exactly the rendered figure pixel matrix.

    px, py = int(disp[0]), int(disp[1])

    # Convert Matplotlib's y-down (from top) to OpenCV's y-down
    H, W = frame.shape[:2]
    py = H - py  # invert y axis

    return px, py


border = {"xmin": -4, "xmax": 4, "ymin": -4, "ymax": 4, "zmin": 0, "zmax": 4}
numRob = 5  # number of robots
dt = 0.01  # time interval [s]
simTime = 70.0  # simulation time [s]
maxVel = 2  # maximum velocity [m/s]
devInput = np.array([[0.25, 0.25, 0.01]]).T  # input deviation in simulation, Vx[m/s], Vy[m/s], yawRate[rad/s]
devObser = 0.1  # observation deviation of distance[m]
ekfStride = 1  # update interval of EKF is simStride*0.01[s]


def world_to_pixel(x, y, border, W, H):
    """
    Convert world coords (x,y) to pixel coords (px, py) for the rendered frame.
    x: horizontal in meters, within [xmin, xmax]
    y: vertical in meters, within [ymin, ymax]
    W: frame width in pixels
    H: frame height in pixels
    """
    px = int((x - border["xmin"]) / (border["xmax"] - border["xmin"]) * W)
    # flip y because image origin is top-left, matplotlib origin is bottom-left
    py = int((border["ymax"] - y) / (border["ymax"] - border["ymin"]) * H)
    return px, py


# ---------------------------------------------------------
# Variables being updated in simulation
# ---------------------------------------------------------
# random initial groundTruth of state [x, y, yaw]' of numRob robots
xTrue = np.random.uniform(-3, 3, (3, numRob))

# [x_ij, y_ij, yaw_ij]' of the second robot in the first robot's view
relativeState = np.zeros((3, numRob, numRob))

data = dataCreate(numRob, border, maxVel, dt, devInput, devObser)
relativeEKF = EKFonSimData(10, 0.1, 0.25, 0.4, 0.1, numRob)

# These will be created in the show_animation branch before the loop
fig = None
canvas = None
pointsTrue = None
pointsEsti = None
pointsTrueHead = None
pointsEstiHead = None
circle = None
time_text = None


def animate(step: int):
    """
    One simulation step:
    - updates xTrue, relativeState
    - updates Matplotlib artists
    - renders figure to Agg canvas
    - shows frame in OpenCV and overlays icons on estimated robot positions
    """
    global xTrue, relativeState
    global pointsTrue, pointsEsti, pointsTrueHead, pointsEstiHead, circle, time_text
    global fig, canvas

    # xEsti will be used later for icon overlay
    if step * dt < 20:
        u = data.calcInput_FlyIn1m(step)
        # u = data.calcInput_PotentialField(step, xTrue)
        # u = data.calcInput_Formation01(step, relativeState)
        xTrue, zNois, uNois = data.update(xTrue, u)
        if step % ekfStride == 0:
            relativeState = relativeEKF.EKF(uNois, zNois, relativeState, ekfStride)

        xEsti = transform.calcAbsPosUseRelaPosWRTRob0(xTrue[:, 0], relativeState, xTrue, numRob)
        distance = np.linalg.norm(xEsti[:2, :] - xTrue[:2, :])

        pointsTrue.set_data(xTrue[0, :], xTrue[1, :])  # groundTruth points
        pointsEsti.set_data(xEsti[0, :], xEsti[1, :])  # estimated points

        pointsTrueHead.set_data(
            xTrue[0, :] + 0.07 * np.cos(xTrue[2, :]),
            xTrue[1, :] + 0.07 * np.sin(xTrue[2, :])
        )
        pointsEstiHead.set_data(
            xEsti[0, :] + 0.07 * np.cos(xEsti[2, :]),
            xEsti[1, :] + 0.07 * np.sin(xEsti[2, :])
        )

        circle.center = (xTrue[0, 0], xTrue[1, 0])
        circle.radius = 0  # zNois[0, 1]  # could visualize distance
        time_text.set_text(f"t={step * dt:.2f}s, distance={distance:.2f}")

    else:
        u = data.come_to_position(step, xTrue)
        # u = data.calcInput_PotentialField(step, xTrue)
        # u = data.calcInput_Formation01(step, relativeState)
        xTrue, zNois, uNois = data.update(xTrue, u)
        if step % ekfStride == 0:
            relativeState = relativeEKF.EKF(uNois, zNois, relativeState, ekfStride)

        xEsti = transform.calcAbsPosUseRelaPosWRTRob0(xTrue[:, 0], relativeState, xTrue, numRob)
        distance = np.linalg.norm(xEsti[:2, :] - xTrue[:2, :])

        pointsTrue.set_data(xTrue[0, :], xTrue[1, :])
        pointsEsti.set_data(xEsti[0, :], xEsti[1, :])

        pointsTrueHead.set_data(
            xTrue[0, :] + 0.07 * np.cos(xTrue[2, :]),
            xTrue[1, :] + 0.07 * np.sin(xTrue[2, :])
        )
        pointsEstiHead.set_data(
            xEsti[0, :] + 0.07 * np.cos(xEsti[2, :]),
            xEsti[1, :] + 0.07 * np.sin(xEsti[2, :])
        )

        circle.center = (xTrue[0, 0], xTrue[1, 0])
        circle.radius = 0  # zNois[0, 1]
        time_text.set_text(f"t={step * dt:.2f}s, distance={distance:.2f}")

    # -------------------------------------------------
    # Matplotlib → Agg canvas → NumPy → OpenCV
    # -------------------------------------------------
    fig.canvas.draw()  # ensure the figure is rendered
    canvas.draw()  # Agg backend draws to its buffer

    buf = canvas.buffer_rgba()
    img = np.asarray(buf, dtype=np.uint8)

    # Convert RGBA to BGR for OpenCV
    frame = cv.cvtColor(img, cv.COLOR_RGBA2BGR)

    # -------------------------------------------------
    # Overlay icon on each estimated robot position
    # -------------------------------------------------
    H, W = frame.shape[:2]
    for i in range(numRob):
        # world coords of estimated robot i
        x = xEsti[0, i]
        y = xEsti[1, i]

        # px, py = world_to_pixel(x, y, border, W, H)
        px, py = mpl_to_cv_coords(ax, x, y, fig, frame)

        # center the icon on the position
        px -= icon_w // 2
        py -= icon_h // 2

        overlay_icon(frame, icon, px, py)

    # Show in OpenCV window
    cv.imshow("Swarm Simulation", frame)


if show_animation:
    # -------------------------------------------------
    # Set up a Matplotlib figure but do NOT use plt.show
    # -------------------------------------------------
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111, aspect='equal')
    ax.set(xlim=(border["xmin"], border["xmax"]), ylim=(border["ymin"], border["ymax"]))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    title = ax.set_title('Simulated swarm')
    pointsTrue,  = ax.plot([], [], linestyle="", marker="o", color="b", label="GroundTruth")
    pointsEsti,  = ax.plot([], [], linestyle="", marker="o", color="r", label="Relative EKF")
    pointsTrueHead,  = ax.plot([], [], linestyle="", marker=".", color="g")
    pointsEstiHead,  = ax.plot([], [], linestyle="", marker=".", color="g")
    ax.legend()
    circle = plt.Circle((0, 0), 0.2, color='black', fill=False)
    ax.add_patch(circle)
    time_text = ax.text(0.01, 0.97, '', transform=ax.transAxes)
    time_text.set_text('')


    num_steps = int(simTime / dt)

    for step in range(num_steps):
        animate(step)

        # Allow user to quit with 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

else:
    # Simulation using while-loop; figure of X-Y-Yaw errors
    # xEsti = transform.calcAbsPosUseRelaPosWRTRob0(xTrue[:,0], relativeState, xTrue, numRob) # absolute states of all robots in world-frame
    # dataForPlot = np.array([xEsti[0,1], xTrue[0,1], xEsti[1,1], xTrue[1,1], xEsti[2,1], xTrue[2,1]]) # x, xGT, y, yGT, yaw, yawGT
    xEsti = relativeState[:,0,:] # relative states in robot0's body-frame
    xTrueRL = transform.calcRelaState(xTrue, numRob) # groundTruth relative states
    dataForPlot = np.array([xEsti[0,1], xTrueRL[0,1], xEsti[1,1], xTrueRL[1,1], xEsti[2,1], xTrueRL[2,1]]) # x, xGT, y, yGT, yaw, yawGT    
    step = 0
    while simTime >= dt*step:
        step += 1
        u = data.calcInput_FlyIn1m(step)
        # u = data.calcInput_PotentialField(step, xTrue)
        # u = data.calcInput_Formation01(step, relativeState)
        # u = data.calcInput_FlyIn1mRob1NoVel(step)
        xTrue, zNois, uNois = data.update(xTrue, u)
        if step % ekfStride == 0:
            relativeState = relativeEKF.EKF(uNois, zNois, relativeState, ekfStride)

        # xEsti = transform.calcAbsPosUseRelaPosWRTRob0(xTrue[:,0], relativeState, xTrue, numRob)
        # dataForPlot = np.vstack([dataForPlot, np.array([xEsti[0,1], xTrue[0,1], xEsti[1,1], xTrue[1,1], xEsti[2,1], xTrue[2,1]])])
        xEsti = relativeState[:,0,:]
        xTrueRL = transform.calcRelaState(xTrue, numRob)
        dataForPlot = np.vstack([dataForPlot, np.array([xEsti[0,1], xTrueRL[0,1], xEsti[1,1], xTrueRL[1,1], xEsti[2,1], xTrueRL[2,1]])]) 

    dataForPlotArray = dataForPlot.T
    timePlot = np.arange(0, len(dataForPlotArray[0]))/100.0
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    plt.margins(x=0)

    ax1.plot(timePlot, dataForPlotArray[0,:])
    ax1.plot(timePlot, dataForPlotArray[1,:])
    ax1.set_ylabel(r"$x_{ij}$ (m)", fontsize=12)
    ax1.grid(True)

    ax2.plot(timePlot, dataForPlotArray[2,:])
    ax2.plot(timePlot, dataForPlotArray[3,:])
    ax2.set_ylabel(r"$y_{ij}$ (m)", fontsize=12)
    ax2.grid(True)

    ax3.plot(timePlot, dataForPlotArray[4,:], label='Relative EKF')
    ax3.plot(timePlot, dataForPlotArray[5,:], label='Ground-truth')
    ax3.set_ylabel(r"$\mathrm{\psi_{ij}}$ (rad)", fontsize=12)
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.grid(True)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.8, 0.6), shadow=True, ncol=1, fontsize=12)
    # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()
