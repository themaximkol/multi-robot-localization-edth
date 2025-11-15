import numpy as np
import matplotlib.pyplot as plt

from dataCreate import dataCreate
from relativeEKF import EKFonSimData
import transform

# ---- global sim config (similar to main_simulation.py) ----

border = {"xmin": -4, "xmax": 4,
          "ymin": -4, "ymax": 4,
          "zmin": 0, "zmax": 4}

numRob = 4  # number of robots
dt = 0.01  # time step [s]
simTime = 70.0  # max sim time [s]
maxVel = 1.0  # max robot velocity [m/s]

# base input noise (paper default) – we will scale this
BASE_DEV_INPUT = np.array([[0.25, 0.25, 0.01]]).T  # Vx, Vy, yawRate noise

# keep UWB noise fixed for this experiment
DEV_OBSER = 0.1  # distance measurement std [m]

# EKF parameters
ekfStride = 1
relativeEKF_params = dict(
    Nwin=10,
    a1=0.1,
    a2=0.25,
    a3=0.4,
    a4=0.1
)

# convergence parameters
POS_THR = 0.01  # [m] threshold for relative position error
NUM_TRIALS = 10  # simulations per noise level

# sweep devInput as: BASE_DEV_INPUT * scale
INPUT_SCALE_START = 0.0
INPUT_SCALE_END = 3.0
INPUT_SCALE_STEP = 0.5


def run_single_sim(dev_input_scale: float, seed: int):
    """
    Run one simulation with devInput = BASE_DEV_INPUT * dev_input_scale.
    Return convergence time in seconds, or None if never converged.
    """
    np.random.seed(seed)

    # random initial true states: [x, y, yaw] x numRob
    xTrue = np.random.uniform(-3, 3, (3, numRob))
    relativeState = np.zeros((3, numRob, numRob))

    # scaled input noise
    devInput = BASE_DEV_INPUT * dev_input_scale

    # data generator with this devInput and fixed DEV_OBSER
    data = dataCreate(numRob, border, maxVel, dt, devInput, DEV_OBSER)

    # EKF instance
    relativeEKF = EKFonSimData(
        relativeEKF_params["Nwin"],
        relativeEKF_params["a1"],
        relativeEKF_params["a2"],
        relativeEKF_params["a3"],
        relativeEKF_params["a4"],
        numRob
    )

    step = 0
    conv_step = None

    while step * dt < simTime:
        step += 1

        # choose one of the input patterns from dataCreate
        u = data.calcInput_FlyIn1m(step)
        # you can experiment with these instead:
        # u = data.calcInput_PotentialField(step, xTrue)
        # u = data.calcInput_Formation01(step, relativeState)

        # update true state + generate noisy measurements
        xTrue, zNois, uNois = data.update(xTrue, u)

        if step % ekfStride == 0:
            relativeState = relativeEKF.EKF(uNois, zNois, relativeState, ekfStride)

            # EKF estimate of relative states in robot0's frame
            xEsti = relativeState[:, 0, :]  # shape (3, numRob)
            # ground-truth relative states in robot0's frame
            xTrueRL = transform.calcRelaState(xTrue, numRob)

            # look at robot1 relative to robot0
            est_vec = xEsti[0:2, 1]  # estimated [x_01, y_01]
            true_vec = xTrueRL[0:2, 1]  # true [x_01, y_01]
            err = float(np.linalg.norm(est_vec - true_vec))

            if err < POS_THR:
                conv_step = step
                break

    if conv_step is None:
        return None
    return conv_step * dt  # seconds


def main():
    scales = np.arange(INPUT_SCALE_START, INPUT_SCALE_END + 1e-9, INPUT_SCALE_STEP)
    results = []

    print("Experiment: devInput scaling vs convergence time")
    print(f"  devObser (range noise) fixed at {DEV_OBSER:.2f} m")
    print(f"  convergence threshold = {POS_THR:.3f} m")

    for s in scales:
        times = []
        for trial in range(NUM_TRIALS):
            t_conv = run_single_sim(s, seed=trial)
            if t_conv is not None:
                times.append(t_conv)

        if times:
            mean_conv = float(np.mean(times))
            std_conv = float(np.std(times))
            print(f"scale = {s:.2f} -> devInput = {BASE_DEV_INPUT.ravel() * s} "
                  f"=> mean conv time = {mean_conv:.2f} s "
                  f"(std = {std_conv:.2f}, {len(times)}/{NUM_TRIALS} runs converged)")
        else:
            mean_conv = None
            print(f"scale = {s:.2f} -> devInput = {BASE_DEV_INPUT.ravel() * s} "
                  f"=> no runs converged")
        results.append((s, mean_conv))

    # plot only converged points
    xs = [r[0] for r in results if r[1] is not None]
    ys = [r[1] for r in results if r[1] is not None]

    if xs:
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Input noise scale (devInput = base * scale)")
        plt.ylabel("Mean convergence time [s]")
        plt.grid(True)
        plt.title(f"Effect of self-motion noise on convergence (devObser={DEV_OBSER})")
        plt.show()
    else:
        print("No converged runs, nothing to plot.")


if __name__ == "__main__":
    main()
