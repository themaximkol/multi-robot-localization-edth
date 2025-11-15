import numpy as np
import matplotlib.pyplot as plt

from dataCreate import dataCreate
from relativeEKF import EKFonSimData
import transform

# --- simulation constants (same as main_simulation.py) ---

border = {"xmin": -4, "xmax": 4,
          "ymin": -4, "ymax": 4,
          "zmin": 0, "zmax": 4}

numRob = 10  # number of robots
dt = 0.01*1  # time step [s]
simTime = 180.0  # max simulation time [s]
maxVel = 1.0  # max robot velocity [m/s]

# input noise (kept constant, like in the paper)
devInput = np.array([[0.25, 0.25, 0.01]]).T  # Vx, Vy, yawRate noise

# EKF parameters: same as main_simulation.py
ekfStride = 1
relativeEKF_params = dict(
    Nwin=10,  # window length (first arg)
    a1=0.1,
    a2=0.25,
    a3=0.4,
    a4=0.1
)

# --- experiment parameters ---

POS_THR = 0.01  # convergence threshold on position error [m]
NOISE_START = 0.0  # start devObser
NOISE_END = 0.2  # end devObser
NOISE_STEP = 0.04  # step size for devObser
NUM_TRIALS = 10  # runs per noise level


def run_single_sim(devObser: float, seed: int) -> float | None:
    """
    Run one simulation for a given distance noise std (devObser).
    Return convergence time in seconds, or None if it never converges.
    """
    # for reproducibility per trial
    np.random.seed(seed)

    # initial true state: [x, y, yaw] for numRob robots
    xTrue = np.random.uniform(-3, 3, (3, numRob))
    relativeState = np.zeros((3, numRob, numRob))

    # create data generator with specified range noise
    data = dataCreate(numRob, border, maxVel, dt, devInput, devObser)

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

        # same input pattern as main_simulation.py
        u = data.calcInput_FlyIn1m(step)
        # alternative options in original code:
        # u = data.calcInput_PotentialField(step, xTrue)
        # u = data.calcInput_Formation01(step, relativeState)

        xTrue, zNois, uNois = data.update(xTrue, u)

        if step % ekfStride == 0:
            relativeState = relativeEKF.EKF(uNois, zNois, relativeState, ekfStride)

            # relative estimated states in robot0's frame
            xEsti = relativeState[:, 0, :]  # shape (3, numRob)
            # ground-truth relative states in robot0's frame
            xTrueRL = transform.calcRelaState(xTrue, numRob)

            # we look at robot1 relative to robot0
            est_vec = xEsti[0:2, 1]  # [x_01_hat, y_01_hat]
            true_vec = xTrueRL[0:2, 1]  # [x_01_true, y_01_true]
            err_vec = est_vec - true_vec
            err_norm = float(np.linalg.norm(err_vec))

            if err_norm < POS_THR:
                conv_step = step
                break

    if conv_step is None:
        return None
    return conv_step * dt  # seconds


def main():
    noise_values = np.arange(NOISE_START, NOISE_END + 1e-9, NOISE_STEP)
    results = []

    print("Noise sweep (devObser) vs convergence time:")
    print("  threshold = {:.4f} m, step = {:.2f} s".format(POS_THR, dt))

    for sigma in noise_values:
        times = []
        for trial in range(NUM_TRIALS):
            t_conv = run_single_sim(sigma, seed=trial)
            if t_conv is not None:
                times.append(t_conv)

        if times:
            mean_conv = float(np.mean(times))
            std_conv = float(np.std(times))
            print(
                f"devObser = {sigma:.2f} m -> mean conv time = {mean_conv:.2f} s (std = {std_conv:.2f}, {len(times)}/{NUM_TRIALS} runs converged)")
        else:
            mean_conv = None
            print(f"devObser = {sigma:.2f} m -> no runs converged")
        results.append((sigma, mean_conv))

    # simple plot of noise vs convergence time (ignoring non-converged cases)
    xs = [r[0] for r in results if r[1] is not None]
    ys = [r[1] for r in results if r[1] is not None]

    if xs:
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("UWB range noise std dev (devObser) [m]")
        plt.ylabel("Mean convergence time [s]")
        plt.grid(True)
        plt.title(f"Convergence threshold = {POS_THR} m")
        plt.show()
    else:
        print("No converged runs to plot.")


if __name__ == "__main__":
    main()
