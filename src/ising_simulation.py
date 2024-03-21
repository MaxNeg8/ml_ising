from ising import generate_configuration, propagate, Trajectory, filter_trajectories, generate_trajectory_name, plot_configuration
from multiprocessing import Process
import numpy as np


def simulate(N, J, B, temperature):
    Ns = [N] if not isinstance(N, list) and not isinstance(N, np.ndarray) else N
    Js = [J] if not isinstance(J, list) and not isinstance(J, np.ndarray) else J
    Bs = [B] if not isinstance(B, list) and not isinstance(B, np.ndarray) else B
    temperatures = [temperature] if not isinstance(temperature, list) and not isinstance(temperature, np.ndarray) else temperature

    for temperature in temperatures:
        for N in Ns:
            for J in Js:
                for B in Bs:
                    configuration = generate_configuration(N=N, random=True)
                    filename = generate_trajectory_name(N=N, J=J, B=B, temperature=temperature, folder="ising_analysis_wolff_traj")
                    propagate(configuration=configuration, n_timestep=1000, n_output=1, J=J, B=B, temperature=temperature, filename=filename, algorithm="wolff")

# Different B simulations at different temperatures
if __name__ == "__main__":
    temperatures = [0.001, 0.3, 1.0, 2.0]

    J = 0
    Bs = np.round(np.arange(-2.0, 2.0, 0.01), 2)
    N = 10

    processes = 4

    # Divide Bs to simulate in blocks of mostly equal size (+/- 1) according to number of processes
    n_Bs = len(Bs)
    Bs_per_block = n_Bs//processes
    residuals = n_Bs%processes
    Bs = [list(Bs[i*Bs_per_block:(i+1)*Bs_per_block]) + ([Bs[-residuals+i]] if i < residuals and residuals > 0 else []) for i in range(processes)]


    procs = []
    for i in range(processes):
        proc = Process(target=simulate, args=(N, J, Bs[i], temperatures))
        procs.append(proc)
        proc.start()
        print(f"Started process {i+1} of {processes} with {len(Bs[i])} Bs")

    for proc in procs:
        proc.join()

# Different temperature simulations
"""
if __name__ == "__main__":
    temperatures = np.round(np.arange(0.01, 3.51, 0.01), 2)

    J = 1
    B = 0
    N = 50

    processes = 8

    # Divide temperatures to simulate in blocks of mostly equal size (+/- 1) according to number of processes
    n_temps = len(temperatures)
    temps_per_block = n_temps//processes
    residuals = n_temps%processes
    temps = [list(temperatures[i*temps_per_block:(i+1)*temps_per_block]) + ([temperatures[-residuals+i]] if i < residuals and residuals > 0 else []) for i in range(processes)]


    procs = []
    for i in range(processes):
        proc = Process(target=simulate, args=(N, J, B, temps[i]))
        procs.append(proc)
        proc.start()
        print(f"Started process {i+1} of {processes} with {len(temps[i])} temperatures")

    for proc in procs:
        proc.join()
"""