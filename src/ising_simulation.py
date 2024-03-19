from ising import generate_configuration, propagate, Trajectory, filter_trajectories, generate_trajectory_name, plot_configuration
from multiprocessing import Process
import numpy as np

def simulate_temperatures(N, J, B, temperatures):
    for temperature in temperatures:
        configuration = generate_configuration(N=N, random=False)
        filename = generate_trajectory_name(N=N, J=J, B=B, temperature=temperature, folder="ising_analysis_traj")
        propagate(configuration=configuration, n_timestep=1000000, n_output=1000, J=J, B=B, temperature=temperature, filename=filename)

if __name__ == "__main__":
    temperatures = np.round(np.arange(0.01, 3.51, 0.01), 2)

    J = 1
    B = 0
    N = 50

    processes = 8

    # Divide temperatures to simulate in blocks of mostly equal size (+/- 1) according to numbr of processes
    n_temps = len(temperatures)
    temps_per_block = n_temps//processes
    residuals = n_temps%processes
    temps = [list(temperatures[i*temps_per_block:(i+1)*temps_per_block]) + ([temperatures[-residuals+i]] if i < residuals and residuals > 0 else []) for i in range(processes)]


    procs = []
    for i in range(processes):
        proc = Process(target=simulate_temperatures, args=(N, J, B, temps[i]))
        procs.append(proc)
        proc.start()
        print(f"Started process {i+1} of {processes} with {len(temps[i])} temperatures")

    for proc in procs:
        proc.join()