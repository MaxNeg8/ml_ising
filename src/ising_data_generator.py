from ising import generate_configuration, propagate, save_configurations_bin
import time
import numpy as np

def simulate(N: int, J : float, B : float, temperature : float, n_samples : int = 1):
    """
    Simulate Ising model for given parameters to sample n_samples uncorrelated configurations
    from the respective configuration space. All parameters can either be given as single
    values or as lists, in which case every parameter in the respective list will be simulated.

    Parameters
    ----------
    N : int
        Number of edge lattice sites

    J : float
        Spin-spin coupling strength

    B : float
        Strength of outer magnetic field

    temperature': float
        Simulation temperature

    n_samples : int
        Number of samples to generate from each configuration space
    """

    Ns = [N] if not isinstance(N, list) and not isinstance(N, np.ndarray) else N
    Js = [J] if not isinstance(J, list) and not isinstance(J, np.ndarray) else J
    Bs = [B] if not isinstance(B, list) and not isinstance(B, np.ndarray) else B
    temperatures = [temperature] if not isinstance(temperature, list) and not isinstance(temperature, np.ndarray) else temperature

    for B in Bs:
        for N in Ns:
            for J in Js:
                filename = "ising_ml_data/" + f"N_{N}_J_{J}_B_{B}_"
                labels = np.array([temperatures[i//n_samples] for i in range(len(temperatures)*n_samples)])
                np.savetxt(filename + "labels_train.csv", labels)
                configurations = np.empty((n_samples*len(temperatures), N, N), dtype=int)
                for i,temperature in enumerate(temperatures):
                    for sample in range(n_samples):
                        configuration = generate_configuration(N=N, random=True)
                        propagate(configuration=configuration, n_timestep=300, n_output=0, J=J, B=B, temperature=temperature, filename=None, algorithm="wolff")
                        configurations[i*n_samples + sample] = configuration
                save_configurations_bin(filename + "data_train", configurations, allow_overwrite=True)


def main():
    temperatures = np.random.random(10000)*4.0

    J = 1
    B = 0
    N = 10

    simulate(N, J, B, temperatures, 5)

if __name__ == "__main__":
    main()
