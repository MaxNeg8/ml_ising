from ising import generate_configuration, propagate, save_configurations_bin
import numpy as np

def simulate(N: int, J : float, B : float, temperature : float, training : bool, n_samples : int = 1):
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

    training : bool
        Whether or not this data set is to be regarded as a training set (influences the file name)

    n_samples : int
        Number of samples to generate from each configuration space
    """

    Ns = [N] if not isinstance(N, list) and not isinstance(N, np.ndarray) else N
    Js = [J] if not isinstance(J, list) and not isinstance(J, np.ndarray) else J
    Bs = [B] if not isinstance(B, list) and not isinstance(B, np.ndarray) else B
    temperatures = [temperature] if not isinstance(temperature, list) and not isinstance(temperature, np.ndarray) else temperature
    train_test = "train" if training else "test"

    total_simulations = len(Ns)*len(Js)*len(Bs)*len(temperatures)*n_samples
    current = 0

    for B in Bs:
        for N in Ns:
            for J in Js:
                filename = "ising_ml_data/" + f"N_{N}_J_{J}_B_{B}_"
                labels = np.array([temperatures[i//n_samples] for i in range(len(temperatures)*n_samples)])
                np.savetxt(filename + f"labels_{train_test}.csv", labels)
                configurations = np.empty((n_samples*len(temperatures), N, N), dtype=int)
                for i,temperature in enumerate(temperatures):
                    for sample in range(n_samples):
                        configuration = generate_configuration(N=N, random=True)
                        propagate(configuration=configuration, n_timestep=300, n_output=0, J=J, B=B, temperature=temperature, filename=None, algorithm="wolff")
                        configurations[i*n_samples + sample] = configuration
                        current += 1
                        print(f"Progress: {current}/{total_simulations} ({current/total_simulations*100:0.2f}%)", end="\r")
                save_configurations_bin(filename + f"data_{train_test}", configurations, allow_overwrite=True)
    print("\nDone.")

def main():
    temperatures = np.random.random(100000)*4.0

    J = 1
    B = 0
    N = 25

    simulate(N, J, B, temperatures, training=True, n_samples=5)

if __name__ == "__main__":
    main()
