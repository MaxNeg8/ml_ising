import numpy as np
from numba import jit
from matplotlib import pyplot as plt

def generate_configuration(N : int, random=True) -> np.ndarray:
    """
    Function that generates a 2D spin configuration for simulation using
    the Ising model.

    Parameters
    ----------
    N : int
        The size of the simulation grid. Grid will be quadratic with dimensions (N x N). N must
        be greater than 0.

    random : bool
        If True, spins will be oriented randomly. Otherwise, the spins will be oriented in
        the same direction (value 1)

    Returns
    -------
    configuration : np.ndarray
        The spin configuration as a 2D NumPy array of integers -1 or 1
    """
    assert N > 0, "System size must be greater than 0"
    if random:
        return np.random.choice([-1, 1], size=(N, N))
    return np.ones((N, N), dtype=int)

@jit(nopython=True)
def compute_energy(configuration : np.ndarray, J : float, B : float):
    """
    Function that computes the total energy of a given configration.

    Parameters
    ----------
    configuration : np.ndarray
        The configuration for which to compute the energy. Must be a 2D NumPy integer array with
        entries in {-1, 1}.

    J : float
        The positive interaction constant for the spins

    B : float
        The strength of the outer magnetic field

    Returns
    -------
    energy : float
        The computed total energy of the system
    """

    assert configuration.ndim == 2, "Configuration is not a 2D NumPy array"
    assert configuration.shape[0] == configuration.shape[1], "Configuration is not a quadratic"
    assert J >= 0, "Interaction constant must be positive"

    N = configuration.shape[0]
    for i in range(N):
        for j in range(N):
            pass

def main():
    print(generate_configuration(10))

if __name__ == "__main__":
    main()
