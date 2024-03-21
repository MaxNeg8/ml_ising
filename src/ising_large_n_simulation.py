import numpy as np
from numba import jit
from multiprocessing import Process

@jit(nopython=True)
def generate_configuration(N : int, random : bool=True) -> np.ndarray:
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
        return np.random.randint(0, 2, size=(N, N)) * 2 - 1
    return np.random.randint(0, 2, size=(N, N)) * 0 + 1

@jit(nopython=True)
def compute_energy(configuration : np.ndarray, J : float, B : float) -> float:
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
    N = configuration.shape[0]

    H = 0.0
    if B == 0.0: # This may seem unelegant, but we save N**2 float multiplications or comparisons this way
        for i in range(N):
            for j in range(N):
                S = configuration[i, j]
                H -= S * (configuration[(i + 1)%N, j] + configuration[i, (j + 1)%N]) # We only need two directions since the other two will be counted anyways through PBC
        H *= J
    else:
        for i in range(N):
            for j in range(N):
                S = configuration[i, j]
                H -= J * S * (configuration[(i + 1)%N, j] + configuration[i, (j + 1)%N]) # We only need two directions since the other two will be counted anyways through PBC
                H -= B * S
    return H

@jit(nopython=True)
def compute_flip_energy(configuration : np.ndarray, position : tuple[int, int], J : float, B : float) -> float:
    """
    Function that computes the change in energy if the spin at the given position is flipped

    Parameters
    ----------
    configuration : np.ndarray
        The configuration for which to compute the energy difference after spin flip. Must be a 2D NumPy 
        integer array with entries in {-1, 1}.

    position : tuple[int, int]
        The index (row, col) of the spin to be flipped

    J : float
        The positive interaction constant for the spins

    B : float
        The strength of the outer magnetic field

    Returns
    -------
    difference : float
        The energy difference that needs to be added to the old energy if the spin at the given position
        was to be flipped

    Example
    -------
    If the current energy of the configuration is E and we try a spin flip at spin index (1, 1), we obtain the new energy
    new_energy of the system using

        >>> dE = compute_flip_energy(configuration, (1, 1), ...)
        >>> new_energy = E + dE

    """
    N = configuration.shape[0]
    i, j = position[0], position[1]

    sum_neighbours = configuration[(i + 1)%N, j] + configuration[(i - 1)%N, j] + configuration[i, (j + 1)%N] + configuration[i, (j - 1)%N]
    return 2 * J * configuration[i, j] * sum_neighbours + 2 * B * configuration[i, j]

@jit(nopython=True)
def compute_row_col_flip_energy(configuration : np.ndarray, row_col : str, row_col_index : int, J : float, B : float) -> float:
    """ 
    Function that computes the change in energy if and entire row or column of spins is flipped

    Parameters
    ----------
    configuration : np.ndarray
        The configuration for which to compute the energy difference after spin flips. Must be a 2D NumPy 
        integer array with entries in {-1, 1}.

    row_col : str
        Either "row" if row is to be flipped or "col" for column flip

    row_col_index : int
        The index of the row or column to be flipped

    J : float
        The positive interaction constant for the spins

    B : float
        The strength of the outer magnetic field

    Returns
    -------
    difference : float
        The energy difference that needs to be added to the old energy if the row/col of spins at the given index
        was to be flipped
    """
    N = configuration.shape[0]
    if row_col == "row":
        sum_neighbours = configuration[(row_col_index + 1)%N] + configuration[(row_col_index - 1)%N]
        return 2 * J * np.sum(configuration[row_col_index] * sum_neighbours) + 2 * B * np.sum(configuration[row_col_index])
    else:
        sum_neighbours = configuration[:, (row_col_index + 1)%N] + configuration[:, (row_col_index - 1)%N]
        return 2 * J * np.sum(configuration[:, row_col_index] * sum_neighbours) + 2 * B * np.sum(configuration[:, row_col_index])
    
def write_to_file(temperature, magnetization):
    filename = "ising_analysis_traj/out.csv"
    with open(filename, "a") as file:
        file.write(f"{temperature},{magnetization}\n")

    
@jit(nopython=True)
def simulation(temperature, N, J, B, n_timestep):
    configuration = generate_configuration(N=N, random=temperature > 2.269185)

    E = compute_energy(configuration, J, B)
    start = N**2
    magnetization = np.zeros(n_timestep - start, dtype=np.intc)

    for i in range(n_timestep):
        if i == start:
            magnetization[i - start] = int(np.sum(configuration))
        if np.random.rand() > 0.1: # Attempt single spin flip
            spin_to_flip = np.random.randint(0, N, size=2)
        
            dE = compute_flip_energy(configuration, spin_to_flip, J, B)
     
            if dE <= 0 or np.random.rand() < np.exp(-dE/temperature): # Only compute exponential if dE > 0, otherwise always accept
                configuration[spin_to_flip[0], spin_to_flip[1]] *= -1
                E += dE
                if i > start:
                    magnetization[i - start] = magnetization[i - start - 1] + int(2*configuration[spin_to_flip[0], spin_to_flip[1]])
            else:
                if i > start:
                    magnetization[i - start] = magnetization[i - start - 1]
        else: # Attempt row or column spin flip
            row_col_selected = np.random.randint(0, N)
            row_col = "row" if i % 2 else "col"
            dE = compute_row_col_flip_energy(configuration, row_col, row_col_selected, J, B)
            if dE <= 0 or np.random.rand() < np.exp(-dE/temperature): # Only compute exponential if dE > 0, otherwise always accept
                if row_col == "row":
                    configuration[row_col_selected, :] *= -1
                    if i > start:
                        magnetization[i - start] = magnetization[i - start - 1] + int(2*np.sum(configuration[row_col_selected, :]))
                else:
                    configuration[:, row_col_selected] *= -1
                    if i > start:
                        magnetization[i - start] = magnetization[i - start - 1] + int(2*np.sum(configuration[:, row_col_selected]))
                E += dE
            else:
                if row_col == "row":
                    if i > start:
                        magnetization[i - start] = magnetization[i - start - 1]
                else:
                    if i > start:
                        magnetization[i - start] = magnetization[i - start - 1]

    return np.abs(np.mean(magnetization[::100000])) / N**2

def simulate(temperatures, N, J, B, n_timestep):
    for temperature in temperatures:
        mag = simulation(temperature, N, J, B, n_timestep)
        write_to_file(temperature, mag)

if __name__ == "__main__":
    temperatures = np.round(np.arange(2.1, 2.401, 0.001), 3)
    N = 1000
    J = 1
    B = 0
    n_timestep = int(100e6)

    processes = 8

    n_temps = len(temperatures)
    temps_per_block = n_temps//processes
    residuals = n_temps%processes
    temps = [list(temperatures[i*temps_per_block:(i+1)*temps_per_block]) + ([temperatures[-residuals+i]] if i < residuals and residuals > 0 else []) for i in range(processes)]

    procs = []
    for i in range(processes):
        proc = Process(target=simulate, args=(temps[i], N, J, B, n_timestep))
        procs.append(proc)
        proc.start()
        print(f"Started process {i+1} of {processes} with {len(temps[i])} temperatures")

    for proc in procs:
        proc.join()