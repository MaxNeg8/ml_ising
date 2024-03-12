import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from typing import Optional
import os

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
        return np.random.choice([-1, 1], size=(N, N))
    return np.ones((N, N), dtype=int)

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
        were to be flipped

    Example
    -------
    If the current energy of the configuration is E and we try a spin flip at spin index (1, 1), we obtain the new energy
    new_energy of the system using

        >>> dE = compute_flip_energy(configuration, (1, 1), ...)
        >>> new_energy = E + dE

    """
    N = configuration.shape[0]
    i, j = position

    sum_neighbours = configuration[(i + 1)%N, j] + configuration[(i - 1)%N, j] + configuration[i, (j + 1)%N] + configuration[i, (j - 1)%N]
    return 2 * J * configuration[i, j] * sum_neighbours + 2 * B * configuration[i, j]

@jit(nopython=True)
def magnetization(configuration : np.ndarray, normalize : bool=True) -> float:
    """
    Function that calculates the magnetization (per spin) of a given configuration.

    Parameters
    ----------
    configuration : np.ndarray
        The configuration for which to compute the magnetization (per spin)

    normalize : bool
        If True, the magnetization per spin is returned. Otherwise, the total magnetization is calculated.

    Returns
    -------
    magnetization : float
        The magnetization (per spin)
    """
    if normalize:
        return np.mean(configuration)
    return np.sum(configuration)

def save_trajectory(filename : str, trajectory : np.ndarray) -> None:
    """
    Function that saves a simulation trajectory as a csv file.

    Parameters
    ----------
    filename : str
        The name of the file the trajectory will be saved to

    trajectory : np.ndarray
        The NumPy array containing the trajectory. Shape is (n_timestep, N, N) and
        the configuration at timestep i is trajectory[i].
    """
    n_timestep = trajectory.shape[0]
    N = trajectory.shape[1]
    flattened_trajectory = np.reshape(trajectory, (n_timestep, N**2))
    np.savetxt(filename, flattened_trajectory, delimiter=",", fmt="%i")

def load_trajectory(filename : str) -> np.ndarray:
    """
    Function that loads a simulation trajectory from the given file.

    Parameters
    ----------
    filename : str
        The name of the file from which to load the trajectory

    Returns
    -------
    trajectory : np.ndarray
        The loaded trajectory of shape (n_timestep, N, N)
    """
    loaded_trajectory = np.loadtxt(filename, delimiter=",").astype(int)
    n_timestep = loaded_trajectory.shape[0]
    N = int(np.sqrt(loaded_trajectory.shape[1]))
    return np.reshape(loaded_trajectory, (n_timestep, N, N))

def propagate(configuration : np.ndarray, n_timestep : int, J : float, B : float, temperature : float, n_output : int=0, filename : str=None, copy : bool=False) -> Optional[np.ndarray]:
    """
    Function that propagates a spin configuration in the Ising model using Markov Chain Monte Carlo and the Metropolis algorithm

    Parameters
    ----------
    configuration : np.ndarray
        The configuration to propagate

    n_timestep : int
        The number of timesteps to propagate for
    
    J : float
        The positive interaction constant for the spins

    B : float
        The strength of the outer magnetic field

    temperature : float
        The temperature to use for the Metropolis criterion

    n_output : int
        The number of timesteps to wait between outputing trajectory frames

    filename : str
        The name of the file to output the trajectory to

    copy : bool
        If True, creates a copy of the original array instead of overwriting it. The propagated copy
        is then returned.

    Returns
    -------
        None, if copy is False. Otherwise, the propagated copy of the configuration.
    """

    assert (n_output == 0 and filename == None) or (n_output != 0 and filename != None), "If you provide a filename or an n_output > 0, you must also provide the other"

    if n_output:
        n_frames = int(np.ceil(n_timestep/n_output))
        trajectory = np.empty((n_frames, configuration.shape[0], configuration.shape[1]))

    N = configuration.shape[0]

    E = compute_energy(configuration, J, B)

    if copy:
        configuration = configuration.copy()

    for i in range(n_timestep):
        spin_to_flip = tuple(np.random.randint(0, N, size=2))
        
        dE = compute_flip_energy(configuration, spin_to_flip, J, B)
     
        if dE <= 0 or np.random.rand() < np.exp(-dE/temperature): # Only compute exponential if dE > 0, otherwise always accept
            configuration[spin_to_flip[0], spin_to_flip[1]] *= -1
            E += dE
        
        if n_output and i % n_output == 0:
            trajectory[i//n_output] = configuration
    
    if n_output:
        save_trajectory(filename, trajectory)

    if copy:
        return configuration

def plot_configuration(configuration : np.ndarray) -> None:
    """
    Function that plots a given configuration as a heatmap.

    Parameters
    ----------
    configuration : np.ndarray
        The configuration to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.imshow(configuration, cmap="summer")
    ax.set_axis_off()

    plt.show()

def animate_trajectory(filename : str) -> None:
    """
    Function that generates an animation of a given trajectory file.

    Parameters
    ----------
    filename : str
        The file containing the trajectory to plot
    """
    if not os.path.isfile(filename):
        raise ValueError(f"File {filename} does not exist")

    trajectory = load_trajectory(filename)

    n_frames = trajectory.shape[0]
    assert n_frames > 0, "Trajectory has no images"

    def draw(frame):
        ax.clear()
        ax.imshow(trajectory[frame], cmap="summer")
        ax.set_title(f"File: {os.path.basename(filename)}, Frame: {frame}/{n_frames}")
        ax.set_axis_off()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    animation = FuncAnimation(fig, draw, frames=n_frames, interval=1/50, repeat=True)
    
    slider = Slider(plt.axes([0.2, 0.02, 0.65, 0.03]), 'Speed', 1, 1000, valinit=500)

    def update_speed(value):
        animation.event_source.interval = 100/value

    slider.on_changed(update_speed)

    plt.show()

def main():
    configuration = generate_configuration(10, True)
    propagate(configuration, 100000, 1, 0, 3, n_output=10, filename="out.csv")
    animate_trajectory("out.csv")

if __name__ == "__main__":
    main()
