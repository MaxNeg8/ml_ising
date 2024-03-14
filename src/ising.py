import numpy as np

from numba import jit

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from typing import Optional, Self

import json
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
        trajectory = np.empty((n_frames, configuration.shape[0], configuration.shape[1]), dtype=int)

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
        Trajectory(trajectory).save(filename)

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

class Trajectory:
    """
    Class that represents a simulation trajectory for the 2D ising model. It allows for
    loading/saving trajectories in a compressed format and analyzing trajectories.
    """

    def __init__(self, trajectory : np.ndarray, name : str = ""):
        """
        Init function for the Trajectory class.

        Parameters
        ----------
        trajectory : np.ndarray
            The simulation trajectory array with shape (n_timestep, N, N) and dtype int.

        name : str
            Name for the simulation trajectory (will be shown in plot title)
        """
        assert isinstance(trajectory, np.ndarray), "Trajectory must be NumPy array"
        assert trajectory.ndim == 3 and trajectory.shape[1] == trajectory.shape[2], "Trajectory must have shape (n_timestep, N, N)"
        assert trajectory.dtype == int, "Trajectory must have datatype int"
 
        self._n_timestep = trajectory.shape[0]
        self._N = trajectory.shape[1]

        self._trajectory = trajectory
        self.name = name

    @classmethod
    def from_file(cls, filename : str, name : str = "") -> Self:
        """
        Function that loads a simulation trajectory from the given file. Decompresses
        the data.

        Parameters
        ----------
        filename : str
            The name of the file from which to load the trajectory

        name : str
            Name for the loaded trajectory. If empty, filename will be used.

        Returns
        -------
        trajectory : Trajectory
            The loaded trajectory object
        """
        with open(filename) as file:
            json_dict = json.load(file)

        initial_configuration = np.array(json_dict["initial"], dtype=int)

        n_timestep = len(json_dict["changes"]) + 1
        N = int(np.sqrt(initial_configuration.size))

        trajectory = np.empty((n_timestep, N**2), dtype=int)
        trajectory[0] = initial_configuration

        changes = json_dict["changes"]

        for i in range(1, n_timestep):
            trajectory[i] = trajectory[i-1]
            trajectory[i, changes[i-1]] *= -1

        return cls(np.reshape(trajectory, (n_timestep, N, N)), os.path.basename(filename))

    def save(self, filename : str) -> None:
        """
        Function that saves the trajectory to file. Compresses the data
        by only storing indices of spins that changed between frames, which produces
        much smaller trajectory files compared to storing each configuration individually.

        Parameters
        ----------
        filename : str
            The name of the file the trajectory will be saved to
        """
    
        initial_configuration = self.trajectory[0].flatten()
        json_dict = {"initial": [int(spin) for spin in initial_configuration]}

        flattened_trajectory = np.reshape(self.trajectory, (self.n_timestep, self.N**2))
        difference = np.abs(flattened_trajectory[1:] - flattened_trajectory[:-1])
    
        json_dict["changes"] = [np.nonzero(row)[0].tolist() for row in difference]
    
        with open(filename, "w") as file:
            json.dump(json_dict, file)

    def animate(self) -> None:
        """
        Function that generates an animation of the trajectory.
        """

        def draw(frame):
            ax.clear()
            ax.imshow(self.trajectory[frame], cmap="summer")
            ax.set_title((f"Name: {self.name}, " if len(self.name) > 0 else "") + f"Frame: {frame}/{self.n_timestep}")
            ax.set_axis_off()

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        animation = FuncAnimation(fig, draw, frames=self.n_timestep, interval=1/50, repeat=True)
    
        slider = Slider(plt.axes([0.2, 0.02, 0.65, 0.03]), 'Speed', 1, 1000, valinit=500)

        def update_speed(value):
            animation.event_source.interval = 100/value

        slider.on_changed(update_speed)

        plt.show()     

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def N(self):
        return self._N

    @property
    def n_timestep(self):
        return self._n_timestep


def main():
    configuration = generate_configuration(10, True)
    propagate(configuration, 100000, 1, 0, 3, n_output=10, filename="out.json")
    Trajectory.from_file("out.json").animate()

if __name__ == "__main__":
    main()
