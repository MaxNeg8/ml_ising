import numpy as np

from numba import jit

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from typing import Optional, Self, Union, List

import json
import os
import time

import struct

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

@jit(nopython=True)
def _wolff_build_cluster(configuration : np.ndarray, start : tuple[int, int], p_add : float, B : float) -> Optional[list]:
    """
    Helper function that builds a cluster of spins with same orientation for the Wolff algorithm and, if B = 0, performs
    the cluster flip

    Parameters
    ----------
    configuration : np.ndarray
        Spin configuration to build clusters for

    start : tuple[int, int]
        Spin position at which to start building the cluster

    p_add : float
        Probability of adding a spin to the cluster if it has the right orientation

    B : float
        The strength of the outer magnetic field

    Returns
    -------
    If B != 0:
        cluster : list
            A list of all the spin indices (row, col) in the cluster that was built

    """
    N = configuration.shape[0]
    sign = configuration[start[0], start[1]]
    to_be_tried = [start]

    return_cluster = not np.isclose(B, 0)

    if return_cluster:
        cluster = [start]
    else:
        configuration[start[0], start[1]] *= -1
    while len(to_be_tried) > 0:
        row, col = to_be_tried.pop()

        neighbours = [((row + 1)%N, col), ((row - 1)%N, col), (row, (col + 1)%N), (row, (col - 1)%N)]
       
        for n in neighbours:
            if return_cluster and n in cluster:
                continue
            if configuration[n[0], n[1]] != sign:
                continue
            if np.random.rand() < p_add:
                to_be_tried.append(n)
                if return_cluster:
                    cluster.append(n)
                else:
                    configuration[n[0], n[1]] *= -1

    if return_cluster:
        return cluster

def _wolff_propagate(configuration : np.ndarray, n_timestep : int, J : float, B : float, temperature : float, n_output : int,
                            mute_output : bool) -> Optional[np.ndarray]:
    """
    Function that propagates a given configuration using the Wolff algorithm (cluster spin flips).

    NOT TO BE USED INDIVIDUALLY BUT ONLY THROUGH THE WRAPPER FUNCTION propagate()

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

    mute_output : bool
        If True, console output is muted.

    Returns
    -------
        None, if n_output is 0. Otherwise, the trajectory produced
    """
    if n_output:
        n_frames = int(np.ceil(n_timestep/n_output))
        trajectory = np.empty((n_frames, configuration.shape[0], configuration.shape[1]), dtype=int)

    N = configuration.shape[0]

    prefix = f"[N={N},J={J},B={B}] "
    if not mute_output:
        start_simulation = time.time()
        elapsed = 0
    
    p_add = 1 - np.exp(-2*J/temperature)

    for i in range(n_timestep):
        start = tuple(np.random.randint(0, N, size=2))

        cluster = _wolff_build_cluster(configuration, start, p_add, B)

        if cluster != None and len(cluster) > 0:
            rows, cols = zip(*cluster)
            dE = 2*B*np.sum(configuration[rows, cols])
            if dE < 0 or np.random.rand() < np.exp(-dE/temperature):
                configuration[rows, cols] *= -1

        if n_output and i % n_output == 0:
            trajectory[i//n_output] = configuration
        
        if not mute_output:
            if i % 10 == 0:
                elapsed = time.time() - start_simulation
            print(prefix, f"Simulation progress: {i/n_timestep*100:0.1f}%, Elapsed: {elapsed:0.1f} s\r", end="")
    
    if not mute_output:
        print(" "*os.get_terminal_size()[0] + "\r", end="")
        print(prefix, f"Done. Elapsed: {elapsed:0.1f} s")

    if n_output:
        return trajectory

def _metropolis_propagate(configuration : np.ndarray, n_timestep : int, J : float, B : float, temperature : float, n_output : int,
                            mute_output : bool) -> Optional[np.ndarray]:
    """
    Function that propagates a given configuration using the Metropolis algorithm (single spin flips).

    NOT TO BE USED INDIVIDUALLY BUT ONLY THROUGH THE WRAPPER FUNCTION propagate()

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

    mute_output : bool
        If True, console output is muted.

    Returns
    -------
        None, if n_output is 0. Otherwise, the trajectory produced
    """
    if n_output:
        n_frames = int(np.ceil(n_timestep/n_output))
        trajectory = np.empty((n_frames, configuration.shape[0], configuration.shape[1]), dtype=int)

    N = configuration.shape[0]

    E = compute_energy(configuration, J, B)
    accepted = 0

    prefix = f"[N={N},J={J},B={B}] "
    if not mute_output:
        start_simulation = time.time()
        elapsed = 0
    
    for i in range(n_timestep):
        if np.random.rand() > 0.1: # Attempt single spin flip
            spin_to_flip = np.random.randint(0, N, size=2)
        
            dE = compute_flip_energy(configuration, spin_to_flip, J, B)
     
            if dE <= 0 or np.random.rand() < np.exp(-dE/temperature): # Only compute exponential if dE > 0, otherwise always accept
                configuration[spin_to_flip[0], spin_to_flip[1]] *= -1
                E += dE
                accepted += 1
        else: # Attempt row or column spin flip
            row_col_selected = np.random.randint(0, N)
            row_col = "row" if i % 2 else "col"
            dE = compute_row_col_flip_energy(configuration, row_col, row_col_selected, J, B)
            if dE <= 0 or np.random.rand() < np.exp(-dE/temperature): # Only compute exponential if dE > 0, otherwise always accept
                if row_col == "row":
                    configuration[row_col_selected, :] *= -1
                else:
                    configuration[:, row_col_selected] *= -1
                E += dE
                accepted += 1

        if n_output and i % n_output == 0:
            trajectory[i//n_output] = configuration
        
        if not mute_output:
            if i % 300 == 0:
                elapsed = time.time() - start_simulation
            print(prefix, f"Simulation progress: {i/n_timestep*100:0.1f}%, Acceptance probability: {accepted/(i+1):0.4f} ({accepted}/{i+1}), Elapsed: {elapsed:0.1f} s\r", end="\r", flush=True)
    
    if not mute_output:
        print(" "*os.get_terminal_size()[0], end="\r")
        print(prefix, f"Done. Acceptance probability: {accepted/n_timestep:0.4f} ({accepted}/{n_timestep}), Elapsed: {elapsed:0.1f} s", flush=True)

    if n_output:
        return trajectory

def propagate(configuration : np.ndarray, n_timestep : int, J : float, B : float, temperature : float, n_output : int=0, 
            filename : str=None, algorithm : str = "metropolis", copy : bool=False, mute_output : bool = True) -> Optional[np.ndarray]:
    """
    Function that propagates a spin configuration in the Ising model using Markov Chain Monte Carlo and the Metropolis or Wolff algorithm

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

    algorithm : str
        Algorithm used for propagation (either "metropolis" or "wolff")

    copy : bool
        If True, creates a copy of the original array instead of overwriting it. The propagated copy
        is then returned.

    mute_output : bool
        If True, console output is muted.

    Returns
    -------
        None, if copy is False. Otherwise, the propagated copy of the configuration.
    """
    algorithm = algorithm.lower()
    assert algorithm in ["metropolis", "wolff"], f"Invalid algorithm '{algorithm}'"
    assert (n_output == 0 and filename == None) or (n_output != 0 and filename != None), "If you provide a filename or an n_output > 0, you must also provide the other"

    if copy:
        configuration = configuration.copy()

    if algorithm == "metropolis":
        trajectory = _metropolis_propagate(configuration, n_timestep, J, B, temperature, n_output, mute_output)
    else:
        trajectory = _wolff_propagate(configuration, n_timestep, J, B, temperature, n_output, mute_output)

    if n_output:
        Trajectory(trajectory).save(filename)

    if copy:
        return configuration

def plot_configuration(configuration : np.ndarray, cluster : list = None) -> None:
    """
    Function that plots a given configuration as a heatmap.

    Parameters
    ----------
    configuration : np.ndarray
        The configuration to plot

    cluster : list
        Cluster to highlight
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.imshow(configuration, cmap="summer", vmin=-1, vmax=1)
    if cluster != None:
        for row, col in cluster:
            ax.plot([col-0.5, col+0.5, col+0.5, col-0.5, col-0.5],[row-0.5, row-0.5, row+0.5, row+0.5, row-0.5], color='red')
    ax.set_axis_off()

    plt.show()

def generate_trajectory_name(N : int, J : float, B : float, temperature : float, folder : str = "trajectories") -> str:
    """
    Function that generates an appropriate file name for a trajectory of a simulation with given
    parameters

    Parameters
    ----------
    N : int
        System width/height

    J : float
        Spin-Spin coupling strength

    B : float
        Strength of magnetic field

    temperature : float
        Simulation temperature

    folder : str
        Name of the folder to which the trajectory will be saved (default: trajectories)

    Returns
    -------
    filename : str
        The name of the trajectory file
    """
    return f"{folder}/ising_{N}_{J}_{B}@{temperature}.json"

def filter_trajectories(N : Union[int, List[int], tuple[int, int]], J : Union[float, List[float], tuple[float, float]], 
                        B : Union[float, List[float], tuple[float, float]], temperature : Union[float, List[float], tuple[float, float]], 
                        folder : Union[str, List[str]] = "trajectories") -> dict:
    """
    Function that filters saved trajectories (that use the naming scheme defined in function generate_trajectory_name()) by
    the relevant simulation parameters and returns their file names.

    Parameters
    ----------
    sim_parameter : val | List[dtype(val)] | tuple[dtype(val), dtype(val)]
        The various simulation parameters, where sim_parameter is one of (N, J, B, temperature) and val is the value
        of the respective parameter. Depending on the datatype given (float/int, list or tuple), this has different
        effects on the filter algorithm.

        float/int:
            If a single number is given, the function will only look for trajectories with the respective parameter
            having exactly this value

        list:
            If a list of numbers is given, the algorithm will include all trajectories where the value of the respective
            parameter is in the list given

        tuple:
            If a tuple of two numbers (val_min, val_max) is given, the function will look for trajectories with
            val_min <= sim_param <= val_max, i.e., with the respective simulation parameter in the range defined
            by the tuple (boundaries included)

    folder : str | List[str]
        The folder in which to look for trajectories. If a list of folders is given, the function will look for
        trajectories in all the folders in the list

    Returns
    -------
    filenames : dict
        A dictionary of file names that match the filter parameters as keys and extracted simulation parameters as a dictionary
        as values
    """
    if isinstance(folder, str):
        folder = [folder]

    parameters = [N, J, B, temperature]

    filenames = {}
    for dir in folder:
        if not os.path.isdir(dir):
            continue # Skip folder if it does not exist
        files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.splitext(f)[1] == '.json' and f.startswith("ising_")]

        for file in files:
            basename = os.path.basename(file)[6:-5] # Cut away ising_ prefix and .json file format
            split_underscore = basename.split("_")
            if len(split_underscore) != 3 or len(basename.split("@")) != 2 or len(split_underscore[2].split("@")) != 2:
                continue # File name does not correspond to format defined in generate_trajectory_name()
            
            param_values = [split_underscore[0], split_underscore[1], split_underscore[2].split("@")[0], basename.split("@")[1]]
            match = True
            for i, parameter in enumerate(parameters):
                comp_value = None
                try:
                    comp_value = float(param_values[i])
                except ValueError:
                    match = False
                    break # Skip file
                
                if isinstance(parameter, list):
                    if comp_value not in parameter:
                        match = False
                        break
                elif isinstance(parameter, tuple):
                    if not (parameter[0] <= comp_value <= parameter[1]):
                        match = False
                        break
                else:
                    if not np.isclose(float(parameter), comp_value):
                        match = False
                        break
            if match:
                filenames[file] = {"N" : float(param_values[0]), "J" : float(param_values[1]), "B" : float(param_values[2]), "temperature" : float(param_values[3])}

    return filenames

def correlate(signal_A : np.ndarray, signal_B : np.ndarray) -> np.ndarray:
    """
    Computes the correlation function between two signals A and B

    Parameters
    ----------
    signal_A : np.ndarray
        Signal A that will be correlated with signal B
    
    signal_B : np.ndarray
        Signal B that will be correlated with signal A

    Returns
    -------
    correlation : np.ndarray
        The computed correlation function of the two signals
    """
    N = len(signal_A)

    signal_A = signal_A.copy() - np.mean(signal_A)
    signal_B = signal_B.copy() - np.mean(signal_B)

    return np.correlate(signal_A, signal_B, mode="full")[-N:] / (np.arange(N, 0, -1) * np.std(signal_A)*np.std(signal_B))
       
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
    
        if len(os.path.dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as file:
            json.dump(json_dict, file)

    def animate(self) -> None:
        """
        Function that generates an animation of the trajectory.
        """

        def draw(frame):
            ax.clear()
            ax.imshow(self.trajectory[frame], cmap="summer", vmin=-1, vmax=1)
            ax.set_title((f"Name: {self.name}, " if len(self.name) > 0 else "") + f"Frame: {frame}/{self.n_timestep}")
            ax.set_axis_off()

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        animation = FuncAnimation(fig, draw, frames=self.n_timestep, interval=1/50, repeat=True)
    
        slider = Slider(plt.axes([0.2, 0.02, 0.65, 0.03]), 'Speed', 1, 1000, valinit=500)

        def update_speed(value):
            animation.event_source.interval = 100/value

        slider.on_changed(update_speed)

        plt.show()     

    def magnetization(self, r_equil : float = 0.0, normalize : bool = True, abs : bool = False, n_blocks : int = 1) -> float:
        """
        Computes the average magnetization (per spin) of the trajectory

        Parameters
        ----------
        r_equil : float
            Ratio of equilibration steps / total steps. The specified percentage of steps will be
            discarded as equilibration steps.

        normalize : bool
            If True, returns the average magnetization per spin instead of the total magnetization

        abs : bool
            If True, computes the absolute value of the magnetization instead

        n_blocks : int
            Number of blocks to use for block averaging. Default is 1, so no block averaging will be done.

        Returns
        -------
        magnetization : float
            Avergae magnetization per spin if normalize is True, else total magnetization
        """
        start_index = int(np.ceil(r_equil*self.n_timestep))
        length = self.n_timestep - start_index
        if length < n_blocks:
            n_blocks = length
        block_size = length//n_blocks
        residuals = length%n_blocks

        magnetizations = []
        for i in range(start_index, self.n_timestep):
            mag = magnetization(self.trajectory[i], normalize=False) # Avoid rounding issues due to small numbers
            magnetizations.append(np.abs(mag) if abs else mag)

        magnetizations = [magnetizations[i*block_size:(i+1)*block_size] + (magnetizations[-residuals:] if residuals > 0 and i == n_blocks - 1 else []) for i in range(n_blocks)]
        block_avg = np.mean([np.mean(magnetization) for magnetization in magnetizations])

        return block_avg/(self.N**2 if normalize else 1.0)

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def N(self):
        return self._N

    @property
    def n_timestep(self):
        return self._n_timestep

def save_configurations_bin(filename : str, configurations : np.ndarray, allow_overwrite : bool = False) -> None:
    """
    Used to save a list of configurations in binary form in order to consume as
    little space as possible. The file format will be the following:

    5 bytes: File header, word "ising" in ascii encoding
    2 bytes: Unsigned short, representing N (edge length) of the configurations
    4 bytes: Unsigned int, storing the number of configurations saved in the file
    Rest of the file: Configurations, where each byte represents 8 spins in binary (spin -1 is 0, spin 1 is 1)
    The last byte will be padded with zeros if necessary

    Paramerers
    ----------
    filename : str
        The name of the file (without extention, .ising will be appended automatically)

    configurations : np.ndarray
        Configurations to be saved in the file. If configurations.ndim == 2: Only one configuration
        will be saved. If configurations.ndim == 3: configurations[i, ...] will be treated as the
        ith configuration

    allow_overwrite : bool
        Whether or not to allow an existing file with the same name to be overwritten (default: False)
    """
    filename += ".ising"

    if os.path.isfile(filename) and not allow_overwrite:
        raise ValueError(f"File {filename} exists")

    if configurations.ndim == 2:
        configs = [configurations.flatten()]
        assert configurations.shape[0] == configurations.shape[1], "Must be square lattice"
        N = configurations.shape[0]
    elif configurations.ndim == 3:
        configs = [configurations[i].flatten() for i in range(configurations.shape[0])]
        assert configurations.shape[1] == configurations.shape[2], f"Must be square lattice (got {configurations.shape[1]}x{configurations.shape[2]})"
        N = configurations.shape[1]
    else:
        raise ValueError("configurations.ndim must be 2 or 3")

    assert 0 < N < 2**16, "N must be between 1 and 65,535 to fit into an unsigned short"

    byte_array = [] # Prepare array for storing all spins, each entry is a byte corresponding to 8 spin values
    current = "" # The current byte being formed
    for config in configs:
        for spin in config:
            current += str((spin+1)//2) # Convert -1/1 to 0/1 and append to current string
            if len(current) == 8:
                byte_array.append(int(current, 2))
                current = ""
    if len(current) != 0:
        current += "0"*(8 - len(current)) # Pad the last byte with zeros at the end
        byte_array.append(int(current, 2))

    # All bytes are stored with little endian
    N_ushort = struct.pack("<H", N) # Store edge length N as unsigned short (2 byte)
    N_configs = struct.pack("<I", len(configs)) # Store number of configurations as unsigned int (4 byte)
 
    with open(filename, "wb") as file:
        file.write("ising".encode("ascii")) # File header
        file.write(N_ushort)
        file.write(N_configs)
        file.write(struct.pack(f"<{len(byte_array)}B", *byte_array)) # Write all bytes as unsigned chars

def load_configurations_bin(filename : str, flatten : bool = False) -> np.ndarray:
    """
    Loads the configurations saved in binary form using save_configurations_bin function and
    returns the loaded data as numpy array (same as input value to the save function)

    Paramerers
    ----------
    filename : str
        The name of the file (including .ising file ending) to be loaded

    flatten : bool
        If this is True, the individual configurations are flattened from shape (N, N) to (N**2, )

    Returns
    -------
    configurations : np.ndarray
        The loaded configurations (2-dim array if only one configuration was saved,
        3-dim array if multiple configurations were saved, in which case configurations[i]
        is the ith configuration of shape (N, N))

        If flattened is True, the returned array will either be 1-dimensional (N**2, ) if
        only one configuration was stored or 2-dimensional (N_configs, N**2) if N_configs
        configurations were stored.
    """
    if not os.path.isfile(filename):
        raise ValueError(f"File {filename} does not exist")

    N = None
    N_configs = None
    read_bytes = None
    with open(filename, "rb") as file:
        try:
            header = file.read(5).decode("ascii") # File header
            assert header == "ising"
        except (AssertionError, UnicodeDecodeError):
            raise ValueError(f"File {filename} does not seem to be ising configuration collection (file header does not match)")
        N = struct.unpack("<H", file.read(2))[0] # Retrieve the edge length (N)
        N_configs = struct.unpack("<I", file.read(4))[0] # Retrieve the number of stored configurations
        data = file.read()
        read_bytes = struct.unpack(f"<{len(data)}B", data) # Retrieve the spins values
    
    if N_configs > 1:
        configurations = np.empty((N_configs, N, N), dtype=int) # Prepare the returned configurations
        for config in range(N_configs):
            for spin in range(N**2):
                bit = config*N**2 + spin # The total bit index of the spin
                byte = bit // 8 # The byte in which to find the spin
                # Here, we perform some bit manipulation in order to get the value of the bit
                # corresponding to the current spin and transforming it back from 0/1 to -1/1
                configurations[config, spin//N, spin%N] = ((read_bytes[byte] >> (7-bit%8)) & 1)*2-1
        if flatten:
            configurations = configurations.reshape((N_configs, N**2))
    else:
        configurations = np.empty((N, N), dtype=int) # Prepare the returned configurations
        for spin in range(N**2):
            byte = spin // 8 # The byte in which to find the spin
            # Here, we perform some bit manipulation in order to get the value of the bit
            # corresponding to the current spin and transforming it back from 0/1 to -1/1
            configurations[spin//N, spin%N] = ((read_bytes[byte] >> (7-spin%8)) & 1)*2-1
        if flatten:
            configurations = configurations.flatten()
    return configurations
    

def main():
    configuration = generate_configuration(100, True)
    propagate(configuration, n_timestep=1000, J=1, B=0, temperature=0.1, n_output=1, filename="out.json", mute_output=False, algorithm="wolff")
    traj = Trajectory.from_file("out.json")
    traj.animate()

if __name__ == "__main__":
    main()
