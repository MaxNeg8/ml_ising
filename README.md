# Machine Learning on the 2D Ising model

In this project, we create simulation code for the [2D Ising model](https://en.wikipedia.org/wiki/Ising_model) 
in Python and subsequently use machine learning (ML) to analyze the configurations produced with 
respect to the simulation temperature.

## The Ising Model simulation environment: ising.py

The file `ising.py` contains a multitude of functions that allow for the simulation of the 2D Ising model. For this,
we assume a hamiltonian of the form

$$\mathcal{H} = - J \sum_{i\, j}^{} s_i s_j - B \sum_{i = 1}^{N^2} s_i$$

where $J$ is the spin coupling constant, $B$ the outer magnetic field, $s_i$ a spin in the system, $N$ the length of
the square box and $\sum_{i\, j}^{}$ denotes the sum over all nearest neighbours.

### Generating a configuration

When we speak of a configuration, we refer to an N by N NumPy integer array where each entry represents a
spin (-1 for spin down, 1 for spin up).
To generate a new configuration of the Ising model that can subsequently be used for simulation, the function
`generate_configuration(N : int, random : bool = True)` is used. Here, `N` is the box length and `random`
initialized the spins in a random state instead of an all-spin-up state (all ones).

### Energy computation

Since, for the implementation of the Metropolis algorithm, we need to be able to compute the energy of the system,
both as a whole as well as the energy difference between spin flips, there are a total of three functions devoted to
this task:

* `compute_energy(configuration : np.ndarray, J : float, B : float)` takes the configuration, the spin coupling constant
and the magnetic field strength and computes the total energy of the system. This is very cost-expensive and scales with
$\mathcal{O}(N^2)$, so it should only be used at the beginning of the simulation.
* `compute_flip_energy(configuration : np.ndarray, position : tuple[int, int], J : float, B : float)` is used to compute
the energy difference between a given configuration and the same configuration but with the spin at `position` flipped. This
returns the energy difference $\Delta E$ in a way such that $E' = E + \Delta E$, where $E'$ is the new energy and $E$ the old
energy.
* `compute_row_col_flip_energy(configuration : np.ndarray, row_col : str, row_col_index : int, J : float, B : float)` works the
same way as `compute_flip_energy`, but instead of flipping a single spin, in this case, an entire row/column is flipped. For this,
`row_col` is a string of either `"row"` or `"col"` which determines whether the index `row_col_index` is meant to represent a row
or a column. The energy returned is to be interpreted the same way as for `compute_flip_energy`.

### Propagation of the system

The main function used for propagation is called
`propagate(configuration : np.ndarray, n_timestep : int, J : float, B : float, temperature : float, n_output : int=0,  filename : str=None, algorithm : str = "metropolis", copy : bool=False, mute_output : bool = True)`.
It takes a configuration and propagates it for `n_timestep` timesteps and at temperature `temperature` using the algorithm `algorithm` (either
`"metropolis"` or `"wolff"`). If `copy` is True, the function returns the propagated configuration as a copy instead of modifying the given
configuration array. `n_output` is used to determine the output frequency to the trajectory, which will be saved in the file `filename` in json
format (see below for details).

#### Metropolis algorithm

For the Ising model, the metropolis algorithm performs reliably albeit very inefficiently, requiring a large number of steps in order to sufficiently
sample the configuration space. Also, for below-critical temperatures, a phenomenon we refer to as _slabbing_ can occur, where the configuration
gets separated into at least two regimes of equal spins that have an interface in the form of a line, at which point the simulation gets stuck, since the
energy barrier it needs to cross in order to get rid of the interface is extremely high. To combat this and encourage the crossing of the barrier, we implement
a Monte Carlo move that is attempted with a certain probability and that tries to flip an entire random row or column at once. This proved to be very
effective.

Nevertheless, with the Metropolis algorithm, the system still does not sample the entire configuration space during one simulation. For example, if, at low temperatures,
the spins align to either +1 or -1, this alignment does not change anymore (again because of an energy barrier), effectively only sampling half of the configuration
space.

The solution to all of these sampling problems is to use a different algorithm called "Wolff", which is tailored to simulating the Ising model.

#### Wolff algorithm

As mentioned before, this algorithm does a good job at quickly sampling the entire configuration space at a given temperature. Correlations die away after 
100 timesteps and there are no issues related to high energy barriers. The algorithm achieves this by, instead of flipping single spins, building clusters
of spins with the same alignment and then flipping those clusters as a whole.

### Trajectories

Trajectories can be saved, loaded and analyzed using the Trajectory class. To understand how it works, see the following examples:

Loading a trajectory from a file:
```python
loaded_trajectory = Trajectory.from_file("path_to_file/")
```

Saving a trajectory (automatically done in the propagate function):
```python
trajectory = ... # Generate the trajectory during simulation
Trajectory(trajectory).save("filename")
```

Accessing the individual configurations in the trajectory:
```python
timestep = 0

loaded_trajectory = Trajectory.from_file("path_to_file/")
initial_configuration = loaded_trajectory.trajectory[timestep] # A NumPy array of shape (n_timestep, N, N)
```

Animating a trajectory:
```python
Trajectory.from_file("path_to_file/").animate()
```

Computing the avergage magnetization of an entire trajectory:
```python
Trajectory.from_file("path_to_file/").magnetization(r_equil = 0.3, normalize = True, abs = True, n_blocks = 10)
# Here, r_equil is the portion of the trajectory to discard as equilibration and not use for computing the observable
# normalize determines whether to return the magnetization per spin or the total magnetization
# abs determines whether to return the absolute value of the magnetization
# n_blocks is the number of blocks to use for computing the block averages
```

#### Trajectory file format

The trajectories are stored in a json format which looks something like this:

```json
{
    "initial": [...],
    "changes": [[...], [...]]
}
```

where `initial` stores a list containing the flattened array of the initial spin configuration
and `changes` holds a list of lists, each list `i` containing the indices of spins that flipped
between frames `i` and `i-1`. Note that these indices also refer to a flattened array of spins.

This approach to storing the trajectory is much more storage efficient compared to naively storing
each frame as a whole, since the latter implementation, in addition to the number of frames produced, 
scales quadratically with system size.

#### Finding trajectories

In order to find trajectories corresponding to desired simulation parameters, the function
`filter_trajectories(N : Union[int, List[int], tuple[int, int]], J : Union[float, List[float], tuple[float, float]], B : Union[float, List[float], tuple[float, float]], temperature : Union[float, List[float], tuple[float, float]], folder : Union[str, List[str]] = "trajectories")`
can be used. Here, the trajectory files inside the folder(s) `folder` (a single folder if `folder` is a string or all folders in the list if `folder`
is a list) are filtered based on the given simulation parameters. Each parameter can be given as either
* A single value. This way, the function only looks for trajectories where the simulation parameter has exactly the given value
* A tuple (x_low, x_high). Here, the function considers trajectories with simulation parameters $x \in [x_{low}, x_{high}]$
* A list. This is the same as providing a single value but the function considers all simulation paramater values in the given list.

The function returns a dictionary, the keys of which are the file names of the matched trajectory files and the values the corresponding
simulation parameters of each trajectory file.

### Simulation analysis

For analyzing the simulations, there are two functions:
* `magnetization(configuration : np.ndarray, normalize : bool=True)` Computes the magnetization (per spin, if `normalize` is True) of a given
configuration
* `correlate(signal_A : np.ndarray, signal_B : np.ndarray)` Computes the correlation function between two signals.

### Plotting configurations

In addition to animating an entire trajectory, the program also includes a function to plot a single configuration named
`plot_configuration(configuration : np.ndarray, cluster : list = None)`. This function also allows for highlighting spins
that belong to a Wolff cluster. For this, the highlighted spins' positions must be given as tuples inside of the `cluster`
list.

### Saving and loading configurations

In order to being able to perform machine learning on data generated by the ising model simulations, we need an easy way of storing multiple configurations
in a file. Those configurations don't always belong to the same trajectory, but may instead stem from different simulations, so we can't use the save/load
functionality of the Trajectory class described before. Also, the saved files should be as small as possible, as we will possibly be generating a very large
amount of data.

This is where the functions `save_configurations_bin(filename : str, configurations : np.ndarray, allow_overwrite : bool = False)`
and `load_configurations_bin(filename : str)` come into play. They allow for easy saving and loading of configurations in a binary
format, which consumes as little space as possible.

The `save_configurations_bin` function expects a filename (without extension) and a NumPy array containing one or multiple Ising configurations.
The NumPy array can either be 2-dimensional (representing an N by N grid of -1 and 1, corresponding to a single Ising configuration) or
3-dimensional (where configurations[i, :, :] corresponds to the ith Ising configuration to be stored). You can also choose whether the function
overwrites existing files with the same name by modifying the `allow_overwrite` parameter.

The `load_configurations_bin` function expects a file name (including the file extension ".ising") and returns the saved configuration exactly
the same way as they were stored.

The file format of a ".ising" binary file is the following:
* 2 bytes: Unsigned short representing the edge length N of each configuration
* 4 bytes: Unsigned integer representing the number of configurations stored in the file
* Rest of the file: Flattened spin configurations represented as bits. The last byte will be padded with zeros at the end if necessary.

Note that all values are stored as little endians.
