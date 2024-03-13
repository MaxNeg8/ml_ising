# Machine Learning on the 2D Ising model

In this project, we create simulation code for the [2D Ising model](https://en.wikipedia.org/wiki/Ising_model) 
in Python and subsequently use machine learning (ML) to analyze the configurations produced with 
respect to the simulation temperature.

# Trajectory file format

The trajectories are stored in a json format which looks somethink like this:

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
