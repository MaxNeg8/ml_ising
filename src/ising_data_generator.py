import numpy as np
from ising import generate_configuration, propagate, save_configurations_bin
from multiprocessing import Process, Pipe
import os
import time

class Job:
    """
    The Job class is a container for simulation jobs to be done. It allows the user to store simulation parameters
    and evenly divide simulation temperatures into sub-jobs, depending on the number of processes that are going
    to run the sub-jobs.

    For example, if the user wants to run the data generator with 4 processes, it would be sensible to divide the
    simulation temperatures into 4 sub-jobs.

    Object variables
    ----------------
    temperatures : list[float]
        The simulation temperatures to be simulated

    Ns : list[int]
        Number of edge lattice sites to be simulated

    Js : list[float]
        The spin-spin coupling strengths to be simulated

    Bs : list[float]
        The outer magnetic field strengths to be simulated

    n_samples : int
        The number of samples to generate per (N, J, B, temperature)

    training : bool
        Whether or not the simulations are for training data

    n_simulations : int
        The total number of simulations that the job object will result in
    """

    def __init__(self, temperatures : list[float], Ns : list[int], Js : list[float], Bs : list[float], n_samples : int, training : bool):
        """
        Init function of the Job class

        Parameters
        ----------
        All parameters are the same as in the class docstring. The simulation parameters (temperatures, Ns, Js, Bs) can be provided
        as lists or as single values.
        """
        # Convert possible single values to lists for uniformity of the datatypes
        self.Ns = [Ns] if not isinstance(Ns, list) and not isinstance(Ns, np.ndarray) else Ns
        self.Js = [Js] if not isinstance(Js, list) and not isinstance(Js, np.ndarray) else Js
        self.Bs = [Bs] if not isinstance(Bs, list) and not isinstance(Bs, np.ndarray) else Bs
        self.temperatures = [temperatures] if not isinstance(temperatures, list) and not isinstance(temperatures, np.ndarray) else temperatures
        # Store additional information
        self.n_samples = n_samples
        self.training = training
        # Compute total number of simulations for this job
        self.n_simulations = len(self.Ns)*len(self.Js)*len(self.Bs)*len(self.temperatures)*self.n_samples

    def subdivide(self, n_jobs : int = os.cpu_count()) -> list:
        """
        The subdivide function divides a Jobs object into n_jobs sub-jobs of approximately equal
        size (+/- 1) and returns the resulting jobs as a list.

        Parameters
        ----------
        n_jobs : int
            The number of jobs to divide into (default: number of CPU cores)

        Returns
        -------
        jobs : list[Job]
            A list of jobs, each representing 1/n_jobs of the original job's size. If the number of simulation
            temperatures is smaller than n_jobs, only len(temperatures) jobs will be created.
        """
        n_temps = len(self.temperatures)
        if n_temps >= n_jobs: # If we have less temperatures than jobs, only create n_temps jobs
            temps_per_block = n_temps//n_jobs # The number of temperatures that can equally be divided into n_jobs blocks
            residuals = n_temps%n_jobs # The remaining number of temperatures with which the block will be filled
            temps = [list(self.temperatures[i*temps_per_block:(i+1)*temps_per_block]) + ([self.temperatures[-residuals+i]] if i < residuals and residuals > 0 else []) for i in range(n_jobs)]
        else:
            temps = [[self.temperatures[i]] for i in range(n_temps)]
        return [Job(temps[i], self.Ns, self.Js, self.Bs, self.n_samples, self.training) for i in range(len(temps))] # Create and return list of sub-jobs

    def __str__(self):
        return f"Data generation job, {len(self.temperatures)} temperatures, {len(self.Ns)} Ns, {len(self.Js)} Js, {len(self.Bs)} Bs, {self.n_samples} n_samples, Training: {self.training}"

    def __repr__(self):
        return f"Data generation job, temperatures={self.temperatures}, Ns={self.Ns}, Js={self.Js}, Bs={self.Bs}, n_samples={self.n_samples}, training={self.training}"

class Result:
    """
    The Result class handles gathering simulation results and saving them to a file as soon as all results for a given
    set of (N, J, B) are ready. This is done in order to free memory as much as possible without having to resort to
    saving data to temporary files.

    Object variables
    ----------------
    length : int
        The total number of configurations to be stored in this set of results (len(temperatures)*n_samples for each set of (N, J, B))

    train_test : str
        A string that is either "train" or "test", depending on whether the results are for a training or a testing data set

    filename_data : str
        The filename for the simulation data

    filename_labels : str
        The filename for the labels (temperatures)

    configurations : np.ndarray
        The configurations that are stored in this set of results

    temperatures : np.ndarray
        The simulation temperatures corresponding to the configurations
    """

    def __init__(self, length, N, J, B, training):
        """
        Init function for the Result class.

        Parameters
        ----------
        N : int
            The number of edge lattice sites for this result set

        J : float
            The spin-spin coupling strength for this result set

        B : float
            The outer magnetic field strength for this result set

        training : bool
            Whether or not this result set is to be regarded as training data
        """
        self.length = length
        self.train_test = "train" if training else "test"
        self.filename_data = f"ising_ml_data/N_{N}_J_{J}_B_{B}_data_{self.train_test}"
        self.filename_labels = f"ising_ml_data/N_{N}_J_{J}_B_{B}_labels_{self.train_test}.csv"
        # Prepare arrays for storing simulation data
        self.configurations = np.empty((length, N, N), dtype=int)
        self.temperatures = np.empty(length, dtype=float)
        # Index of the next block of simulation data
        self._offset = 0

    def store(self, configs, temps):
        """
        The store function is used to append data to the result set whenever it is ready. The object
        will automatically keep track of where to store the data. Also, when the result set is full,
        it will automatically save the data to the corresponding files.

        Parameters
        ----------
        configs : np.ndarray
            The configurations to be appended to the result data set (shape must be (N_configs, N, N))
        
        temps : np.ndarray
            The simulation temperatures at which the configurations were generated (shape must be (N_configs, ))

        Returns
        -------
        True, if the result data set is full and was saved to disk successfully. False otherwise.
        """
        self.configurations[self._offset:self._offset+len(configs)] = configs
        self.temperatures[self._offset:self._offset+len(configs)] = temps
        self._offset += len(configs) # Update index for next block of simulation data
        if self._offset == self.length: # If we have reached the end of the array, the result set is full and we save it to disk
            self.save()
            return True
        return False

    def save(self):
        """
        The save function is automatically called by the store function when the result data set is full.
        It saves self.configurations to a binary .ising file and self.temperatures to a .csv file.
        """
        np.savetxt(self.filename_labels, self.temperatures, delimiter=",")
        save_configurations_bin(self.filename_data, self.configurations, allow_overwrite=True)

def handle_job(job, conn):
    """
    The handle job function is the function that is run inside of a process. It is able to do all simulations
    included in a Job object and communicate with the parent process through a Pipe. This works the following way:
    
    - When a simulation for a set of (N, J, B) is done, the simulation result will be sent through the pipe to the
    parent process in the form of a list with entries: [
            (N, J, B), # The parameters corresponding to the simulation result
            configurations, # The generated configurations
            temperatures # The simulation temperatures used
        ]
    - After each individual temperature simulation, the number of simulations that this process has done is sent through the
    pipe as an integer
    - When the process is finished, the string "done" is sent through the pipe and the process sleeps for 60 seconds before
    terminating

    It is important that the process does not terminate on itself, since simulation data of the last run might still be stored
    inside of the Pipe. If this data is not received by the parent process before the child process terminates, it might be
    corrupted (see multiprocessing documentation). So, we send a termination signal through the pipe, letting the parent
    process know that the child process can be terminated after the data in the pipe was read.

    Parameters
    ----------
    job : Job
        The job the process should run

    conn : Connection
        The connection to the parent process. This must be writeable.
    """
    total = 0 # Total number of simulations that this process has done
    for N in job.Ns:
        for J in job.Js:
            for B in job.Bs:
                # Prepare arrays for this block of simulation data
                configurations = np.empty((len(job.temperatures)*job.n_samples, N, N), dtype=int)
                temperatures = np.empty(len(job.temperatures)*job.n_samples)
                for i,temperature in enumerate(job.temperatures):
                    for sample in range(job.n_samples):
                        configuration = generate_configuration(N=N, random=True)
                        propagate(configuration=configuration, n_timestep=300, n_output=0, J=J, B=B, temperature=temperature, filename=None, algorithm="wolff")
                        configurations[i*job.n_samples + sample] = configuration
                        temperatures[i*job.n_samples + sample] = temperature
                        total += 1
                    conn.send(total) # Send the number of simulations performed to the parent process
                conn.send([(N, J, B), configurations, temperatures]) # Send the simulation results to the parent process
    conn.send("done") # Send terminate signal to the parent process
    time.sleep(60) # Wait for the parent process to terminate this process

def main():
    """
    The main function handles simulation parameter input, job creation, subdivision and process management.
    It spawns the appropriate number of processes, communicates with them to receive simulation results,
    store results in Result objects and terminated processes when they are done.
    """
    # Simulation parameters
    temperatures = np.random.random(100000)*4.0
    Js = [1]
    Bs = [0]
    Ns = [10]
    n_samples = 5
    training = True

    main_job = Job(temperatures, Ns, Js, Bs, n_samples, training) # The job containing all simulations to be done
    total_simulations = main_job.n_simulations # The total number of simulations to be done over all processes
    jobs = main_job.subdivide()

    processes = [] # For storing processes along with connections
    results = {} # Temporary storage for Result objects
    simulations_done = np.zeros(len(jobs), dtype=int) # Simulations done by each process
        
    job_dict = {} # For linking processes to jobs

    for i, job in enumerate(jobs):
        receive_conn, send_conn = Pipe(duplex=False)
        process = Process(target=handle_job, args=(job, send_conn))
        processes.append((process, receive_conn))
        process.start()
        job_dict[process] = (i, job)

    while len(processes) > 0:
        for state in processes:
            process, conn = state
            if not process.is_alive():
                # If we reach this, then the process ended without being terminated by us, so the data in its Pipe might
                # not have been read into the corresponding Result object. To prevent data loss/corruption, we exit here.
                raise EOFError("Process died without being properly terminated")
            if conn.poll():
                signal = conn.recv()
                if isinstance(signal, int): # Status update of number of simulations performed
                    simulations_done[job_dict[process][0]] = signal
                elif isinstance(signal, str) and signal == "done": # Termination signal
                    processes.remove((process, conn))
                    process.terminate()
                elif isinstance(signal, list): # Simulation results
                    params, configs, temps = signal
                    complete = None # Will hold the return value of the Result.store function
                    try:
                        complete = results[params].store(configs, temps) # Temporarily store the simulation result in the corresponding Result object
                    except KeyError: # In case the dictionary does not yet contain a Result object for the simulation parameters (N, J, T)
                        job = job_dict[process][1]
                        result = Result(len(main_job.temperatures)*main_job.n_samples, *params, training=job.training) # Create a new Result object
                        complete = result.store(configs, temps)
                        results[params] = result # Store Result object in dictionary
                    finally:
                        if complete:
                            del results[params] # If the result set is full and has been saved to disk, we delete it from the dictionary to free up memory
                else:
                    # If we reach this, then the signal that came through the pipe is neither a string, and integer nor a list, so we don't know what to do with it
                    raise ValueError(f"Got unexpected signal from connection: {signal}")
        time.sleep(0.0001) # Polling delay
        print(f"Progress: {np.sum(simulations_done)}/{total_simulations} ({np.sum(simulations_done)/total_simulations*100:0.2f}%)", end="\r")
    print("\nDone.")

if __name__ == "__main__":
    main()
