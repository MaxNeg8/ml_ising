import numpy as np
from ising import generate_configuration, propagate, save_configurations_bin
from multiprocessing import Process, Pipe
import os
import time

class Job:

    def __init__(self, temperatures : list[float], Ns : list[int], Js : list[float], Bs : list[float], n_samples : int, training : bool):
        self.Ns = [Ns] if not isinstance(Ns, list) and not isinstance(Ns, np.ndarray) else Ns
        self.Js = [Js] if not isinstance(Js, list) and not isinstance(Js, np.ndarray) else Js
        self.Bs = [Bs] if not isinstance(Bs, list) and not isinstance(Bs, np.ndarray) else Bs
        self.temperatures = [temperatures] if not isinstance(temperatures, list) and not isinstance(temperatures, np.ndarray) else temperatures
        self.n_samples = n_samples
        self.training = training
        self.n_simulations = len(self.Ns)*len(self.Js)*len(self.Bs)*len(self.temperatures)*self.n_samples

    def subdivide(self, n_jobs : int = os.cpu_count()) -> list:
        n_temps = len(self.temperatures)
        if n_temps >= n_jobs:
            temps_per_block = n_temps//n_jobs
            residuals = n_temps%n_jobs
            temps = [list(self.temperatures[i*temps_per_block:(i+1)*temps_per_block]) + ([self.temperatures[-residuals+i]] if i < residuals and residuals > 0 else []) for i in range(n_jobs)]
        else:
            temps = [[self.temperatures[i]] for i in range(n_temps)]
        return [Job(temps[i], self.Ns, self.Js, self.Bs, self.n_samples, self.training) for i in range(len(temps))]

    def __str__(self):
        return f"Data generation job, {len(self.temperatures)} temperatures, {len(self.Ns)} Ns, {len(self.Js)} Js, {len(self.Bs)} Bs, {self.n_samples} n_samples, Training: {self.training}"

    def __repr__(self):
        return f"Data generation job, temperatures={self.temperatures}, Ns={self.Ns}, Js={self.Js}, Bs={self.Bs}, n_samples={self.n_samples}, training={self.training}"

class Result:

    def __init__(self, length, N, J, B, training):
        self.length = length
        self.train_test = "train" if training else "test"
        self.filename_data = f"ising_ml_data/N_{N}_J_{J}_B_{B}_data_{self.train_test}"
        self.filename_labels = f"ising_ml_data/N_{N}_J_{J}_B_{B}_labels_{self.train_test}.csv"
        self.configurations = np.empty((length, N, N), dtype=int)
        self.temperatures = np.empty(length, dtype=float)
        self.offset = 0

    def store(self, configs, temps):
        self.configurations[self.offset:self.offset+len(configs)] = configs
        self.temperatures[self.offset:self.offset+len(configs)] = temps
        self.offset += len(configs)
        if self.offset == self.length:
            self.save()
            return True
        return False

    def save(self):
        np.savetxt(self.filename_labels, self.temperatures, delimiter=",")
        save_configurations_bin(self.filename_data, self.configurations, allow_overwrite=True)

def handle_job(job, conn):
    total = 0
    for N in job.Ns:
        for J in job.Js:
            for B in job.Bs:
                configurations = np.empty((len(job.temperatures)*job.n_samples, N, N), dtype=int)
                temperatures = np.empty(len(job.temperatures)*job.n_samples)
                for i,temperature in enumerate(job.temperatures):
                    for sample in range(job.n_samples):
                        configuration = generate_configuration(N=N, random=True)
                        propagate(configuration=configuration, n_timestep=300, n_output=0, J=J, B=B, temperature=temperature, filename=None, algorithm="wolff")
                        configurations[i*job.n_samples + sample] = configuration
                        temperatures[i*job.n_samples + sample] = temperature
                    total += 1
                    conn.send(total)
                conn.send([(N, J, B), configurations, temperatures])
    conn.send("done")
    time.sleep(60)

def main():
    temperatures = np.random.random(100)*4.0
    Js = [1]
    Bs = [0]
    Ns = [4]
    n_samples = 5
    training = True

    main_job = Job(temperatures, Ns, Js, Bs, n_samples, training)
    total_simulations = main_job.n_simulations
    jobs = main_job.subdivide()

    processes = []
    results = {}
    simulations_done = np.zeros(len(jobs), dtype=int)
        
    job_dict = {}

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
                raise EOFError("Process died without being properly terminated")
            if conn.poll():
                signal = conn.recv()
                if isinstance(signal, int):
                    simulations_done[job_dict[process][0]] = signal
                elif isinstance(signal, str) and signal == "done":
                    processes.remove((process, conn))
                    process.terminate()
                elif isinstance(signal, list):
                    params, configs, temps = signal
                    complete = None
                    try:
                        complete = results[params].store(configs, temps)
                    except KeyError:
                        job = job_dict[process][1]
                        result = Result(len(main_job.temperatures)*main_job.n_samples, *params, training=job.training)
                        complete = result.store(configs, temps)
                        results[params] = result
                    finally:
                        if complete:
                            del results[params]
                else:
                    raise ValueError(f"Got unexpected signal from connection: {signal}")
        time.sleep(0.001)
        print(f"Progress: {np.sum(simulations_done)*main_job.n_samples}/{total_simulations} ({np.sum(simulations_done)*main_job.n_samples/total_simulations*100:0.2f}%)", end="\r")
    print("\nDone.")

if __name__ == "__main__":
    main()
