#!/usr/bin/env python
"""
optimize_command.py

A utility that uses Optuna to optimize a command-line script or program based on parameters 
defined in a configuration YAML file.

Example usage:
    optimize_command.py -c config.yaml -d

Options:
    -c, --config_file         Path to the YAML configuration file.
    -d, --delete_existing     Delete existing Optuna study if it exists.
    -v, --visualize_only      Generate visualizations only, without running optimization.
    -s, --server_file         Optional file listing servers/jobs (format: jobs/servername).

See template YAML for example configuration and command.
"""

import argparse
import os
import subprocess
import threading
import time
import socket
import yaml
import optuna
import optuna.visualization as vis

description = __doc__

# ----------------------------
# Argument Parser
# ----------------------------
def parse_args():
    """basic arg parsing"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-c', '--config_file',
        required=True,
        help='Path to the YAML configuration file.'
    )
    parser.add_argument(
        '-d', '--delete_existing',
        action='store_true',
        help='Delete existing Optuna study if it exists.'
    )
    parser.add_argument(
        '-v', '--visualize_only',
        action='store_true',
        help='Only generate visualizations; do not run optimization.'
    )
    parser.add_argument(
        '-s', '--server_file',
        help='Optional file listing servers and jobs (format: jobs/servername).'
    )
    return parser.parse_args()

# ----------------------------
# Job Monitor
# ----------------------------
class JobMonitor(threading.Thread):
    def __init__(self, process, start_time, timeout, error_phrases, log_file):
        super().__init__()
        self.process = process
        self.start_time = start_time
        self.timeout = timeout
        self.error_phrases = error_phrases
        self.log_file = log_file
        self.status = 'running'

    def run(self):
        while self.process.poll() is None:
            if time.time() - self.start_time > self.timeout:
                self.process.terminate()
                self.status = "timeout"
                return

            with open(self.log_file, 'r') as fh:
                log = fh.read()
                if any(phrase in log for phrase in self.error_phrases):
                    self.process.terminate()
                    self.status = "error"
                    return

            time.sleep(1)

        self.status = 'completed' if self.process.returncode == 0 else 'error'

# ----------------------------
# Utility Functions
# ----------------------------
def parse_config(path):
    """read config YAML and unpack"""
    with open(path) as fh:
        config = yaml.safe_load(fh)
    return (
        config['COMMAND'],
        config['TIMEOUT'],
        config['OPTUNA_SETTINGS'],
        config['OPTIMIZE_ARGUMENTS'],
        config['ERROR_KEYS']
    )

def parse_server_file(path):
    """load job count/server from file"""
    with open(path, 'r') as fh:
        return [(int(line.split('/')[0]), line.split('/')[1].strip()) for line in fh]

def run_command(command, timeout, error_phrases, log_file):
    """run the command and monitor it"""
    start = time.time()
    with open(log_file, 'w') as fh:
        process = subprocess.Popen(' '.join(command), shell=True, stdout=fh, stderr=fh)
        monitor = JobMonitor(process, start, timeout, error_phrases, log_file)
        monitor.start()
        monitor.join()
    print(f"Process status: {monitor.status}")
    return time.time() - start

def run_local_optuna_workers(n_jobs, study_name, storage):
    """run workers locally using python -c"""
    cmd = (
        f"python -c 'import optuna; "
        f"study = optuna.load_study(study_name=\"{study_name}\", storage=\"{storage}\"); "
        f"study.optimize(lambda trial: None, n_trials=None, n_jobs={n_jobs})'"
    )
    os.system(cmd)

def run_remote_optuna_workers(server_jobs, study_name, storage):
    """run remote jobs via SSH"""
    local_host = socket.gethostname()
    threads = []

    for jobs, server in server_jobs:
        if server == local_host:
            thread = threading.Thread(target=run_local_optuna_workers, args=(jobs, study_name, storage))
        else:
            remote_cmd = (
                f"ssh {server} \"python -c 'import optuna; "
                f"study = optuna.load_study(study_name=\\\"{study_name}\\\", storage=\\\"{storage}\\\"); "
                f"study.optimize(lambda trial: None, n_trials=None, n_jobs={jobs})'\""
            )
            thread = threading.Thread(target=os.system, args=(remote_cmd,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def save_visualizations(study, output_dir):
    """dump all optuna plots to HTML files"""
    os.makedirs(output_dir, exist_ok=True)
    vis.plot_optimization_history(study).write_html(f"{output_dir}/optimization_history.html")
    vis.plot_param_importances(study).write_html(f"{output_dir}/param_importances.html")
    vis.plot_parallel_coordinate(study).write_html(f"{output_dir}/parallel_coordinate.html")
    vis.plot_slice(study).write_html(f"{output_dir}/slice_plot.html")
    vis.plot_contour(study).write_html(f"{output_dir}/contour_plot.html")
    print(f"Visualizations saved to: {output_dir}")

# ----------------------------
# Optuna Functions
# ----------------------------
def objective(trial, base_cmd, arguments, timeout, error_phrases, log_file):
    """basic optuna trial"""
    trial_cmd = base_cmd.copy()

    for arg in arguments:
        if 'tuples' in arg:
            values = trial.suggest_categorical(arg['paired_arg_name'], arg['tuples'])
            for name, val in zip(arg['names'], values):
                trial_cmd.append(f"{name}{val}" if name.endswith("=") else f"{name} {val}")
        else:
            name = arg['name']
            r = arg['range_values']
            suggest = arg['suggest_type']

            if suggest == 'categorical':
                val = trial.suggest_categorical(name, r)
            elif suggest == 'int':
                val = trial.suggest_int(name, r[0], r[-1])
            elif suggest == 'float':
                val = trial.suggest_float(name, r[0], r[-1])
            else:
                raise ValueError(f"Unsupported suggest type: {suggest}")

            trial_cmd.append(f"{name}{val}" if name.endswith("=") else f"{name} {val}")

    try:
        return run_command(trial_cmd, timeout, error_phrases, log_file)
    except Exception as e:
        print(f"Error: {e}")
        return float('inf')

def optimize_command(config_file, delete_existing=False, visualize_only=False, server_file=None):
    """top-level optimization logic"""
    cmd, timeout, optuna_settings, opt_args, error_keys = parse_config(config_file)
    timeout = float('inf') if timeout == 0 else timeout
    log_file = "optimize_command.log"

    study_name = optuna_settings.get('study_name')
    storage = optuna_settings.get('storage')
    opt_args_clean = {
        k: v for k, v in optuna_settings.items()
        if k in ['study_name', 'storage', 'direction', 'sampler', 'pruner', 'load_if_exists']
    }

    if delete_existing and study_name and storage:
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f"Deleted study: {study_name}")

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Loaded study: {study_name}")
    except KeyError:
        if visualize_only:
            print("Cannot visualize. Study not found.")
            return
        study = optuna.create_study(**opt_args_clean)
        print(f"Created new study: {study_name}")

    if visualize_only:
        save_visualizations(study, output_dir="optuna_visualizations")
        return

    if server_file:
        server_jobs = parse_server_file(server_file)
        run_remote_optuna_workers(server_jobs, study_name, storage)
    else:
        study.optimize(
            lambda trial: objective(trial, cmd, opt_args, timeout, error_keys, log_file),
            n_trials=optuna_settings.get('max_trials', 50),
            n_jobs=1
        )
        print(f"Best parameters: {study.best_params}")
        print(f"Best runtime: {study.best_value}")

    save_visualizations(study, output_dir="optuna_visualizations")

def main():
    args = parse_args()
    optimize_command(
        config_file=args.config_file,
        delete_existing=args.delete_existing,
        visualize_only=args.visualize_only,
        server_file=args.server_file
    )

if __name__ == "__main__":
    main()