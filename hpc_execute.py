import argparse
import json
import subprocess
from pathlib import Path
from hpc_data_set_names import data_set_names
from hpc_algorithm_names import algorithm_names
from available_algorithms import recbole_algorithm_names, lenskit_algorithm_names, recpack_algorithm_names
from settings import cluster_email
from algorithm_config import retrieve_configurations


def execute(mode):
    num_folds = 5

    skip_outfile_exists = True
    skip_output_exists = True

    num_configurations = 0
    for algorithm_name in algorithm_names:
        num_configurations += len(retrieve_configurations(algorithm_name=algorithm_name))
    num_jobs = len(data_set_names) * num_configurations * num_folds
    job_counter = 0

    for data_set_name in data_set_names:
        for algorithm_name in algorithm_names:
            if algorithm_name in recbole_algorithm_names:
                resource = "gpu"
            elif algorithm_name in lenskit_algorithm_names:
                resource = "cpu"
            elif algorithm_name in recpack_algorithm_names:
                resource = "cpu"
            else:
                raise ValueError(f"Algorithm {algorithm_name} not found.")
            configurations = retrieve_configurations(algorithm_name=algorithm_name)
            for algorithm_config_index in range(len(configurations)):
                for fold in range(num_folds):
                    job_counter += 1
                    model_path = Path(
                        f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
                        f"config_{algorithm_config_index}/fold_{fold}/")
                    model_path.mkdir(parents=True, exist_ok=True)

                    if skip_outfile_exists:
                        exists = False
                        for file in model_path.iterdir():
                            if mode == "fit":
                                if ".out" in file.name and "RSDL_fit" in file.name:
                                    if file.stat().st_size > 0:
                                        print(f"HPC output file {file.name} exists. Skipping job {job_counter}.")
                                        exists = True
                            elif mode == "predict":
                                if ".out" in file.name and "RSDL_predict" in file.name:
                                    if file.stat().st_size > 0:
                                        print(f"HPC output file {file.name} exists. Skipping job {job_counter}.")
                                        exists = True
                            elif mode == "evaluate":
                                if ".out" in file.name and "RSDL_evaluate" in file.name:
                                    if file.stat().st_size > 0:
                                        print(f"HPC output file {file.name} exists. Skipping job {job_counter}.")
                                        exists = True
                            else:
                                raise ValueError(f"Mode {mode} not found.")
                        if exists:
                            continue

                    if skip_output_exists:
                        if mode == "fit":
                            fit_log = model_path.joinpath("fit_log.json")
                            if fit_log.is_file():
                                with open(fit_log, "r") as file:
                                    fit_log_dict = json.load(file)
                                model_file = Path(fit_log_dict["model_file"])
                                if model_file.is_file():
                                    print(f"Fit log {fit_log.name} and model file {model_file.name} exist. "
                                          f"Skipping job {job_counter}.")
                                    continue
                                # else:
                                #    print(f"Model file {model_file.name} does not exist.")
                            # else:
                            #    print(f"Fit log file {fit_log.name} does not exist.")
                        elif mode == "predict":
                            predict_log = model_path.joinpath("predict_log.json")
                            predictions_file = model_path.joinpath("predictions.json")
                            if (predict_log.is_file() and predict_log.stat().st_size > 0 and
                                    predictions_file.is_file() and predictions_file.stat().st_size > 0):
                                print(
                                    f"Predict log {predict_log.name} and predictions file {predictions_file.name} exist. "
                                    f"Skipping job {job_counter}.")
                                continue
                            # else:
                            #    print(f"Predict log file {predict_log.name} and/or "
                            #          f"predictions file {predictions_file.name} do not exist.")
                        elif mode == "evaluate":
                            evaluate_log = model_path.joinpath("evaluate_log.json")
                            if evaluate_log.is_file() and evaluate_log.stat().st_size > 0:
                                print(f"Evaluate log {evaluate_log.name} exists. Skipping job {job_counter}.")
                                continue
                            # else:
                            #    print(f"Evaluate log file {evaluate_log.name} does not exist.")
                        else:
                            raise ValueError(f"Mode {mode} not found.")

                    if resource == "gpu":
                        bash_partition = "#SBATCH --partition=gpu\n" \
                                         "#SBATCH --gres=gpu:1\n"
                    elif resource == "cpu":
                        bash_partition = "#SBATCH --partition=short,medium,long\n"
                    else:
                        raise ValueError(f"Resource {resource} not found.")

                    script = "#!/bin/bash\n" \
                             "#SBATCH --nodes=1\n" \
                             "#SBATCH --cpus-per-task=1\n" \
                             "#SBATCH --mail-type=FAIL\n" \
                             f"#SBATCH --mail-user={cluster_email}\n" \
                             f"{bash_partition}" \
                             "#SBATCH --time=00:32:00\n" \
                             "#SBATCH --mem=48G\n" \
                             f"#SBATCH --output={model_path}/%x_%j.out\n" \
                             "module load singularity\n" \
                             "singularity exec --nv --pwd /mnt --bind ./:/mnt ./data_loader.sif python -u " \
                             f"./execution_master.py --mode {mode} --data_set_name {data_set_name} " \
                             f"--algorithm_name {algorithm_name} --algorithm_config {algorithm_config_index} " \
                             f"--fold {fold}\n"
                    script_name = f"__RSDL_{mode}_{data_set_name}_{algorithm_name}_{fold}_{algorithm_config_index}.sh"
                    subprocess.run(["sbatch", "-J", script_name], input=script, universal_newlines=True)
                    print(f"Submitted job {job_counter}/{num_jobs}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HPC Execute")
    parser.add_argument('--mode', type=str, required=True)

    args = parser.parse_args()

    execute(args.mode)
