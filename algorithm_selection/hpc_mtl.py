import subprocess
from pathlib import Path
from ..settings import cluster_email, out_folder

metrics = ["NDCG@10"]
splits = ["loo"]
eval_ids = [i for i in range(72)]
objectives = ["regression", "ranking"]
multi_labels = [0]
algo_ids = [i for i in range(46)]
models = ["linear_regression", "k_nearest_neighbors", "xgboost", "random_forest", "autogluon", "autogluon_best",
          "autogluon_best_opt"]

Path(out_folder).mkdir(exist_ok=True)
job_counter = 0
num_jobs = len(metrics) * len(splits) * len(eval_ids) * len(objectives) * len(multi_labels) * len(algo_ids) * len(
    models)
for metric in metrics:
    for split in splits:
        for eval_id in eval_ids:
            for objective in objectives:
                for multi_label in multi_labels:
                    for algo_id in algo_ids:
                        for model in models:
                            job_counter += 1
                            pred_path = Path(
                                f"./labels/"
                                f"{metric}_{split}_{eval_id}_{objective}_{multi_label}_{algo_id}_{model}_pred.csv")
                            true_path = Path(
                                f"./labels/"
                                f"{metric}_{split}_{eval_id}_{objective}_{multi_label}_{algo_id}_{model}_true.csv")
                            if (pred_path.is_file() and pred_path.stat().st_size > 0 and
                                    true_path.is_file() and true_path.stat().st_size > 0):
                                print(f"Skipped job {job_counter}/{num_jobs}.")
                                continue
                            script_name = f"__MTL_{metric}_{split}_{eval_id}_{objective}_{multi_label}_{algo_id}_{model}"
                            script = "#!/bin/bash\n" \
                                     "#SBATCH --nodes=1\n" \
                                     "#SBATCH --cpus-per-task=5\n" \
                                     "#SBATCH --mail-type=FAIL\n" \
                                     f"#SBATCH --mail-user={cluster_email}\n" \
                                     "#SBATCH --partition=short,medium\n" \
                                     "#SBATCH --time=30:00\n" \
                                     "#SBATCH --mem=18G\n" \
                                     f"#SBATCH --output=./{out_folder}/%x_%j.out\n" \
                                     "module load singularity\n" \
                                     "singularity exec --pwd /mnt --bind ./:/mnt ./mtl.sif python -u " \
                                     f"./process_evaluation.py --metric {metric} --split {split} --eval_id {eval_id} " \
                                     f"--objective {objective} --multi_label {multi_label} --algo_id {algo_id} " \
                                     f"--model {model}"

                            subprocess.run(["sbatch", "-J", script_name], input=script, universal_newlines=True)
                            print(f"Submitted job {job_counter}/{num_jobs}.")
