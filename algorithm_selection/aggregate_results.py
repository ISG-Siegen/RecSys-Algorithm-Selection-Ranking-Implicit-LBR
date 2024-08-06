from pathlib import Path
import pandas as pd

metrics = ["NDCG@10"]
splits = ["loo"]
eval_ids = [i for i in range(72)]
objectives = ["regression", "ranking"]
multi_labels = [0]
algo_ids = [i for i in range(46)]
models = ["linear_regression", "k_nearest_neighbors", "xgboost", "random_forest", "autogluon", "autogluon_best",
          "autogluon_best_opt"]

results = []
for metric in metrics:
    for split in splits:
        for eval_id in eval_ids:
            for objective in objectives:
                for multi_label in multi_labels:
                    for algo_id in algo_ids:
                        for model in models:
                            pred_path = Path(f"./labels/{metric}_{split}_{eval_id}_"
                                             f"{objective}_{multi_label}_{algo_id}_{model}_pred.csv")
                            true_path = Path(f"./labels/{metric}_{split}_{eval_id}_"
                                             f"{objective}_{multi_label}_{algo_id}_{model}_true.csv")
                            if not pred_path.is_file() or not true_path.is_file():
                                continue
                            with open(pred_path, "r") as f:
                                lines = f.readlines()
                                pred = lines[0]
                            with open(true_path, "r") as f:
                                lines = f.readlines()
                                true = lines[0]
                            results.append((metric, split, eval_id, objective, multi_label, algo_id, model, pred, true))

results_df = pd.DataFrame(results,
                          columns=["metric", "split", "eval_id", "objective", "multi_label", "algo_id", "model", "pred",
                                   "true"])
results_df.to_csv("results_agg.csv", index=False)
