import json
from pathlib import Path
import pandas as pd


def collect_log_data():
    metadata_logs = []
    fit_logs = []
    predict_logs = []
    evaluate_logs = []

    for data_set_folder in Path("./").iterdir():
        if data_set_folder.is_file():
            continue
        data_set_name = data_set_folder.name
        print(data_set_name)
        for algorithm_folder in data_set_folder.iterdir():
            if algorithm_folder.is_file():
                continue
            if "checkpoint" in algorithm_folder.name:
                algorithm_name = algorithm_folder.name.split("_")[-1]
                for config_folder in algorithm_folder.iterdir():
                    if config_folder.is_file():
                        continue
                    if "config" in config_folder.name:
                        for fold_folder in config_folder.iterdir():
                            if fold_folder.is_file():
                                continue
                            if "fold" in fold_folder.name:
                                fold_name = fold_folder.name.split("_")[-1]
                                if (fold_folder / "fit_log.json").exists():
                                    with open(fold_folder / "fit_log.json", "r") as file:
                                        fit_log = {
                                            "data_set_name": data_set_name,
                                            "algorithm_name": algorithm_name,
                                            "fold_name": fold_name
                                        }
                                        content = json.load(file)
                                        fit_log.update(content)
                                        fit_logs.append(fit_log)
                                if (fold_folder / "predict_log.json").exists():
                                    with open(fold_folder / "predict_log.json", "r") as file:
                                        predict_log = {
                                            "data_set_name": data_set_name,
                                            "algorithm_name": algorithm_name,
                                            "fold_name": fold_name
                                        }
                                        content = json.load(file)
                                        predict_log.update(content)
                                        predict_logs.append(predict_log)
                                if (fold_folder / "evaluate_log.json").exists():
                                    with open(fold_folder / "evaluate_log.json", "r") as file:
                                        evaluate_log = {
                                            "data_set_name": data_set_name,
                                            "algorithm_name": algorithm_name,
                                            "fold_name": fold_name
                                        }
                                        content = json.load(file)
                                        evaluate_log.update(content)
                                        evaluate_logs.append(evaluate_log)
            if "atomic" in algorithm_folder.name:
                if (algorithm_folder / "metadata.json").exists():
                    with open(algorithm_folder / "metadata.json", "r") as file:
                        metadata_log = {
                            "data_set_name": data_set_name
                        }
                        content = json.load(file)
                        metadata_log.update(content)
                        metadata_logs.append(metadata_log)

    metadata = pd.DataFrame(metadata_logs)
    fit = pd.DataFrame(fit_logs)
    predict = pd.DataFrame(predict_logs)
    evaluate = pd.DataFrame(evaluate_logs)

    metadata.to_csv("metadata.csv", index=False)
    fit.to_csv("fit.csv", index=False)
    predict.to_csv("predict.csv", index=False)
    evaluate.to_csv("evaluate.csv", index=False)


def merge_log_data():
    fit = pd.read_csv("fit.csv")
    predict = pd.read_csv("predict.csv")
    evaluate = pd.read_csv("evaluate.csv")

    merged = pd.merge(predict, evaluate, on=["data_set_name", "algorithm_name", "algorithm_config_index", "fold"],
                      suffixes=("", "_drop"))
    merged.drop([col for col in merged.columns if "_drop" in col], axis=1, inplace=True)

    merged = pd.merge(fit, merged, on=["data_set_name", "algorithm_name", "algorithm_config_index", "fold"],
                      suffixes=("", "_drop"))
    merged.drop([col for col in merged.columns if "_drop" in col], axis=1, inplace=True)

    merged.drop(columns=["fold_name", "fold", "model_file", "algorithm_configuration"], inplace=True)

    merged.to_csv("merged.csv", index=False)

    merged = merged.groupby(["data_set_name", "algorithm_name", "algorithm_config_index"]).mean().reset_index()
    merged.to_csv("merged_mean.csv", index=False)


def create_metaset():
    merged_mean = pd.read_csv("merged_mean.csv")

    metadata = pd.read_csv("metadata.csv")
    metadata = metadata.set_index("data_set_name")
    metadata.drop(columns=["feedback_type"], inplace=True)

    metrics = list(merged_mean.columns[-15:])

    for metric in metrics:
        metric_set = merged_mean[["data_set_name", "algorithm_name", "algorithm_config_index", metric]].copy()
        metric_set.loc[:, "algorithm"] = metric_set["algorithm_name"] + "_" + metric_set[
            "algorithm_config_index"].astype(str)
        metric_set.drop(columns=["algorithm_name", "algorithm_config_index"], inplace=True)

        metric_set = metric_set.pivot(index="data_set_name", columns="algorithm", values=metric)
        metric_set.dropna(inplace=True)

        metaset = pd.merge(metadata, metric_set, left_index=True, right_index=True)
        metaset.to_csv(f"metaset_{metric}.csv", index=False)


if __name__ == "__main__":
    collect_log_data()
    merge_log_data()
    create_metaset()
