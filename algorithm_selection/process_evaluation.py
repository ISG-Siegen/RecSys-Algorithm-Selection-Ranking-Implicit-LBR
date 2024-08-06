import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from autogluon.tabular import TabularPredictor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def meta_learn(metric, split, eval_id, objective, multi_label, algo_id, model):
    print("Starting meta learning.")
    print(f"metric: {metric}")
    print(f"split: {split}")
    print(f"eval_id: {eval_id}")
    print(f"objective: {objective}")
    print(f"multi_label: {multi_label}")
    print(f"algo_id: {algo_id}")
    print(f"model: {model}")

    metaset = pd.read_csv(f"metaset_{metric}.csv")

    metadata = metaset.columns[:12]
    algorithms = metaset.columns[12:]

    x_train = y_train = x_test = y_test = None

    if split == "loo":
        for idx, dataset in enumerate(metaset.index):
            if idx == eval_id:
                x_train = metaset.drop(index=dataset)[metadata]
                y_train = metaset.drop(index=dataset)[algorithms]
                x_test = metaset.loc[[dataset]][metadata]
                y_test = metaset.loc[[dataset]][algorithms]
                break
    elif split == "kfold":
        for idx, (train_index, test_index) in enumerate(
                KFold(n_splits=5, shuffle=True, random_state=42).split(metaset)):
            if idx == eval_id:
                x_train = metaset.iloc[train_index][metadata]
                y_train = metaset.iloc[train_index][algorithms]
                x_test = metaset.iloc[test_index][metadata]
                y_test = metaset.iloc[test_index][algorithms]
                break
    else:
        raise ValueError("Invalid split.")

    if x_train is None or y_train is None or x_test is None or y_test is None:
        raise ValueError("Invalid eval_id.")

    if objective == "ranking":
        y_train = y_train.rank(axis=1, method="first", ascending=False)
        y_test = y_test.rank(axis=1, method="first", ascending=False)
    elif objective == "regression":
        pass
    else:
        raise ValueError("Invalid objective type.")

    if multi_label != 0:
        pass
    else:
        target_algorithm = None
        for idx, algorithm in enumerate(list(algorithms)):
            if idx == algo_id:
                target_algorithm = algorithm
                y_train = y_train[algorithm]
                y_test = y_test[algorithm]
                break

        if model == "autogluon" or model == "autogluon_best" or model == "autogluon_best_opt":
            predictor = TabularPredictor(label=target_algorithm, problem_type="regression",
                                         eval_metric="mean_squared_error", verbosity=2,
                                         path=f"./autogluon_logs/"
                                              f"{metric}_{split}_{eval_id}_{objective}_{algo_id}_{model}.log")
            if model == "autogluon_best_opt":
                predictor.fit(train_data=pd.concat([x_train, y_train], axis=1), presets="best_quality",
                              time_limit=60 * 20, num_cpus=5, num_bag_folds=8, num_bag_sets=20, num_stack_levels=0)
            elif model == "autogluon_best":
                predictor.fit(train_data=pd.concat([x_train, y_train], axis=1), presets="best_quality",
                              time_limit=60 * 20, num_cpus=5)
            else:
                predictor.fit(train_data=pd.concat([x_train, y_train], axis=1), presets="medium_quality",
                              time_limit=60 * 5, num_cpus=5)
            y_pred = predictor.predict(x_test).values
            predictor.delete_models(models_to_delete=predictor.model_names(), dry_run=False)
        elif model == "random_forest":
            predictor = RandomForestRegressor()
            param_grid = {
                'n_estimators': [100, 500, 1000],
                'max_features': ['sqrt', None],
                'max_depth': [None, 10, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'criterion': ['squared_error', 'absolute_error'],
            }
            grid_search = GridSearchCV(estimator=predictor, param_grid=param_grid,
                                       cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error',
                                       error_score='raise')
            grid_search.fit(x_train, y_train)

            results = grid_search.cv_results_
            results_df = pd.DataFrame(results)
            Path("./grid_search_logs/").mkdir(exist_ok=True)
            results_df.to_csv(
                f"./grid_search_logs/grid_search_results_"
                f"{metric}_{split}_{eval_id}_{objective}_{multi_label}_{algo_id}.csv", index=False)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(x_test)
        elif model == "k_nearest_neighbors":
            predictor = KNeighborsRegressor()
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 40, 50],
                'p': [1, 2]
            }
            grid_search = GridSearchCV(estimator=predictor, param_grid=param_grid,
                                       cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error',
                                       error_score='raise')
            grid_search.fit(x_train, y_train)

            results = grid_search.cv_results_
            results_df = pd.DataFrame(results)
            Path("./grid_search_logs/").mkdir(exist_ok=True)
            results_df.to_csv(
                f"./grid_search_logs/grid_search_results_"
                f"{metric}_{split}_{eval_id}_{objective}_{multi_label}_{algo_id}.csv", index=False)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(x_test)
        elif model == "xgboost":
            predictor = XGBRegressor()
            param_grid = {
                'eta': [0.01, 0.1, 0.3],
                'gamma': [0, 0.5, 1],
                'max_depth': [4, 5, 6, 7],
                'min_child_weight': [1, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
            }
            grid_search = GridSearchCV(estimator=predictor, param_grid=param_grid,
                                       cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error',
                                       error_score='raise')
            grid_search.fit(x_train, y_train)

            results = grid_search.cv_results_
            results_df = pd.DataFrame(results)
            Path("./grid_search_logs/").mkdir(exist_ok=True)
            results_df.to_csv(
                f"./grid_search_logs/grid_search_results_"
                f"{metric}_{split}_{eval_id}_{objective}_{multi_label}_{algo_id}.csv", index=False)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(x_test)
        elif model == "linear_regression":
            predictor = LinearRegression()
            predictor.fit(x_train, y_train)
            y_pred = predictor.predict(x_test)
        else:
            raise ValueError("Invalid model type.")

        y_test = y_test.values
        Path("./labels/").mkdir(exist_ok=True)
        np.savetxt(f"./labels/{metric}_{split}_{eval_id}_{objective}_{multi_label}_{algo_id}_{model}_pred.csv",
                   y_pred, delimiter=",")
        np.savetxt(f"./labels/{metric}_{split}_{eval_id}_{objective}_{multi_label}_{algo_id}_{model}_true.csv",
                   y_test, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MTL")
    parser.add_argument('--metric', dest='metric', type=str, required=True)
    parser.add_argument('--split', dest='split', type=str, required=True)
    parser.add_argument('--eval_id', dest='eval_id', type=int, required=True)
    parser.add_argument('--objective', dest='objective', type=str, required=True)
    parser.add_argument('--multi_label', dest='multi_label', type=int, required=True)
    parser.add_argument('--algo_id', dest='algo_id', type=int, required=True)
    parser.add_argument('--model', dest='model', type=str, required=True)
    args = parser.parse_args()

    meta_learn(metric=args.metric, split=args.split, eval_id=args.eval_id, objective=args.objective,
               multi_label=args.multi_label, algo_id=args.algo_id, model=args.model)
