import pandas as pd
from scipy.stats import spearmanr, kendalltau

data = pd.read_csv("results_agg.csv")

for meta_leaner in ["autogluon", "random_forest", "autogluon_best", "linear_regression", "autogluon_best_opt",
                    "k_nearest_neighbors", "xgboost"]:
    for metric in ["NDCG@10"]:
        for objective in ["ranking", "regression"]:
            objective_meta_leaner = data.loc[
                (data["metric"] == metric) & (data["objective"] == objective) & (data["model"] == meta_leaner)].copy()
            objective_meta_leaner.drop(columns=["metric", "objective", "split", "multi_label", "model"], inplace=True)

            objective_performance = []

            unique_eval_ids = objective_meta_leaner["eval_id"].unique()
            for eval_id in unique_eval_ids:
                eval_data = objective_meta_leaner.loc[objective_meta_leaner["eval_id"] == eval_id]
                spearman_corr, spearman_p_value = spearmanr(eval_data["pred"], eval_data["true"])
                kendall_corr, kendall_p_value = kendalltau(eval_data["pred"], eval_data["true"])

                rank_ascending = None
                if objective == "regression":
                    rank_ascending = False
                elif objective == "ranking":
                    rank_ascending = True

                top1_true_indices = eval_data['true'].rank(method='first', ascending=rank_ascending).nsmallest(1).index
                top3_true_indices = eval_data['true'].rank(method='first', ascending=rank_ascending).nsmallest(3).index
                top5_true_indices = eval_data['true'].rank(method='first', ascending=rank_ascending).nsmallest(5).index
                top1_predicted_indices = eval_data['pred'].rank(method='first', ascending=rank_ascending).nsmallest(
                    1).index
                top3_predicted_indices = eval_data['pred'].rank(method='first', ascending=rank_ascending).nsmallest(
                    3).index
                top5_predicted_indices = eval_data['pred'].rank(method='first', ascending=rank_ascending).nsmallest(
                    5).index

                top1_recall = len(set(top1_true_indices).intersection(top1_predicted_indices)) / 1
                top3_recall = len(set(top3_true_indices).intersection(top3_predicted_indices)) / 3
                top5_recall = len(set(top5_true_indices).intersection(top5_predicted_indices)) / 5

                top1_hit_in_top3 = len(set(top1_true_indices).intersection(top3_predicted_indices))
                top1_hit_in_top5 = len(set(top1_true_indices).intersection(top5_predicted_indices))
                top3_hit_in_top5 = min(1, len(set(top3_true_indices).intersection(top5_predicted_indices)))

                objective_performance.append(
                    {"eval_id": eval_id, "spearman_corr": spearman_corr, "spearman_p_value": spearman_p_value,
                     "kendall_corr": kendall_corr, "kendall_p_value": kendall_p_value, "top1_recall": top1_recall,
                     "top3_recall": top3_recall, "top5_recall": top5_recall, "top1_hit_in_top3": top1_hit_in_top3,
                     "top1_hit_in_top5": top1_hit_in_top5, "top3_hit_in_top5": top3_hit_in_top5})

            objective_performance = pd.DataFrame(objective_performance)
            objective_performance.to_csv(f"results_agg_{objective}_{metric}_{meta_leaner}.csv", index=False)

pass
