import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math


def corr_plot():
    for metric in ["NDCG@10"]:
        corr_plot_ranking = []
        corr_plot_regression = []

        for meta_leaner in ["linear_regression", "k_nearest_neighbors", "xgboost", "random_forest", "autogluon",
                            "autogluon_best", "autogluon_best_opt"]:
            for objective in ["ranking", "regression"]:
                results = pd.read_csv(f"results_agg_{objective}_{metric}_{meta_leaner}.csv")

                plot_df = pd.DataFrame(
                    results[["spearman_corr", "spearman_p_value"]].copy())
                plot_df["learner"] = f"{meta_leaner}_{objective}"

                if objective == "ranking":
                    corr_plot_ranking.append(plot_df)
                elif objective == "regression":
                    corr_plot_regression.append(plot_df)

        corr_plot_ranking = pd.concat(corr_plot_ranking)
        corr_plot_regression = pd.concat(corr_plot_regression)

        corr_plot_ranking.rename(
            columns={"spearman_corr": "Spearman Correlation", "kendall_corr": "Kendall Correlation",
                     "learner": "Meta Model"},
            inplace=True)
        corr_plot_ranking.replace({"autogluon_ranking": "AutoGluon Medium",
                                   "random_forest_ranking": "Random Forest",
                                   "autogluon_best_ranking": "AutoGluon Best (No Bagging)",
                                   "linear_regression_ranking": "Linear Regression",
                                   "autogluon_best_opt_ranking": "AutoGluon Best (Bagging)",
                                   "k_nearest_neighbors_ranking": "K Nearest Neighbors",
                                   "xgboost_ranking": "XGBoost"}, inplace=True)

        corr_plot_regression.rename(
            columns={"spearman_corr": "Spearman Correlation", "kendall_corr": "Kendall Correlation",
                     "learner": "Meta Model"},
            inplace=True)
        corr_plot_regression.replace({"autogluon_regression": "AutoGluon Medium",
                                      "random_forest_regression": "Random Forest",
                                      "autogluon_best_regression": "AutoGluon Best (No Bagging)",
                                      "linear_regression_regression": "Linear Regression",
                                      "autogluon_best_opt_regression": "AutoGluon Best (Bagging)",
                                      "k_nearest_neighbors_regression": "K Nearest Neighbors",
                                      "xgboost_regression": "XGBoost"}, inplace=True)

        print(corr_plot_ranking.groupby("Meta Model").median())
        print(corr_plot_regression.groupby("Meta Model").median())
        print((corr_plot_ranking.groupby("Meta Model").median() - corr_plot_regression.groupby(
            "Meta Model").median()).mean().values[0])

        print(corr_plot_ranking.groupby("Meta Model").mean())
        print(corr_plot_regression.groupby("Meta Model").mean())
        print((corr_plot_ranking.groupby("Meta Model").mean() - corr_plot_regression.groupby(
            "Meta Model").mean()).mean().values[0])

        fig, axes = plt.subplots(2, 1, figsize=(6, 6))

        sns.boxplot(ax=axes[0], x="Meta Model", hue="Meta Model", y="Spearman Correlation", data=corr_plot_ranking,
                    legend=False)
        axes[0].set_title("Correlation of Meta Model Predictions for Ranking Task")
        axes[0].set_xticks([])
        axes[0].set_xlabel("")
        axes[0].set_ylim(-0.1, 1.0)
        axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        sns.boxplot(ax=axes[1], x="Meta Model", hue="Meta Model", y="Spearman Correlation", data=corr_plot_regression,
                    legend=True)
        axes[1].set_title("Correlation of Meta Model Predictions for Regression Task")
        axes[1].set_xticks([])
        axes[1].set_xlabel("")
        axes[1].set_ylim(-0.1, 1.0)
        axes[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=2, frameon=False)

        plt.tight_layout()
        plt.savefig(f"corr_plot_{metric}.pdf")
        plt.show()


def recall_plot():
    for metric in ["NDCG@10"]:
        recall_plot_ranking = []

        for meta_leaner in ["linear_regression", "k_nearest_neighbors", "xgboost", "random_forest", "autogluon",
                            "autogluon_best", "autogluon_best_opt"]:
            for objective in ["ranking", "regression"]:
                results = pd.read_csv(f"results_agg_{objective}_{metric}_{meta_leaner}.csv")

                plot_df = pd.DataFrame(results[["top1_recall", "top3_recall"]].copy())
                plot_df["learner"] = f"{meta_leaner}_{objective}"

                if objective == "ranking":
                    recall_plot_ranking.append(plot_df)

        recall_plot_ranking = pd.concat(recall_plot_ranking)

        recall_plot_ranking.rename(
            columns={"top1_recall": "Top 1 Recall", "top3_recall": "Top 3 Recall", "learner": "Meta Model"},
            inplace=True)
        recall_plot_ranking.replace({"autogluon_ranking": "AutoGluon\nMedium",
                                     "random_forest_ranking": "Random\nForest",
                                     "autogluon_best_ranking": "AutoGluon Best\n(No Bagging)",
                                     "linear_regression_ranking": "Linear\nRegression",
                                     "autogluon_best_opt_ranking": "AutoGluon Best\n(Bagging)",
                                     "k_nearest_neighbors_ranking": "K Nearest\nNeighbors",
                                     "xgboost_ranking": "XGBoost"}, inplace=True)
        recall_plot_ranking["Top 3 Recall"] = recall_plot_ranking["Top 3 Recall"].apply(
            lambda x: math.floor(x * 100) / 100)

        print(recall_plot_ranking.groupby("Meta Model").mean())

        fig, axes = plt.subplots(2, 1, figsize=(7, 7))

        colors = sns.color_palette("tab10", n_colors=4)
        top1_palette = [colors[0]] + [colors[3]]
        top3_palette = [colors[0]] + [colors[1]] + [colors[2]] + [colors[3]]

        hist00 = sns.histplot(ax=axes[0], y="Meta Model", hue="Top 1 Recall", multiple="dodge", shrink=0.7,
                              data=recall_plot_ranking, palette=top1_palette, legend=True)
        axes[0].set_title("Recall@1 of Ranking Meta-Model Predictions Per Dataset")
        axes[0].set_xlim(0, 55)
        axes[0].set_ylabel("")
        sns.move_legend(hist00, loc="lower right", ncol=1, frameon=False, title="")
        for count in [10, 20, 30, 40]:
            axes[0].axvline(x=count, color='gray', linestyle='--', linewidth=1)

        hist01 = sns.histplot(ax=axes[1], y="Meta Model", hue="Top 3 Recall", multiple="dodge", shrink=0.8,
                              data=recall_plot_ranking, palette=top3_palette, legend=True)
        axes[1].set_title("Recall@3 of Ranking Meta-Model Predictions Per Dataset")
        axes[1].set_xlim(0, 55)
        axes[1].set_ylabel("")
        sns.move_legend(hist01, loc="lower right", ncol=1, frameon=False, title="")
        for count in [10, 20, 30, 40]:
            axes[1].axvline(x=count, color='gray', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.savefig(f"recall_plot_{metric}.pdf")
        plt.show()


if __name__ == "__main__":
    corr_plot()
    recall_plot()
