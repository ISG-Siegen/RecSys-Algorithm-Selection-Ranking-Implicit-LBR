import itertools


def retrieve_configurations(algorithm_name):
    configuration_space = {}
    if algorithm_name == "Pop":
        configuration_space["epochs"] = [1]
    elif algorithm_name == "ItemKNN":
        configuration_space["epochs"] = [1]
        configuration_space["k"] = [50, 100]
    elif algorithm_name == "BPR":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "NeuMF":
        configuration_space["mf_embedding_size"] = [64]
        configuration_space["mlp_embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "DMF":
        configuration_space["user_embedding_size"] = [64]
        configuration_space["item_embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "LightGCN":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "MultiVAE":
        configuration_space["latent_dimension"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "MultiDAE":
        configuration_space["latent_dimension"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "MacridVAE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "LINE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "CDAE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "ENMF":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "RecVAE":
        configuration_space["latent_dimension"] = [200]
        configuration_space["learning_rate"] = [0.01, 0.001]
    elif algorithm_name == "EASE":
        configuration_space["epochs"] = [1]
        configuration_space["reg_weight"] = [100, 250]
    elif algorithm_name == "NCEPLRec":
        configuration_space["epochs"] = [1]
        configuration_space["rank"] = [200, 450]
    elif algorithm_name == "SimpleX":
        configuration_space["embedding_size"] = [64]
        configuration_space["gamma"] = [0.3, 0.5]
    elif algorithm_name == "Random":
        configuration_space["epochs"] = [1]
    elif algorithm_name == "PopScore":
        configuration_space["score_method"] = ["quantile", "count"]
    elif algorithm_name == "ItemItem":
        configuration_space["nnbrs"] = [50, 100]
        configuration_space["min_nbrs"] = [1]
        configuration_space["min_sim"] = [1e-6]
    elif algorithm_name == "UserUser":
        configuration_space["nnbrs"] = [50, 100]
        configuration_space["min_nbrs"] = [1]
        configuration_space["min_sim"] = [1e-6]
    elif algorithm_name == "ImplicitMF":
        configuration_space["features"] = [25, 50]
        configuration_space["iterations"] = [20]
        configuration_space["reg"] = [0.01]
    elif algorithm_name == "SVD":
        configuration_space["num_components"] = [50, 100]
    elif algorithm_name == "NMF":
        configuration_space["num_components"] = [50, 100]
        configuration_space["alpha"] = [0.0]
    elif algorithm_name == "ItemKNNRP":
        configuration_space["K"] = [50, 100]
        configuration_space["similarity"] = ["cosine"]

    experiments = [dict(zip(configuration_space.keys(), v)) for v in itertools.product(*configuration_space.values())]
    return experiments
