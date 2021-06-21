import pickle
from modt.modt import MoDT
from modt._initialization import *

data_input = pickle.load(open("datasets/adult_input.pd", "rb"))
data_target = pickle.load(open("datasets/adult_target.pd", "rb"))

parameters = {
    "X": data_input,
    "y": data_target,
    "n_experts": 10,
    "iterations": 1,
    "max_depth": 2,
    "init_learning_rate": 10,
    "learning_rate_decay": 1,
    "initialize_with": "pass_method",
    "initialization_method": Kmeans_init(),
    "feature_names": None,
    "class_names": None,
    "use_2_dim_gate_based_on": "feature_importance",
    "use_2_dim_clustering": False,
    "black_box_algorithm": None,
    }

parameters_fit = {
    "optimization_method": "least_squares_linear_regression",
    "add_noise": False,
    "use_posterior": False,
    }


modt = MoDT(**parameters)
print(modt.learn_rate)
modt.fit(**parameters_fit)
modt.score_internal_disjoint()
