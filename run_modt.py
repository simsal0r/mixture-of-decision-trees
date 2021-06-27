import pickle
from modt.modt import MoDT
from modt._initialization import *

data_input = pickle.load(open("datasets/iris_input.pd", "rb"))
data_target = pickle.load(open("datasets/iris_target.pd", "rb"))

parameters = {
    "X": data_input,
    "y": data_target,
    "n_experts": 3,
    "iterations": 50,
    "max_depth": 2,
    "init_learning_rate": 10,
    "learning_rate_decay": 1,
    "initialization_method": Kmeans_init(),
    "feature_names": None,
    "class_names": None,
    "use_2_dim_gate_based_on": "feature_importance",
    "use_2_dim_clustering": False,
    "black_box_algorithm": None,
    }

parameters_fit = {
    "optimization_method": "least_squares_linear_regression",
    "early_stopping": "likelihood",
    "use_posterior": False,
    }


modt = MoDT(**parameters)
print(modt.learn_rate)
modt.fit(**parameters_fit)
print(modt.gating_values)
modt.score_internal_disjoint()
