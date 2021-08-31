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
    "max_depth": 1,
    "init_learning_rate": 10,
    "learning_rate_decay": 1,
    "initialization_method": Random_init(42),
    "use_2_dim_gate_based_on": "feature_importance_lr_max",
    "use_2_dim_clustering": False,
    "black_box_algorithm": None,
    "feature_names": None,
    "class_names": None,
    "save_likelihood": False,
    "verbose": True,    
    }

parameters_fit = {
    "optimization_method": "least_squares_linear_regression",
    "early_stopping": "accuracy",
    "use_posterior": False,
    }

modt = MoDT(**parameters)
modt.fit(**parameters_fit)
print("Training accuracy:", modt.score_internal_disjoint())

