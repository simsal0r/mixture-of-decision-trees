import pickle

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *
from modt.visualization import *


data_input = pickle.load(open("datasets/generated6_input.np", "rb"))
data_target = pickle.load(open("datasets/generated6_target.np", "rb"))

# data_input = pickle.load(open("datasets/adult_input.pd", "rb"))
# data_target = pickle.load(open("datasets/adult_target.pd", "rb"))

parameters = {
    "X": data_input,
    "y": data_target,
    "n_experts": 3,
    "iterations": 10,
    "max_depth": 2,
    "init_learning_rate": 10,
    "learning_rate_decay": 1,
    "initialize_with": "random",
    "initialization_method": None,
    "feature_names": None,
    "class_names": None,
    "use_2_dim_gate_based_on": None,
    "use_2_dim_clustering": False,
    "black_box_algorithm": None,
    }

parameters_fit = {
    "optimization_method": "least_squares_linear_regression",
    "add_noise": False,
    "use_posterior": False,
    }


modt = MoDT(**parameters)

modt.fit(**parameters_fit)

print(modt.score(data_input, data_target))

#visualize_decision_area(modt.predict_disjoint, modt.X_original, modt.y)
