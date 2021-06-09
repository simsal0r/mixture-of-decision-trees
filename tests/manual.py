import pickle

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *
from modt.visualization import *


data_input = pickle.load(open("datasets/generated6_input.np", "rb"))
data_target = pickle.load(open("datasets/generated6_target.np", "rb"))

parameters = {
    "X": data_input,
    "y": data_target,
    "n_experts": 10,
    "iterations": 10,
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
#print(modt.learn_rate)
modt.fit(**parameters_fit)

#modt.train_disjoint_trees()
#visualize_decision_area(modt.predict_disjoint, modt.X_original, modt.y)
