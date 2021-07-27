import pickle

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *

#data_input = pickle.load(open("../datasets/iris_input.pd", "rb"))
#data_target = pickle.load(open("../datasets/iris_target.pd", "rb"))

data_input = pickle.load(open("datasets/iris_input.pd", "rb"))
data_target = pickle.load(open("datasets/iris_target.pd", "rb"))

parameters = {
    "X": data_input,
    "y": data_target,
    "n_experts": 2,
    "iterations": 50,
    "max_depth": 2,
    "init_learning_rate": 100,
    "learning_rate_decay": 0.995,
    "initialization_method": BGM_init(),
    "feature_names": None,
    "class_names": None,
    "use_2_dim_gate_based_on": "feature_importance_pca_loadings",
    "use_2_dim_clustering": False,
    "black_box_algorithm": None,
    }

parameters_fit = {
    "optimization_method": "least_squares_linear_regression",
    "early_stopping": False,
    "use_posterior": False,
    }


modt = MoDT(**parameters)
#print(modt.learn_rate)
modt.fit(**parameters_fit)
#print(modt.gating_values)
print(modt.score_internal_disjoint())

modt.train_disjoint_trees(iteration=modt.best_iteration, tree_algorithm="h2o")

