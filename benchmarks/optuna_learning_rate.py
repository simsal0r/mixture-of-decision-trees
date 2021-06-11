import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *
from modt.visualization import *
from modt.utility import *

from datetime import datetime
import pickle

import optuna
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

storage_name = "sqlite:///benchmarks/optuna_results.sqlite3"
study_name = "{} learning_rate".format(datetime.now().strftime("%Y.%m.%d %H:%M:%S"))

data_input = pickle.load(open("../datasets/generated6_input.np", "rb"))
data_target = pickle.load(open("../datasets/generated6_target.np", "rb"))

#study = optuna.create_study(study_name=study_name,directions=["maximize","minimize"], storage=storage_name)
n_trials=100

X, y = shuffle(data_input, data_target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def modt_wrapper(X_train, y_train, learning_rate):
    parameters = {
    "X": X_train,
    "y": y_train,
    "n_experts": 3,
    "iterations": 20,
    "max_depth": 2,
    "init_learning_rate": learning_rate,
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
    return modt

def cross_validation(X, y, learning_rate):

    rkf = RepeatedKFold(n_splits=5, n_repeats=2)
    accuracies = []
    for train_idx, test_idx in rkf.split(X_train):
        modt = modt_wrapper(X[train_idx], y[train_idx], learning_rate)
        accuracies.append(modt.score(X[test_idx], y[test_idx]))
    
    return np.mean(accuracies)
