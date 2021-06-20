import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *
from modt.visualization import *
from modt.utility import *

from datetime import datetime
import pickle
import gc

import optuna
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

storage_name = "sqlite:///benchmarks/optuna_results_initialization.sqlite3"
study_name = "{} init".format(datetime.now().strftime("%Y.%m.%d %H:%M:%S"))

parameters = {
    "X": None,
    "y": None,
    "n_experts": 3,
    "iterations": 100,
    "max_depth": 2,
    "init_learning_rate": 100,
    "learning_rate_decay": 0.995,
    "initialize_with": "pass_method",
    "initialization_method": "BGM_init(n_components=n_experts, weight_cutoff=0.0) FI 2dim",
    "feature_names": None,
    "class_names": None,
    "use_2_dim_gate_based_on": None,
    "use_2_dim_clustering": False,
    "black_box_algorithm": None,
    }

parameters_fit = {
    "optimization_method": "least_squares_linear_regression",
    "early_stopping": False,
    "use_posterior": False,
    }

# data_input_name = "datasets/generated6_input.np"
# data_target_name = "datasets/generated6_target.np"
# data_input_name = "datasets/steel_input.pd"
# data_target_name = "datasets/steel_target.pd"
data_input_name = "datasets/banknote_input.pd"
data_target_name = "datasets/banknote_target.pd"
parameters["max_depth"] = 1
parameters["n_experts"] = 2

parameters["use_2_dim_gate_based_on"] = "feature_importance"
parameters["use_2_dim_clustering"] = True

data_input = pickle.load(open(data_input_name, "rb"))
data_target = pickle.load(open(data_target_name, "rb"))

study = optuna.create_study(study_name=study_name, directions=["maximize"], sampler=TPESampler(n_startup_trials=25), storage=storage_name)
study.set_system_attr("Data X", data_input_name)
study.set_system_attr("Data y", data_target_name)
n_trials = 100

X, y = shuffle(data_input, data_target, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

use_dataframe = False
if isinstance(X_train, pd.core.frame.DataFrame):
    use_dataframe = True
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

def cross_validation(**kwargs):  
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    train_score_i_0 = []  # First iteration
    train_score_i_9 = []  # Tenth iteration
    completed_iterations = []

    rkf = RepeatedKFold(n_splits=5, n_repeats=1)
    for train_idx, val_idx in rkf.split(X_train):
        if use_dataframe:
            X_temp = X_train.iloc[train_idx]
            y_temp = y_train.iloc[train_idx]
            X_temp.reset_index(inplace=True, drop=True)
            y_temp.reset_index(inplace=True, drop=True)
            modt = modt_wrapper(X_temp, y_temp, **kwargs)
        else:
            modt = modt_wrapper(X_train[train_idx], y_train[train_idx], **kwargs)

        train_accuracies.append(modt.score_internal_disjoint())

        if use_dataframe:
            X_temp = X_train.iloc[val_idx]       
            y_temp = y_train.iloc[val_idx]
            X_temp.reset_index(inplace=True, drop=True)
            y_temp.reset_index(inplace=True, drop=True)
            val_accuracies.append(modt.score(X_temp, y_temp))
        else:
            val_accuracies.append(modt.score(X_train[val_idx], y_train[val_idx]))
            
        test_accuracies.append(modt.score(X_test, y_test))
        train_score_i_0.append(modt.all_accuracies[0])
        train_score_i_9.append(modt.all_accuracies[9])
        completed_iterations.append(modt.completed_iterations)
    
    return (np.mean(train_accuracies),
            np.mean(val_accuracies),
            np.mean(test_accuracies),
            np.mean(train_score_i_0),
            np.mean(train_score_i_9),
            np.mean(completed_iterations))

def modt_wrapper(X, y):
    parameters["initialization_method"] = BGM_init(n_components=parameters["n_experts"], weight_cutoff=0.0)
    parameters["X"] = X
    parameters["y"] = y
    modt = MoDT(**parameters)
    modt.fit(**parameters_fit)
    return modt

distributions = {

}

for idx_trial in range(n_trials):
    if idx_trial == 0:
        for key, value in parameters.items():
            study.set_system_attr(key, str(value))
        for key, value in parameters_fit.items():
            study.set_system_attr(key, str(value))

    trial = study.ask(distributions)  

    (train_score, 
     validation_score,
     test_score,
     train_score_i_0,
     train_score_i_9,
     completed_iterations) = cross_validation()

    trial.set_user_attr("train_score", train_score)
    trial.set_user_attr("test_score", test_score)
    trial.set_user_attr("train_score_i_0", train_score_i_0)
    trial.set_user_attr("train_score_i_9", train_score_i_9)
    trial.set_user_attr("completed_iterations", completed_iterations)

    study.tell(trial, validation_score) 

    print("Completed trial:", idx_trial)
    gc.collect() 

print("Best parameters of study:", study.best_params) 




