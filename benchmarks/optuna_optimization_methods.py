import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *
from modt.visualization import *
from modt.utility import *

from datetime import datetime
import pickle
import gc
from timeit import default_timer as timer

import optuna
from optuna.samplers import RandomSampler
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

datasets = [
    #["generated6_input.np", "generated6_target.np"],
    #["adult_input.pd","adult_target.pd"],
    #["sensorless_input.pd","sensorless_target.pd"],
    ["steel_input.pd","steel_target.pd"],
]

parameters = {
    "X": None,
    "y": None,
    "n_experts": None,
    "iterations": 100,
    "max_depth": None,
    "init_learning_rate": None,
    "learning_rate_decay": None,
    "initialize_with": "random",
    "initialization_method": None,
    "feature_names": None,
    "class_names": None,
    "use_2_dim_gate_based_on": None,
    "use_2_dim_clustering": False,
    "black_box_algorithm": None,
    }

parameters_fit = {
    "optimization_method": None,
    "early_stopping": False,
    "use_posterior": None,
    }

distributions = {
    "init_learning_rate": optuna.distributions.UniformDistribution(10,150),
    "learning_rate_decay": optuna.distributions.UniformDistribution(0.975,1),
    "n_experts": optuna.distributions.IntUniformDistribution(2,5),
    "max_depth": optuna.distributions.IntUniformDistribution(1,3),
    "use_2_dim_gate_based_on": optuna.distributions.CategoricalDistribution(["feature_importance", "feature_importance_lda", "PCA", None]),

    "use_posterior": optuna.distributions.CategoricalDistribution([True, False]),
                }

optimization_methods = ["least_squares_linear_regression","ridge_regression","lasso_regression","matmul"]

storage_name = "sqlite:///optuna_results_optimization_methods.sqlite3"

start = timer()
n_trials = 500
rows = []
for dataset in datasets:
    print("Starting",dataset[0],"...")
    data_input = pickle.load(open("../datasets/" + dataset[0], "rb"))
    data_target = pickle.load(open("../datasets/" + dataset[1], "rb"))

    study_name = "{} optimization_methods {}".format(datetime.now().strftime("%Y.%m.%d %H:%M:%S"), dataset[0])
    study = optuna.create_study(study_name=study_name, directions=["maximize"], sampler=RandomSampler(), storage=storage_name)
    study.set_system_attr("Data X",  dataset[0])
    study.set_system_attr("Data y", dataset[1])

    use_dataframe = False
    if isinstance(data_input, pd.core.frame.DataFrame):
        use_dataframe = True

    for idx_trial in range(n_trials):

        #Set random parameters
        trial = study.ask(distributions)  
        parameters["init_learning_rate"] = trial.params["init_learning_rate"]
        parameters["learning_rate_decay"] = trial.params["learning_rate_decay"]
        parameters["n_experts"] = trial.params["n_experts"]
        parameters["max_depth"] = trial.params["max_depth"]
        parameters["use_2_dim_gate_based_on"] = trial.params["use_2_dim_gate_based_on"]

        parameters_fit["use_posterior"] = trial.params["use_posterior"]        
            
        accuracies_methods_of_folds = {}
        for method in optimization_methods:
            accuracies_methods_of_folds["train_" + str(method)] = []
            accuracies_methods_of_folds["val_" + str(method)] = []

        rkf = RepeatedKFold(n_splits=4, n_repeats=1)
        for train_idx, val_idx in rkf.split(data_input):
            
            if use_dataframe:
                X_temp = data_input.iloc[train_idx]
                y_temp = data_target.iloc[train_idx]
                X_temp.reset_index(inplace=True, drop=True)
                y_temp.reset_index(inplace=True, drop=True)
            else:
                X_temp = data_input[train_idx]
                y_temp = data_target[train_idx]

            # Insert k-fold dataset params
            parameters["X"] = X_temp
            parameters["y"] = y_temp

            modt = MoDT(**parameters)

            for method in optimization_methods:
                # Insert method params
                parameters_fit["optimization_method"] = method

                modt.fit(**parameters_fit)
                accuracies_methods_of_folds["train_" + str(method)].append(modt.score_internal_disjoint())

                if use_dataframe:
                    X_temp = data_input.iloc[val_idx]
                    y_temp = data_target.iloc[val_idx]
                    X_temp.reset_index(inplace=True, drop=True)
                    y_temp.reset_index(inplace=True, drop=True)
                else:
                    X_temp = data_input[val_idx]
                    y_temp = data_target[val_idx]
                accuracies_methods_of_folds["val_" + str(method)].append(modt.score(X_temp, y_temp))

        dict_results = {}
        for method in optimization_methods:
            dict_results["train_" + str(method)] = np.mean(accuracies_methods_of_folds["train_" + str(method)])
            dict_results["val_" + str(method)] = np.mean(accuracies_methods_of_folds["val_" + str(method)])

        meta_accuracies = []
        for key, value in dict_results.items():
            trial.set_user_attr(key, value)
            meta_accuracies.append(value)
        
        study.tell(trial, np.mean(meta_accuracies)) 

        print("Completed trial:", idx_trial)
        gc.collect() 

print("Duration", timer() - start)
    



