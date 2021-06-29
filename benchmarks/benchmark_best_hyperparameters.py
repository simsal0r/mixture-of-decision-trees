import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *
from modt.visualization import *
from modt.utility import *

import pickle
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import normalize
from sklearn.model_selection import RepeatedKFold

#     optuna_hyperparameters_per_dataset.py
#  -> analysis_hyperparameters.ipynb
#  -> benchmark_best_hyperparameters.py
#  -> analysis_hyperparameters_runs.ipynb

parameters = {
    "X": None,
    "y": None,
    "n_experts": 3,
    "iterations": 100,
    "max_depth": 2,
    "init_learning_rate": None,
    "learning_rate_decay": None,
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

df = pd.read_pickle("dataframes/df_top10_hyperparameters_per_dataset.pd")
datasets = np.unique(df["Data X"])
repeats = 2

def k_fold(parameters,parameters_fit,n_repeats):

    parameters = parameters
    parameters_fit = parameters_fit
    data_input = pickle.load(open("../datasets/" + parameters["X"], "rb"))
    data_target = pickle.load(open("../datasets/" + parameters["y"], "rb"))    

    use_dataframe = False
    if isinstance(data_input, pd.core.frame.DataFrame):
        use_dataframe = True

    accuracies_training = []
    accuracies_validation = []   
    rkf = RepeatedKFold(n_splits=4, n_repeats=n_repeats)
    for train_idx, val_idx in rkf.split(data_input):
        
        if use_dataframe:
            X_temp = data_input.iloc[train_idx].reset_index(inplace=False, drop=True)
            y_temp = data_target.iloc[train_idx].reset_index(inplace=False, drop=True)
        else:
            X_temp = data_input[train_idx]
            y_temp = data_target[train_idx]

        # Insert k-fold dataset params
        parameters["X"] = X_temp
        parameters["y"] = y_temp

        modt = MoDT(**parameters)
        modt.fit(**parameters_fit)
        accuracies_training.append(modt.score_internal_disjoint())

        if use_dataframe:
            X_temp = data_input.iloc[val_idx].reset_index(inplace=False, drop=True)
            y_temp = data_target.iloc[val_idx].reset_index(inplace=False, drop=True)
        else:
            X_temp = data_input[val_idx]
            y_temp = data_target[val_idx]
        accuracies_validation.append(modt.score(X_temp, y_temp))

    dict_results = {}
    dict_results["accuracy_train"] = accuracies_training
    dict_results["accuracy_val"] = accuracies_validation

    return dict_results
    # dict_results["std_train"] = np.std(accuracies_training)
    # dict_results["std_val"] = np.std(accuracies_validation)    

start = timer()
results_row = []
for dataset in datasets:
    print("Starting dataset", dataset)
    accuracies_training = []
    accuracies_validation = []  
    for _, row in df[df["Data X"] == dataset].iterrows():
        parameters["X"] = row["Data X"]
        parameters["y"] = row["Data y"]
        if row.initialization_method == "str":  # Random saved as str in df
            parameters["initialization_method"] = "random"
        elif row.initialization_method == "Kmeans_init":
            parameters["initialization_method"] = Kmeans_init()
        elif row.initialization_method == "KDTmeans_init":
            alpha = row["params_alpha"]
            beta = row["params_beta"]
            gamma = row["params_gamma"]
            parameters["initialization_method"] = KDTmeans_init(alpha=alpha, beta=beta, gamma=gamma)
        elif row.initialization_method == "BGM_init": 
            weight_cutoff = row["params_weight_cutoff"]
            weight_concentration_prior_type = row["params_weight_concentration_prior_type"]
            weight_concentration_prior = row["params_weight_concentration_prior"]
            mean_precision_prior = row["params_mean_precision_prior"]
            parameters["initialization_method"] = BGM_init(weight_concentration_prior=weight_concentration_prior, weight_cutoff=weight_cutoff, weight_concentration_prior_type=weight_concentration_prior_type,mean_precision_prior=mean_precision_prior)
        else:
            raise ValueError("Can't interpret initialization method.")
        parameters["init_learning_rate"] = row["params_init_learning_rate"]
        parameters["learning_rate_decay"] = row["params_learning_rate_decay"]
        parameters_fit["optimization_method"] = row["params_optimization_method"]
        parameters_fit["use_posterior"] = row["params_use_posterior"]

        k_fold_results = k_fold(parameters, parameters_fit, n_repeats=repeats)
        accuracies_training.append(k_fold_results["accuracy_train"])
        accuracies_validation.append(k_fold_results["accuracy_val"])

        # 2 GATE LDA

    row = {
        "dataset" : row["Data X"],
        "acc_train" : np.mean(accuracies_training),
        "acc_val" : np.mean(accuracies_validation),
        "std_train" : np.std(accuracies_training),
        "std_val" : np.std(accuracies_validation),
    }
    results_row.append(row)

df_results = pd.DataFrame(results_row)
pickle.dump(df_results, open("dataframes/df_runs_with_hyperparameters_per_dataset.pd", "wb"))
print("Duration", timer() - start)

    


