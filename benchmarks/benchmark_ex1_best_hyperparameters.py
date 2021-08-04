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

#     optuna_ex1_hyperparameters_per_dataset.py
#  -> analysis_ex1_hyperparameters.ipynb
#  -> benchmark_ex1_best_hyperparameters.py
#  -> analysis_ex1_hyperparameters_best.ipynb

parameters = {
    "X": "overwritten",
    "y": "overwritten",
    "n_experts": "overwritten",
    "iterations": 100,
    "max_depth": "overwritten",
    "init_learning_rate": "overwritten",
    "learning_rate_decay": "overwritten",
    "initialization_method": "overwritten",
    "feature_names": None,
    "class_names": None,
    "use_2_dim_gate_based_on": "overwritten",
    "use_2_dim_clustering": "overwritten", 
    "black_box_algorithm": None,
    }

parameters_fit = {
    "optimization_method": "overwritten",
    "early_stopping": False,
    }

df = pd.read_pickle("dataframes/ex1_df_top10_hyperparameters_per_dataset_FG_e3_d2.pd")
#df = pd.read_pickle("dataframes/ex1_df_top10_hyperparameters_per_dataset_2D_e3_d2.pd") #CHANGE
datasets = np.unique(df["Data X"])
repeats = 5 # For each found hyperparameter

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
    for _, row in df[df["Data X"] == dataset].iterrows():  # For each hyperparameter combination
        parameters["X"] = row["Data X"]
        parameters["y"] = row["Data y"]
        parameters["init_learning_rate"] = row["params_init_learning_rate"]
        parameters["learning_rate_decay"] = row["params_learning_rate_decay"]
        parameters_fit["optimization_method"] = row["params_optimization_method"]
        parameters['use_2_dim_gate_based_on'] = row['params_use_2_dim_gate_based_on']
        parameters['use_2_dim_clustering'] = row['params_use_2_dim_clustering']
        parameters['max_depth'] = row['params_max_depth']
        parameters['n_experts'] = row['params_n_experts']

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

        k_fold_results = k_fold(parameters, parameters_fit, n_repeats=repeats)
        accuracies_training.append(k_fold_results["accuracy_train"])
        accuracies_validation.append(k_fold_results["accuracy_val"])

    row = {
        "dataset" : row["Data X"],
        "acc_train" : np.mean(accuracies_training),
        "acc_val" : np.mean(accuracies_validation),
        "std_train" : np.std(accuracies_training),
        "std_val" : np.std(accuracies_validation),
    }
    results_row.append(row)

df_results = pd.DataFrame(results_row)
pickle.dump(df_results, open("dataframes/ex1_df_runs_with_hyperparameters_per_dataset_FG_e3_d2.pd", "wb")) #CHANGE
print("Duration", timer() - start)

    


