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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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

repeats = 1 # For each found hyperparameter
SETUP = "2D"  # "FG" or "2D"    

df = pd.read_pickle("dataframes/ex1_df_top10_hyperparameters_per_dataset_{}_e3_d2.pd".format(SETUP))
datasets = np.unique(df["Data X"])

def run_modt(parameters,parameters_fit,n_repeats):

    parameters = parameters
    parameters_fit = parameters_fit

    data_complete_input = pickle.load(open("../datasets/" + parameters["X"], "rb"))
    data_complete_target = pickle.load(open("../datasets/" + parameters["y"], "rb"))    

    shuffled_X, shuffled_y = shuffle(data_complete_input,data_complete_target, random_state=7)
    data_input_train, data_input_test, data_target_train, data_target_test = train_test_split(shuffled_X, shuffled_y, test_size=0.25, random_state=7)

    if isinstance(data_input_train, pd.core.frame.DataFrame):
        data_input_train.reset_index(inplace=True, drop=True)
        data_input_test.reset_index(inplace=True, drop=True)        
        data_target_train.reset_index(inplace=True, drop=True)
        data_target_test.reset_index(inplace=True, drop=True)

    parameters["X"] = data_input_train
    parameters["y"] = data_target_train

    accuracies_training = []
    accuracies_validation = []   

    for _ in range(n_repeats):
        
        modt = MoDT(**parameters)
        modt.fit(**parameters_fit)
        accuracies_training.append(modt.score_internal_disjoint())
        accuracies_validation.append(modt.score(data_input_test, data_target_test))

    dict_results = {}
    dict_results["accuracy_train"] = accuracies_training
    dict_results["accuracy_val"] = accuracies_validation

    return dict_results


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

        results = run_modt(parameters, parameters_fit, n_repeats=repeats)
        accuracies_training.append(results["accuracy_train"])
        accuracies_validation.append(results["accuracy_val"])

    row = {
        "dataset" : row["Data X"],
        "acc_train" : np.mean(accuracies_training),
        "acc_val" : np.mean(accuracies_validation),
        "std_train" : np.std(accuracies_training),
        "std_val" : np.std(accuracies_validation),
    }
    results_row.append(row)

df_results = pd.DataFrame(results_row)
pickle.dump(df_results, open("dataframes/ex1_df_runs_with_hyperparameters_per_dataset_{}_e3_d2.pd".format(SETUP), "wb")) 
print("Duration", timer() - start)

    


