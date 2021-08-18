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
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

#     optuna_ex1_hyperparameters_per_dataset.py
#  -> analysis_ex1_hyperparameters.ipynb
#  -> benchmark_ex1_best_hyperparameters.py
#  -> analysis_ex1_hyperparameters_best.ipynb

SETUP = "2D"  # "FG" or "2D"

storage_name = "sqlite:///optuna_databases/optuna_ex1_parameter_tuning_{}_e3_d2.sqlite3".format(SETUP)  

datasets = [
    ["abalone_input.pd","abalone_target.pd"], 
    #["adult_input.pd","adult_target.pd"], # Large
    ["banknote_input.pd","banknote_target.pd"], # Easy
    #["bank_input.pd","bank_target.pd"], # Large
    ["breast_cancer_input.np","breast_cancer_target.np"],
    ["cars_input.pd","cars_target.pd"], 
    ["contraceptive_input.pd","contraceptive_target.pd"], 
    ["generated6_input.np","generated6_target.np"],
    #["hrss_input.pd","hrss_target.pd"], # Large
    ["iris_input.pd","iris_target.pd"],
    ["steel_input.pd","steel_target.pd"],
    ["students_input.pd","students_target.pd"],
    #["sensorless_input.pd","sensorless_target.pd"], # Very Large dataset
]

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

distributions = {
    "init_learning_rate": optuna.distributions.UniformDistribution(1,300),
    "learning_rate_decay": optuna.distributions.UniformDistribution(0.75,1),
    "n_experts": optuna.distributions.IntUniformDistribution(3, 3),
    "max_depth": optuna.distributions.IntUniformDistribution(2, 2),
    "optimization_method": optuna.distributions.CategoricalDistribution(["least_squares_linear_regression","ridge_regression","lasso_regression"]),
                }

if SETUP == "2D":
    distributions["use_2_dim_gate_based_on"] = optuna.distributions.CategoricalDistribution(["feature_importance",
                                                                             "feature_importance_lda_max",
                                                                             "feature_importance_lr",  
                                                                             "feature_importance_lr_max",  
                                                                             "feature_importance_pca_loadings",
                                                                             "feature_importance_xgb",
                                                                             "PCA"])
    distributions["use_2_dim_clustering"] = optuna.distributions.CategoricalDistribution([True])                                                                         
elif SETUP == "FG":                                                                        
    distributions["use_2_dim_gate_based_on"] =  optuna.distributions.CategoricalDistribution([None])
    distributions["use_2_dim_clustering"] = optuna.distributions.CategoricalDistribution([False])
else:
    raise ValueError("Invalid setup.")

distributions_KDT_means = {
    "alpha": optuna.distributions.UniformDistribution(0.5,10),
    "beta": optuna.distributions.UniformDistribution(0.001,0.5),
    "gamma": optuna.distributions.UniformDistribution(0.001,0.5),
}

distributions_BGM = {
    "mean_precision_prior": optuna.distributions.UniformDistribution(0.1, 1),
    "weight_concentration_prior_type": optuna.distributions.CategoricalDistribution(["dirichlet_process", "dirichlet_distribution"]),
    "weight_concentration_prior": optuna.distributions.UniformDistribution(0.1, 1),
    "weight_cutoff": optuna.distributions.UniformDistribution(0.0,0.0),
}

initialization_methods = ["random", Kmeans_init(), KDTmeans_init(), BGM_init()]

start = timer()
n_trials = 1 # per initialization_method
n_startup_trials = 50 # 25 of X is random instead of the TPE algorithm.  

for dataset in datasets:

    print("Starting",dataset[0],"...")
    data_complete_input = pickle.load(open("../datasets/" + dataset[0], "rb"))
    data_complete_target = pickle.load(open("../datasets/" + dataset[1], "rb"))

    shuffled_X, shuffled_y = shuffle(data_complete_input,data_complete_target, random_state=7)
    data_input, _, data_target, _ = train_test_split(shuffled_X, shuffled_y, test_size=0.25, random_state=7)
  
    for initialization_method in initialization_methods:
        
        parameters["initialization_method"] = initialization_method

        study_name = "{} Hyper: {} {}".format(datetime.now().strftime("%Y.%m.%d %H:%M:%S"), dataset[0], initialization_method.__class__.__name__)
        study = optuna.create_study(study_name=study_name, directions=["maximize"], sampler=TPESampler(n_startup_trials=n_startup_trials), storage=storage_name)
        study.set_system_attr("initialization_method", initialization_method.__class__.__name__)
        study.set_system_attr("Data X",  dataset[0])
        study.set_system_attr("Data y", dataset[1])

        use_dataframe = False
        if isinstance(data_input, pd.core.frame.DataFrame):
            use_dataframe = True

        for idx_trial in range(n_trials):

            if isinstance(initialization_method, KDTmeans_init):
                trial = study.ask({**distributions, **distributions_KDT_means})
                alpha = trial.params["alpha"]
                beta = trial.params["beta"]
                gamma = trial.params["gamma"]
                parameters["initialization_method"] = KDTmeans_init(alpha=alpha, beta=beta, gamma=gamma)
            elif isinstance(initialization_method, BGM_init):
                trial = study.ask({**distributions, **distributions_BGM})
                mean_precision_prior = trial.params["mean_precision_prior"]
                weight_concentration_prior_type = trial.params["weight_concentration_prior_type"]
                weight_concentration_prior = trial.params["weight_concentration_prior"]
                weight_cutoff = trial.params["weight_cutoff"]
                parameters["initialization_method"] = BGM_init(mean_precision_prior=mean_precision_prior,
                                                                weight_concentration_prior_type=weight_concentration_prior_type,
                                                                weight_concentration_prior=weight_concentration_prior,
                                                                weight_cutoff=weight_cutoff)
            else:
                trial = study.ask(distributions)

            parameters["init_learning_rate"] = trial.params["init_learning_rate"]
            parameters["learning_rate_decay"] = trial.params["learning_rate_decay"]
            parameters["n_experts"] = trial.params["n_experts"]
            parameters["max_depth"] = trial.params["max_depth"]
            parameters["use_2_dim_gate_based_on"] = trial.params["use_2_dim_gate_based_on"]
            parameters["use_2_dim_clustering"] = trial.params["use_2_dim_clustering"]

            parameters_fit["optimization_method"] = trial.params["optimization_method"]      
                
            accuracies_training = []
            accuracies_validation = []    
            rkf = RepeatedKFold(n_splits=4, n_repeats=1)
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
            dict_results["accuracy_train"] = np.mean(accuracies_training)
            dict_results["accuracy_val"] = np.mean(accuracies_validation)
            dict_results["std_train"] = np.std(accuracies_training)
            dict_results["std_val"] = np.std(accuracies_validation)

            for key, value in dict_results.items():
                trial.set_user_attr(key, value)
            
            study.tell(trial, np.mean(accuracies_validation)) 

            print("Completed trial:", idx_trial, initialization_method.__class__.__name__)
            gc.collect() 

print("Duration", timer() - start)






