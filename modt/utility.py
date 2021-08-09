import pickle

import numpy as np
import pandas as pd
from sklearn import tree

def tree_accuracy(X, y, depth):
    """Output the score of a scikit-learn DT with given max_depth for comparison"""
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X, y)
    return(clf.score(X,y))

def pickle_disjoint_data(modt, iteration, filepathprefix=""):
    """Save/export each subset of the data, separated by the gating function"""
    gating_values = modt.all_gating_values[iteration]
    gate = np.argmax(gating_values, axis=1)
    gating_values_hard = np.zeros([modt.n_input, modt.n_experts])
    gating_values_hard[np.arange(0, modt.n_input), gate] = 1

    for index_expert in range(modt.n_experts):
        mask = gating_values_hard[:, index_expert] == 1
        df = modt.X_original_pd[mask].copy()
        df["target"] = modt.y_original[mask]

        pickle.dump(df, open(filepathprefix + "output/disjoint_data_e_{}.pd".format(index_expert), "wb"))

def optuna_optimization(dataset_input, dataset_target, n_experts, max_depth, gating_2D, runs, k_folds=4, delete_studies=True, only_random=True):

    import sys, os
    sys.path.insert(1, os.path.join(sys.path[0], ".."))
    from modt.modt import MoDT
    from modt._initialization import KDTmeans_init
    from modt._initialization import Kmeans_init
    from modt._initialization import BGM_init

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


    storage_name = "sqlite:///../temp/optuna_optimization.sqlite3" 

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
        "early_stopping": True,
        }

    if gating_2D:
        gate_selection = optuna.distributions.CategoricalDistribution(["feature_importance",  
                                                                       "feature_importance_lda", 
                                                                       "feature_importance_lda_max",
                                                                       "feature_importance_lr",  
                                                                       "feature_importance_lr_max",  
                                                                       "feature_importance_pca_loadings",
                                                                       "feature_importance_xgb",
                                                                       "PCA"])
    else:
        gate_selection = optuna.distributions.CategoricalDistribution([None])                                                                            


    distributions = {
        "init_learning_rate": optuna.distributions.UniformDistribution(1,150),
        "learning_rate_decay": optuna.distributions.UniformDistribution(0.975,1),
        "n_experts": optuna.distributions.IntUniformDistribution(n_experts, n_experts),
        "max_depth": optuna.distributions.IntUniformDistribution(max_depth, max_depth),
        "use_2_dim_gate_based_on": gate_selection,
        "use_2_dim_clustering": optuna.distributions.CategoricalDistribution([gating_2D]), # Set to True for 2D  
        "optimization_method": optuna.distributions.CategoricalDistribution(["least_squares_linear_regression","ridge_regression","lasso_regression"]),

                    }

    distributions_KDT_means = {
        "alpha": optuna.distributions.UniformDistribution(0.7,4),
        "beta": optuna.distributions.UniformDistribution(0.0001,0.5),
        "gamma": optuna.distributions.UniformDistribution(0.0001,0.5),
    }

    distributions_BGM = {
        "mean_precision_prior": optuna.distributions.UniformDistribution(0.1, 1),
        "weight_concentration_prior_type": optuna.distributions.CategoricalDistribution(["dirichlet_process", "dirichlet_distribution"]),
        "weight_concentration_prior": optuna.distributions.UniformDistribution(0.1, 1),
        "weight_cutoff": optuna.distributions.UniformDistribution(0.0, 0.1), # If max not 0.0, BGM can output fewer outputs
    }

    if only_random:
        initialization_methods = ["random"]
    else:
        initialization_methods = ["random", Kmeans_init(), KDTmeans_init(), BGM_init()]

    start = timer()
    n_trials = runs # per initialization_method
    n_startup_trials = int(runs/10) + 5 # random starting points instead of the TPE algorithm.  

    data_input = dataset_input
    data_target = dataset_target

    for initialization_method in initialization_methods:
        
        parameters["initialization_method"] = initialization_method

        study_name = "{} Hyper: {} {}".format(datetime.now().strftime("%Y.%m.%d %H:%M:%S"), "Adhoc", initialization_method.__class__.__name__)
        study = optuna.create_study(study_name=study_name, directions=["maximize"], sampler=TPESampler(n_startup_trials=n_startup_trials), storage=storage_name)
        study.set_system_attr("initialization_method", initialization_method.__class__.__name__)

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
            rkf = RepeatedKFold(n_splits=k_folds, n_repeats=1)
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

            #print("Completed trial:", idx_trial, initialization_method.__class__.__name__)
            gc.collect() 

    print("Duration:", timer() - start, "Returning trials...")

    studies = optuna.study.get_all_study_summaries(storage=storage_name)
    loaded_study = optuna.load_study(study_name=studies[0].study_name, storage=storage_name)
    df = loaded_study.trials_dataframe()
    for key, value in loaded_study.system_attrs.items():
        df[key] = value
    for study in studies[1:]:
        loaded_study = optuna.load_study(study_name=study.study_name , storage=storage_name)
        df_new = loaded_study.trials_dataframe()
        for key, value in loaded_study.system_attrs.items():
            df_new[key] = value
        df = pd.concat([df, df_new])
        if delete_studies:
            optuna.delete_study(study_name=study.study_name , storage=storage_name)
    df.reset_index(inplace=True, drop=True)
    return df    







