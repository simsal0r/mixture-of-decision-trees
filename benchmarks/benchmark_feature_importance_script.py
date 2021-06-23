# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize
from sklearn.model_selection import RepeatedKFold


if True:
    df_intersections = pickle.load(open("df_intersections.pd", "rb"))
else:
    runs = 100
    rows = []
    for dataset in datasets:
        data_input = pickle.load(open("../datasets/" + dataset[0], "rb"))
        data_target = pickle.load(open("../datasets/" + dataset[1], "rb"))
        parameters["X"] = data_input
        parameters["y"] = data_target
        modt = MoDT(**parameters)
        intersections = []
        for _ in range(runs):
            intersections.append(fi_intersect(modt.X,modt.y))
        intersection = np.sum(intersections) / runs
        dict1 = {
            "dataset" : dataset[0],
            "n_features" : modt.X.shape[1],
            "intersection" : intersection,
        }
        rows.append(dict1)
    df = pd.DataFrame(rows)
    df_intersections = df


# %%
df_intersections["n_features"] = df_intersections["n_features"] -1

# %%
parameters = {
    "X": None,
    "y": None,
    "n_experts": 2,
    "iterations": 75,
    "max_depth": 2,
    "init_learning_rate": 100,
    "learning_rate_decay": 0.995,
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

datasets = [
    ["banknote_input.pd","banknote_target.pd"],
    ["adult_input.pd","adult_target.pd"],
    ["bank_input.pd","bank_target.pd"],
    ["breast_cancer_input.np","breast_cancer_target.np"],
    ["hrss_input.pd","hrss_target.pd"],
    ["iris_input.pd","iris_target.pd"],
    ["occupancy_input.pd","occupancy_target.pd"],
    ["pdm6_input.pd","pdm6_target.pd"],
    ["sensorless_input.pd","sensorless_target.pd"],
    ["steel_input.pd","steel_target.pd"],
]

start = timer()
runs = 2
rows = []
for dataset in datasets:
    print("Starting",dataset[0],"...")
    data_input = pickle.load(open("../datasets/" + dataset[0], "rb"))
    data_target = pickle.load(open("../datasets/" + dataset[1], "rb"))
    
    use_dataframe = False
    if isinstance(data_input, pd.core.frame.DataFrame):
        use_dataframe = True
        
      
    dimensionality_reduction = ["feature_importance", "feature_importance_lda", "PCA", None]
    dict_results = {
        "dataset" : dataset[0]
    } 
    
    for method in dimensionality_reduction:
        print("Starting", method,"...")
        parameters["use_2_dim_gate_based_on"] = method

        train_accuracies = []
        val_accuracies = []
        rkf = RepeatedKFold(n_splits=5, n_repeats=runs)
        for train_idx, val_idx in rkf.split(data_input):
            if use_dataframe:
                X_temp = data_input.iloc[train_idx]
                y_temp = data_target.iloc[train_idx]
                X_temp.reset_index(inplace=True, drop=True)
                y_temp.reset_index(inplace=True, drop=True)
            else:
                X_temp = data_input[train_idx]
                y_temp = data_target[train_idx]

            parameters["X"] = X_temp
            parameters["y"] = y_temp
            modt = MoDT(**parameters)
            modt.fit(**parameters_fit)
            train_accuracies.append(modt.score_internal_disjoint())

            if use_dataframe:
                X_temp = data_input.iloc[val_idx]
                y_temp = data_target.iloc[val_idx]
                X_temp.reset_index(inplace=True, drop=True)
                y_temp.reset_index(inplace=True, drop=True)
            else:
                X_temp = data_input[val_idx]
                y_temp = data_target[val_idx]
            val_accuracies.append(modt.score(X_temp, y_temp))

        train_accuracy = np.mean(train_accuracies)
        val_accuracy = np.mean(val_accuracies)
        dict_results[str(method) + "_train"] = train_accuracy
        dict_results[str(method) + "_test"] = val_accuracy
        
    rows.append(dict_results)
    
print("Duration", timer() - start)
df_performance = pd.DataFrame(rows)



# %%
pickle.dump(df_performance, open("df_fi_performance2.pd", "wb"))


# %%
df_c = pd.concat([df_intersections.reset_index(drop=True), df_performance], axis=1)
df_c


# %%
df_c = df_c.loc[:,~df_c.columns.duplicated()]


# %%
df_c[["dataset","n_features","intersection","feature_importance_test","feature_importance_lda_test","PCA_test","None_test"]]
pickle.dump(df_performance, open("df_fi_methods2.pd", "wb"))



