import pickle

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *
from modt.utility import *
from modt.visualization import *

data_input = pickle.load(open("../datasets/steel_input.pd", "rb"))
data_target = pickle.load(open("../datasets/steel_target.pd", "rb"))
parameters = {
    "X": data_input,
    "y": data_target,
    "n_experts": 6,
    "iterations": 100,
    "max_depth": 1,
    "init_learning_rate": 100,
    "learning_rate_decay": 0.995,
    "initialization_method": "random",
    "use_2_dim_gate_based_on": "feature_importance_lda",
    "save_likelihood": False,
    }

parameters_fit = {
    "optimization_method": "least_squares_linear_regression",
    "early_stopping": False,
    }

modt = MoDT(**parameters)
modt.fit(**parameters_fit)

plot_gating(modt,iteration=modt.best_iteration,title=False,axis_digits=True,inverse_transform_standardization=True,jitter=True,legend=True)
plt.show()

E = modt.get_expert(modt.X,internal=True)

print(np.unique(E, return_counts=True))

modt.all_theta_gating

#pickle_disjoint_data(modt, modt.best_iteration)

#modt.train_disjoint_trees(iteration=modt.best_iteration, tree_algorithm="h2o")

