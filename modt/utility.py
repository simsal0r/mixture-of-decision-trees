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
