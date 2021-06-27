import unittest
import pickle
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *

import numpy as np


class TestMoDT(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMoDT, self).__init__(*args, **kwargs)
        self.data_input = pickle.load(open("datasets/iris_input.pd", "rb"))
        self.data_target = pickle.load(open("datasets/iris_target.pd", "rb"))

    def set_default_paramas(self):
        self.parameters = {
                           "X": self.data_input,
                           "y": self.data_target,
                           "n_experts": 3,
                           "iterations": 2,
                           "max_depth": 2,
                           "init_learning_rate": 10,
                           "learning_rate_decay": 1,
                           "initialization_method": "random",
                           "use_2_dim_gate_based_on": None,
                           "use_2_dim_clustering": False,
                           "black_box_algorithm": None,
                           "feature_names": None,
                           "class_names": None,
                           "save_likelihood" : False,
                           "verbose" : False,
                           }

        self.parameters_fit = {
                               "optimization_method": "ridge_regression",
                               "early_stopping": False,
                               "use_posterior": False,
                              }

    @staticmethod
    def fit_modt(parameters, parameters_fit):
        modt = MoDT(**parameters)
        modt.fit(**parameters_fit)
        return modt


    def test_fi_DT(self):
        self.set_default_paramas()
        self.parameters["use_2_dim_gate_based_on"] = "feature_importance"
        self.assertTrue(TestMoDT.fit_modt(self.parameters, self.parameters_fit).duration_fit is not None)

    def test_fi_lda(self):
        self.set_default_paramas()
        self.parameters["use_2_dim_gate_based_on"] = "feature_importance_lda"
        self.assertTrue(TestMoDT.fit_modt(self.parameters, self.parameters_fit).duration_fit is not None)

    def test_prediction_input_has_other_features(self):
        self.set_default_paramas()
        X = pickle.load(open("datasets/adult_input.pd", "rb"))
        y = pickle.load(open("datasets/adult_target.pd", "rb"))
        X_temp = X.iloc[0:50]
        y_temp = y.iloc[0:50]
        X_temp.reset_index(inplace=True, drop=True)
        y_temp.reset_index(inplace=True, drop=True)
        self.parameters["X"] = X_temp
        self.parameters["y"] = y_temp

        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)

        X_temp = X.iloc[50:100]
        y_temp = y.iloc[50:100]
        X_temp.reset_index(inplace=True, drop=True)
        y_temp.reset_index(inplace=True, drop=True)

        predictions = test_model.predict(X_temp)

        self.assertTrue(len(predictions) == 50)

    def test_fit_twice(self):
        self.set_default_paramas()
        modt = MoDT(**self.parameters)
        modt.fit(**self.parameters_fit)
        self.assertTrue(len(modt.all_theta_gating) == 2)
        modt.fit(**self.parameters_fit)
        self.assertTrue(len(modt.all_theta_gating) == 2)

    def test_save_likelihood(self):
        self.set_default_paramas()
        self.parameters["iterations"] = 25
        self.parameters["save_likelihood"] = True
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(len(test_model.all_likelihood) == 25)

    def test_early_stopping_accuracy(self):
        self.set_default_paramas()
        self.parameters["iterations"] = 500
        self.parameters_fit["iterations"] = 500
        self.parameters_fit["early_stopping"] = "accuracy"
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.completed_iterations < 499)
        self.assertTrue(test_model.completed_iterations >= 5)

    def test_early_stopping_likelihood(self):
        self.set_default_paramas()
        self.parameters["iterations"] = 500
        self.parameters_fit["iterations"] = 500
        self.parameters_fit["early_stopping"] = "likelihood"
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.completed_iterations < 499)
        self.assertTrue(test_model.completed_iterations >= 5)

        self.parameters_fit["early_stopping"] = True
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.completed_iterations < 499)
        self.assertTrue(test_model.completed_iterations >= 5)



if __name__ == '__main__':
    unittest.main()
