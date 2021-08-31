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
        self.parameters = None
        self.parameters_fit = None

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

    def test_y_mapping(self):
        self.set_default_paramas()
        data_input = np.array([[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]]).T
        data_target = np.array(["a","b","c","d","e","f","g"])
        self.parameters["X"] = data_input
        self.parameters["y"] = data_target
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.duration_fit is not None)
        self.assertTrue(np.all(test_model._map_y(test_model.y) == data_target))

    def test_parameters(self):
        self.set_default_paramas()
        self.assertTrue(TestMoDT.fit_modt(self.parameters, self.parameters_fit).duration_fit is not None)

        self.parameters["n_experts"] = 0
        with self.assertRaises(ValueError):
            TestMoDT.fit_modt(self.parameters, self.parameters_fit)

        self.parameters["n_experts"] = 10
        self.assertTrue(TestMoDT.fit_modt(self.parameters, self.parameters_fit).duration_fit is not None)

        self.parameters["n_experts"] = 1
        self.assertTrue(TestMoDT.fit_modt(self.parameters, self.parameters_fit).duration_fit is not None)

        self.parameters["n_experts"] = 3
        self.assertTrue(TestMoDT.fit_modt(self.parameters, self.parameters_fit).duration_fit is not None)

    def test_pandas_input(self):
        self.set_default_paramas()
        data_input = pickle.load(open("datasets/banknote_input.pd", "rb"))
        data_target = pickle.load(open("datasets/banknote_target.pd", "rb"))
        self.parameters["X"] = data_input
        self.parameters["y"] = data_target

        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.duration_fit is not None)
        self.assertTrue(test_model.predict(data_input) is not None)         

    def test_pandas_input_categorical(self):
        self.set_default_paramas()
        data_input = pickle.load(open("datasets/adult_input.pd", "rb"))
        data_target = pickle.load(open("datasets/adult_target.pd", "rb"))
        self.parameters["X"] = data_input
        self.parameters["y"] = data_target

        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.duration_fit is not None)
        self.assertTrue(test_model.predict(data_input) is not None)     

    def test_random_init(self):
        self.set_default_paramas() 
        self.parameters["use_2_dim_gate_based_on"] = None
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.duration_fit is not None)

    def test_kmeans_init(self):
        self.set_default_paramas()
        self.parameters["initialization_method"] = Kmeans_init(theta_fittig_method="lda")
        n_experts = [2,3,10]
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.init_labels is not None)

        self.parameters["initialization_method"] = Kmeans_init(theta_fittig_method="lr")
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.init_labels is not None)

    def test_kDTmeans_init(self):
        self.set_default_paramas()
        self.parameters["initialization_method"] = KDTmeans_init(alpha=1,beta=0.05,gamma=0.1,theta_fittig_method="lda")
        n_experts = [1,2,3,10]
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.all_DT_clusters != [])

    def test_kDTmeans_init_large_dataset(self):
        self.set_default_paramas()
        data_input = pickle.load(open("datasets/breast_cancer_input.np", "rb"))
        data_target = pickle.load(open("datasets/breast_cancer_target.np", "rb"))
        self.parameters["X"] = data_input
        self.parameters["y"] = data_target
        self.parameters["initialization_method"] = KDTmeans_init(alpha=1,beta=0.05,gamma=0.1,theta_fittig_method="lda")
        n_experts = [1,2,3,10]
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.all_DT_clusters != [])

    def test_DBSCAN_init(self):
        self.set_default_paramas()
        data_input = pickle.load(open("datasets/breast_cancer_input.np", "rb"))
        data_target = pickle.load(open("datasets/breast_cancer_target.np", "rb"))
        self.parameters["X"] = data_input
        self.parameters["y"] = data_target        
        self.parameters["initialization_method"] = DBSCAN_init(theta_fittig_method="lda",eps=0.9,min_samples=5)
        n_experts = [1,2,3]
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.init_labels is not None)

    def test_Boosting_init(self):
        self.set_default_paramas()
        self.parameters["initialization_method"] = Boosting_init()
        n_experts = [1,2,3,10]
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.init_labels is not None)

    def test_BGM_init(self):
        self.set_default_paramas()
        self.parameters["initialization_method"] = BGM_init()
        n_experts = [1,2,3,10]
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.init_labels is not None)

    def test_2dim_sanity_check(self):
        self.set_default_paramas()
        self.parameters["use_2_dim_gate_based_on"] = None
        self.parameters["use_2_dim_clustering"] = True
        with self.assertRaises(ValueError):
            TestMoDT.fit_modt(self.parameters, self.parameters_fit)

    def test_2dim(self):
        self.set_default_paramas()
        self.parameters["use_2_dim_gate_based_on"] = "feature_importance"
        self.parameters["use_2_dim_clustering"] = True
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model._select_X_internal()[1].shape[1] == 3)
        self.assertTrue(test_model.duration_fit is not None)

        self.parameters["use_2_dim_gate_based_on"] = "feature_importance"
        self.parameters["use_2_dim_clustering"] = False
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model._select_X_internal()[1].shape[1] == 3)
        self.assertTrue(test_model.duration_fit is not None)

        self.parameters["use_2_dim_gate_based_on"] = "PCA"
        self.parameters["use_2_dim_clustering"] = True
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model._select_X_internal()[1].shape[1] == 3)
        self.assertTrue(test_model.duration_fit is not None)

        self.parameters["use_2_dim_gate_based_on"] = "PCA"
        self.parameters["use_2_dim_clustering"] = False
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model._select_X_internal()[1].shape[1] == 3)
        self.assertTrue(test_model.duration_fit is not None)

    def test_2dim_initialization_method(self):
        self.set_default_paramas()
        self.parameters["initialization_method"] = Kmeans_init(theta_fittig_method="lda")

        self.parameters["use_2_dim_gate_based_on"] = "feature_importance"
        self.parameters["use_2_dim_clustering"] = True
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model._select_X_internal()[1].shape[1] == 3)
        self.assertTrue(test_model.duration_fit is not None)

        self.parameters["use_2_dim_gate_based_on"] = "feature_importance"
        self.parameters["use_2_dim_clustering"] = False
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model._select_X_internal()[1].shape[1] == 3)
        self.assertTrue(test_model.duration_fit is not None)

        self.parameters["use_2_dim_gate_based_on"] = "PCA"
        self.parameters["use_2_dim_clustering"] = True
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model._select_X_internal()[1].shape[1] == 3)
        self.assertTrue(test_model.duration_fit is not None)

        self.parameters["use_2_dim_gate_based_on"] = "PCA"
        self.parameters["use_2_dim_clustering"] = False
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model._select_X_internal()[1].shape[1] == 3)
        self.assertTrue(test_model.duration_fit is not None)

    def test_predict(self):
        self.set_default_paramas()
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(len(test_model.predict(X=self.data_input)) == len(self.data_input))

    def test_score(self):
        self.set_default_paramas()
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.score(self.data_input, self.data_target) > 0)        

    def test_score_internal(self):
        self.set_default_paramas()
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.score_internal(0) > 0)   

    def test_score_internal_disjoint(self):
        self.set_default_paramas()
        test_model = TestMoDT.fit_modt(self.parameters, self.parameters_fit)
        self.assertTrue(test_model.score_internal_disjoint() > 0)   

if __name__ == '__main__':
    unittest.main()
