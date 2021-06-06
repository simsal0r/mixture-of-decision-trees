import unittest
import pickle
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT
from modt._initialization import *

class TestMoDT(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMoDT, self).__init__(*args, **kwargs)
        self.data_input = pickle.load(open("datasets/iris_input.pd", "rb"))
        self.data_target = pickle.load(open("datasets/iris_target.pd", "rb"))
        self.parameters = None

    def set_paramas(self):
        self.parameters = {
                           "X": self.data_input,
                           "y": self.data_target,
                           "n_experts": 3,
                           "iterations": 1,
                           "max_depth": 2,
                           "init_learning_rate": 10,
                           "learning_rate_decay": 1,
                           "initialize_with": "random",
                           "initialization_method": None,
                           "feature_names": None,
                           "class_names": None,
                           "use_2_dim_gate_based_on": "feature_importance",
                           "use_2_dim_clustering": False,
                           "black_box_algorithm": None,

                           "optimization_method": "least_squares_linear_regression",
                           "add_noise": False,
                           "use_posterior": False,}

    @staticmethod
    def fit_modt(X,
                y,
                n_experts,
                iterations,
                max_depth,
                init_learning_rate,
                learning_rate_decay,
                initialize_with,
                initialization_method,
                feature_names,
                class_names,
                use_2_dim_gate_based_on,
                use_2_dim_clustering,
                black_box_algorithm,

                optimization_method,
                add_noise,
                use_posterior):

        modt = MoDT(X=X,
                    y=y,
                    n_experts=n_experts,
                    iterations=iterations,
                    init_learning_rate=init_learning_rate,
                    learning_rate_decay=learning_rate_decay,
                    max_depth=max_depth,
                    initialize_with=initialize_with,
                    initialization_method=initialization_method,
                    feature_names=feature_names,
                    class_names=class_names,
                    black_box_algorithm=black_box_algorithm,
                    use_2_dim_gate_based_on=use_2_dim_gate_based_on,
                    use_2_dim_clustering=use_2_dim_clustering)

        modt.fit(optimization_method=optimization_method,
                add_noise=add_noise,
                use_posterior=use_posterior)

        return modt

    def test_parameters(self):
        self.set_paramas()
        self.assertTrue(TestMoDT.fit_modt(**self.parameters).duration_fit is not None)

        self.parameters["n_experts"] = 0
        self.assertRaises(ValueError)

        self.parameters["n_experts"] = 10
        self.assertTrue(TestMoDT.fit_modt(**self.parameters).duration_fit is not None)

        self.parameters["n_experts"] = 1
        self.assertTrue(TestMoDT.fit_modt(**self.parameters).duration_fit is not None)

        self.parameters["n_experts"] = 3
        self.assertTrue(TestMoDT.fit_modt(**self.parameters).duration_fit is not None)

    def test_kmeans_init(self):
        self.set_paramas()
        self.parameters["initialize_with"] = "pass_method"
        self.parameters["initialization_method"] = Kmeans_init(theta_fittig_method="lda")
        n_experts = [1,2,3,10]
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(**self.parameters)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.init_labels is not None)

        self.parameters["initialization_method"] = Kmeans_init(theta_fittig_method="lr")
        for n in n_experts:
            self.parameters["n_experts"] = n
            test_model = TestMoDT.fit_modt(**self.parameters)
            self.assertTrue(test_model.duration_fit is not None)
            self.assertTrue(test_model.init_labels is not None)




if __name__ == '__main__':
    unittest.main()
