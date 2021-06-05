import unittest
import pickle
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from modt.modt import MoDT


def fit_modt(X,
             y,
             n_experts,
             iterations,
             max_depth,
             init_learning_rate,
             learning_rate_decay,
             initialize_with,
             initalization_method,
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
                initalization_method=initalization_method,
                feature_names=feature_names,
                class_names=class_names,
                black_box_algorithm=black_box_algorithm,
                use_2_dim_gate_based_on=use_2_dim_gate_based_on,
                use_2_dim_clustering=use_2_dim_clustering)

    modt.fit(optimization_method=optimization_method,
             add_noise=add_noise,
             use_posterior=use_posterior)

    return modt


class TestMoDT(unittest.TestCase):

    def test_parameters(self):
        data_input = pickle.load(open("datasets/iris_input.pd", "rb"))
        data_target = pickle.load(open("datasets/iris_target.pd", "rb"))
        parameters = {
            "X": data_input,
            "y": data_target,
            "n_experts": 3,
            "iterations": 1,
            "max_depth": 2,
            "init_learning_rate": 10,
            "learning_rate_decay": 1,
            "initialize_with": "random",
            "initalization_method": None,
            "feature_names": None,
            "class_names": None,
            "use_2_dim_gate_based_on": "feature_importance",
            "use_2_dim_clustering": False,
            "black_box_algorithm": None,

            "optimization_method": "least_squares_linear_regression",
            "add_noise": False,
            "use_posterior": False,
        }

        self.assertTrue(fit_modt(**parameters).duration_fit is not None)

        parameters["n_experts"] = 0
        self.assertRaises(ValueError)

        parameters["n_experts"] = 10
        self.assertTrue(fit_modt(**parameters).duration_fit is not None)

        parameters["n_experts"] = 1
        self.assertTrue(fit_modt(**parameters).duration_fit is not None)

        parameters["n_experts"] = 3
        self.assertTrue(fit_modt(**parameters).duration_fit is not None)




if __name__ == '__main__':
    unittest.main()
