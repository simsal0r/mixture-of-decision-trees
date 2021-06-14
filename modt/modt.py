import numpy as np
import pandas as pd
from functools import lru_cache
from timeit import default_timer as timer
import pickle

from sklearn.linear_model import LinearRegression
from scipy.special import softmax
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture


class MoDT():

    def __init__(self,
                 X,
                 y,
                 n_experts,
                 iterations,
                 max_depth,
                 init_learning_rate=2,
                 learning_rate_decay=0.95,
                 initialize_with="random",
                 initialization_method=None,
                 use_2_dim_gate_based_on=None,
                 use_2_dim_clustering=False,
                 black_box_algorithm=None,
                 feature_names=None,
                 class_names=None,
                 verbose=False):

        self.verbose = verbose
        self.verbose_detailed = False
        self.X_contains_categorical = False

        if np.array(X).ndim == 1:
            raise ValueError("X must have at least 2 dimensions.")

        self.n_features_of_X = X.shape[1]
        self.n_input = X.shape[0]
        self.y_map = None
        self.n_experts = n_experts
        self.max_depth = max_depth
        self.iterations = iterations
        self.completed_iterations = None
        #self.counter_stale_iterations = 0
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learn_rate = [self.init_learning_rate * (self.learning_rate_decay ** max(float(i), 0.0)) for i in range(iterations)]
        self.gating_values = None
        self.DT_experts = None
        self.DT_experts_disjoint = None
        self.all_DTs = []
        self.all_theta_gating = []
        self.all_gating_values = []
        self.all_accuracies = []
        self.best_iteration = None
        self.posterior_probabilities = None
        self.confidence_experts = None
        self.counter_stale_iterations = 0  
        self.use_2_dim_gate_based_on = use_2_dim_gate_based_on
        self.use_2_dim_clustering = use_2_dim_clustering

        # Plotting & Debugging
        self.duration_fit = None
        self.duration_initialization = None
        self.init_labels = None
        self.dbscan_mask = None
        self.dbscan_selected_labels = None
        # Debugging & Plotting kDTmeans
        self.all_DT_clusters = []
        self.all_clustering_accuracies = []
        self.all_cluster_labels = []
        self.all_cluster_centers = []

        self._check_argument_validity()

        (self.X,  # Bias and standardization will be added
         self.X_original,  # Original input as numpy array 
         self.X_original_pd,
         self.X_one_hot, # TODO: Remove
         self.y,
         self.y_original,
         self.feature_names,
         self.class_names
         ) = self._interpret_input(X, y, feature_names, class_names)

        if black_box_algorithm is not None:
            self.y_before_surrogate = self.y
            self.y = self._transform_y_with_surrogate_model(black_box_algorithm)

        self.scaler = self._create_scaler(self.X)  # Standardization on input. Scaler also needed for prediction of new observations.
        self.X = self._preprocess_X(self.X)  # Apply standardization and add bias

        self.X_top_2_mask = self._get_2_dim_feature_importance_mask()  # Always calculate Top 2 features for plotting; 2D + Bias
        self.X_2_dim = self.X[:, self.X_top_2_mask]  # For plotting; 2D + bias; components in case of PCA

        if self.use_2_dim_gate_based_on is not None:
            if self.use_2_dim_gate_based_on == "feature_importance":
                pass
            elif self.use_2_dim_gate_based_on == "PCA":
                self.X_2_dim = self._perform_PCA()
            else:
                raise Exception("Invalid method for gate dimensionality reduction.")

        # Initialize gating values
        self.theta_gating = self._initialize_theta(initialize_with, initialization_method)
        self.init_theta = self.theta_gating.copy()

    def _check_argument_validity(self):
        if self.use_2_dim_gate_based_on is None and self.use_2_dim_clustering:
            raise ValueError("Argument incompatibility.")
        if self.n_experts <= 0:
            raise ValueError("More than 0 experts required.")

    def _interpret_input(self, X, y, feature_names, class_names):
        X_one_hot = None
        X_original_pd = None
        feature_names_new = None

        if isinstance(X, pd.core.frame.DataFrame):  # Pandas
            # Categorical treatment
            if np.intersect1d(['object', 'category'], X.dtypes.values.astype(str)).size > 0:
                self.X_contains_categorical = True
                X_new = self._one_hot_encode(X)
            else:
                X_new = np.array(X)
            X_original = np.array(X)
            X_original_pd = X
            feature_names_new = np.array(X.columns)

        elif isinstance(X, np.ndarray):  # Numpy
            X_new = X
            X_original = X
            # TODO: Insert categorical treatment
        else:
            raise Exception("X must be Pandas DataFrame or Numpy array.")

        y_new, y_original, class_names_new = self._interpret_input_y(y)

        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                raise Exception("Feature names list length ({}) inconsistent with X ({}).".format(len(feature_names), X.shape[1]))
            feature_names_new = np.array(feature_names)

        if class_names is not None:
            class_names_new = np.array(class_names)
            class_names_new = class_names_new.astype(str)

        return X_new, X_original, X_original_pd, X_one_hot, y_new, y_original, feature_names_new, class_names_new

    def _interpret_input_y(self, y):
        class_names_new = None
        if isinstance(y, pd.core.series.Series):  # Pandas 
            y_new, class_names = pd.factorize(y)
            class_names_new = np.array(class_names)
        elif isinstance(y, pd.core.frame.DataFrame):
            if y.columns.size > 1:
                raise ValueError("y has more than one column.")
            y = y.iloc[:, 0]
            y_new, class_names = pd.factorize(y)
            class_names_new = np.array(class_names)
        elif isinstance(y, np.ndarray):  # Numpy
            y_new = pd.factorize(y)[0]
        else:
            raise ValueError("y must be Pandas series Pandas DataFrame or Numpy array.")

        indexes_unique = np.unique(y, return_index=True)[1]
        keys = [y[idx] for idx in indexes_unique] 
        values = [y_new[idx] for idx in indexes_unique] 
        self.y_map = dict(zip(values, keys))

        return y_new, y, class_names_new

    def _map_y(self, y_prediction, reverse=False):
        if reverse:  # Unseen value is outputted as is # TODO: Remove?
            y_map = {v: k for k, v in self.y_map.items()}
            return np.array([(y_map[y] if y in y_map else y) for y in y_prediction])
        else:
            return np.array([self.y_map[y] for y in y_prediction])

    def _one_hot_encode(self, X):
        X_one_hot = pd.get_dummies(X, columns=list(X.select_dtypes(include=['object', 'category']).columns))
        X_one_hot = np.array(X_one_hot)
        return X_one_hot

    def _get_2_dim_feature_importance_mask(self):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.X, self.y)
        # TODO: Rework
        features_10 = []  # Features that have more than 10 unique values
        for column in range(0, self.X.shape[1]):
            if np.unique(self.X[:, column]).size > 10:
                features_10.append(column)
            else:
                pass

        if len(features_10) > 2:
            top_features_idx_all = np.argsort(-clf.feature_importances_)
            mask = np.repeat(False, self.X.shape[1])
            features_10_idx = []

            for feature in features_10:
                features_10_idx.append(np.where(top_features_idx_all == feature)[0][0])

            mask[features_10_idx] = True

            top_features_idx = top_features_idx_all[mask][:2]  # Select 2 best features

            if self.verbose:
                print("Top 2 Feature Importance:", clf.feature_importances_[top_features_idx])
                print("Top 2 Feature Importance w/ features with few unique values:", clf.feature_importances_[np.argsort(-clf.feature_importances_)[:2]])

            self.top_features_idx = top_features_idx
        else:
            self.top_features_idx = np.argsort(-clf.feature_importances_)[:2]

        # X with only 2 dimensions (+1 bias) for interpretable gates
        mask = list(self.top_features_idx)
        mask.append(-1)  # We also need the last (bias) column for regression etc.
        return mask

    def _select_X_internal(self):
        if self.use_2_dim_clustering:
            X = self.X_2_dim
            X_gate = self.X_2_dim
        elif self.use_2_dim_gate_based_on:  # (but not 2_dim_clustering)
            X = self.X
            X_gate = self.X_2_dim
        else:
            X = self.X
            X_gate = self.X

        return X, X_gate

    def _transform_X_into_2_dim_for_prediction(self, X, method):
        if method == "feature_importance":
            X = X[:, self.X_top_2_mask]
        elif method == "PCA":
            X = self.pca.transform(X)
            X = np.append(X, np.ones([X.shape[0], 1]), axis=1)  # Bias
        else:
            raise ValueError("Invalid method for gate dimensionality reduction.")

        return X

    def _perform_PCA(self):
        pca = PCA(n_components=2)
        pca.fit(self.X)
        self.pca = pca  # Needed s.t. we can transform incoming prediction data
        if self.verbose:
            print("PCA explained variance:", pca.explained_variance_ratio_)
        X_pca = pca.transform(self.X)
        X_pca = np.append(X_pca, np.ones([X_pca.shape[0], 1]), axis=1)
        return X_pca

    def _create_scaler(self, X):
        return(StandardScaler().fit(X))

    def _preprocess_X(self, X):
        """Perform standardization and add bias."""
        X = self.scaler.transform(X)
        return np.append(X, np.ones([X.shape[0], 1]), axis=1)  # Add bias

    def _initialize_theta(self, initialize_with, initialization_method=None):
        start = timer()

        if initialize_with == "random":
            if self.use_2_dim_gate_based_on is not None:
                n_features = 3
            else:
                n_features = self.X.shape[1]
            initialized_theta = np.random.rand(n_features, self.n_experts)
        elif initialize_with == "pass_method":
            initialized_theta = initialization_method._calculate_theta(self)
        else:
            raise Exception("Invalid initalization method specified.")

        end = timer()
        self.duration_initialization = end - start
        if self.verbose:
            print("Duration initialization:", self.duration_initialization)

        return initialized_theta
 
    def predict(self, X):
        iteration = self.best_iteration

        if self.X_contains_categorical:
            X = self._one_hot_encode(X)
        X_gate = self._preprocess_X(X)
        if self.use_2_dim_gate_based_on is not None:  # feature importance or PCA
            X_gate = self._transform_X_into_2_dim_for_prediction(X_gate, method=self.use_2_dim_gate_based_on)

        DTs = self.DT_experts_disjoint        
        gating = self._gating_softmax(X_gate, self.all_theta_gating[iteration])
        selected_gates = np.argmax(gating, axis=1)

        predictions = [DTs[tree_index].predict(X) for tree_index in range(0, len(DTs))]
        predictions_gate_selected = np.array([predictions[gate][index] for index, gate in enumerate(selected_gates)]).astype("int")
        return self._map_y(predictions_gate_selected)

    def predict_internal(self,iteration):
        if self.use_2_dim_gate_based_on is not None:  # feature importance or PCA
            X_gate = self._transform_X_into_2_dim_for_prediction(self.X, method=self.use_2_dim_gate_based_on)
        else:
            X_gate = self.X

        DTs = self.all_DTs[iteration]
        gating = self._gating_softmax(X_gate, self.all_theta_gating[iteration])
        selected_gates = np.argmax(gating, axis=1)

        predictions = [DTs[tree_index].predict(self.X) for tree_index in range(0, len(DTs))]
        return np.array([predictions[gate][index] for index, gate in enumerate(selected_gates)]).astype("int")


    # def predict_disjoint(self, X):
    #     """Wrapper for disjoint predictions."""

    #     if self.DT_experts_disjoint is None:
    #         raise Exception("Disjoint DTs must be trained.")
    #     return self.predict(X, transform=False, disjoint_trees=True)

    def predict_with_expert(self, X, expert, iteration="best"):
        if iteration == "best":
            iteration = self.best_iteration
        X = self._preprocess_X(X)
        DTs = self.all_DTs[iteration]
        return DTs[expert].predict(X)

    def get_expert(self, X_gate, iteration="best", internal=False):
        if iteration == "best":
            iteration = self.best_iteration        
        if not internal:
            X_gate = self._preprocess_X(X_gate)
            if self.use_2_dim_gate_based_on is not None:
                X_gate = self._transform_X_into_2_dim_for_prediction(X_gate, method=self.use_2_dim_gate_based_on)

        gating = self._gating_softmax(X_gate, self.all_theta_gating[iteration])
        return np.argmax(gating, axis=1)

    def fit(self, optimization_method="default", early_stopping=True, use_posterior=False, **optimization_kwargs):
        start = timer()

        _, X_gate = self._select_X_internal()
        self.completed_iterations = self.iterations

        for iteration in range(0, self.iterations):
            self._e_step(X_gate)

            self._log_values_to_array()  # After E step: Theta, DTs and gating values are in sync
            self.all_accuracies.append(self.score_internal(iteration=iteration))

            if early_stopping:
                if self._no_accuracy_change(iteration):
                    if self.verbose:
                        print("Stopped at iteration: {}".format(iteration))
                    self.completed_iterations = iteration
                    break
                
            if iteration != self.iterations - 1:  # We can skip last M step because DT would not be re-trained again anyway
                self._m_step(X_gate,
                            iteration,
                            optimization_method=optimization_method,
                            use_posterior=use_posterior,
                            **optimization_kwargs)

        self.best_iteration = np.argmax(self.all_accuracies)
        self.train_disjoint_trees(iteration=self.best_iteration, tree_algorithm="sklearn")
            
        end = timer()
        self.duration_fit = end - start
        if self.verbose:
            print("Duration EM fit:", self.duration_fit)

    def _e_step(self, X_gate):
        self.gating_values = self._gating_softmax(X=X_gate, theta_gating=self.theta_gating)
        self.gating_values = np.nan_to_num(self.gating_values)
        self._train_trees()

        self.posterior_probabilities = self._posterior_probabilties(self.DT_experts)

    def _m_step(self, X_gate, iteration, optimization_method, use_posterior, **optimization_kwargs):
        theta_new = self._update_theta(X_gate, optimization_method, use_posterior, **optimization_kwargs)
        self.theta_gating += self.learn_rate[iteration] * theta_new

    def _update_theta(self, X_gate, optimization_method, use_posterior, **optimization_kwargs):
        if use_posterior is True:
            optimization_target = self.posterior_probabilities
        else:
            optimization_target = self.posterior_probabilities - self.gating_values

        if optimization_method == "least_squares_linear_regression":
            model = LinearRegression(fit_intercept=False).fit(X_gate, optimization_target)
            theta_new = model.coef_.T
        elif optimization_method == "ridge_regression":
            max_iter = optimization_kwargs.get("rr_max_iter")
            model = Ridge(alpha=1, fit_intercept=False, max_iter=max_iter, solver="auto").fit(X_gate, optimization_target)
            theta_new = theta_new = model.coef_.T
        elif optimization_method == "lasso_regression":
            model = Lasso(alpha=1, fit_intercept=False).fit(X_gate, optimization_target)
            theta_new = theta_new = model.coef_.T
        elif optimization_method == "moet1":
            theta_new = X_gate.T @ optimization_target
        elif optimization_method == "moet2":
            theta_new = self._theta_recalculation_moet2(posterior_probabilities=self.posterior_probabilities, gating=self.gating_values, X=X_gate)
        else:
            raise Exception("Invalid opimization method selected.")

        if self.verbose and self.verbose_detailed:
            score = model.score(X_gate, optimization_target)
            print("Score of {} = {}".format(optimization_method, score))

        return theta_new

    def _train_trees(self):
        DT_experts = [None for i in range(self.n_experts)]
        for index_expert in range(0, self.n_experts):
            DT_experts[index_expert] = tree.DecisionTreeClassifier(max_depth=self.max_depth)
            DT_experts[index_expert].fit(X=self.X, y=self.y, sample_weight=self.gating_values[:, index_expert])  # Training weighted by gating values
        self.DT_experts = DT_experts

    def _log_values_to_array(self):
        # Plotting & Debugging
        self.all_theta_gating.append(self.theta_gating.copy())  # TODO: Theta and gating values not in sync
        self.all_DTs.append(self.DT_experts.copy())
        self.all_gating_values.append(self.gating_values.copy())

    def train_disjoint_trees(self, iteration, tree_algorithm="sklearn"):
        """Train DTs with one-hot gates as weights""" 
        gating_values = self.all_gating_values[iteration]
        gate = np.argmax(gating_values, axis=1)
        gating_values_hard = np.zeros([self.n_input, self.n_experts])
        gating_values_hard[np.arange(0, self.n_input), gate] = 1

        DT_experts_disjoint = [None for i in range(self.n_experts)]
        if tree_algorithm == "sklearn":
            if self.X_contains_categorical:
                X = self._one_hot_encode(self.X_original_pd)
            else:
                X = self.X_original
            for index_expert in range(0, self.n_experts):
                DT_experts_disjoint[index_expert] = tree.DecisionTreeClassifier(max_depth=self.max_depth)
                DT_experts_disjoint[index_expert].fit(X=X, y=self.y, sample_weight=gating_values_hard[:, index_expert])
        elif tree_algorithm == "optimal_trees":
            from interpretableai import iai
            for index_expert in range(0, self.n_experts):
                mask = gating_values_hard[:, index_expert] == 1
                X = self.X_original_pd[mask].copy()

                X.loc[:, X.dtypes == 'object'] = X.select_dtypes(['object']).apply(lambda x: x.astype('category'))  # Optimal trees cant handle object dtypes
                pickle.dump(X, open("output/iai_X_e{}.p".format(index_expert), "wb"))

                y = self.y_original[mask]
                pickle.dump(y, open("output/iai_y_e{}.p".format(index_expert), "wb"))
                # grid = iai.GridSearch(iai.OptimalTreeClassifier(),max_depth=2)
                # X1 = pickle.load( open("output/iai_X.p", "rb" ))
                # y1 = pickle.load( open ("output/iai_y.p", "rb"))
                # z = grid.fit(X=X1, y=y1)
        elif tree_algorithm == "h2o":
            for index_expert in range(0, self.n_experts):
                DT_experts_disjoint[index_expert] = tree.DecisionTreeClassifier(max_depth=self.max_depth)
                DT_experts_disjoint[index_expert].fit(X=self.X_original, y=self.y, sample_weight=gating_values_hard[:, index_expert])
        else:
            raise Exception("Invalid tree algorithm.")

        self.DT_experts_disjoint = DT_experts_disjoint

    def _theta_recalculation_moet2(self, posterior_probabilities, gating, X):
        R = np.zeros([self.n_experts * self.n_features_of_X, self.n_experts * self.n_features_of_X])

        for idx_input in range(self.n_input):
            for idx_expert in range(self.n_experts):
                pom1 = gating[idx_input, idx_expert] * (1 - gating[idx_input, idx_expert])
                pom2 = np.zeros([self.n_experts, self.n_features_of_X])
                pom2[idx_expert] = X[idx_input]
                pom2 = pom2.reshape(-1, 1)  # Flatten
                R += pom1 * (pom2 @ pom2.T)

        e = self._update_theta(X, optimization_method="moet1", use_posterior=False)

        if np.linalg.cond(R) < 1e7:
            return (np.linalg.inv(R) @ e.flatten()).reshape(-1, self.n_experts)
        else:
            return e

    def _no_accuracy_change(self, iteration):
        max_stale_iterations = 20
        min_iterations = 5
        if self.counter_stale_iterations >= max_stale_iterations:
            return True
        if iteration >= min_iterations:
            difference = self.all_accuracies[iteration] - self.all_accuracies[iteration-1]
            difference_cycle = self.all_accuracies[iteration] - self.all_accuracies[iteration-2]
            # print(difference,np.abs(difference) < 0.0001)
            if np.abs(difference) < 0.0001 or np.abs(difference_cycle) < 0.0001:
                self.counter_stale_iterations += 1
            else:
                self.counter_stale_iterations = 0
        return False

    # Returns gating probabilities for experts column-wise for all datapoints x
    def _gating_softmax(self, X, theta_gating):
        # Use theta to calculate raw gating values
        linear_model = np.matmul(X, theta_gating)
        if linear_model.shape[1] != self.n_experts:
            raise Exception()
        # Softmax # substract the row's max from each row
        exp_linear_model = np.exp(linear_model - np.max(linear_model, axis=1, keepdims=True))
        return(
            exp_linear_model / np.expand_dims(exp_linear_model.sum(axis=1), axis=1)
        )

    # h function
    def _posterior_probabilties(self, DT_experts):
        confidence_correct = np.zeros([self.n_input, self.n_experts])  # TODO: See Nina version (?)
        for expert_index in range(0, self.n_experts):
            dt = DT_experts[expert_index]
            dt_probability = dt.predict_proba(self.X)
            confidence_correct[:, expert_index] = dt_probability[np.arange(self.n_input), self.y.flatten().astype(int)]

        multiplication = self.gating_values * confidence_correct
        return(multiplication / np.expand_dims(np.sum(multiplication, axis=1), axis=1))

    def score(self, X , y):
        if len(X) != len(y):
            raise ValueError("X and y have different lengths.")

        predicted_labels = self.predict(X)
        accuracy = (np.count_nonzero(predicted_labels == np.array(y)) / len(X))
        return accuracy

    def score_internal(self, iteration):
        predicted_labels = self.predict_internal(iteration)
        accuracy = (np.count_nonzero(predicted_labels.astype(int) == self.y.astype(int)) / self.n_input)
        return accuracy

    def score_internal_disjoint(self):
        if self.X_contains_categorical:
            X = self.X_original_pd
        else:
            X = self.X_original
        predicted_labels = self.predict(X)
        accuracy = (np.count_nonzero(predicted_labels == self.y_original) / self.n_input)
        return accuracy

    def estimate_n_experts(self, range1=range(1, 7)):
        start = timer()
        n_components_range = range(1, 7)
        lowest_bic = np.infty
        bic = []
        estimated_n_components = None
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type="full")
            gmm.fit(self.X)
            bic.append(gmm.bic(self.X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                estimated_n_components = n_components

        end = timer()
        duration = end - start
        if self.verbose:
            print("Duration:", duration)
            print("N_expert tested:", range1)
            print("BIC:", bic)
            print("BIC %: ", np.around(bic / np.sum(bic), 2))
            print("Estimated experts", estimated_n_components)
        return estimated_n_components

    def _transform_y_with_surrogate_model(self, black_box_algorithm):
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.svm import SVC
        start = timer()

        if black_box_algorithm == "ANN":
            clf = MLPClassifier().fit(self.X, self.y)
        elif black_box_algorithm == "DT":
            clf = tree.DecisionTreeClassifier().fit(self.X, self.y)
        elif black_box_algorithm == "Adaboost":
            clf = AdaBoostClassifier().fit(self.X, self.y)
        elif black_box_algorithm == "SVM":
            clf = SVC().fit(self.X, self.y)
        else:
            raise Exception("Invalid black box algorithm specified.")

        labels = np.array(clf.predict(self.X))

        if np.sum(np.in1d(labels, np.arange(0, len(np.unique(labels)))) is False) > 0:
            if self.verbose:
                print("Warning: Surrogate model has removed at least one class entirely.")
                labels = pd.factorize(labels)[0]

        end = timer()
        duration = end - start

        if self.verbose:
            print("Duration training surrogate model:", duration)
            print(black_box_algorithm, "accuracy:", clf.score(self.X, self.y))

        return labels
