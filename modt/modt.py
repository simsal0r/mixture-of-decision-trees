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
                 class_names=None):

        self.verbose = True
        self.X_contains_categorical = False

        self.n_features_of_X = X.shape[1]
        self.n_input = X.shape[0]
        self.n_experts = n_experts
        self.max_depth = max_depth
        self.iterations = iterations
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learn_rate = [self.init_learning_rate * (self.learning_rate_decay ** max(float(i), 0.0)) for i in range(iterations)]
        self.gating_values = None
        self.DT_experts = None
        self.DT_experts_disjoint = None
        self.all_DTs = []
        self.posterior_probabilities = None
        self.confidence_experts = None
        self.no_improvements = 0  # Counter for adding noise
        self.use_2_dim_gate_based_on = use_2_dim_gate_based_on
        self.use_2_dim_clustering = use_2_dim_clustering

        # Plotting & Debugging
        self.duration_fit = None
        self.duration_initialization = None
        self.init_labels = None
        self.all_theta_gating = []
        self.all_gating_values = []
        self.dbscan_mask = None
        self.regression_target = None
        self.dbscan_selected_labels = None
        # Debugging & Plotting kDTmeans
        self.all_DT_clusters = []
        self.all_clustering_accuracies = []
        self.all_cluster_labels = []
        self.all_cluster_centers = []


        self._check_argument_validity()

        (self.X,
         self.X_original,
         self.X_original_pd,
         self.X_one_hot,
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

        self.X_top_2_mask = self._get_2_dim_feature_importance_mask()  # Always calculate Top 2 features for plotting
        self.X_2_dim = self.X[:, self.X_top_2_mask]  # For plotting

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
            raise ValueError("Argument incompatibility")
        if self.n_experts <= 0:
            raise ValueError("More than 0 experts required.")

    def _interpret_input(self, X, y, feature_names, class_names):
        X_one_hot = None
        X_original_pd = None
        feature_names_new = None
        class_names_new = None
        # X
        # Pandas
        if isinstance(X, pd.core.frame.DataFrame):
            # Categorical treatment
            if np.intersect1d(['object', 'category'], X.dtypes.values.astype(str)).size > 0:
                self.X_contains_categorical = True
                X_one_hot = pd.get_dummies(X, columns=list(X.select_dtypes(include=['object', 'category']).columns))
                X_one_hot = np.array(X_one_hot)
                X_new = X_one_hot
            else:
                X_new = np.array(X)
            X_original = np.array(X)
            X_original_pd = X
            feature_names_new = np.array(X.columns)
        # Numpy
        elif isinstance(X, np.ndarray):
            X_new = X
            X_original = X
            # TODO: Insert categorical treatment
        else:
            raise Exception("X must be Pandas DataFrame or Numpy array.")

        # Y
        # Pandas
        if isinstance(y, pd.core.series.Series):
            y_new, class_names = pd.factorize(y)
            class_names_new = np.array(class_names)
        elif isinstance(y, pd.core.frame.DataFrame):
            if y.columns.size > 1:
                raise Exception("y has more than one column.")
            y = y.iloc[:, 0]
            y_new, class_names = pd.factorize(y)
            class_names_new = np.array(class_names)
        # Numpy
        elif isinstance(y, np.ndarray):
            y_new = pd.factorize(y)[0]
        else:
            raise Exception("y must be Pandas series Pandas DataFrame or Numpy array.")
        y_original = y

        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                raise Exception("Feature names list length ({}) inconsistent with X ({}).".format(len(feature_names), X.shape[1]))
            feature_names_new = np.array(feature_names)

        if class_names is not None:
            class_names_new = np.array(class_names)
            class_names_new = class_names_new.astype(str)

        return X_new, X_original, X_original_pd, X_one_hot, y_new, y_original, feature_names_new, class_names_new

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
        # TODO: If less than 2 true

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
        X = self.scaler.transform(X)
        return np.append(X, np.ones([X.shape[0], 1]), axis=1)  # Add bias

    def _initialize_theta(self, initialize_with, initialization_method=None):
        start = timer()

        if initialize_with == "random":
            if self.use_2_dim_gate_based_on is not None:
                n_features = 3
            else:
                n_features = self.n_features_of_X
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
 
    def predict_hard_iteration(self, X, iteration, internal=False, disjoint_trees=False):
        if not internal:
            X = self._preprocess_X(X)
        if self.use_2_dim_gate_based_on is not None:
            X_gate = self._transform_X_into_2_dim_for_prediction(X, method=self.use_2_dim_gate_based_on)
        else:
            X_gate = X

        # if disjoint_trees:
        #     DTs = self.DT_experts_disjoint
        #     gating = self._gating_softmax(X_gate,self.all_theta_gating[self.iterations-1])
        #     selected_gates = np.argmax(gating,axis=1)
        # else:
        DTs = self.all_DTs[iteration]
        gating = self._gating_softmax(X_gate, self.all_theta_gating[iteration])
        selected_gates = np.argmax(gating, axis=1)

        predictions = [DTs[tree_index].predict(X) for tree_index in range(0, len(DTs))]
        return np.array([predictions[gate][index] for index, gate in enumerate(selected_gates)]).astype("int")

    def predict_hard(self, X, disjoint_trees=False):  # Final iteration
        if disjoint_trees:
            return self.predict_hard_iteration(X, iteration=self.iterations - 1, disjoint_trees=True)
        else:
            return self.predict_hard_iteration(X, iteration=self.iterations - 1)

    def predict_with_expert_iteration(self, X, expert, iteration):
        X = self._preprocess_X(X)
        DTs = self.all_DTs[iteration]
        return DTs[expert].predict(X)

    def get_expert_iteration(self, X, iteration, internal=False):
        if not internal:
            X = self._preprocess_X(X)
            if self.use_2_dim_gate_based_on is not None:
                X = self._transform_X_into_2_dim_for_prediction(X, method=self.use_2_dim_gate_based_on)

        gating = self._gating_softmax(X, self.all_theta_gating[iteration])
        return np.argmax(gating, axis=1)

    def fit(self, optimization_method="default", add_noise=False, use_posterior=False, **optimization_kwargs):
        start = timer()

        _, X_gate = self._select_X_internal()

        for iteration in range(0, self.iterations):
            self._e_step(X_gate, first_iteration=(iteration == 0))
            self._log_values_to_array()  # Theta, DTs and gating values are in sync
            if iteration == self.iterations - 1:  # We can skip last M step because DT would not be re-trained again anyway
                break
            self._m_step(X_gate,
                         iteration,
                         first_iteration=(iteration == 0),
                         optimization_method=optimization_method,
                         use_posterior=use_posterior,
                         add_noise=add_noise,
                         **optimization_kwargs)
            
        end = timer()
        self.duration_fit = end - start
        if self.verbose:
            print("Duration EM fit:", self.duration_fit)

    def _e_step(self, X_gate, first_iteration):
        self.gating_values = self._gating_softmax(X=X_gate, theta_gating=self.theta_gating)
        self.gating_values = np.nan_to_num(self.gating_values)
        self._train_trees()

        self.posterior_probabilities = self._posterior_probabilties(self.DT_experts)

        # if first_iteration:
        #     self._log_values_to_array()

    def _m_step(self, X_gate, iteration, first_iteration, optimization_method, use_posterior, add_noise, **optimization_kwargs):
        theta_new = self._update_theta(X_gate, optimization_method, use_posterior, **optimization_kwargs)
        self.theta_gating += self.learn_rate[iteration] * theta_new

        if add_noise:
            self.theta_gating += self._theta_noise(first_iteration, iteration)

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

        if self.verbose is True:
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

    def train_disjoint_trees(self, tree_algorithm="sklearn"):
        # DTs trained with one-hot gates as weights
        gate = np.argmax(self.gating_values, axis=1)
        gating_values_one_hot = np.zeros([self.n_input, self.n_experts])
        gating_values_one_hot[np.arange(0, self.n_input), gate] = 1

        DT_experts_disjoint = [None for i in range(self.n_experts)]
        if tree_algorithm == "sklearn":
            if self.X_contains_categorical:
                X = self.X_one_hot
            else:
                X = self.X_original
            for index_expert in range(0, self.n_experts):
                DT_experts_disjoint[index_expert] = tree.DecisionTreeClassifier(max_depth=self.max_depth)
                DT_experts_disjoint[index_expert].fit(X=X, y=self.y, sample_weight=gating_values_one_hot[:, index_expert])
        elif tree_algorithm == "optimal_trees":
            from interpretableai import iai
            for index_expert in range(0, self.n_experts):
                mask = gating_values_one_hot[:, index_expert] == 1
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
                DT_experts_disjoint[index_expert].fit(X=self.X_original, y=self.y, sample_weight=gating_values_one_hot[:, index_expert])
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

    def _theta_noise(self, first_iteration, iteration):
        noise_threshold = 5
        if self.no_improvements >= noise_threshold:
            max_noise = np.mean(np.abs(self.theta_gating)) / 10

            noise = np.random.random_integers(-max_noise, max_noise, self.theta_gating.size).reshape(self.theta_gating.shape)
            self.no_improvements = noise_threshold - 5
            print("Noise inserted at iteration:", iteration)
            return noise
        else:
            if first_iteration:
                pass
            else:
                difference = self._accuracy_score(iteration) - self._accuracy_score(iteration - 1)
                # print(difference,np.abs(difference) < 0.0001)
                if difference < 0 or np.abs(difference) < 0.0001:
                    self.no_improvements += 1
                else:
                    self.no_improvements = 0
        return 0

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

    @lru_cache(maxsize=100)
    def _accuracy_score(self, iteration):
        predicted_labels = self.predict_hard_iteration(self.X, iteration, internal=True)
        accuracy = (np.count_nonzero(predicted_labels.astype(int) == self.y) / self.n_input)
        return accuracy

    def _accuracy_score_disjoint(self):
        predicted_labels = self.predict_hard_iteration(self.X_original, iteration=None, internal=True, disjoint_trees=True)
        accuracy = (np.count_nonzero(predicted_labels.astype(int) == self.y) / self.n_input)
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
