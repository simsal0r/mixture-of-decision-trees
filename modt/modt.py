import numpy as np
import pandas as pd
from timeit import default_timer as timer
import pickle

from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

from modt._initialization import *

class MoDT():

    def __init__(self,
                 X,
                 y,
                 n_experts,
                 iterations,
                 max_depth,
                 init_learning_rate=100,
                 learning_rate_decay=0.995,
                 initialization_method="random",
                 use_2_dim_gate_based_on=None,
                 use_2_dim_clustering=False,
                 black_box_algorithm=None,
                 feature_names=None,
                 class_names=None,
                 save_likelihood=False,
                 verbose=False):

        self.verbose = verbose
        self.verbose_detailed = False
        self.X_contains_categorical = False
        self.save_likelihood = save_likelihood

        if np.array(X).ndim == 1:
            raise ValueError("X must have at least 2 dimensions.")

        self.n_features_of_X = X.shape[1]
        self.n_input = X.shape[0]
        self.y_map = None
        self.n_experts = n_experts
        self.max_depth = max_depth
        self.iterations = iterations
        self.initialization_method = initialization_method
        self.init_learning_rate = init_learning_rate
        self.use_2_dim_gate_based_on = use_2_dim_gate_based_on
        self.use_2_dim_clustering = use_2_dim_clustering
        self.learning_rate_decay = learning_rate_decay
        self.learn_rate = [self.init_learning_rate * (self.learning_rate_decay ** i) for i in range(iterations)]

        self._check_argument_validity()

        (self.X,  # Bias and standardization will be added later
         self.X_original,  # Original input as numpy array 
         self.X_original_pd,
         self.X_one_hot, # TODO: Remove
         self.y,
         self.y_original,
         self.feature_names,
         self.feature_names_one_hot,
         self.class_names
         ) = self._interpret_input(X, y, feature_names, class_names)

        if black_box_algorithm is not None:
            self.y_before_surrogate = self.y
            self.y = self._transform_y_with_surrogate_model(black_box_algorithm)

        self.scaler = self._create_scaler(self.X)  # Create standardization model. Also needed for prediction of new observations.
        self.X = self._preprocess_X(self.X)  # Apply standardization and add bias

        self.X_top_2_mask = None
        self.X_2_dim = None
        self._setup_2_dimensional_gate()  # Sets the above variables


    def _initialize_fitting_variables(self):
        """Initalize model variables that are need for the fitting process"""

        self.gating_values = None
        self.DT_experts = None
        self.DT_experts_disjoint = None
        self.DT_experts_alternative_algorithm = None
        self.all_DTs = []
        self.all_theta_gating = []
        self.all_gating_values = []
        self.all_accuracies = []
        self.all_likelihood = []
        self.best_iteration = None
        self.completed_iterations = None
        #self.counter_stale_iterations = 0
        self.posterior_probabilities = None
        self.confidence_experts = None
        self.counter_stale_iterations = 0  

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
        
        # Initialize gating values
        self.theta_gating = self._initialize_theta(self.initialization_method)
        self.init_theta = self.theta_gating.copy()

    def _check_argument_validity(self):
        if self.use_2_dim_gate_based_on is None and self.use_2_dim_clustering:
            raise ValueError("Cant use 2D initalization if no initialization is selected.")
        if self.use_2_dim_clustering != False and self.use_2_dim_clustering != True: 
            raise ValueError("use_2_dim_clustering must be True or False")            
        if self.n_experts <= 0:
            raise ValueError("More than 0 experts required.")
        if self.iterations < 0:
            raise ValueError("At least 1 iteration is necessary.")

    def _interpret_input(self, X, y, feature_names, class_names):
        X_one_hot = None
        X_original_pd = None
        feature_names_new = None
        feature_names_one_hot = None

        if isinstance(X, pd.core.frame.DataFrame):  # Pandas
            # Categorical treatment
            if np.intersect1d(['object', 'category'], X.dtypes.values.astype(str)).size > 0:
                self.X_contains_categorical = True
                X_new, feature_names_one_hot = self._one_hot_encode(X)
            else:
                X_new = np.array(X)
            X_original = np.array(X)
            X_original_pd = X
            feature_names_new = np.array(X.columns)

        elif isinstance(X, np.ndarray):  # Numpy
            X_new = X
            X_original = X
            # TODO: Insert categorical treatment

            if feature_names is not None:
                if len(feature_names) != X.shape[1]:
                    raise Exception("Feature names list length ({}) inconsistent with X ({}).".format(len(feature_names), X.shape[1]))
                feature_names_new = np.array(feature_names) 

        else:
            raise Exception("X must be Pandas DataFrame or Numpy array.")

        y_new, y_original, class_names_new = self._interpret_input_y(y)

        if class_names is not None:
            class_names_new = np.array(class_names)
            class_names_new = class_names_new.astype(str)

        return X_new, X_original, X_original_pd, X_one_hot, y_new, y_original, feature_names_new, feature_names_one_hot, class_names_new

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
        """Convert unique values of categorical features into new binary (one-hot) features"""
        X_one_hot = pd.get_dummies(X, columns=list(X.select_dtypes(include=['object', 'category']).columns))
        X_one_hot_columns = X_one_hot.columns
        X_one_hot = np.array(X_one_hot)
        return X_one_hot, X_one_hot_columns

    def _create_scaler(self, X):
        """Create a standardization model."""
        return(StandardScaler().fit(X))

    def _preprocess_X(self, X):
        """Apply standardization and add bias column."""
        X = self.scaler.transform(X)
        return np.append(X, np.ones([X.shape[0], 1]), axis=1)  # Add bias                

    def _setup_2_dimensional_gate(self):
        """Delegate gate dimensionality reduction."""

        self.X_top_2_mask = self._get_2_dim_feature_importance_mask(method="DT")  # Always calculate Top 2 features for plotting; 2D + Bias
        self.X_2_dim = self.X[:, self.X_top_2_mask]  # For plotting; 2D + bias; components in case of PCA

        if self.use_2_dim_gate_based_on is not None:
            if isinstance(self.use_2_dim_gate_based_on, list):
                X_top_2_mask = self.use_2_dim_gate_based_on
                X_top_2_mask.append(-1)
                self.X_top_2_mask = X_top_2_mask
                self.X_2_dim = self.X[:, self.X_top_2_mask] 
            elif self.use_2_dim_gate_based_on == "feature_importance":
                pass  # This default choice has been calculated above.
            elif self.use_2_dim_gate_based_on == "PCA":
                self.X_2_dim = self._perform_PCA()
            else:
                if self.use_2_dim_gate_based_on == "feature_importance_lda":
                    self.X_top_2_mask = self._get_2_dim_feature_importance_mask(method="LDA")                       
                elif self.use_2_dim_gate_based_on == "feature_importance_lda_max":
                    self.X_top_2_mask = self._get_2_dim_feature_importance_mask(method="LDA_max")                       
                elif self.use_2_dim_gate_based_on == "feature_importance_lr":
                    self.X_top_2_mask = self._get_2_dim_feature_importance_mask(method="LR")  
                elif self.use_2_dim_gate_based_on == "feature_importance_lr_max":
                    self.X_top_2_mask = self._get_2_dim_feature_importance_mask(method="LR_max")  
                elif self.use_2_dim_gate_based_on == "feature_importance_xgb":
                    self.X_top_2_mask = self._get_2_dim_feature_importance_mask(method="XGB")  
                elif self.use_2_dim_gate_based_on == "feature_importance_pca_loadings":
                    self.X_top_2_mask = self._get_2_dim_feature_importance_mask(method="PCA_loadings")  
                else:
                    raise Exception("Invalid method for gate dimensionality reduction.")
                self.X_2_dim = self.X[:, self.X_top_2_mask] 


    def _get_2_dim_feature_importance_mask(self, method="DT"):

        X_without_bias = self.X[:,:-1]

        if method == "DT":
            clf = tree.DecisionTreeClassifier()
            clf.fit(X_without_bias, self.y)
            importances = clf.feature_importances_
        elif method == "XGB":
            clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
            clf.fit(X_without_bias, self.y)
            importances = clf.feature_importances_
        elif method == "LDA": 
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_without_bias, self.y)
            importances = np.sum(np.abs(clf.coef_), axis=0) / np.sum(np.sum(np.abs(clf.coef_), axis=0))
        elif method == "LDA_max": 
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_without_bias, self.y)
            importances = np.max(np.abs(clf.coef_) ,axis=0) / np.sum(np.max(np.abs(clf.coef_), axis=0))
        elif method == "LR": 
            clf = LogisticRegression(solver='liblinear')
            clf.fit(X_without_bias, self.y)
            importances = np.sum(np.abs(clf.coef_), axis=0) / np.sum(np.sum(np.abs(clf.coef_), axis=0))
        elif method == "LR_max": 
            clf = LogisticRegression(solver='liblinear')
            clf.fit(X_without_bias, self.y)
            importances = np.max(np.abs(clf.coef_), axis=0) / np.sum(np.max(np.abs(clf.coef_), axis=0))
        elif method == "PCA_loadings":
            pca = PCA(n_components=2).fit(X_without_bias)
            loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))

            #  Features with the hightest correlation in the first two components
            idx_feature1 = loadings[0].sort_values(ascending=False).index[0]
            idx_feature2 = loadings[1].sort_values(ascending=False).index[0]

            if idx_feature1 == idx_feature2:
                idx_feature2 = loadings[0].sort_values(ascending=False).index[1]

            mask = [idx_feature1, idx_feature2, -1] # Add bias (last) column
            return mask
        else:
            raise ValueError("Invalid method for feature importance.")

        importances = -importances  # Reverse sign for sorting convenience 
        features_10 = []  # Preferably use features that have more than 10 unique values
        for column in range(X_without_bias.shape[1]):
            if np.unique(X_without_bias[:, column]).size > 10:
                features_10.append(column)
            else:
                pass

        if len(features_10) > 2:
            top_features_idx_all = np.argsort(importances)
            mask = np.repeat(False, X_without_bias.shape[1])
            features_10_idx = []

            for feature in features_10:
                features_10_idx.append(np.where(top_features_idx_all == feature)[0][0])

            mask[features_10_idx] = True
            top_features_idx = top_features_idx_all[mask][:2]  # Select 2 best features
            self.top_features_idx = top_features_idx

            if self.verbose:
                print("Top 2 feature importance with features that have at least 10 unique values:", importances[top_features_idx])
                print("Top 2 feature importance including features w/ fewer than 10 unique values:", importances[np.argsort(importances)[:2]])
        else:
            self.top_features_idx = np.argsort(importances)[:2]
            if self.verbose:
                print("Top 2 feature importance including features w/ fewer than 10 unique values:", importances[np.argsort(importances)[:2]])

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
        if method != "PCA":
            X = X[:, self.X_top_2_mask]
        else:
            X = self.pca.transform(X)
            X = np.append(X, np.ones([X.shape[0], 1]), axis=1)  # Bias

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

    def _initialize_theta(self, initialization_method="random"):
        start = timer()

        if initialization_method == "random" or isinstance(initialization_method, Random_init):
            if self.use_2_dim_gate_based_on is not None:
                n_features = 3
            else:
                n_features = self.X.shape[1]
            if isinstance(initialization_method, Random_init):
                np.random.seed(initialization_method.seed)
            initialized_theta = np.random.rand(n_features, self.n_experts)
            
        elif (isinstance(initialization_method, Kmeans_init) or 
              isinstance(initialization_method, KDTmeans_init) or
              isinstance(initialization_method, DBSCAN_init) or
              isinstance(initialization_method, Boosting_init) or
              isinstance(initialization_method, BGM_init)
             ):
            initialized_theta = initialization_method._calculate_theta(self)
        else:
            raise Exception("Invalid initalization method specified.")

        end = timer()
        self.duration_initialization = end - start
        if self.verbose:
            print("Duration initialization:", self.duration_initialization)

        return initialized_theta

    def _reapply_feature_encoding(self, X):

        X = pd.get_dummies(X, columns=list(X.select_dtypes(include=['object', 'category']).columns))
        unseen_features_of_prediction_input = np.setdiff1d(X.columns, self.feature_names_one_hot)
        missing_features_of_prediction_input = np.setdiff1d(self.feature_names_one_hot, X.columns)

        if unseen_features_of_prediction_input.size > 0:
            X = X.drop(columns=unseen_features_of_prediction_input, axis=1)
        if missing_features_of_prediction_input.size > 0:
            X[missing_features_of_prediction_input] = 0

        return np.array(X)  
 
    def predict(self, X):
        iteration = self.best_iteration

        if self.X_contains_categorical:
            X = self._reapply_feature_encoding(X)
           
        X_gate = self._preprocess_X(X)        
        if self.use_2_dim_gate_based_on is not None:  # 2D gating function
            X_gate = self._transform_X_into_2_dim_for_prediction(X_gate, method=self.use_2_dim_gate_based_on)

        DTs = self.DT_experts_disjoint        
        gating = self._gating_softmax(X_gate, self.all_theta_gating[iteration])
        selected_gates = np.argmax(gating, axis=1)

        predictions = [DTs[tree_index].predict(X) for tree_index in range(0, len(DTs))]
        predictions_gate_selected = np.array([predictions[gate][index] for index, gate in enumerate(selected_gates)]).astype("int")
        return self._map_y(predictions_gate_selected)

    def predict_internal(self, iteration, return_complete=False):
        if self.use_2_dim_gate_based_on is not None:  # 2D gating function
            X_gate = self._transform_X_into_2_dim_for_prediction(self.X, method=self.use_2_dim_gate_based_on)
        else:
            X_gate = self.X

        DTs = self.all_DTs[iteration]
        gating = self._gating_softmax(X_gate, self.all_theta_gating[iteration])
        selected_gates = np.argmax(gating, axis=1)

        predictions = [DTs[tree_index].predict(self.X) for tree_index in range(0, len(DTs))]
        if return_complete:
            return np.array(predictions)
        else:
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

    def fit(self, optimization_method="least_squares_linear_regression", early_stopping=True, use_posterior=False, **optimization_kwargs):
        
        self._initialize_fitting_variables()
        if early_stopping is True:
            early_stopping = "likelihood"

        start = timer()

        _, X_gate = self._select_X_internal()
        self.completed_iterations = self.iterations - 1

        for iteration in range(0, self.iterations):

            self._e_step(X_gate)
            self._log_values_to_array()  # After E step: Theta, DTs and gating values are in sync
            
            self.all_accuracies.append(self.score_internal(iteration=iteration))
            if self.save_likelihood or early_stopping == "likelihood":
                self.all_likelihood.append(self._likelihood())
            if early_stopping == "likelihood" or early_stopping == "accuracy":
                if self._check_for_convergence(iteration, based_on=early_stopping):
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

        self.best_iteration = self.argmax_last(self.all_accuracies)
        self.train_disjoint_trees(iteration=self.best_iteration, tree_algorithm="sklearn_default")
            
        end = timer()
        self.duration_fit = end - start
        if self.verbose:
            print("Duration EM fit:", self.duration_fit)

    def _e_step(self, X_gate):
        """
        E-step of the EM algorithm. 
        Calculate gating values from theta.
        Train DTs.
        Calculate expectation.
        """

        self.gating_values = self._gating_softmax(X=X_gate, theta_gating=self.theta_gating)
        self.gating_values = np.nan_to_num(self.gating_values)
        self._train_trees()

        self.posterior_probabilities = self._posterior_probabilties()

    def _m_step(self, X_gate, iteration, optimization_method, use_posterior, **optimization_kwargs):
        """M-step of the EM algorithm. Update current theta."""

        theta_new = self._update_theta(X_gate, optimization_method, use_posterior, **optimization_kwargs)
        self.theta_gating += self.learn_rate[iteration] * theta_new

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

    def _posterior_probabilties(self, return_predictions=False):
        """Expectation of the E-step"""

        confidence_correct = np.zeros([self.n_input, self.n_experts]) 
        for expert_index in range(0, self.n_experts):
            dt = self.DT_experts[expert_index]
            dt_probability = dt.predict_proba(self.X)
            confidence_correct[:, expert_index] = dt_probability[np.arange(self.n_input), self.y.flatten().astype(int)]

        multiplication = self.gating_values * confidence_correct
        if return_predictions:
            return multiplication / np.expand_dims(np.sum(multiplication, axis=1), axis=1), confidence_correct
        else:
            return multiplication / np.expand_dims(np.sum(multiplication, axis=1), axis=1)        

    def _update_theta(self, X_gate, optimization_method, use_posterior, **optimization_kwargs):
        """Calculate an updated theta that will be added to the current theta."""

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
        elif optimization_method == "matmul":
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

    def train_disjoint_trees(self, iteration, tree_algorithm):
        """Train DTs on separate subsets""" 

        gating_values = self.all_gating_values[iteration]
        gate = np.argmax(gating_values, axis=1)
        gating_values_hard = np.zeros([self.n_input, self.n_experts])
        gating_values_hard[np.arange(0, self.n_input), gate] = 1
        DT_experts_disjoint = [None for _ in range(self.n_experts)]
        DT_experts_alternative_algorithm = [None for _ in range(self.n_experts)]

        if tree_algorithm == "sklearn_default":
            if self.X_contains_categorical:
                X, _ = self._one_hot_encode(self.X_original_pd)
            else:
                X = self.X_original
            for index_expert in range(0, self.n_experts):
                DT_experts_disjoint[index_expert] = tree.DecisionTreeClassifier(max_depth=self.max_depth)
                DT_experts_disjoint[index_expert].fit(X=X, y=self.y, sample_weight=gating_values_hard[:, index_expert])

            self.DT_experts_disjoint = DT_experts_disjoint

        # Alternative algorithms

        elif tree_algorithm == "optimal_trees":

            from interpretableai import iai  #  Commercial software

            for index_expert in range(self.n_experts):
                mask = gating_values_hard[:, index_expert]
                mask = mask.astype(bool)
                X = self.X_original_pd[mask].copy()  
                y = self.y_original[mask].copy()  

                 # Optimal trees cannot handle object dtypes
                X.loc[:, X.dtypes == 'object'] = X.select_dtypes(['object']).apply(lambda x: x.astype('category')) 

                DT_experts_alternative_algorithm[index_expert] = iai.GridSearch(iai.OptimalTreeClassifier(), max_depth=self.max_depth)
                DT_experts_alternative_algorithm[index_expert].fit(X, y)
            
            self.DT_experts_alternative_algorithm = DT_experts_alternative_algorithm

        elif tree_algorithm == "h2o":

            # Allows multi-class but performs binary prediction

            from modt._alternative_DTs import H2o_classifier

            server = H2o_classifier(max_depth = -1)
            server.start_server()

            for index_expert in range(0, self.n_experts):
                mask = gating_values_hard[:, index_expert]
                mask = mask.astype(bool)
                X = self.X_original_pd[mask].copy()  
                y = self.y_original[mask].copy()  

                DT_experts_alternative_algorithm[index_expert] = H2o_classifier(max_depth=self.max_depth)
                DT_experts_alternative_algorithm[index_expert].fit(X=X, y=y, expert_identifier=index_expert)
                DT_experts_alternative_algorithm[index_expert].plot()

            self.DT_experts_alternative_algorithm = DT_experts_alternative_algorithm

            server.stop_server()

        else:
            raise Exception("Invalid tree algorithm.")

    def _theta_recalculation_moet2(self, posterior_probabilities, gating, X):
        """Computationally expensive; used in MOET paper"""
        R = np.zeros([self.n_experts * self.n_features_of_X, self.n_experts * self.n_features_of_X])

        for idx_input in range(self.n_input):
            for idx_expert in range(self.n_experts):
                pom1 = gating[idx_input, idx_expert] * (1 - gating[idx_input, idx_expert])
                pom2 = np.zeros([self.n_experts, self.n_features_of_X])
                pom2[idx_expert] = X[idx_input]
                pom2 = pom2.reshape(-1, 1)  # Flatten
                R += pom1 * (pom2 @ pom2.T)

        e = self._update_theta(X, optimization_method="matmul", use_posterior=False)

        if np.linalg.cond(R) < 1e7:
            return (np.linalg.inv(R) @ e.flatten()).reshape(-1, self.n_experts)
        else:
            return e

    def _check_for_convergence(self, iteration, based_on="likelihood"):
        max_stale_iterations = 20
        min_iterations = 5
        if self.counter_stale_iterations >= max_stale_iterations:
            return True
        if iteration >= min_iterations:
            if based_on == "likelihood":
                difference = self.all_likelihood[iteration] - self.all_likelihood[iteration-1]
                difference_cycle = self.all_likelihood[iteration] - self.all_likelihood[iteration-2]
                if np.abs(difference) < self.all_likelihood[iteration-2] * 0.0025 or np.abs(difference_cycle) < self.all_likelihood[iteration-2] * 0.0025:
                    self.counter_stale_iterations += 1
                else:
                    self.counter_stale_iterations = 0
            elif based_on == "accuracy":                
                difference = self.all_accuracies[iteration] - self.all_accuracies[iteration-1]
                difference_cycle = self.all_accuracies[iteration] - self.all_accuracies[iteration-2]
                if np.abs(difference) < 0.0001 or np.abs(difference_cycle) < 0.0001:
                    self.counter_stale_iterations += 1
                else:
                    self.counter_stale_iterations = 0                
            else:
                raise ValueError("Invalid method for convergence check. Must be accuracy or likelihood.")
            # print(difference,np.abs(difference) < 0.0001)

        return False

    def score(self, X , y):
        """Calculate prediction accuracy on a possibly new dataset."""
        if len(X) != len(y):
            raise ValueError("X and y have different lengths.")

        predicted_labels = self.predict(X)
        accuracy = (np.count_nonzero(predicted_labels == np.array(y).flatten()) / len(X))
        return accuracy

    def score_internal(self, iteration):
        """Calculate prediction accuracy on the training data for specific iteration."""
        predicted_labels = self.predict_internal(iteration)
        accuracy = (np.count_nonzero(predicted_labels == self.y) / self.n_input)
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

    def _likelihood(self):
        """
        Negative likelihood function / loss.
        Formula taken from: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6215056 (Twenty Years of Mixture of Experts).
        """

        g = self.gating_values
        h, predictions_experts  = self._posterior_probabilties(return_predictions=True)
        loss = - np.sum(np.sum(h * (np.log(g + 1e-05) + np.log(predictions_experts + 1e-05)), axis=1))
        return loss

        # if hard_loss:
        #     g_argmax = np.argmax(g, axis=1)
        #     g_hard = np.zeros((g.shape[0],g.shape[1]))
            
        #     for idx in range(g_hard.shape[0]):
        #         g_hard[idx, g_argmax[idx]] = 1

        #     loss_h = - np.sum(np.sum(h * (np.log(g_hard + 1e-05) + np.log(predictions_experts + 1e-05)), axis=1))
        #     return loss, loss_h

    @staticmethod
    def argmax_last(list):
        """
        Returns the index of the maximum value of a list.
        If there are multiple maxima, the last value is chosen.
        """
        reverse_list = list[::-1]
        return len(reverse_list) - np.argmax(reverse_list) - 1
