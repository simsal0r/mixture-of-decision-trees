import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.utils.multiclass import unique_labels


def _fit_theta(self_modt,X_gate,labels,theta_fittig_method):
    self_modt.init_labels = labels
    if theta_fittig_method == "lda":
        return _theta_calculation_lda(self_modt,X_gate,labels)
    elif theta_fittig_method == "lr":
        return _theta_calculation_lr(self_modt,X_gate,labels)
    else:
        raise Exception("Invalid theta fitting method. Use lda or lr.")

def _get_desired_theta_dimensions(self_modt):
    if self_modt.use_2_dim_gate_based_on is not None:
        n_features = 3
    else:
        n_features = self_modt.X.shape[1]
    return (n_features, self_modt.n_experts)

def _random_initialization_fallback(shape):
    return np.random.rand(shape[0], shape[1])

def _theta_calculation_lr(self_modt,X,y):
    if np.sum(np.in1d(y, np.arange(0,len(np.unique(y)))) == False) > 0:
        if self_modt.verbose:
            print("Initialization label names contain gaps, factorizing using pandas...")
        y = pd.factorize(y)[0]
    expert_target_matrix = np.zeros((self_modt.n_input,self_modt.n_experts))
    expert_target_matrix[np.arange(0,self_modt.n_input),y[np.arange(0,self_modt.n_input)]] = 1
    lr = LinearRegression(fit_intercept=False).fit(X,expert_target_matrix)
    if self_modt.verbose:
        print("Initialization LR score:", lr.score(X, expert_target_matrix))

    return lr.coef_.T

def _theta_calculation_lda(self_modt, X, y):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    if self_modt.verbose:
        print("Initialization LDA score:", clf.score(X, y))
    if self_modt.n_experts == 2: # special case 2 experts; here both regions are separeted by the same discriminant
        theta = np.zeros((X.shape[1], self_modt.n_experts))
        theta[:,0] = clf.coef_
        theta[:,1] = clf.coef_ * -1
    else:
        theta = clf.coef_.T
    theta[-1] = clf.intercept_ # Replace bias row with intercept of LDA

    desired_shape = _get_desired_theta_dimensions(self_modt)

    if theta.shape != (desired_shape):
        if self_modt.verbose or True:  # TODO: Change
            print("LDA separation unsuccessful. Gate initialized randomly.")
        return _random_initialization_fallback(desired_shape)

        # n_experts = clf.coef_.T.shape[1]
        # if n_experts != self_modt.n_experts:
        #     print("LDA has eliminated an empty region. Setting number of experts to {}.".format(n_experts))
        #     self_modt.n_experts = n_experts

    
    return theta

class Random_init():

    def __init__(self, seed=None):
        self.seed = seed

class Kmeans_init():

    def __init__(self,theta_fittig_method="lda"):
        self.theta_fittig_method = theta_fittig_method

    def _calculate_theta(self,self_modt):
        X, X_gate = self_modt._select_X_internal()
        kmeans = KMeans(n_clusters=self_modt.n_experts).fit(X)
        labels = kmeans.labels_
        self_modt.init_labels = labels
        #expert_target_matrix = np.zeros((self_modt.n_input, self_modt.n_experts))
        #expert_target_matrix[np.arange(0, self_modt.n_input), labels[np.arange(0, self_modt.n_input)]] = 1

        return _fit_theta(self_modt, X_gate, labels, self.theta_fittig_method)

    
class KDTmeans_init():

    def __init__(self,alpha=1,beta=0.05,gamma=0.1,theta_fittig_method="lda"):
        self.theta_fittig_method = theta_fittig_method
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def _calculate_theta(self,self_modt):

        X, X_gate = self_modt._select_X_internal()
        n_features = X.shape[1]

        n_cluster = self_modt.n_experts
        kDTmeans_iterations = 20 #Test 1

        kmeans = KMeans(n_clusters=self_modt.n_experts).fit(X)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        self_modt.all_cluster_labels.append(labels)
        self_modt.all_cluster_centers.append(cluster_centers)

        for iteration in range(0,kDTmeans_iterations):    
            DT_clusters = [None for i in range(n_cluster)]
            updated_cluster_distances = np.zeros((self_modt.n_input,n_cluster))
            
            for cluster_idx in range(0,n_cluster):
                distances = distance.cdist([cluster_centers[cluster_idx,:]], X, 'euclidean').flatten()
                weights = 1.0/(distances**self.alpha + self.beta)

                DT_clusters[cluster_idx] = tree.DecisionTreeClassifier(max_depth = 2)
                DT_clusters[cluster_idx].fit(X=X, y=self_modt.y, sample_weight=weights)

                confidence = DT_clusters[cluster_idx].predict_proba(X=X)[np.arange(self_modt.n_input),self_modt.y]
                updated_cluster_distances[:,cluster_idx] = distances / (confidence + self.gamma)
                
            self_modt.all_DT_clusters.append(DT_clusters.copy())    
            cluster_labels = np.argmin(updated_cluster_distances,axis=1)
            new_centers = np.zeros((n_cluster,n_features))
            
            for cluster_idx in range(0,n_cluster):
                new_centers[cluster_idx,:] = np.mean(X[cluster_labels==cluster_idx,:],axis=0)
                
            ## Plotting & Debugging ## 
            DT_predictions = np.zeros((self_modt.n_input,n_cluster))
            for cluster_idx in range(0,n_cluster):    
                DT_predictions[:,cluster_idx] = DT_clusters[cluster_idx].predict(X=X)
            predicted_labels = DT_predictions[np.arange(self_modt.n_input),cluster_labels] #TODO: does not work for more than 2 clusters use f1 or sth
            accuracy = (np.count_nonzero(predicted_labels.astype(int) == self_modt.y) / self_modt.n_input)
            
            self_modt.all_clustering_accuracies.append(accuracy)
            self_modt.all_cluster_labels.append(cluster_labels)   
            self_modt.all_cluster_centers.append(new_centers)
            ## ----- ##
            
            if np.allclose(cluster_centers,new_centers):
                if self_modt.verbose:
                    print("Convergence at iteration",iteration)
                break
            else:
                cluster_centers = new_centers
        
        return _fit_theta(self_modt, X_gate, labels, self.theta_fittig_method)

class DBSCAN_init():

    def __init__(self,theta_fittig_method="lda",eps=0.035,min_samples=25):
        self.theta_fittig_method = theta_fittig_method
        self.eps = eps
        self.min_samples = min_samples

    def _calculate_theta(self,self_modt):
        X, X_gate = self_modt._select_X_internal()

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < self_modt.n_experts:
            raise Exception("DBSCAN parameters yield {} clusters but at least {} required".format(n_clusters, self_modt.n_experts))
        self_modt.init_labels = labels  
        core_samples_mask = np.zeros_like(labels, dtype=bool)  # Array of 0
        #core_samples_mask[db.core_sample_indices_] = True  # Core samples exclude outliers and weak cluster members
        mask = core_samples_mask

        # Get only one cluster for each expert, exclude noise cluster (-1) if top cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        top_label_indices = counts.argsort()[-(self_modt.n_experts + 1):][::-1]
        top_labels = unique_labels[top_label_indices]
        if -1 in top_labels:
            top_labels = top_labels[top_labels != -1]
        else:
            top_labels = top_labels[:-1]
        mask[np.isin(labels, top_labels)] = True
        self_modt.dbscan_mask = mask  # Plotting
        no_small_labels = labels[mask]

        no_small_labels_temp = no_small_labels.copy()
        # Rename clusters starting with 0
        for number, unique_label in enumerate(top_labels):
            no_small_labels[no_small_labels_temp == unique_label] = number

        self_modt.dbscan_selected_labels = no_small_labels

        return _fit_theta(self_modt, X_gate[mask], no_small_labels, self.theta_fittig_method)

class Boosting_init():

    def __init__(self,theta_fittig_method="lda"):
        self.theta_fittig_method = theta_fittig_method

    def _calculate_theta(self,self_modt):
        X, X_gate = self_modt._select_X_internal()

        DTC = tree.DecisionTreeClassifier(max_depth=self_modt.max_depth)
        clf = AdaBoostClassifier(base_estimator=DTC, n_estimators=self_modt.n_experts)
        clf.fit(X, self_modt.y)
        if self_modt.verbose:
            print("AdaBoost model score:", clf.score(X, self_modt.y))

        confidence_correct = np.zeros([self_modt.n_input, self_modt.n_experts])
        for expert_index in range(0, self_modt.n_experts):
            dt = clf.estimators_[expert_index]
            dt_probability = dt.predict_proba(X)
            confidence_correct[:, expert_index] = dt_probability[np.arange(self_modt.n_input), self_modt.y.flatten().astype(int)]

            # Set the most confident expert to 1; the others to 0.
        labels = np.argmax(confidence_correct, axis=1)
        self_modt.init_labels = labels

        return _fit_theta(self_modt, X_gate, labels, self.theta_fittig_method)


class BGM_init():
    def __init__(self,
                 theta_fittig_method="lda",
                 n_components=7,
                 covariance_type="full",
                 init_params="random",
                 max_iter=500,
                 mean_precision_prior=0.8,
                 weight_concentration_prior_type="dirichlet_distribution",
                 weight_concentration_prior=0.25,
                 weight_cutoff=0.05):
        self.theta_fittig_method = theta_fittig_method
        self.n_components = n_components
        self.covariance_type = covariance_type 
        self.init_params = init_params
        self.max_iter = max_iter
        self.mean_precision_prior = mean_precision_prior
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.weight_cutoff = weight_cutoff
                    
    def _calculate_theta(self,self_modt):

        X, X_gate = self_modt._select_X_internal()

        parameters = {
                       "n_components" : self_modt.n_experts,  # method can result in fewer but not more experts
                       "covariance_type" : self.covariance_type,
                       "init_params" : self.init_params,
                       "max_iter" : self.max_iter,
                       "mean_precision_prior" : self.mean_precision_prior,
                       "weight_concentration_prior_type" : self.weight_concentration_prior_type,  # dirichlet_distribution -> more uniform than dirichlet_process
                       "weight_concentration_prior" : self.weight_concentration_prior  # Lower -> Fewer components
        }

        try:
            bgm = BayesianGaussianMixture(**parameters)
            bgm.fit(X)
        except ValueError:
            try:
                print("Covariance matrix ill-defined, increasing reg_covar...")
                parameters["reg_covar"] = 1e-5
                bgm = BayesianGaussianMixture(**parameters)
                bgm.fit(X)
            except ValueError:
                print("Covariance matrix ill-defined, initializing randomly...")
                shape = _get_desired_theta_dimensions(self_modt)
                return _random_initialization_fallback(shape)

        probabilities = bgm.predict_proba(X)
        probabilities[:, bgm.weights_ < self.weight_cutoff] = -1
        labels = np.argmax(probabilities, axis=1)  # Label number range can have gaps
        
        # Rename labels starting with 0
        labels_temp = labels.copy()
        unique_labels = np.unique(labels_temp)
        for number, unique_label in enumerate(unique_labels):
            labels[labels_temp == unique_label] = number
        self_modt.init_labels = labels

        n_experts = np.sum(bgm.weights_ >= self.weight_cutoff)
        if self_modt.verbose:
            print("BGM estimates {} experts of max {} in iteration {}".format(n_experts, self_modt.n_experts, bgm.n_iter_))
            if (np.sum(np.sum(probabilities, axis=1) == 0) != 0):
                print("Warning: Some datapoints have zero weight in the BGM model.")
        if n_experts < 2:
            n_experts = 2
            print("BGM estimated less than 2 experts. Resetting to 2. Increasing weight_concentration_prior might yield more experts.")
        
        self_modt.n_experts = n_experts

        return _fit_theta(self_modt,X_gate,labels,self.theta_fittig_method)



        