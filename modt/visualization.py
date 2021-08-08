import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

# Colors of the regions. Need to be visible below the scatter points.
COLOR_SCHEMA = ["#E24A33",
                "#8EBA42",    
                "#81D0DB",
                "#FBC15E",
                "#B97357",
                "#988ED5",
                "#348ABD",]   

def rand_jitter(arr):
    """Add small amount of noise to array."""
    stdev = .005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def plot_gating(modt,
                iteration,
                point_size=4,
                title=True, 
                axis_digits=False,
                axis_ticks=True,
                jitter=False,
                inverse_transform_standardization=False):

    # Colors of the regions. Need to be visible below the scatter points.
    color_schema = COLOR_SCHEMA              
    
    #plt.figure(figsize=(3,2))
    ax = plt.gca()
    y = modt.y

    if modt.use_2_dim_gate_based_on is not None:
        X = modt.X_2_dim
    else:
        X = modt.X_original
        if X.shape[1] != 2:
            raise ValueError("X must have 2 dimensions for visualization. Use 2D gate if dataset has more dimensions.")

    if jitter:
        ax.scatter(rand_jitter(X[:, 0]), rand_jitter(X[:, 1]), c=y, s=point_size, 
                        clim=(y.min(), y.max()), zorder=3)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y, s=point_size, clim=(y.min(), y.max()), zorder=3)                              
        
    ax.axis('tight')
    if not axis_ticks:
        ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim() 
    
    # Add feature names to axis
    if modt.feature_names_one_hot is not None:
        feature_names = modt.feature_names_one_hot
    else:
        feature_names = modt.feature_names
    if feature_names is None:
        print("Warning: Dataset does not include feature names.")
    else:
        names_selected_features = feature_names[modt.X_top_2_mask[:-1]]
        ax.set_xlabel(names_selected_features[0], fontsize=12)        
        ax.set_ylabel(names_selected_features[1], fontsize=12)

    # Overwrite axis ticks by reversing the standardization of the original ticks
    if inverse_transform_standardization: 
        mask = modt.X_top_2_mask[:-1]
        placeholder = np.zeros((len(ax.get_xticks()),modt.X.shape[1]-1))
        placeholder[:,mask[0]] = ax.get_xticks()
        new_x_ticks = modt.scaler.inverse_transform(placeholder)[:,mask[0]]
        new_x_ticks = np.around(new_x_ticks,1)
        ax.set_xticklabels(new_x_ticks)
        
        placeholder = np.zeros((len(ax.get_yticks()),modt.X.shape[1]-1))
        placeholder[:,mask[1]] = ax.get_yticks()
        new_y_ticks = modt.scaler.inverse_transform(placeholder)[:,mask[1]]
        new_y_ticks = np.around(new_y_ticks,1)
        ax.set_yticklabels(new_y_ticks) 
        
    if not axis_digits:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # Create 200*200 sample points. 
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if modt.use_2_dim_gate_based_on is not None:
        grid = np.append(grid, np.ones([grid.shape[0], 1]),axis=1) # Bias
        Z = modt.get_expert(grid, iteration, internal=True).reshape(xx.shape)
    else:
        Z = modt.get_expert(grid, iteration, internal=False).reshape(xx.shape)

    # Create a contour plot with the results Z -> regions
    n_classes = len(np.unique(Z))
    ax.contourf(xx, yy, Z, alpha=0.6,
                           levels=np.arange(n_classes + 1) - 0.5,
                           colors=color_schema,
                           zorder=1)
    
    if title:
        plt.title("iteration: {}".format(iteration))

def plot_initialization(modt,
                        point_size=4,
                        true_labels=False,
                        jitter=False): 
    
    #plt.figure(figsize=(3,2))
    ax = plt.gca()

    if true_labels:
        y = modt.y
    elif modt.init_labels is not None:
        y = modt.init_labels
    else:
        y = np.zeros(modt.y.shape[0])

    if modt.use_2_dim_gate_based_on is not None:
        X = modt.X_2_dim
    else:
        X = modt.X_original
        if X.shape[1] != 2:
            raise ValueError("X must have 2 dimensions for visualization. Use 2D gate if dataset has more dimensions.")

    if jitter:
        ax.scatter(rand_jitter(X[:, 0]), rand_jitter(X[:, 1]), c=y, s=point_size, 
                        clim=(y.min(), y.max()))
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y, s=point_size, clim=(y.min(), y.max()))                              
        
    ax.axis('tight')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

        


def visualize_gating(modt,
                     iteration,
                     ax=None,
                     cmap='rainbow',
                     enable_scatter=True,
                     low_alpha=False,
                     title=True, 
                     axis_digits=False,
                     axis_ticks=False):
    #plt.figure(figsize=(8,4))
    ax = plt.gca()
    y = modt.y

    if modt.use_2_dim_gate_based_on is not None:
        X = modt.X_2_dim
    else:
        X = modt.X_original
        if X.shape[1] != 2:
            raise ValueError("X must have 2 dimensions for visualization.")
    
    # Plot the training points
    if enable_scatter:
        if low_alpha:
            ax.scatter(X[:, 0], X[:, 1], c=y, s=1, alpha=0.1,
                    clim=(y.min(), y.max()), zorder=3)
        else:
            ax.scatter(X[:, 0], X[:, 1], c=y, s=1, 
                    clim=(y.min(), y.max()), zorder=3)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y, s=1, alpha=0,
                clim=(y.min(), y.max()), zorder=3)   
    ax.axis('tight')
    if not axis_digits:
        ax.axis('off')
    #xlim = [0,1]
    #ylim = [0,1]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if not axis_ticks:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))

    grid = np.c_[xx.ravel(), yy.ravel()]

    if modt.use_2_dim_gate_based_on is not None:
        grid = np.append(grid, np.ones([grid.shape[0], 1]),axis=1) # Bias
        Z = modt.get_expert(grid, iteration, internal=True).reshape(xx.shape)
    else:
        Z = modt.get_expert(grid, iteration, internal=False).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(Z))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)

    if title:
        plt.title("Iteration: {}".format(iteration))

    #return plt
    #plt.show()

def visualize_decision_area(predictor, X, y, enable_scatter=True, axis_digits=False):
    """Plot prediction class areas in 2D.""" 

    if X.shape[1] != 2:
        raise ValueError("X must have 2 dimensions.")

    ax = plt.gca()
    
    # Plot the training points
    if enable_scatter:
        ax.scatter(X[:, 0], X[:, 1], c=y, s=1,
                clim=(y.min(), y.max()), zorder=3)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y, s=1, alpha=0,
                clim=(y.min(), y.max()), zorder=3)         
    ax.axis('tight')
    if not axis_digits:
        ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = predictor(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    #plt.show()

def plot_initialization_gates(modt, point_size=3):
    plt.subplot(1, 2, 1)
    plot_initialization(modt,point_size=point_size)
    plt.subplot(1, 2, 2)
    plot_gating(modt,iteration=0,point_size=point_size,title=False,axis_digits=False,inverse_transform_standardization=False)
    # plt.subplot(1, 3, 3)
    # plot_gating(modt,iteration=modt.best_iteration,title=False,axis_digits=False,inverse_transform_standardization=False)


def plot_training(modt):
    plt.subplot(1, 3, 1)
    accuracy_line(modt)
    plt.subplot(1, 3, 2)
    plt.plot(modt.all_likelihood)
    plt.subplot(1, 3, 3)
    theta_development(modt)    

def accuracy_line(modt, color="#348ABD"):
    accuracy = modt.all_accuracies
    print("Min: ", min(accuracy), "Max: ", max(accuracy))
    plt.plot(accuracy, c=color)

def theta_development(modt):
    theta = np.array(modt.all_theta_gating)
    for theta_variable in (theta.reshape(theta.shape[0] , -1).T):
        plt.plot(theta_variable.flatten())

def plot_dt(modt, expert, size=(15,10), iteration="best",feature_names=None,class_names=None):
    if iteration is None:
        iteration = modt.iterations - 1

    plt.figure(figsize=size)
    dt = modt.all_DTs[iteration][expert]
    tree.plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True, rounded = True)

def plot_disjoint_dt(modt, expert, tree_algorithm="sklearn", size=(15,10), feature_names=None, class_names=None):

    if tree_algorithm == "sklearn":
        plt.figure(figsize=size)
        try:  # Plotting fails if filled=True and tree empty
            tree.plot_tree(modt.DT_experts_disjoint[expert], feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
        except: 
            tree.plot_tree(modt.DT_experts_disjoint[expert])

    elif tree_algorithm == "optimal_trees":
        for dt_tree in modt.DT_experts_disjoint:
            dt_tree.get_learner()
    else:
        raise Exception("")
