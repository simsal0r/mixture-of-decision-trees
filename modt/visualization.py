import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

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


def accuracy_line(modt, color="#348ABD"):
    # accuracy = []
    # for iteration in range(0,modt.iterations):
    #     accuracy.append(modt.accuracy_score(iteration))
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
