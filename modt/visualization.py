import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dtreeviz.trees import *

import numpy as np
import pandas as pd
from sklearn import tree

from modt.utility import pickle_disjoint_data

# Colors of the regions. Need to be visible below the scatter points.
COLOR_SCHEME_REGIONS = ["#E24A33",
                        "#8EBA42",    
                        "#81D0DB",
                        "#FBC15E",
                        "#B97357",
                        "#988ED5",
                        "#348ABD",
                        "#808a89",]  

COLOR_SCHEME_SCATTER = ["#440154", # dark purple
                        "#21918c", # blue green
                        "#fde725", # neon yellow
                        "#5ec962", # green
                        "#ffbaab", # pastel
                        "#ff0000", # red
                        "#f7f7f7", # white grey
                        "#f700ff", # pink
                        "#4dff00", # ugly green
                        "#000000", # black
                        "#3b528b", # blue
                        "#276921", # dark green
                        "#87ffdf", # turquoise
                        "#016685", # dark blue
                        "#313985", # dark purple blue
                        ]                      

def color_coder(x):  
    idx = x % len(COLOR_SCHEME_REGIONS)
    return COLOR_SCHEME_REGIONS[idx]

def color_coder_scatter(x):  
    idx = x % len(COLOR_SCHEME_SCATTER)
    return COLOR_SCHEME_SCATTER[idx]

def rand_jitter(arr):
    """Add small amount of noise to array."""
    stdev = .005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def plot_gating(modt,
                iteration,
                point_size=4,
                rasterize=False,
                title=True, 
                axis_digits=False,
                axis_ticks=True,
                jitter=False,
                inverse_transform_standardization=False,
                legend=False,
                legend_classes=False):
            
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
        ax.scatter(rand_jitter(X[:, 0]), rand_jitter(X[:, 1]), color=list(map(color_coder_scatter, y)), s=point_size, clim=(y.min(), y.max()), zorder=3, rasterized=rasterize)
    else:
        ax.scatter(X[:, 0], X[:, 1], color=list(map(color_coder_scatter, y)), s=point_size, clim=(y.min(), y.max()), zorder=3, rasterized=rasterize)                              
        
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
        ...
        #print("Warning: Dataset does not include feature names.")
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
        grid = np.append(grid, np.ones([grid.shape[0], 1]), axis=1) # Bias

    Z = modt.get_expert(grid, iteration, plotting=True).reshape(xx.shape) 

    # Create a contour plot with the results Z -> regions
    n_classes = len(np.unique(Z))
    ax.contourf(xx, yy, Z, alpha=0.6,
                           levels=np.arange(modt.n_experts + 1) - 0.5,
                           colors=COLOR_SCHEME_REGIONS,
                           zorder=1)

    if legend:
        legend_elements = []
        for region in np.unique(Z):
            legend_elements.append(Line2D([], [], color=color_coder(region), marker='$\\blacksquare$', linestyle='None', markersize=12, label="DT {}".format(str(region))))
        legend_regions = ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1.02), loc="upper left")
        ax.add_artist(legend_regions)
        
    # Class names
    if modt.class_names is None:
        class_names = list(np.unique(modt.y_original))
    else:
        class_names = list(modt.class_names)
        
    #import pdb; pdb.set_trace()
        
    if legend_classes:     
        legend_elements_classes = []
        for class0 in np.unique(modt.y):
            legend_elements_classes.append(Line2D([], [], color=color_coder_scatter(class0), marker='o',
                                                  linestyle='None', markersize=10, label= class_names[class0] ))
        lengend_classes = plt.legend(handles=legend_elements_classes, bbox_to_anchor=(1, -0.02), loc="lower left")
        ax.add_artist(lengend_classes)
    
    if title:
        plt.title("iteration: {}".format(iteration))

def plot_initialization(modt,
                        point_size=4,
                        rasterize=False,
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
        ax.scatter(rand_jitter(X[:, 0]), rand_jitter(X[:, 1]), color=list(map(color_coder_scatter, y)), s=point_size, 
                        clim=(y.min(), y.max()), rasterized=rasterize)
    else:
        ax.scatter(X[:, 0], X[:, 1], color=list(map(color_coder_scatter, y)), s=point_size, clim=(y.min(), y.max()), rasterized=rasterize)                              
        
    ax.axis('tight')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

def visualize_decision_area(modt,
                point_size=4,
                rasterize=False,
                axis_digits=False,
                axis_ticks=True,
                jitter=False,
                inverse_transform_standardization=False):
    """
    Plot prediction class areas in 2D.
    Only possible for native 2D datasets.
    """ 

    iteration = modt.best_iteration

    ax = plt.gca()
    y = modt.y

    X = modt.X_original
    if X.shape[1] != 2:
        raise ValueError("X must have 2 dimensions for decision area visualization.")

    if modt.use_2_dim_gate_based_on is not None:
        raise ValueError("Set modt.use_2_dim_gate_based_on to None.")

    if jitter:
        ax.scatter(rand_jitter(X[:, 0]), rand_jitter(X[:, 1]), color=list(map(color_coder_scatter, y)), s=point_size, clim=(y.min(), y.max()), zorder=3, rasterized=rasterize)
    else:
        ax.scatter(X[:, 0], X[:, 1], color=list(map(color_coder_scatter, y)), s=point_size, clim=(y.min(), y.max()), zorder=3, rasterized=rasterize)                              
        
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
        ...
        #print("Warning: Dataset does not include feature names.")
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

    #grid = np.append(grid, np.ones([grid.shape[0], 1]),axis=1) # Bias
    Z = modt.predict(grid, preprocessing=True).reshape(xx.shape) #TODO: Stdz is applied here

    # Create a contour plot with the results Z -> regions
    n_classes = len(np.unique(Z))
    ax.contourf(xx, yy, Z, alpha=0.6,
                           levels=np.arange(n_classes + 1) - 0.5,
                           #colors=COLOR_SCHEME_REGIONS,
                           zorder=1)


# def visualize_decision_area_old(predictor, X, y,rasterize=False, enable_scatter=True, axis_digits=False):
#     """Plot prediction class areas in 2D.""" 

#     if X.shape[1] != 2:
#         raise ValueError("X must have 2 dimensions.")

#     ax = plt.gca()
    
#     # Plot the training points
#     if enable_scatter:
#         ax.scatter(X[:, 0], X[:, 1], c=y, s=1,
#                 clim=(y.min(), y.max()), zorder=3, rasterized=rasterize)
#     else:
#         ax.scatter(X[:, 0], X[:, 1], c=y, s=1, alpha=0,
#                 clim=(y.min(), y.max()), zorder=3, rasterized=rasterize)         
#     ax.axis('tight')
#     if not axis_digits:
#         ax.axis('off')
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
    
#     xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
#                          np.linspace(*ylim, num=200))
#     Z = predictor(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

#     # Create a color plot with the results
#     n_classes = len(np.unique(y))
#     contours = ax.contourf(xx, yy, Z, alpha=0.3,
#                            levels=np.arange(n_classes + 1) - 0.5,
#                            zorder=1)

#     ax.set(xlim=xlim, ylim=ylim)
#     #plt.show()        

# def visualize_gating(modt,
#                      iteration,
#                      ax=None,
#                      cmap='rainbow',
#                      enable_scatter=True,
#                      low_alpha=False,
#                      title=True, 
#                      axis_digits=False,
#                      axis_ticks=False):
#     #plt.figure(figsize=(8,4))
#     ax = plt.gca()
#     y = modt.y

#     if modt.use_2_dim_gate_based_on is not None:
#         X = modt.X_2_dim
#     else:
#         X = modt.X_original
#         if X.shape[1] != 2:
#             raise ValueError("X must have 2 dimensions for visualization.")
    
#     # Plot the training points
#     if enable_scatter:
#         if low_alpha:
#             ax.scatter(X[:, 0], X[:, 1], c=y, s=1, alpha=0.1,
#                     clim=(y.min(), y.max()), zorder=3)
#         else:
#             ax.scatter(X[:, 0], X[:, 1], c=y, s=1, 
#                     clim=(y.min(), y.max()), zorder=3)
#     else:
#         ax.scatter(X[:, 0], X[:, 1], c=y, s=1, alpha=0,
#                 clim=(y.min(), y.max()), zorder=3)   
#     ax.axis('tight')
#     if not axis_digits:
#         ax.axis('off')
#     #xlim = [0,1]
#     #ylim = [0,1]
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     if not axis_ticks:
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])

#     xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
#                          np.linspace(*ylim, num=200))

#     grid = np.c_[xx.ravel(), yy.ravel()]

#     if modt.use_2_dim_gate_based_on is not None:
#         grid = np.append(grid, np.ones([grid.shape[0], 1]),axis=1) # Bias
#         Z = modt.get_expert(grid, iteration, internal=True).reshape(xx.shape)
#     else:
#         Z = modt.get_expert(grid, iteration, internal=False).reshape(xx.shape)

#     # Create a color plot with the results
#     n_classes = len(np.unique(Z))
#     contours = ax.contourf(xx, yy, Z, alpha=0.3,
#                            levels=np.arange(n_classes + 1) - 0.5,
#                            zorder=1)

#     ax.set(xlim=xlim, ylim=ylim)

#     if title:
#         plt.title("Iteration: {}".format(iteration))

#     #return plt
#     #plt.show()



def plot_initialization_gates(modt, point_size=3, rasterize=False):
    plt.subplot(1, 2, 1)
    plot_initialization(modt,point_size=point_size, rasterize=rasterize)
    plt.subplot(1, 2, 2)
    plot_gating(modt,iteration=0, point_size=point_size, rasterize=rasterize, title=False, axis_digits=False, inverse_transform_standardization=False)
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

def plot_disjoint_dt(modt, expert, asymmetric=False, size=(15,10), feature_names=None, class_names=None):

    if feature_names is None:
        feature_names = modt.feature_names
    if class_names is None:
        class_names = modt.class_names

    if asymmetric:
        DT = modt.DT_experts_alternative_algorithm[expert]
    else:
        DT = modt.DT_experts_disjoint[expert]

    plt.figure(figsize=size)
    try:  # Plotting fails if filled=True and tree empty
        tree.plot_tree(DT, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
    except: 
        tree.plot_tree(DT)

def plot_dt_dtreeviz(modt, expert, colors="pretty", fancy="True", asymmetric=False):
    """DT plotting using dtreeviz"""

    if modt.X_contains_categorical:
        raise Exception("Currently cannot print trees with categorical values.")

    colors_pretty = {'classes': [
        None, # 0 classes
        None, # 1 class
        ["#E24A33","#8EBA42"],
        ["#E24A33","#8EBA42","#81D0DB"],
        ["#E24A33","#8EBA42","#81D0DB","#FBC15E"],
        ["#E24A33","#8EBA42","#81D0DB","#FBC15E","#B97357"],
        ["#E24A33","#8EBA42","#81D0DB","#FBC15E","#B97357","#988ED5"],
        ["#E24A33","#8EBA42","#81D0DB","#FBC15E","#B97357","#988ED5","#348ABD"],
        ["#FEFEBB",'#edf8b1','#c7e9b4','#7fcdbb','#1d91c0','#225ea8','#fdae61','#f46d43'], # 8
        ["#FEFEBB",'#c7e9b4','#41b6c4','#74add1','#4575b4','#313695','#fee090','#fdae61','#f46d43'], # 9
        ["#FEFEBB",'#c7e9b4','#41b6c4','#74add1','#4575b4','#313695','#fee090','#fdae61','#f46d43','#d73027'] # 10
        ]}

    # colors that can be seen on the colored gates
    colors_visible = {'classes':   [
        None, # 0 classes
        None, # 1 class
        ["#440154", "#21918c"],
        ["#440154", "#21918c", "#fde725"],
        ["#440154", "#21918c", "#fde725", "#5ec962"],            
        ["#440154", "#21918c", "#fde725", "#5ec962", "#ffbaab"],
        ["#440154", "#21918c", "#fde725", "#5ec962", "#ffbaab", "#ff0000"],
        ["#440154", "#21918c", "#fde725", "#5ec962", "#ffbaab", "#ff0000", "#f7f7f7"],
        ["#440154", "#21918c", "#fde725", "#5ec962", "#ffbaab", "#ff0000", "#f7f7f7", "#f700ff"],
        ["#440154", "#21918c", "#fde725", "#5ec962", "#ffbaab", "#ff0000", "#f7f7f7", "#f700ff", "#4dff00"],
        ["#440154", "#21918c", "#fde725", "#5ec962", "#ffbaab", "#ff0000", "#f7f7f7", "#f700ff", "#4dff00", "#000000"]
        ]}

    if colors == "pretty":
        color_scheme = colors_pretty
    elif colors == "visible":
        color_scheme = colors_visible
    else:
        color_scheme = {}

    pickle_disjoint_data(modt, modt.best_iteration)

    df = pd.read_pickle("output/disjoint_data_e_{}.pd".format(expert))
    df["target"] = modt._map_y(df["target"],reverse=True)

    df = df.sort_values(by="target")

    if modt.class_names is None:
        class_names = list(np.unique(modt.y_original))
    else:
        class_names=list(modt.class_names)

    if modt.feature_names is None:
        feature_names = ["feature" + str(x) for x in range(modt.X_original.shape[1])]
    else:
        feature_names = modt.feature_names

    if asymmetric:
        if modt.DT_experts_alternative_algorithm is None:
            raise Exception("Train asymmetric trees first.")
        else:
            classifier = modt.DT_experts_alternative_algorithm[expert]
    else:
        classifier = modt.DT_experts_disjoint[expert]
    try:
        viz = dtreeviz(classifier, 
                    df.iloc[:,:-1], 
                    df["target"],
                    feature_names=feature_names, 
                    class_names=class_names,
                    fancy=fancy,
                    colors=color_scheme
                    )                            
        viz.view() 
    except Exception as e: 
        print("Plotting failed. Exception:", e)



