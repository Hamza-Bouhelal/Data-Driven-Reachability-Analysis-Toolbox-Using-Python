from itertools import product
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import numpy as np
import random
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
from Zonotope import Zonotope
from scipy.spatial import ConvexHull

def get_points_2d(zono, step):
    """Return the points that compose the Zonotope 'zono, at dims 'step' and 'step' + 1.'."""
    tempx = []
    tempy = []
    gens = zono.generators()[step:step+2]
    center = zono.center()[step:step+2]
    for point in range(2**(gens.shape[1])):
        curr = format(point, 'b').zfill(gens.shape[1])
        val = [center[0][0], center[1][0]]
        for g in range(gens.shape[1]):
            if curr[g] == '1':
                val[0] += gens[0][g]
                val[1] += gens[1][g]
            else:
                val[0] -= gens[0][g]
                val[1] -= gens[1][g]
        tempx.append(val[0])
        tempy.append(val[1])
    return tempx, tempy


def plot_zono(ax, Z, step, color):
    """Plots dims 'step' and 'step' + 1 of the zonotope Z"""
    x, y = get_points_2d(Z, step)
    all = []
    for i in range(len(x)):
        all.append([x[i], y[i]])
    all.append(all[0])
    all = np.array(all)
    hull = ConvexHull(points=all,
              qhull_options='QG4')
    temp = list(hull.vertices)
    temp.append(temp[0])
    temp = np.array(temp)
    ax.plot(all[temp,0], all[temp,1], color=color, lw=1)


def plot_results(data, plot=True, save='', titles=[]):
    """Plots all 2d dimensions of the zonotopes inside the list 'data' or lists of zonotope in 'data'.
    
    plot: if True, plots the results.
    save: if not empty, saves the results in the directory 'save'.
    titles: if not empty, plots the results with the titles in the list 'titles'.
    """
    if isinstance(data, list) and False not in [isinstance(ele, Zonotope) for ele in data]:
        data = [data]
    if not isinstance(save, str) or not isinstance(plot, bool):
        raise TypeError("save : String, plot : Boolean")
    if False in [isinstance(result, list) for result in data] or False in [[isinstance(ele, Zonotope) for ele in result] for result in data]:
        raise TypeError("data : List of lists of Zonotopes or list of Zonotopes")
    if save != '':
        final_directory = makedir(save)
    colors = ['red', 'green', 'blue', 'black', 'orange',
              'purple', 'brown', 'gray']
    for i in range(0, int(data[0][0].center().shape[0]/2)):
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot()
        legends = []
        ind = 0
        for result in data:
            color = colors[random.randint(0, len(colors)-1)]
            colors.pop(colors.index(color))
            try:
                legends.append(mpatches.Patch(color=color, label=titles[ind]))
                ind += 1
            except:
                pass
            for ele in result:
                plot_zono(ax, ele, i*2, color)
        ax.set_xlabel('Dim ' + str(i*2 + 1))
        ax.set_ylabel('Dim ' + str(i*2+ 2))
        if len(legends) == len(data):
            plt.legend(handles=legends)
        if save != '':
            plt.savefig(f"{final_directory}/ Plot n{i+1}.png")
        if plot:
            plt.show()
        if i == int(data[0][0].center().shape[0]/2)-1:
            if data[0][0].center().shape[0] % 2 == 1:
                plt.rcParams["figure.autolayout"] = True
                fig = plt.figure()
                ax = fig.add_subplot()
                legends = []
                ind = 0
                for result in data:
                    color = colors[random.randint(0, len(colors)-1)]
                    legends.append(mpatches.Patch(
                        color=color, label=titles[ind]))
                    ind += 1
                    for ele in result:
                        plot_zono(ax, ele, i*2 + 1, color)
                ax.set_xlabel('Dim ' + str(i*2 + 2))
                ax.set_ylabel('Dim ' + str(i*2 + 3))
                if len(legends) == 2:
                    plt.legend(handles=legends)
                if save != '':
                    plt.savefig(f"{final_directory}/ Plot n{i+2}.png")
                if plot:
                    plt.show()


def makedir(save):
    """Creates the directory 'save', if it already exists, creates a directory with the name 'save' (1), etc."""
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, save)
    t = 1
    if os.path.exists(final_directory):
        temp = final_directory + f' ({str(t)})'
        while os.path.exists(temp):
            t += 1
            temp = final_directory + f' ({str(t)})'
        final_directory = temp
    os.mkdir(final_directory)
    print("\nCreated directory at : ", final_directory)
    return final_directory

