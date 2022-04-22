from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import numpy as np
import random
sys.path.append("D:\\Desktop\\thesis\\brsl\\scripts\\reachability")
from Zonotope import Zonotope
import itertools


def plot_zono_center_and_generators(ax, ele, step, color):
    #toplot = get_points_from_generator(ele.generators()).tolist()
    points = np.array(ele.generators())
    try:
        #try to plot 3 next dimensions        
        ax.scatter(ele.center()[step], ele.center()[step+1], ele.center()[step+2],color=color)
    except:
        #plot 2d instead
        ax.scatter(ele.center()[step], ele.center()[step+1], color=color, marker='o')
 

def plot_results(data, plot=True, save='', titles=[]):
    """
    split dim_x into set of 3 dimensions
    if dim_x is different from 3k then plot all 3k dims
    until either only 2 or 1 dims are left to plot
    """

    if isinstance(data, list) and False not in [isinstance(ele, Zonotope) for ele in data]:
        data = [data]
    if not isinstance(save, str) or not isinstance(plot, bool):
        raise TypeError("save : String, plot : Boolean")
    if False in [isinstance(result, list) for result in data] or False in [[isinstance(ele, Zonotope) for ele in result] for result in data]:
        raise TypeError("data : List of lists of Zonotopes")
    if save != '':
        final_directory = makedir(save)
    colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta']
    if data[0][0].center().shape[0] % 2 != 0:
        for i in range(0, int(data[0][0].center().shape[0]/3)+1):
            plt.rcParams["figure.autolayout"] = True
            fig = plt.figure()
            if i == int(data[0][0].center().shape[0]/3):
                if data[0][0].center().shape[0]%3 == 2:
                    ax = fig.add_subplot()
                else:
                    break
            else:
                ax = fig.add_subplot(projection="3d")
            legends = []
            ind = 0
            for result in data:
                color = colors[random.randint(0, len(colors)-1)]
                if(ind<len(titles)):
                    legends.append(mpatches.Patch(color=color, label=titles[ind]))
                    ind += 1
                for ele in result:
                    plot_zono_center_and_generators(ax, ele, i*3, color)
            plt.legend(handles=legends)
            if save != '':
                plt.savefig(f"{final_directory}/ Plot n{i+1}.png")
            if plot:
                plt.show()
    else:
        for i in range(0, int(data[0][0].center().shape[0]/2)):
            plt.rcParams["figure.autolayout"] = True
            fig = plt.figure()
            ax = fig.add_subplot()
            legends = []
            ind = 0
            for result in data:
                color = colors[random.randint(0, len(colors)-1)]
                legends.append(mpatches.Patch(color=color, label=titles[ind]))
                ind += 1
                for ele in result:
                    plot_zono_center_and_generators(ax, ele, i*2, color)
            if len(legends) == 2:
                plt.legend(handles=legends)
            if save != '':
                plt.savefig(f"{final_directory}/ Plot n{i+1}.png")
            if plot:
                plt.show()

def makedir(save):
        current_directory = os.getcwd()
        final_directory =  os.path.join(current_directory, save) 
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

        """for points in toplot:
            for i in range(step,step+3):
                points[i] += ele.center()[i, 0]
        xs, ys, zs = zip(*toplot[step:step+3])
        ax.plot(xs, ys, zs)
        points = np.array(ele.generators())
        ax.fill(points[step] + ele.center()[step], points[step+1] + ele.center()[step+1], points[step + 2] + ele.center()[step + 2], color=color)
        for i in range(ele.generators().shape[1]):
            #if sqrt(ele.center()[0+step]**2 + ele.center()[1+step]**2 + ele.center()[2+step]**2 - ele.generators()[0+step, i]**2 - ele.generators()[1+step, i]**2 - ele.generators()[2+step, i]**2) > 0.01:
            ax.quiver(ele.center()[0+step], ele.center()[1+step], ele.center()[2+step], ele.generators()[0+step, i], ele.generators()[1+step, i], ele.generators()[2+step, i])  """