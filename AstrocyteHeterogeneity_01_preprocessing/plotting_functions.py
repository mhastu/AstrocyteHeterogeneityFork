#%%
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import random

from matplotlib import cm

from matplotlib.lines import Line2D

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns

import os
import os.path as os_p

from Cell import Cell
from CellsData import CellsData

#%%
def calc_bin_width(data) -> float:
    bin_width = (max(data) - min(data)) / np.sqrt(len(data))
    return bin_width

def bar_plot(labels, values, title, ylabel):
    figure = plt.figure(figsize=(10,10))
    ax = figure.add_subplot()
    ax.bar(labels, values)

    ax.set_title(title)
    ax.grid()
    ax.set_ylabel(ylabel)
    return figure, ax
        
def plot_vs(X:pd.DataFrame, labels:list = ["Diameter", "Volumes"], units:list = ["$\mu m^2$", "$\mu m^2$"], Title: str = "OutliersAnnotation", annotation: bool = True):
    n = X.index.values
    fig, ax = plt.subplots(figsize = (10,10))

    ax.scatter( X[labels[0]],  X[labels[1]])
    if annotation:
        for i, txt in enumerate(n):
            #print(i, txt)
            ax.annotate(txt, (X[labels[0]][txt], X[labels[1]][txt]))
    ax.set_xlabel(labels[0] + " in " + units[0])
    ax.set_ylabel(labels[1] + " in " + units[0])
    ax.grid()
    fig.savefig(f"{Title}"+ labels[0] + "vs" + labels[1] + ".pdf")

def plot_Histo(data, ax, title, bins = 25,  axes = ["nr of points", "counts"], units:list = ["$\mu m^2$", "$\mu m^2$"],pdf = True, rwidth = 0.8, dist_names = ['alpha', 'beta', 'rdist']):
    """Plot a histogram of Data
    axes -> labels of axes x and y
    units -> units of axes x and y
    pdf -> if a pdf should be drawn"""
    ax.hist(data, bins = bins)
    ax.set_title(title + f" of {len(data)} cells ")
    ax.set_xlabel(axes[0] + " in " + units[0])
    ax.set_ylabel(axes[1] + " in " + units[1])
    ax.grid()

    if pdf:
        ax2 = ax.twinx()
        
        ax2.set_ylabel("Probability density")
        get_ProbDensFkt(data, ax2, dist_names)
        #ax2.plot(x, y, label = "Probability density", color = "r")
    #fig.savefig("{Title}.pdf")
    return ax

def plot_cube(cells, ThresholdCube = True, offset = 2):
    print("Plotting the Cube...")
    cube =  plt.figure( figsize=(10, 8))
    ax = cube.add_subplot(121, projection='3d')

    min_x = cells.limits[0][0]
    max_x = cells.limits[0][1]
    min_y = cells.limits[1][0]
    max_y = cells.limits[1][1]
    min_z = cells.limits[2][0]
    max_z = cells.limits[2][1]

    print(max_x, min_x, "\n", max_y, min_y, "\n", max_z, min_z)
    # print cube / sphere https://www.tutorialspoint.com/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    #z = 0
    ax.plot([max_x, max_x], [min_y, max_y], min_z, color = "b", label = "Preperate Cube")
    ax.plot([min_x, max_x], [min_y, min_y], min_z, color = "b")
    ax.plot([min_x, min_x], [min_y, max_y], min_z, color = "b")
    ax.plot([min_x, max_x], [max_y, max_y], min_z, color = "b")
    #z = max
    ax.plot([max_x, max_x], [min_y, max_y], max_z, color = "b")
    ax.plot([min_x, max_x], [min_y, min_y], max_z, color = "b")
    ax.plot([min_x, min_x], [min_y, max_y], max_z, color = "b")
    ax.plot([min_x, max_x], [max_y, max_y], max_z, color = "b")
    #y = min
    ax.plot([max_x, max_x], [min_y, min_y], [min_z, max_z], color = "b")
    ax.plot([min_x, min_x], [min_y, min_y], [min_z, max_z], color = "b")
    #y = max
    ax.plot([max_x, max_x], [max_y, max_y], [min_z, max_z], color = "b")
    ax.plot([min_x, min_x], [max_y, max_y], [min_z, max_z], color = "b")
    ax.set_xlabel("x Direction in $\mu m$")
    ax.set_ylabel("y Direction in $\mu m$")
    ax.set_zlabel("z Direction in $\mu m$")

    if ThresholdCube:

        max_x = max_x - offset
        max_y = max_y - offset
        max_z = max_z - offset
        min_x = min_x + offset
        min_y = min_y + offset
        min_z = min_z + offset
        
    ax.legend()
    return cube, ax

def get_ProbDensFkt(data, ax, dist_names = ['norm', 'alpha', 'beta']):
    # https://erdogant.github.io/distfit/pages/html/index.html best fitting distr
    # https://stackoverflow.com/questions/6615489/fitting-distributions-goodness-of-fit-p-value-is-it-possible-to-do-this-with/16651524#16651524 27.02.2023

    size = len(data)
    x = np.linspace(min(data), max(data), 100)

    #null hypothesis normally distributed
    mean = np.mean(data)
    sig = np.std(data)
    print(f"Mean = {mean}, \nStd = {sig}")
    ax.plot(x, scipy.stats.norm.pdf(x, mean, sig) * size, ".-",label = "Normal distribution")

    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        params = dist.fit(data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        if arg:
            pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale) * size
        else:
            pdf_fitted = dist.pdf(x, loc=loc, scale=loc) * size
        ax.plot(x, pdf_fitted, label=dist_name)
        #ax.set_xlim(0,47)
    ax.legend(loc='best')

def ellipse_params(A, centroid):
    # from: https://stackoverflow.com/questions/41955492/how-to-plot-efficiently-a-large-number-of-3d-ellipsoids-with-matplotlib-axes3d 21.02.2023
    # from https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py 14.03.2023
    U, D, V = np.linalg.svd(A)
    rx, ry, rz = 1/np.sqrt(D)
    # r = R.from_euler("xyz", degrees = True)
    # V = R.as_matrix(r)
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    xs = rx * np.outer(np.cos(u), np.sin(v))
    ys = ry * np.outer(np.sin(u), np.sin(v))
    zs = rz * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(xs)):
        for j in range(len(xs)):
            [xs[i,j],ys[i,j],zs[i,j]] = np.dot([xs[i,j],ys[i,j],zs[i,j]], V) + centroid
        
    return xs, ys, zs, V, rx, ry, rz

def plot_cell_rot(fig, cell, mode = "rough", ell = True):
    ax1 = fig.add_subplot(221, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax2 = fig.add_subplot(222, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax3 = fig.add_subplot(223, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")
    ax4 = fig.add_subplot(224, projection = "3d", xlabel = "$\mu m$", ylabel = "$\mu m$", zlabel = "$\mu m$")

    if mode == None:
        plot_cell(cell, ax1, plot_ell= ell)
        plot_cell(cell, ax2, plot_ell= ell)
        plot_cell(cell, ax3, plot_ell= ell)
        plot_cell(cell, ax4, plot_ell= ell)
        # x, y, z = ellipse_params(cell.A, cell.c)
        # ax1.plot_surface(x, y, z,  rstride=3, color = "b", cstride=3, linewidth=0.1, alpha=0.2, shade=True)
        # ax2.plot_surface(x, y, z,  rstride=3, color = "b", cstride=3, linewidth=0.1, alpha=0.2, shade=True)
        # ax3.plot_surface(x, y, z,  rstride=3, color = "b", cstride=3, linewidth=0.1, alpha=0.2, shade=True)
        # ax4.plot_surface(x, y, z,  rstride=3, color = "b", cstride=3, linewidth=0.1, alpha=0.2, shade=True)
        
        
        # ax1.scatter(cell.ellipse_points[:, 0], cell.ellipse_points[:, 1], cell.ellipse_points[:, 2], color = "r")
        # ax2.scatter(cell.ellipse_points[:, 0], cell.ellipse_points[:, 1], cell.ellipse_points[:, 2], color = "r")
        # ax3.scatter(cell.ellipse_points[:, 0], cell.ellipse_points[:, 1], cell.ellipse_points[:, 2], color = "r")
        # ax4.scatter(cell.ellipse_points[:, 0], cell.ellipse_points[:, 1], cell.ellipse_points[:, 2], color = "r")
        # mode = "Ellipsepoints"
    else:
        plot_cell(ax1, cell, plot_ell= ell, mode = mode)
        plot_cell(ax2,cell,  plot_ell= ell, mode = mode)
        plot_cell(ax3,cell,  plot_ell= ell, mode = mode)
        plot_cell(ax4,cell,  plot_ell= ell, mode = mode)

    ax1.set_title("Iso view")
    ax2.set_title("View onto xz-plane")
    ax3.set_title("View onto xy-plane")
    ax4.set_title("View onto yz-plane")

    ax2.view_init(0, 90) #frontal
    ax2.set_yticks([])
    ax3.view_init(90, 90)
    ax3.set_zticks([])
    ax4.view_init(0, 0)
    ax4.set_xticks([])
    if not os_p.isdir("./CellPlots"):
        os.mkdir("./CellPlots")
    plt.savefig(f"./CellPlots/{cell.cellID} 3D View plot {mode}.pdf")

def plot_circle_3d(center, diameter, ax):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:30j]
    x = center[0] + diameter *np.cos(u) * np.sin(v)
    y = center[1] + diameter * np.sin(u) * np.sin(v)
    z = center[2] + diameter * np.cos(v)

    ax.plot_surface(x, y, z, alpha = 0.5)
    
def plot_cell(ax, cell: Cell, color = "r", mode = "rough", plot_ell = False, plot_circ = False, Annotate = False, percent_points = 0.8):
    
    if mode != "rough" and mode != "fine" and mode != "scatter":
        print("please input accepted mode like 'rough', 'fine' or 'scatter'")
        return
    if  cell.rough_branches == {}:
            cell._get_branches()
    if mode == "rough":

        for id, branch in cell.rough_branches.items():
            #print(id)
        
            x = np.array(branch["PtPositionX"])
            y = np.array(branch["PtPositionY"])
            z = np.array(branch["PtPositionZ"])
            dia = np.array(branch["PtDiameter"])
            depth = np.array(branch["Depth"])

            #https://stackoverflow.com/questions/38079366/matplotlib-line3dcollection-multicolored-line-edges-are-jagged
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            for ii in range(len(x)-1):
                
                segii=segments[ii]
                ax.plot(segii[:,0],segii[:,1],segii[:,2], "-",color= color,linewidth=dia[ii], alpha = 1 / (depth[ii] + 1))
                
                if Annotate:
                    if ii == 0:
                    
                        #print(depth[ii])
                        ax.text(segii[0,0],segii[0,1],segii[0,2], str((id, depth[ii])))
                        pass

    if mode == "fine":
        for id, branch in cell.fine_branches.items():
            
            x = np.array(branch[:, 0])
            #print(np.shape(x))
            y = np.array(branch[:, 1])
            z = np.array(branch[:, 2])
            #dia = np.array(branch["PtDiameter"])
            #depth = np.array(branch["Depth"])

            #https://stackoverflow.com/questions/38079366/matplotlib-line3dcollection-multicolored-line-edges-are-jagged
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #print(id, segments)
            for ii in range(len(x)-1):
                #print(ii)
                segii=segments[ii]
                #print(ii, segii)
                ax.plot(segii[:,0],segii[:,1],segii[:,2], "-",color= color) #, alpha = 1 / (depth[ii] + 1)
    
    if mode == "scatter":
        x = np.array(cell.BranchingPointsData["PtPositionX"])
        y = np.array(cell.BranchingPointsData["PtPositionY"])
        z = np.array(cell.BranchingPointsData["PtPositionZ"])
        depth = np.array(cell.BranchingPointsData["Depth"])
        ax.scatter(x, y, z) #, alpha = 1/(depth+ 1)

    if plot_ell:
        try:
            x, y, z, V, rx, ry, rz = ellipse_params(cell.A , cell.c) 
        except AttributeError:
            # run ellipsoid fitting
            # The following code could probably run with parameters from EllipseParameters.csv,
            # but V is not stored there and the code currently depends on V.
            # TODO: fix this by implementing a consistent storing and loading of the ellipsoid parameters
            cell._get_ellipsoid()
            x, y, z, V, rx, ry, rz = ellipse_params(cell.A , cell.c)
        ax.plot_surface(x, y, z,  rstride=3, color = "b", cstride=3, linewidth=0.1, alpha=0.2, shade=True)
        #plot principal axes:
        point1 = np.array([0, 0, rz])
        point2 = np.array([0, ry, 0])
        point3 = np.array([rx, 0, 0])
        cent = cell.c
        point_rot1 = np.dot(point1, V) + cent
        point_rot2 = np.dot(point2, V) + cent
        point_rot3 = np.dot(point3, V) + cent

        ax.plot([cent[0], point_rot1[0]], [cent[1], point_rot1[1]], [cent[2], point_rot1[2]], color = "b")
        ax.plot([cent[0], point_rot2[0]], [cent[1], point_rot2[1]], [cent[2], point_rot2[2]], color = "g")
        ax.plot([cent[0], point_rot3[0]], [cent[1], point_rot3[1]], [cent[2], point_rot3[2]], color = "m")
        
    if plot_circ:
        r = cell.fit_sphere(percent_points)
        plot_circle_3d(cell.somaLoc, r, ax )
    
def plot_finevsrough(ax, cell, BranchID):
    """ Plots the fine branch against the rough branch to visualice the disparities
    cell -> cell object from which the branches are taken
    BranchID -> id of the branch that is to be plotted"""

    fB = cell.fine_branches[BranchID]
    rB = cell.rough_branches[BranchID]

    rx = np.array(rB["PtPositionX"])
    ry = np.array(rB["PtPositionY"])
    rz = np.array(rB["PtPositionZ"])

    fx = fB[:, 0]
    fy = fB[:, 1]
    fz = fB[:, 2]

    ax.plot(rx, ry, rz)
    ax.plot(fx, fy, fz)

def ShollAnalysis(main):
    """Creates the sholl analysis plot based on the shell features dataframe contained in the main object"""
    data = main.Shell_FeatureVector.copy()
    figure = plt.figure(figsize=(10,10))
    ax = figure.add_subplot()

    ax.errorbar(data.columns, data.mean(), yerr=data.std(), ecolor = "red")
    data.mean().plot(ax = ax, ls = "--", label = "mean")
    data.min().plot(ax = ax, ls = "--", label = "min")
    data.max().plot(ax = ax, ls = "--", label = "max")
    ax.legend()
    data.transpose().plot(ax = ax, alpha = 0.2, legend = False)
    ax.locator_params(tight=True, nbins=8)
    ax.set_title(f"Sholl analysis of the dataset") 
    
    ax.grid(True)
    ax.set_ylabel("Nr of Points inside shell")
    ax.set_xlabel("Radius of shell from soma in steps of 2.5 $\mu$m")
    figure.savefig("./ResultPlots_" + main.Type + "/ShollAnalysis_" + main.scaler +".png")

def plot_cube(ax, cells,  ThresholdCube = True, zoom = False, offset = 2, Surfaces = True, subplots = 121, cell = None):
    print("Plotting the Cube...")
    
    #ax.set_title("Sample cube")
    min_x = cells.limits[0][0]
    max_x = cells.limits[0][1]
    min_y = cells.limits[1][0]
    max_y = cells.limits[1][1]
    min_z = cells.limits[2][0]
    max_z = cells.limits[2][1]

    if zoom:
        ax.set_title("Sample zoomed in at corner")
        max_x = min_x + 45
        max_y = min_y + 45
        max_z = min_z + 45
        #min x -> max x
        ax.plot([min_x, max_x], [min_y, min_y], min_z, color = "b", label = "Sample Cube")
        #min y -> max y
        ax.plot([min_x, min_x], [min_y, max_y], min_z, color = "b")
        #min z -> maxz
        ax.plot([min_x, min_x], [min_y, min_y], [min_z, max_z], color = "b")
        
    # print cube / sphere https://www.tutorialspoint.com/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    else: 
        #z = 0
        ax.plot([max_x, max_x], [min_y, max_y], min_z, color = "b", label = "Sample Cube")
        ax.plot([min_x, max_x], [min_y, min_y], min_z, color = "b")
        ax.plot([min_x, min_x], [min_y, max_y], min_z, color = "b")
        ax.plot([min_x, max_x], [max_y, max_y], min_z, color = "b")
        #z = max
        ax.plot([max_x, max_x], [min_y, max_y], max_z, color = "b")
        ax.plot([min_x, max_x], [min_y, min_y], max_z, color = "b")
        ax.plot([min_x, min_x], [min_y, max_y], max_z, color = "b")
        ax.plot([min_x, max_x], [max_y, max_y], max_z, color = "b")
        #y = min
        ax.plot([max_x, max_x], [min_y, min_y], [min_z, max_z], color = "b")
        ax.plot([min_x, min_x], [min_y, min_y], [min_z, max_z], color = "b")
        #y = max
        ax.plot([max_x, max_x], [max_y, max_y], [min_z, max_z], color = "b")
        ax.plot([min_x, min_x], [max_y, max_y], [min_z, max_z], color = "b")

    if Surfaces:
        print("Surfaces")
        alpha = 0.1
        x = np.arange(min_x, max_x, 0.5)
        y = np.arange(min_y, max_y, 0.5)
        z = np.arange(min_z, max_z, 0.5)
        #Z sides:
        if zoom:
            Xm, Ym = np.meshgrid(x,y)
            ax.plot_surface(Xm, Ym, np.full_like(Xm, min_z), color = "g", linewidth = 0, antialiased=False, alpha = alpha)

            Ym, Zm = np.meshgrid(y, z)
            ax.plot_surface(np.full_like(Ym, min_x), Ym, Zm, color = "m", linewidth = 0, antialiased=False, alpha = alpha)

        else:             
            Xm, Ym = np.meshgrid(x,y)
            ax.plot_surface(Xm, Ym, np.full_like(Xm, max_z), color = "g", linewidth = 0, antialiased=False, alpha = alpha)
            ax.plot_surface(Xm, Ym, np.full_like(Xm, min_z), color = "g", linewidth = 0, antialiased=False, alpha = alpha)

            Zm, Xm = np.meshgrid(z, x)
            ax.plot_surface(Xm, np.full_like(Xm, max_y), Zm, color = "k", linewidth = 0, antialiased=False, alpha = alpha)
            ax.plot_surface(Xm, np.full_like(Xm, min_y), Zm, color = "k", linewidth = 0, antialiased=False, alpha = alpha)

            Ym, Zm = np.meshgrid(y, z)
            ax.plot_surface(np.full_like(Ym, max_x), Ym, Zm, color = "m", linewidth = 0, antialiased=False, alpha = alpha)
            ax.plot_surface(np.full_like(Ym, min_x), Ym, Zm, color = "m", linewidth = 0, antialiased=False, alpha = alpha)

    if ThresholdCube:
        max_x = max_x - offset
        max_y = max_y - offset
        max_z = max_z - offset
        min_x = min_x + offset
        min_y = min_y + offset
        min_z = min_z + offset

        if zoom: 
            #min x -> max x
            ax.plot([min_x, max_x], [min_y, min_y], min_z, color = "r", label = "Threshold Edge")
            #min y -> max y
            ax.plot([min_x, min_x], [min_y, max_y], min_z, color = "r")
            #min z -> max z
            ax.plot([min_x, min_x], [min_y, min_y], [min_z, max_z], color = "r")
            
        else:
            #z =  min
            ax.plot([max_x, max_x], [min_y, max_y], min_z, color = "r", label = "Threshold Edge")
            ax.plot([min_x, max_x], [min_y, min_y], min_z, color = "r")
            ax.plot([min_x, min_x], [min_y, max_y], min_z, color = "r")
            ax.plot([min_x, max_x], [max_y, max_y], min_z, color = "r")
            #z = max
            ax.plot([max_x, max_x], [min_y, max_y], max_z, color = "r")
            ax.plot([min_x, max_x], [min_y, min_y], max_z, color = "r")
            ax.plot([min_x, min_x], [min_y, max_y], max_z, color = "r")
            ax.plot([min_x, max_x], [max_y, max_y], max_z, color = "r")
            #y = min
            ax.plot([max_x, max_x], [min_y, min_y], [min_z, max_z], color = "r")
            ax.plot([min_x, min_x], [min_y, min_y], [min_z, max_z], color = "r")
            #y = max
            ax.plot([max_x, max_x], [max_y, max_y], [min_z, max_z], color = "r")
            ax.plot([min_x, min_x], [max_y, max_y], [min_z, max_z], color = "r")
    ax.set_xlabel("$\mu m$")
    ax.set_ylabel("$\mu m$")
    ax.set_zlabel("$\mu m$")
    if cell != None: 
        plot_cell(ax, cell)

def plot_EdgeCell(ax, edgeCell,cube, width = 3, surface = True, zoom = False, color = "r"): 
        cell = edgeCell[0]
        print(cell)
        hits = edgeCell[1]
        ax.set_title(f"cell {cell.cellID} classified as Edge cell with \n {hits} points within threshold volume")
        min_x = cube[0][0]
        max_x = cube[0][1]
        min_y = cube[1][0]
        max_y = cube[1][1]
        min_z = cube[2][0]
        max_z = cube[2][1]
        limits = cell.find_edgepoints()
        tolerance = limits + width
        if surface:
            #plotting only touching surfaces
            x = np.arange(tolerance[0,0], tolerance[0,1], 1)
            y = np.arange(tolerance[1,0], tolerance[1,1], 1)
            z = np.arange(tolerance[2,0], tolerance[2,1], 1)
            alpha = 0.2
            if limits[0, 0] <= min_x + 2:
                Ym, Zm = np.meshgrid(y, z)
                ax.plot_surface(np.full_like(Ym, min_x), Ym, Zm, color = "m", linewidth = 0, antialiased=False, alpha = alpha)
            elif limits[0, 1] >= max_x - 2:
                Ym, Zm = np.meshgrid(y, z)
                ax.plot_surface(np.full_like(Ym, max_x), Ym, Zm, color = "m", linewidth = 0, antialiased=False, alpha = alpha)

            if limits[1, 0] <= min_y + 2: 
                Zm, Xm = np.meshgrid(z, x)
                ax.plot_surface(Xm, np.full_like(Xm, min_y), Zm, color = "c", linewidth = 0, antialiased=False, alpha = alpha)
            elif limits[1, 1] >= max_y - 2:
                Zm, Xm = np.meshgrid(z, x)
                ax.plot_surface(Xm, np.full_like(Xm, max_y), Zm, color = "c", linewidth = 0, antialiased=False, alpha = alpha)

            if limits[2, 0] <= min_z + 2:
                Xm, Ym = np.meshgrid(x,y) 
                ax.plot_surface(Xm, Ym, np.full_like(Xm, min_z), color = "g", linewidth = 0, antialiased=False, alpha = alpha)
            elif limits[2, 1] >= max_z - 2:
                Xm, Ym = np.meshgrid(x,y)
                ax.plot_surface(Xm, Ym, np.full_like(Xm, max_z), color = "g", linewidth = 0, antialiased=False, alpha = alpha)            
   
        plot_cell(ax, cell, color = color)
        
        if zoom:
            ax.set_xlim(tolerance[0, 0], tolerance[0, 1])
            ax.set_ylim(tolerance[1, 0], tolerance[1, 1])
            ax.set_zlim(tolerance[2, 0], tolerance[2, 1])

        ax.set_xlabel("$\mu m$")
        ax.set_ylabel("$\mu m$")
        ax.set_zlabel("$\mu m$")
        #ax.legend()

def plot_somas(ax, cells: dict, label = None, color = "r"):
        print("Plotting the Cellsomas")
        #plotting of soma as point and maby dentrite terminals? maby with a reduced ellipse size?
        #fig, ax = main.get_cube()
        no_somaDia = []
        somaLoc = np.zeros((len(cells.items()), 4))
        i = 0
        for id, cell in cells.items():
            if isinstance(cell,list):
                cell = cell[0]

            somaID = cell.somaID
            #somaDia = cell.somaLoc
            somaLoc[i] = cell.somaLoc
            i += 1
        
        ax.scatter(somaLoc[:, 0], somaLoc[:, 1], somaLoc[:, 2], c = color, label = label)

def explainedVar(main, start_pca, opt_pca):
    fig_explained_variance = plt.figure()
    ax = fig_explained_variance.add_subplot(121)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.plot(range(main.std_FeatureVector.shape[1]), start_pca.explained_variance_ratio_.cumsum() * 100, marker = 0, ls = "--") 
    ax.axhline(y = 80, ls = "--", color = "r")
    ax.set_title("Explained Variance over Components")
    ax.set_ylabel("Cumulative Explained Variance in %")
    ax.set_xlabel("Number of Components")

    ax = fig_explained_variance.add_subplot(122)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.bar(opt_pca.get_feature_names_out(), opt_pca.explained_variance_ratio_* 100)
    ax.set_ylabel("Explained Variance in %")
    ax.set_xlabel("PCA output components")
    ax.set_title("Explained Variance by Components")
    fig_explained_variance.tight_layout()
    fig_explained_variance.savefig("./ResultPlots_" + main.Type + "/PCA_Featurevariance_" + main.scaler +".png")

def classification(main, features, pca, labels, Name, plot_cells):
    """ main = main object
    pca == feature vector reduced by PCA
    best k-means = best
    silh = silhouette score of clustering
    Name = Name of features plotted
    plot_cells = If cells should be plotted"""
    ## Plot data visualisation
    if type(features) == pd.DataFrame:
        features = np.array(features)
    if len(np.unique(labels)) == 1:
        silh = 0.1
    else:
        silh = silhouette_score(features, labels)
    ClassificationPlot = plt.figure()

    # markers = Line2D.markers
    # datasetmarkers = main.datasetmarkers
    # datasets = np.unique(datasetmarkers)
    # marker_keys = list(markers)
    # epoched_features = np.zeros(())
    # for dataset in np.unique(datasetmarkers): #epoching features into datasets

    #     pass
    # markers = markers[0: len(np.unique(main.datasetmarkers))]
    if np.shape(features.shape)[0] == 1:
        print("Only 1 feature")
        ax = ClassificationPlot.add_subplot(111)
        for i in np.unique(labels):
            ax.hist(features[labels == i], histtype='bar', alpha = 0.5, color= main.cmap(int(i)), log = True)
            ax.set_xlabel(f"Principal Component 1: Explained Variance = {np.round(pca.explained_variance_ratio_[0] * 100, 2)}%")
            ax.set_ylabel("Log counts")
        
    else:
        gs = ClassificationPlot.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)

        ax = ClassificationPlot.add_subplot(gs[1, 0])
        ax_histx = ClassificationPlot.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = ClassificationPlot.add_subplot(gs[1, 1], sharey=ax)
        a = 0
        b = 1
        x = features[:, a] #best fisher or PC0
        y = features[:, b] #second best fisher or PC1

        binsx = np.linspace(min(x), max(x), 35) 
        binsy = np.linspace(min(y), max(y), 35) 
        
        
        if pca != None:
            ClassificationPlot.suptitle(f"Classification based on PCA and {Name} \n Silhouette score = {np.round(silh, 2)}")
            ax.set_xlabel(f"Principal Component 1: Explained Variance = {np.round(pca.explained_variance_ratio_[0] * 100, 2)}%")
            ax.set_ylabel(f"Principal Component 2: Explained Variance = {np.round(pca.explained_variance_ratio_[1] * 100, 2)}%")
        else:
            ClassificationPlot.suptitle(f"Classification based on best features \n Silhouette score = {np.round(silh, 2)}")
            ax.set_xlabel(main.fisher_features[a])
            ax.set_ylabel(main.fisher_features[b])

        #centers = main.centroids(best_clustering.labels_, features)
        #print(centers)
        for i in np.unique(labels):

            ##plotting 2 randomly chosen cells from the clusters:
            if plot_cells:
                cells = main.cells.loc[labels == i].copy()
                # print(np.array(cells.loc[:, "Cell Object"]))
                # print(cells.loc[:, "Cell Object"].to_list())
                # print("Shape test: ", np.shape(cells)[0])
                if np.shape(cells)[0] == 1:
                    random_cells = random.sample(cells.index.to_list(), 1)
                else: 
                    random_cells = random.sample(cells.index.to_list(), 2)
                #print("RandomCell", random_cells)

                fig = plt.figure(figsize = (10,5))
                j = 1
                for ind in random_cells:
                    cellplot = fig.add_subplot(1, len(random_cells), j, projection = "3d")
                    plot_cell(cellplot, cells.loc[ind, "Cell Object"], Annotate= False, color = main.cmap(int(i)))
                    cellplot.set_title(f"Cell {ind} of cluster {i}")

                    #print(x[ind], y[ind])
                    j+=1
                
                fig.suptitle(f"Cells of cluster {i+1}")
                fig.tight_layout()
                fig.savefig("./ResultPlots_" + main.Type + f"/CellsCLuster{i+1}.png")

                
            
            xi = x[labels == i]
            cxi = np.mean(xi)
            yi = y[labels == i]
            cyi = np.mean(yi)
            #print(xi, yi)
            #plot centroids
            ax.scatter(xi, yi, color = main.cmap(int(i)))
            ax.scatter(cxi, cyi,marker = "x", color = main.cmap(int(i)), label = f"Cluster {i +1}")
            ax.annotate(f"C{i+1}", (cxi, cyi))
            
            ax_histx.hist(xi ,binsx, histtype='bar', alpha = 0.5, color= main.cmap(int(i)))
            ax_histy.hist(yi,binsy, histtype='bar', alpha = 0.5, orientation='horizontal', color= main.cmap(int(i)))
            #
            if len(xi) <= 2:
                pass
            elif pca == None:
                pass
            # elif i == 1:
            #     pass
            else:
                pass
                # #x_c , y_c = np.linspace(min(x), max(x)), np.linspace(min(), max(x))
                # X, Y = np.mgrid[min(x) - 5: max(x) + 5: .1, min(y)- 5: max(y) + 5:.1]
                # #print("X,Y", np.shape(min(x): max(x)))
                # cov = np.cov((xi, yi))
                # rv = multivariate_normal([cxi, cyi], cov)
                # data = np.dstack((X,Y))
                # z = rv.pdf(data)

                # # figure = plt.figure()
                # # ax = figure.add_subplot(projection = "3d")
                # # ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                # #        linewidth=0, antialiased=False)
                # #print("Z:", z, np.shape(z))
                # #print(main.cmap(int(i))[:3])
                # ax.contour(X,Y, z, colors = [main.cmap(int(i))[:3]], levels = 3, extent=(-1, 1, -1, 1), alpha = 0.5)
                # #print(cov)

        ax_histx.set_ylabel("Counts", loc = "top")
        ax_histy.set_xlabel("Counts")
    ClassificationPlot.legend(prop={'size': 10})
    
    ClassificationPlot.tight_layout()

    ClassificationPlot.savefig("./ResultPlots_" + main.Type + "/2D_Clustering_" + Name + "_" + main.scaler +".png")

def fisher(main, fischer_matrix, fisher_scores):
    fig_fisherscores = plt.figure(figsize = (6,6))
    ax = fisher_scores.plot.barh()
    ax.set_xlabel("Fisher Inforamation Value")
    ax.set_ylabel("Features")
    ax.set_title("Fisher information values per feature")
    
    #ax.set_xscale("log")
    #ax.set_xlim(left = 0, right = max(fisher_scores))
    fig_fisherscores.add_axes(ax)
    fig_fisherscores.tight_layout()
        
    fig_fishermatrix = plt.figure(figsize= (8,6))
    ax = sns.heatmap(fischer_matrix, annot=False, cbar = True, robust = True, cbar_kws = {"shrink": 0.5},xticklabels = True, yticklabels = True)
    ax.set_title("Fisher information matrix of the features")
    fig_fishermatrix.add_axes(ax)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(),  ha='right', fontsize=5)
    
    ax.set_yticklabels(ax.get_yticklabels(), ha='right', fontsize=5)
    fig_fishermatrix.tight_layout()
    
    fig_fishermatrix.savefig("./ResultPlots_" + main.Type + "/FisherInfoMatrix_" + main.scaler +".png")
    fig_fisherscores.savefig("./ResultPlots_" + main.Type + "/FisherInfoScores_" + main.scaler +".png")    

def MVEE_Results(datasets: list[str], **kwargs):
    path = os_p.abspath("../datasets/")
    for dataset in datasets:
        # CellsData needs FeatureVector.csv to be present
        # (even though we don't need features here)
        cells = CellsData(os_p.join(path,  dataset + "/"),
                          dataset,
                          Features_to_use = [False, False, False, False],
                          Ellipsoids = True,
                          **kwargs).cells
        random_cell_id, random_cell = random.choice(list(cells.items()))

        fig = plt.figure(figsize = (8,8))
        plot_cell_rot(fig, random_cell)
        fig.suptitle(f"Cell {random_cell_id} with bounding ellipsoid")
        fig.tight_layout()
        if not os_p.isdir("./ResultPlots_" + dataset):
            os.mkdir("./ResultPlots_" + dataset)
        fig.savefig("./ResultPlots_" + dataset + "/MVEE_CellRot")

def EdgeCellRemoval(main): #PLT FKT
    #Methods figure--------------------
    fig = plt.figure(figsize= (10, 5))
    ax1 = fig.add_subplot(121, projection= "3d")
    ax1.set_title("Entire sample cube")
    plot_cube(ax1, main.cells, Surfaces= True)

    ax2 = fig.add_subplot(122, projection = "3d")
    plot_cube(ax2, main.cells, Surfaces= True, zoom = True)
    ax2.legend()
    # plot_EdgeCell(ax2, c, cells.limits)
    fig.suptitle("Visual representation of sample cube with threshold border")
    fig.tight_layout()
    fig.savefig("./MethodPlots/EdgeCellRemoval")

    #Results figure of 5 random cells --------------------------
    fig = plt.figure(figsize= (10,12))
    ax1 = fig.add_subplot(321, projection = "3d")
    ax1.set_title("United plot of edge cells")
    i = 2
    #plot random cells...
    random_cells = random.sample(main.cells.edgeCells.items(), 5)
    for cell_hits in random_cells:
        
        ax2= fig.add_subplot(3, 2, i, projection = "3d")
        plot_cell(ax1, cell_hits[1][0], color= f"C{i}")
        plot_EdgeCell(ax2, cell_hits[1], main.cells.limits, surface= True, zoom = True, color = f"C{i}")
        i += 1

    plot_cube(ax1, main.cells)
    
    fig.suptitle("Plots of 5 randomly chosen Edge cells")    
    fig.tight_layout()
    fig.savefig(f"./ResultPlots_" + main.Type + "/EdgeCellRemoval")

    #Soma figure -----------------------
    fig = plt.figure(figsize= (10, 5))
    ax1 = fig.add_subplot(121, projection= "3d")
    ax2 = fig.add_subplot(122, projection= "3d")
    
    plot_somas(ax1, main.cells.cells, color = "b", label = "Non Edge Cells")
    plot_somas(ax1, main.cells.edgeCells, label = "Edge Cells")
    plot_cube(ax1, main.cells, Surfaces= True)
    
    plot_somas(ax2, main.cells.cells, color = "b")
    plot_cube(ax2, main.cells, Surfaces= True)

    ax1.set_title("Plot of all cell somas")
    ax2.set_title("Plot of remaining cells somas")

    fig.suptitle("Cell somas in sample cube after edge cell removal process")
    fig.tight_layout()
    fig.savefig(f"./ResultPlots_" + main.Type + "/EdgeCellRemoval_Somas")

def FeatureVector(main):
    #features = [ "Volume", "Roundness", "Sphericity", "N_Endpoints", "av_Branchdiameter", "Alpha", "Beta", "Main Axis"]
    features = main.FeatureVector.columns
    units = main.units
    idx = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    for j in range(round(len(features)/ 8 + 0.5)):
        
        i = 1
        fig = plt.figure(figsize = (10, 12))
        features_j = features[j*8: 8+j*8]
        units_j = units[j*8: 8+j*8]
        #print(features_j, units_j)
        for feature, unit, idx_i in zip(features_j, units_j, idx):
            
            data = main.FeatureVector.loc[:, feature]
            nr_bins  = int(np.sqrt(len(data)))
            ax = fig.add_subplot(4, 2, i)
            if feature == "Main Axis":
                plot_Histo(data, ax, "Main Axis length", pdf = False,
                                axes = ["Main Axis length", "count"], units = [unit, "a.u."], bins = nr_bins)
            else:
                plot_Histo(data, ax, feature, pdf = False,
                                axes = [feature, "count"], units = [unit, "a.u."], bins = nr_bins )
            ax.grid()
            ax.annotate(idx_i, (0, 0.9), xycoords = ax)
            i += 1
        fig.tight_layout()
        fig.savefig("./ResultPlots_" + main.Type + "/FeatureVectorHist"+str(j)+".png")

def plot_largest_cells(main, figsize = (10, 6)):
    
    #Plot 3 largest volume cells:
    idx = np.argsort(np.array(main.FeatureVector.loc[:, "Volume"]))[-5:-1]
    j = 1
    fig = plt.figure(figsize = figsize)
    fig.suptitle("3 lagest cells according to Volume")
    for i in idx:
        ax = fig.add_subplot(2,2, j, projection = "3d")
        dataset, ID = main.datasetmarkers.iloc[i], main.datasetmarkers.index[i]
        
        cell = main.datasets_dict[dataset].cells[ID]
        plot_cell(ax, cell)
        
        ax.set_title(dataset + " " + str(ID))
        j+=1
    fig.tight_layout()
    fig.savefig("./ResultPlots_" + main.Type + "/4LargestCells_Volume.png")

    idx = np.argsort(np.array(main.FeatureVector.loc[:, "N_Endpoints"]))[-5:-1]
    j = 1
    fig = plt.figure(figsize = figsize)
    fig.suptitle("3 lagest cells according to N_Endpoints")
    for i in idx:
        ax = fig.add_subplot(2,2, j, projection = "3d")
        dataset, ID = main.datasetmarkers.iloc[i], main.datasetmarkers.index[i]
        
        cell = main.datasets_dict[dataset].cells[ID]
        plot_cell(ax, cell)

        ax.set_title(dataset + " " + str(ID))
        j+=1
    fig.tight_layout()
    fig.savefig("./ResultPlots_" + main.Type + "/4LargestCells_NEndpoints.png")

    av_Volumes = main.FeatureVector.loc[:, "Volume"] / np.mean(main.FeatureVector.loc[:, "Volume"] )
    av_Npoints = main.FeatureVector.loc[:, "N_Endpoints"] / np.mean(main.FeatureVector.loc[:, "N_Endpoints"] )
    av_size = (av_Volumes + av_Npoints) / 2
    idx = np.argsort(av_size)[-5:-1]
    j = 1
    fig = plt.figure(figsize = figsize)
    fig.suptitle("3 lagest cells according to both")
    for i in idx:
        
        ax = fig.add_subplot(2,2, j, projection = "3d")
        dataset, ID = main.datasetmarkers.iloc[i], main.datasetmarkers.index[i]
        
        cell = main.datasets_dict[dataset].cells[ID]
        plot_cell(ax, cell)

        ax.set_title(dataset + " " + str(ID))
        j+=1
    fig.tight_layout()
    fig.savefig("./ResultPlots_" + main.Type + "/4LargestCells_both.png")

def plot_sparsevsthick_cells(main,  figsize = (10, 6)):
    
    idx1 = np.argsort(np.array(main.FeatureVector.loc[:, "N_Branchingpoints"]))[-3:-1]
    idx2 = np.argsort(np.array(main.FeatureVector.loc[:, "N_Branchingpoints"]))[0:2]

    j = 1
    fig = plt.figure(figsize = figsize)
    fig.suptitle("2 dense vs 2 sparse cells according to N_Endpoints")
    for i in idx1:
        ax = fig.add_subplot(2,2, j, projection = "3d")
        dataset, ID = main.datasetmarkers.iloc[i], main.datasetmarkers.index[i]
        
        cell = main.datasets_dict[dataset].cells[ID]
        plot_cell(ax, cell)

        ax.set_title("Dense " + dataset + " " + str(ID))
        j+=1
    
    for i in idx2:
        ax = fig.add_subplot(2,2, j, projection = "3d")
        dataset, ID = main.datasetmarkers.iloc[i], main.datasetmarkers.index[i]
        
        cell = main.datasets_dict[dataset].cells[ID]
        plot_cell(ax, cell)

        ax.set_title("Sparse " + dataset + " " + str(ID))
        j+=1
    fig.tight_layout()
    fig.savefig("./ResultPlots_" + main.Type + "/4SparsevsThickCells.png")

def eval_diverent_k(main, k, silhuette, wcss, gaps, CalinskiHarabasz, DaviesBouldin,  figsize = (6, 4)):
        #optimal_clusters_wcss, silhuette, wcss, gaps, CalinskiHarabasz, DaviesBouldin = main.estimate_n_clusters(k, 4, main.scores_pca)

        fig, (ax1,ax2, ax3, ax4) = plt.subplots(4, figsize = figsize)
        
        ax1.plot(silhuette)
        ax1.set_title(f"Silhuette score over k \n k = {np.nanargmax(silhuette)}")
        ax1.scatter(np.nanargmax(silhuette), np.nanmax(silhuette), marker = "x", c = "r")

        ax2.plot(gaps)
        ax2.set_title("Gap score over k")
        ax3.plot(CalinskiHarabasz)
        ax3.set_title(f"Calinski Habarasz score over k \n k = {np.nanargmax(CalinskiHarabasz)}")
        ax3.scatter(np.nanargmax(CalinskiHarabasz), np.nanmax(CalinskiHarabasz), marker = "x", c = "r")
        ax4.plot(DaviesBouldin)
        ax4.set_title(f"Davies Bouldin score over k \n k = {np.nanargmin(DaviesBouldin)}") #best is min
        ax4.scatter(np.nanargmin(DaviesBouldin), np.nanmin(DaviesBouldin), marker = "x", c = "r")

        fig.tight_layout()
        
        fig.savefig("./ResultPlots_" + main.Type + "/DiverntKevaluations_" + main.scaler + ".png")
    
        pass

























































def hide():
# --------------------------
# Unsused Plotting Funktions
# --------------------------
# def plot N_pointHistographs
    # fig = plt.figure()
    # i = 1
    # loc: str = "FeatureVectors", 
    # artefact_removal = None 
    # Branches = True
    # Shells = True
    # Quadrants = False
    # Ellipsoids = True
    # reload_features = False 
    # threshold = None
    # path = os_p.abspath(".")

    # #results_path = os_p.join(os_p.abspath(".."),  loc, "United")

    # for dataset in datasets:
    #     ax = fig.add_subplot(3,2,i)

    #     cells = cells = CellsData(os_p.join(path,  dataset + "//"), set, reload_features= reload_features, 
    #                         Branches= Branches, Shells = Shells, Quadrants = Quadrants,
    #                         Ellipsoids= Ellipsoids, artefact_removal = artefact_removal)

    #     if threshold != None:
    #         cells.FeatureVector = cells.FeatureVector.loc[cells.FeatureVector["N_Branchingpoints"] >= 100]

    #     bins = np.arange(0, 8000, 200)
    #     ax.hist(cells.FeatureVector["N_Branchingpoints"], bins = bins)
    #     ax.set_title(dataset)


    #     i += 1
    # fig.tight_layout()
    # fig.savefig("Histogram_of_N_branchingpooints_dataset_TH_" + str(threshold)+ ".pdf")


# def silhouete_k(main, iterations):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     av_sihls = np.zeros((len(np.unique(main.Storage._optimal_clusters_wcss)), 2))
#     i = 0
#     for j in np.unique(main.Storage._optimal_clusters_wcss): 
#         ax1.hist(main.Storage._optimal_clusters_wcss[main.Storage._optimal_clusters_wcss == j], color = main.cmap(int(j)),
#                 alpha = 1, bins = range(11),align = "left", rwidth = 0.8, label = f"K = {int(j)}")
#         mean , var = np.mean(main.Storage._silhouette[main.Storage._optimal_clusters_wcss == j]), np.var(main.Storage._silhouette[main.Storage._optimal_clusters_wcss == j])
#         #print(f"Cluster k = {j}:", mean, var)
#         av_sihls[i, :] = np.array([mean, var])
#         i += 1

#     ax1.set_xlabel("Optimal clusters scores")
#     ax1.set_ylabel("Counts")
    
#     ax1.legend()
#     fig.suptitle(f"Optimal k for clustering including average silhouette")
#     fig.savefig("./ResultPlots_" + main.Type + "/KClustering_" + main.scaler +".png")

# def gapvwcss(main):
#     fig_gapvswcss = plt.figure()
#     x = range(1, main.k)
#     ax1 = fig_gapvswcss.add_subplot(111)
#     ax1.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
#     ax2 = ax1.twinx()

#     ax1.axvline(int(np.round(main.w_meanclusters, 1)), label = "weighted mean knee wcss", ls = "--", color = "orange")
    
#     for j in np.unique(main.Storage._optimal_clusters_wcss):
#         #print(main.Storage._wcss[main.Storage._optimal_clusters_wcss == j])
#         ax1.plot(x, np.mean(main.Storage._wcss[main.Storage._optimal_clusters_wcss == j], 0),
#                 marker = 0, ls = "-", color = main.cmap(int(j)), alpha = 1, label = f"k = {int(j)}")
#         ax2.plot(x, np.mean(main.Storage._gaps[main.Storage._optimal_clusters_wcss == j], 0),
#                 marker = 'x', ls = "-.", color = main.cmap(int(j)), alpha = 0.8, label = f"mean gap score for k = {int(j)}")
#         #Silh score
            
#         #print(np.shape(main.Storage._silhouette_list[main.Storage._optimal_clusters_wcss == j])[0])
        
        
#     ax1.set_title(f"Sum of squared in between cluster distances (WCSS) vs gapp score over {iterations} iterations")
#     ax1.set_ylabel("WCSS")
#     ax2.set_ylabel("Gap score")
#     #ax3.set_ylabel("Silhuette Score")
#     ax1.set_xlabel("Number of clusters")
#     #fig_sillvsdistance.tight_layout()
#     ax1.legend(loc = "center right")

#     fig_gapvswcss.suptitle(f"Ellbow method")

#     fig_gapvswcss.savefig("./ResultPlots_" + main.Type + "/Iter_Ellbow_" + main.scaler +".png")
   
# def silhouette(main):
#     fig_silhouttescore = plt.figure()
#     ax = fig_silhouttescore.add_subplot(111)
#     for j in np.unique(main.Storage._optimal_clusters_wcss):

#         for i in range(np.shape(main.Storage._silhouette_list[main.Storage._optimal_clusters_wcss == j])[0]):
#                     ax.plot(main.Storage._silhouette_list[main.Storage._optimal_clusters_wcss == j][i],
#                             color = main.cmap(int(j)) , alpha = 0.2)
#     ax.legend()
#     ax.set_title(f"Silhouette score")
#     ax.set_ylabel("Silhouette Score")
#     ax.set_xlabel("Number of clusters")
#     #ax1.legend()
#     fig_silhouttescore.savefig("./ResultPlots_" + main.Type + "/Iter_SilhouetteScore_" + main.scaler +".png")
    pass