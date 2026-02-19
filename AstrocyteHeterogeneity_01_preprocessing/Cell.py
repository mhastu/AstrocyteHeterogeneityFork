#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_functions as fkt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from scipy.spatial.transform import Rotation as R

def calculate_angle(matrix):
    angles = []
    for i in range(len(matrix) - 1):
        for j in range(i + 1, len(matrix)):
            dot_product = np.dot(matrix[i], matrix[j])
            norm_a = np.linalg.norm(matrix[i])
            norm_b = np.linalg.norm(matrix[j])
            cosine_angle = dot_product / (norm_a * norm_b)
            angle_radians = np.arccos(cosine_angle)
            angle_degrees = np.rad2deg(angle_radians)
            angles.append(angle_degrees)
    return angles

#%%
class Cell:
    def __init__(self, CellID: int, BranchingPointsData, diaData, points, connections) -> None:
        
        self._cellID = CellID
        self.BranchingPointsData, diaData, self.points, self.connections = BranchingPointsData, diaData, points, connections
        self.BranchingPointsData.loc[:, "PtDiameter"] = diaData.loc[:, "PtDiameter"]
        self.BranchingPointsData = self.BranchingPointsData[["PtPositionX", "PtPositionY", "PtPositionZ", "PtDiameter", "ID", "Depth", "Type"]]
        self.fine_branches = {}
        self.rough_branches = {}
        self._find_soma()
        self.soma_radius = 5  #Softcode soma radius!
        
    @property
    def cellID(self):
        return self._cellID
    
    @property
    def features(self):
        return self._features

    @property
    def sholl_features(self):
        return self._sholl_features
    
    @property
    def quadrant_features(self):
        return self._Quadrant_shell
    
    @property
    def branch_features(self):
        return self._branch_features
    
    def load_features(self):
        path = f"./Cellsdata/CellsBranchingPointsData/cell{self.cellID}.csv"
        self.BranchingPointsData = pd.read_csv(path, sep = ",", index_col= 0)

    def get_features(self, percent_points = 0.70):
        self._features = pd.Series()
        
        self._features.loc["Volume"], self._features.loc["Roundness"], self._features.loc["Sphericity"], self._features.loc["Aspect Ratio"]  = self._get_VolumeRoundnessSphereicity()
            
        self._features.loc["N_Endpoints"] = self._get_N_Endpoints()
        self._features.loc["N_Branchingpoints"] = self._get_N_Branchingpoints()
        self._features.loc["Radius"]  = self.fit_sphere(percent_points)
        self._features.loc["Primary Angles"] = self._get_primary_branche_angles()
        self._features.loc["Fractal Dimension"] = max(self.BranchingPointsData["Depth"])

        #self._features.loc["av_Branchdiameter"] = self._get_avBranchDiameter()
        

    def fit_sphere(self, percent_points, max_iter = 100): #TODO

        point_vectors = np.array(self.BranchingPointsData.loc[:, ["PtPositionX", "PtPositionY", "PtPositionZ"]].loc[self.BranchingPointsData.loc[:, "Type"] == "Dendrite Terminal"] - self.somaLoc[0:3], dtype= float)

        r_vec = np.sqrt(np.sum(point_vectors**2, 1))
        r= max(r_vec) / 2
        i = 0
        perc_err1 = np.shape(point_vectors[r_vec <= r])[0] / np.shape(point_vectors)[0] -percent_points
        
        dr = r/10
        while abs(perc_err1) >= 0.01:
            if perc_err1 >= 0:
                r = r - dr
            else:
                r = r + dr

            if i == 0:
                pass
            else:
                if abs(perc_err1) < abs(perc_err2):
                    dr *= 0.8
                else:
                    dr *= 1.2

            perc_err2 = perc_err1
            perc_err1 = np.shape(point_vectors[r_vec <= r])[0] / np.shape(point_vectors)[0] - percent_points
            
            i += 1

            if i == max_iter:
                return r

        return r

    def find_edgepoints(self):
        """Generate the edgepoitns of the cell in every direction"""
        max_x = max(self.points[:, 0])
        max_y = max(self.points[:, 1])
        max_z = max(self.points[:, 2])
        min_x = min(self.points[:, 0])
        min_y = min(self.points[:, 1])
        min_z = min(self.points[:, 2])
        return np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])
    
    def return_features(self, General = False, **kwargs):
        if General == True:
            self.get_features(**kwargs)
        else: 
            try:
                self._features
            except AttributeError:
                self.get_features(**kwargs)
            else:
                pass

        return self._features
    
    def _get_primary_branche_angles(self): #TODO
        primary_branches = self.BranchingPointsData[self.BranchingPointsData.loc[:, "Depth"] == 1.0].to_numpy()[:, 0:3]
        r_vec = primary_branches - self.somaLoc[0:3]
        return np.mean(calculate_angle(r_vec))
    
    def _get_main_branches(self, depth = 7):
        """generate a dictionary containing the branching infomation
        Quite time consuming"""
        self._connect_Datasets()
        print("Generating the fine Branches...", self.cellID)

        indexes = np.where(self.connections[:, 0] != self.connections[:, 1] - 1)[0]
        
        #first Branch
        startpoint = 0
        endpoint = indexes[0]
        mergepoint = self.connections[0, 0]#
        
        #Dict with key = mergepoint, keys = point pos, point ID
        self.fine_branches[0] = self.points[startpoint:endpoint, :]
        self.rough_branches[0] = self.BranchingPointsData.loc[(self.BranchingPointsData["corresponding Index"] <= endpoint)].copy()
        
        #go over all mergepoints by indexes
        for i in range(len(indexes)-1):
            mergepoint = self.connections[indexes[i], 0]
            
            if self.BranchingPointsData.loc[self.BranchingPointsData["corresponding Index"] == mergepoint]["Depth"].to_numpy() <= depth:
                
                endpoint = indexes[i+1]    
                startpoint = indexes[i] + 1
                
                self.fine_branches[i+1] = np.vstack((self.points[mergepoint], self.points[startpoint:endpoint, :]))
                self.rough_branches[i+1] = self.BranchingPointsData.loc[self.BranchingPointsData["corresponding Index"] == mergepoint].copy()
                self.rough_branches[i+1] = self.rough_branches[i+1].append(self.BranchingPointsData.loc[(self.BranchingPointsData["corresponding Index"] <= endpoint) & (self.BranchingPointsData["corresponding Index"] >= startpoint)])
            else:
                pass

    def _get_branches(self):
        """generate a dictionary containing the branching inforamtion
        Quite time consuming"""
        self._connect_Datasets()
        print("Generating the fine Branches...", self.cellID)
        
        indexes = np.where(self.connections[:, 0] != self.connections[:, 1] - 1)[0]
        #first Branch
        startpoint = 0
        endpoint = indexes[0]
        mergepoint = self.connections[0, 0]#
    
        #Dict with key = mergepoint, keys = point pos, point ID
        self.fine_branches[0] = self.points[startpoint:endpoint, :]
        self.rough_branches[0] = self.BranchingPointsData.loc[(self.BranchingPointsData["corresponding Index"] <= endpoint)].copy()
        #go over all mergepoints by indexes
        for i in range(len(indexes)-1):
            mergepoint = self.connections[indexes[i], 0]
            endpoint = indexes[i+1]    
            startpoint = indexes[i] + 1

            self.fine_branches[i+1] = np.vstack((self.points[mergepoint], self.points[startpoint:endpoint, :]))
            self.rough_branches[i+1] = self.BranchingPointsData.loc[self.BranchingPointsData["corresponding Index"] == mergepoint].copy()
            self.rough_branches[i+1] = self.rough_branches[i+1].append(self.BranchingPointsData.loc[(self.BranchingPointsData["corresponding Index"] <= endpoint) & (self.BranchingPointsData["corresponding Index"] >= startpoint)])

    def _connect_Datasets(self):
        """Function that matches the branching to the fine point data for connection information"""
        
        # create np array for excel data [x, y, z, dia] from dataframes sorted together by ID
        
        pos = self.BranchingPointsData[["PtPositionX", "PtPositionY", "PtPositionZ"]].copy().to_numpy() #Extract only Position Data
        correspondingIdx = np.zeros(np.size(pos,0))
        
        for i, position in enumerate(pos):
            correspondingIdx[i] = int(np.argmin((position[0] - self.points[:, 0])**2 + (position[1] - self.points[:, 1])**2 + (position[2] - self.points[:, 2])**2))
        
        self.BranchingPointsData.loc[:, "corresponding Index"] = correspondingIdx
      
    def get_CellRadius(self, type = "mean"):
        """Generate a rough estiamte of the cells radius.
        type = str: way of estimation radius, possible are 'mean' and 'max'"""
        self._find_soma()
        #takes only branch terminals into account
        #either takes the max or the mean in distance to soma
        sx = self.somaLoc[0]
        sy = self.somaLoc[1]
        sz = self.somaLoc[2]

        x = np.array(self.BranchingPointsData["PtPositionX"].loc[self.BranchingPointsData["Type"] == "Dendrite Terminal"])
        y = np.array(self.BranchingPointsData["PtPositionY"].loc[self.BranchingPointsData["Type"] == "Dendrite Terminal"])
        z = np.array(self.BranchingPointsData["PtPositionZ"].loc[self.BranchingPointsData["Type"] == "Dendrite Terminal"])

        if type == "mean":
            self.radius = np.mean(np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2))
        elif type == "max":
            self.radius = np.max(np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2))
        else:
            print("allowed types are mean and max")
        
        return self.radius
  
    def _find_soma(self):
        """Functions withs stores the soma location and ID"""
        self.somaID = self.BranchingPointsData.loc[self.BranchingPointsData["Type"] == "Dendrite Beginning"].loc[0, "ID"]
        self.somaLoc = np.array(self.BranchingPointsData.loc[self.BranchingPointsData["Type"] == "Dendrite Beginning"].loc[0, ["PtPositionX", "PtPositionY", "PtPositionZ", "PtDiameter"]])

    def get_shell_features(self, step = 5, nr_shells = 20, start_radius = 5):
        """Generation of the Sholl features
        step = Int: length of the shell step size
        nr_shells = Int: max number of shells calculated
        start_radius = Int: radius at withc to start shells"""
        #Gnerating shells
        self._BP_shell = pd.Series(dtype = float)
        self._sholl_features = pd.Series(dtype = float)
        to = start_radius
        shell_vector = np.zeros((nr_shells, 2))
        for i in range(nr_shells):
            
            shell_vector[i, :] = [to, to + step]
            to += step

        for i, shell in enumerate(shell_vector):
            nr_points = len(self._get_shellpoints(shell))

            self._BP_shell[shell[1]] = nr_points

        # New sholl features
        self._BP_shell.replace(to_replace = 0, value = np.nan, inplace = True)
        self._BP_shell.dropna(how = "all", inplace = True)
        self._sholl_features["Process maximum"] = self._BP_shell.max()
        #calculate primary branches by addung ub number of branchingpoints with depth  1
        self._sholl_features["Primary Branches"] = len(self.BranchingPointsData[self.BranchingPointsData.loc[:, "Depth"] == 1.0])
        self._sholl_features["Critical Value"] = self._BP_shell.idxmax()
        self._sholl_features["Maximum Radius"] = max(self._BP_shell.index)     

    def get_quadrant_features(self, percent_points = 0.70): #TODO max rad, dynamic computing
        try:
            r = [self.soma_radius, self.max_radius]
        except AttributeError:
            self.max_radius = self.fit_sphere(percent_points)
        self._Quadrant_shell = pd.Series(dtype = float)        
        
        self._Quadrant_shell["Q1"]  = self._get_quadrant_points(r, [0, 90], [0,90])
        self._Quadrant_shell["Q2"]  = self._get_quadrant_points(r, [90, 180], [0,90])
        self._Quadrant_shell["Q3"]  = self._get_quadrant_points(r, [-180, -90], [0,90])
        self._Quadrant_shell["Q4"]  = self._get_quadrant_points(r, [-90, 0], [0,90])
        self._Quadrant_shell["Q5"]  = self._get_quadrant_points(r, [0, 90], [-90,0])
        self._Quadrant_shell["Q6"]  = self._get_quadrant_points(r, [90, 180], [-90,0])
        self._Quadrant_shell["Q7"]  = self._get_quadrant_points(r, [-180, -90], [-90,0])
        self._Quadrant_shell["Q8"]  = self._get_quadrant_points(r, [-90, 0], [-90,0])           

        #self._Quadrant_shell.replace(to_replace = 0, value = np.nan, inplace = True)
        self._Quadrant_shell.dropna(how = "all", inplace = True)
        
    def _get_quadrant_points(self, r_quad, phi_quad, theta_quad):
        point_vectors = np.array(self.BranchingPointsData.loc[:, ["PtPositionX", "PtPositionX", "PtPositionZ"]] - self.somaLoc[0:3], dtype= float)

        r = np.sqrt(np.sum(point_vectors**2, 1))
        phi = np.arcsin(point_vectors[:, 2] / r) / np.pi * 180
        theta = np.arctan(point_vectors[:, 1] / point_vectors[:, 0]) / np.pi * 180
        polarcoords = np.array([r, phi, theta])

        points_withinQuadrant = polarcoords[:, polarcoords[0, :] >= r_quad[0]]

        points_withinQuadrant = points_withinQuadrant[:, points_withinQuadrant[0, :] < r_quad[1]]
        points_withinQuadrant = points_withinQuadrant[:, points_withinQuadrant[1, :] >= phi_quad[0]]
        points_withinQuadrant = points_withinQuadrant[:, points_withinQuadrant[1, :] < phi_quad[1]]
        points_withinQuadrant = points_withinQuadrant[:, points_withinQuadrant[2, :] >= theta_quad[0]]
        points_withinQuadrant = points_withinQuadrant[:, points_withinQuadrant[2, :] < theta_quad[1]]

        return np.shape(points_withinQuadrant)[1]
    
    def _get_shellpoints(self, shell):
        #http://kylebarbary.com/nestle/index.html
        #https://www.youtube.com/watch?v=DV4GjHH-fvc sphere plot
        # https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
        #calculate shperic shells
        points_withinShell = self.BranchingPointsData.loc[
            np.sqrt((self.BranchingPointsData.loc[:, "PtPositionX"]- self.somaLoc[0])**2 + 
                    (self.BranchingPointsData.loc[:, "PtPositionY"]- self.somaLoc[1])**2 +
                    (self.BranchingPointsData.loc[:, "PtPositionZ"]- self.somaLoc[2])**2) <= shell[1]].copy()
        points_withinShell = points_withinShell.loc[np.sqrt((points_withinShell.loc[:, "PtPositionX"]- self.somaLoc[0])**2 + (points_withinShell.loc[:, "PtPositionY"]- self.somaLoc[1])**2 + (points_withinShell.loc[:, "PtPositionZ"]- self.somaLoc[2]) ** 2) > shell[0]]
        
        return points_withinShell
 
    def _get_VolumeRoundnessSphereicity(self, tol = 0.01, depth = 10):
        try:
            a, b, c = self.ell_params.loc[["a", "b", "c"]]
        except AttributeError:
            self._get_ellipsoid(tol, depth)
            a, b, c = self.ell_params[0:3]
        else: 
            pass
        
        aspect_ratio = a / c

        #a, b, c = self.ell_params.loc[["a", "b", "c"]]
        volume = 4 / 3 * np.pi * a * b * c
        roundness = 1/3 * (a+b+c)/ max([a,b,c])
        #spericity, ell surface estiamtion
        p = 1.6075
        r_sphere = np.cbrt(3/4 *(np.pi * volume))
        sphere_surface = 4 * np.pi * r_sphere
        ell_surface = 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3)**(1/p)

        sphericity = sphere_surface / ell_surface
       
        return volume, roundness, sphericity, aspect_ratio

    def _get_N_Branchingpoints(self):
        N_Branchingpoints = self.BranchingPointsData.loc[self.BranchingPointsData.loc[:, "Type"] == "Dendrite Branch"].shape[0] 
        return N_Branchingpoints
    
    def _get_N_Endpoints(self):
        N_endpoints = self.BranchingPointsData.loc[self.BranchingPointsData.loc[:, "Type"] == "Dendrite Terminal"].shape[0] 
        return N_endpoints
     
    def get_branch_features(self, depth = 1): 
        if self.rough_branches:
            pass
        else:
            self._get_main_branches(depth)

        branchlenghts = np.zeros(len(self.rough_branches.keys()))
        N_Branches_perBranch = np.zeros(len(self.rough_branches.keys()))
        mean_dia_list = np.zeros(len(self.rough_branches.keys()))
        
        i = 0
        for id, branch in self.rough_branches.items():
            
            if len(branch) <= 1:
                pass
            else:
                startdepth = branch.iloc[0, 5]
                if startdepth <= 10: 
                    x = branch.loc[:, "PtPositionX"].to_numpy()
                    y = branch.loc[:, "PtPositionY"].to_numpy()
                    z = branch.loc[:, "PtPositionZ"].to_numpy()

                    #calc pointwise length of the branch
                    dist = np.sqrt((x[0: np.size(x) - 1] - x[1: np.size(x)])**2 + (y[0: np.size(y) - 1] - y[1: np.size(y)])**2 +(z[0: np.size(z) - 1] - z[1: np.size(z)])**2)
                    if sum(dist == 0):
                        print(id)
                    rough_branchlength = np.sum(dist)

                    dia = branch.loc[:, "PtDiameter"].to_numpy()
                    
                    av_dia = np.sum((dia[0: np.size(dia) - 1] + dia[1: np.size(dia)])/2 * dist) / rough_branchlength
                    
                    branch.loc[:, "Distance to prio point"] = np.hstack((0, dist))
                    nr_branches = len(x)
                    
                    mean_dia_list[i] = av_dia
                                    
                    branchlenghts[i] = rough_branchlength
                    #start_depths[i] = startdepth
                    N_Branches_perBranch[i] = nr_branches
                    i += 1
        
        mean_dia_list = mean_dia_list[mean_dia_list != 0]
        
        max_dia = max(mean_dia_list)
        std_dia = np.std(mean_dia_list)
        #min_dia = min(mean_dia_list)
        mean_dia = np.mean(mean_dia_list)

        max_branchlenght = max(branchlenghts)
        mean_branchlenght = np.mean(branchlenghts[branchlenghts != 0])
        std_branchlengtgh = np.std(branchlenghts[branchlenghts != 0])
        mean_N_Branches_perBranch = np.mean(N_Branches_perBranch[N_Branches_perBranch != 0])
        max_N_Branches_perBranch = max(N_Branches_perBranch)

        index = ["mean dia", "max dia", "std dia", "max branchlenght", "mean branchlenght", "std branchlenght","mean N_Branches_perBranch", "max N_Branches_perBranch"]
        self._branch_features = pd.Series([mean_dia, max_dia, std_dia, max_branchlenght, mean_branchlenght, std_branchlengtgh, mean_N_Branches_perBranch, max_N_Branches_perBranch], index= index)
        self._branch_features.replace(to_replace = 0, value = np.nan, inplace = True)
        self._branch_features.dropna(how = "all", inplace = True)
        self._branch_features.replace(to_replace = np.nan, value = 0, inplace = True)

    def _get_ellipsoid(self, tol = 0.01, depth = 10):
        # https://gist.github.com/Gabriel-p/4ddd31422a88e7cdf953 11.02.2023
        # http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
        # http://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440
        # https://minillinim.github.io/GroopM/dev_docs/groopm.ellipsoid-pysrc.html

        """
        Finds the ellipse equation in "center form"
        (x-c).T * A * (x-c) = 1
        """

        if max(self.BranchingPointsData.loc[:, "Depth"]) <= depth * 2:
            
            depth = max(self.BranchingPointsData.loc[:, "Depth"])/2
        #take outermost points
        xp = self.BranchingPointsData.loc[(self.BranchingPointsData["Type"] == "Dendrite Terminal") & (self.BranchingPointsData["Depth"] >= depth)]["PtPositionX"] #depth as variable
        yp = self.BranchingPointsData.loc[(self.BranchingPointsData["Type"] == "Dendrite Terminal") & (self.BranchingPointsData["Depth"] >= depth)]["PtPositionY"]
        zp = self.BranchingPointsData.loc[(self.BranchingPointsData["Type"] == "Dendrite Terminal") & (self.BranchingPointsData["Depth"] >= depth)]["PtPositionZ"]
        self.ellipse_points   = np.array((xp, yp, zp)).T
        
        N, d = self.ellipse_points.shape
        Q = np.column_stack((self.ellipse_points, np.ones(N))).T
    
        err = tol+1.0
        u = np.ones(N)/N

        while err > tol:
            # assert u.sum() == 1 # invariant
            X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        
            M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
            jdx = np.argmax(M)
            step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
            new_u = (1-step_size)*u
            new_u[jdx] += step_size
            err = np.linalg.norm(new_u-u)
            u = new_u
        
        self.c = np.dot(u, self.ellipse_points)
        self.A = np.linalg.inv(np.dot(np.dot(self.ellipse_points  .T, np.diag(u)), self.ellipse_points)
                - np.multiply.outer(self.c, self.c))/d
        
        U, D, V = np.linalg.svd(self.A)
        a, b, c = 1 / np.sqrt(D)
           
        # is one for a = b = c, get smaller the large the difference between a and b,c
        r = R.from_matrix(V)
        angels = r.as_euler("xyz", degrees = True)
        self.ell_params = a, b, c,angels[0], angels[1], angels[2], self.c[0], self.c[1], self.c[2]
        return self.ell_params


# %%
