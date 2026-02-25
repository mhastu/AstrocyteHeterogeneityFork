#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pymatreader import read_mat
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


import os.path as os_p

from scipy.spatial.transform import Rotation as R

from Cell import Cell
#%%
class CellsData:
    def __init__(self, path: str, dataset: str, Remove_EdgeCells = True, 
                offset = 2, Ellipsoids = False, Features_to_use = [True, True, True, False],
                Features_to_reload = [False, False, False, False],
                artefact_removal = None, **kwargs) -> None:
        """Here the cells data structur is initiated
        Inputs:
            path         ->     Path of stored Matlab data Lenk_Manuscript\C10/)
            dataset      ->     name of the dataset used
            Remove_EdgeCells -> If edge cells should be removed
            offset      ->      offet from the sample cube in /mu m
            Ellipsoid   ->      If Ellipsoid data should be loaded
            reload_features ->  if the feature should be generated or loaded from datafiles
            Branches    ->      if branching data should be generated for all cells
            Quadrants   ->      if Quadrants should be calculated
            artefact_removal    None, if artefact cells should be removed if yes, int to set number of min Bp threshold"""
        
        self.path = path
        print(self.path)
        self.dataset = dataset
        print("Loading Data...")
        # pymatreader ignores matlab tables, so we have to save them as csv in matlab first, then load them here:
        with open(os_p.join(path + "diameterData.csv")) as data:
            self.diaData = pd.read_csv(data)

        with open(path + "positionData.csv") as data:
            self.BranchingPointData = pd.read_csv(data)
        #display(self.BranchingPointData)
        self.FPoints = read_mat(os_p.join(path, "matlab_and_excel_data.mat"))["vFilamentsPoints"]
        self.FEdges = read_mat(os_p.join(path,"matlab_and_excel_data.mat"))["vFilamentsEdges"]

        self.cells = {}
        for cellID in range(len(self.FPoints)):
            cells_to_delete = []
            if len(self.FPoints[cellID]) <= 3: #Delete Cells with less than 2 entries
                cells_to_delete.append(cellID)
        
            else:
                connections = self.FEdges[cellID]  #np array
                points = self.FPoints[cellID] #np array
                Cell_Index = 100000000 + cellID
                Cell_BranchingPointData = self.BranchingPointData.loc[self.BranchingPointData.FilamentID == Cell_Index].sort_values("ID", ignore_index = True) # Position of Branching Points
                Cell_diaData = self.diaData.loc[self.diaData.FilamentID == Cell_Index].sort_values("ID", ignore_index = True) # Diameter at branching Points

                self.cells[cellID] = Cell(cellID, Cell_BranchingPointData, Cell_diaData, points, connections)
        
        for cell in cells_to_delete:
            self.FPoints = np.delete(self.FPoints, (cell), axis=0)
            self.FEdges = np.delete(self.FEdges, (cell), axis=0)    
        
        self._get_cube()
        
        if Remove_EdgeCells:
            print("Removing Edge Cells...")
            self.remove_Cells(offset = offset, threshold= artefact_removal)

        if Ellipsoids:
            self.load_EllipsoidData_from_file()
     
        if Features_to_reload != None:
            print("Reload Features...")
            self.gen_FeatureVector(Features_to_use = Features_to_use, Features_to_reload = Features_to_reload,
                          **kwargs)
        else:
            print("Generate Features...")
            self.load_FeatureVector(Features_to_use = Features_to_use)
            print("done generating Features")
            # self.load_FeatureVector(Ellipsoids)
            # self.load_Shell_FeatureVector()
            # self.FeatureVector = pd.concat([self.FeatureVector, self.Shell_FeatureVector], axis= "columns")

            # if Branches:
            #     self.load_Branch_FeatureVector()
            #     self.FeatureVector = pd.concat([self.FeatureVector, self.Branch_FeatureVector], axis= "columns")   
        
    def get_celln(self):
        return len(self.cells)

    def return_Cells(self):
        return self.cells

    def return_Cell(self, cellID):
        print(f"Returning Cell with ID of {cellID}")
        return list(self.cells[cellID])

    def return_CellIDs(self):
        return self.cells.keys()

    def _get_cube(self):
        min_x = []
        max_x = []
        min_y = []
        max_y = []
        min_z = []
        max_z = []
        
        for i, cell in enumerate(self.FPoints):
            
            if len(cell) <= 3:
                pass
            else:
                min_x.append(np.min(cell[:, 0]))
                max_x.append(np.max(cell[:, 0]))
                min_y.append(np.min(cell[:, 1]))
                max_y.append(np.max(cell[:, 1]))
                min_z.append(np.min(cell[:, 2]))
                max_z.append(np.max(cell[:, 2]))
        
        min_x = min(min_x)
        max_x = max(max_x)
        min_y = min(min_y)
        max_y = max(max_y)
        min_z = min(min_z)
        max_z = max(max_z)

        self.limits = [[min_x, max_x], [min_y, max_y], [min_z, max_z]]

    def remove_Cells(self, offset: float = 2., mode: str = "hardlimit", limit: int = 20, threshold = None):
        self.cells_total = self.get_celln()
        print("Cell Nr before removals: ", self.cells_total)
        #remove Artefactcells
        if threshold != None:
            self._get_artefactCells(threshold)
            for key in self.artefactCells.keys():
                del self.cells[key]

            self.cells_used = self.get_celln()
            print("Cell Nr after artefact removal: ", self.cells_used)
        #remove Edgecells
        self._get_edgeCells(offset, mode, limit)

        for key in self.edgeCells.keys():
            del self.cells[key]
        
        self.cells_used = self.get_celln()
        print("Cell Nr after removal: ", self.cells_used)
        print("Nr of removed cells: ", len(self.edgeCells))
        
    def _get_artefactCells(self, threshold): #TODO real artifact removal
        print("Finding Artefact Cells...")

        self.artefactCells = {}
        for ID, cell in self.cells.items():
            N_Bps = cell._get_N_Branchingpoints()
            if N_Bps <= threshold:
                self.artefactCells[ID] = [cell, N_Bps]

    def _get_edgeCells(self, offset: float = 2., mode: str = "hardlimit", limit: int = 50):
        print("Finding Edge Cells...")
        if mode != "hardlimit" and mode != "percentage":
            print(f"Incorrect mode was chosen, pleas give either \"hardlimit\" or \"percentage\" as parameters")
            return
        
        min_x = self.limits[0][0]
        max_x = self.limits[0][1]
        min_y = self.limits[1][0]
        max_y = self.limits[1][1]
        min_z = self.limits[2][0]
        max_z = self.limits[2][1]

        self.edgeCells = {}
        #find and filter out edge cells
        for ID, cell in self.cells.items():
            #print(cell[2], cell[2][0])
            j = np.sum(cell.points[:, 0] <= min_x + offset) + np.sum(cell.points[:, 0] >= max_x - offset)
            j += np.sum(cell.points[:, 1] <= min_y + offset) + np.sum(cell.points[:, 1] >= max_y - offset)
            j += np.sum(cell.points[:, 2] <= min_z + offset) + np.sum(cell.points[:, 2] >= max_z - offset)
            #different filter implementations
            if mode == "hardlimit":
                if j >= limit:
                    #print(ID, j)
                    self.edgeCells[ID] = [cell, j]

            elif mode == "percentage":
                nr_points = len(cell[3][0])
                percent = j / nr_points * 100
                if percent >= limit:
                    self.edgeCells[ID] = [cell]
                    
    def plot_somas(self, cells: dict):
        print("Plotting the Cellsomas")
        #plotting of soma as point and maby dentrite terminals? maby with a reduced ellipse size?
        #fig, ax = self.get_cube()
        no_somaDia = []
        cube, ax = self.plot_cube()
        for id, cell in cells.items():
            somaID = cell.BranchingPointData.loc[cell.BranchingPointData["Type"] == "Dendrite Beginning"].loc[0, "ID"]
            somaDia = cell.BranchingPointData.loc[cell.BranchingPointData["Type"] == "Dendrite Beginning"].loc[0, "PtDiameter"]
            somaLoc = np.array(cell.BranchingPointData.loc[cell.BranchingPointData["Type"] == "Dendrite Beginning"].loc[0, ["PtPositionX", "PtPositionY", "PtPositionZ"]])

            if somaID not in cell.BranchingPointData["ID"].unique():     
                no_somaDia.append(id)

            ax.scatter(somaLoc[0], somaLoc[1], somaLoc[2], s = somaDia)
        ax.set_title("Soma plot of material")
        ax.set_xlabel("x-Position in $\mu$m")
        ax.set_ylabel("y-Position in $\mu$m")
        ax.set_zlabel("z-Position in $\mu$m")
        cube.savefig("./CellPlots/SomaPlotCube.pdf")
        return no_somaDia, cube, ax

    def gen_Ellipsoids(self):
        self.Ellipse_params = pd.DataFrame(index = self.cells.keys(), columns = ["a", "b", "c",  "Alpha", "Beta", "Gamma", "cx", "cy", "cz"])
        
        for id, cell in self.cells.items():
            
        # https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix 14.03.2023
        # https://www.sciencedirect.com/science/article/pii/S1877750318304757 01.03.2022 Ellipsoid orientation, roundness
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html 14.03.2023
            
            angels = np.zeros((3))
            a, b, c, angels[0], angels[1], angels[2], cell.c[0], cell.c[1], cell.c[2] = cell._get_ellipsoid()
            
            self.Ellipse_params.loc[id] = a, b, c,angels[0], angels[1], angels[2], cell.c[0], cell.c[1], cell.c[2]
    
        self.save_ellipsoids()
       
    def load_EllipsoidData_from_file(self):
        try:
            self.Ellipse_params = pd.read_csv(self.path + "EllipseParameters.csv", sep = ",", index_col= 0)
            
            for id, cell, in self.cells.items():
                #print(id)
                cell.ell_params = self.Ellipse_params.loc[id]
                #Read a,b,c and the angle values into cell objects
                        
        except FileNotFoundError:
            print("Generationg the ellipsoids...")
            self.gen_Ellipsoids()
        
    def save_ellipsoids(self):
        self.Ellipse_params.to_csv(self.path + "EllipseParameters.csv", sep = ",")
 
    def gen_UniqueFeatureVectors(self, General = False, Branches = False, Shells = False,
                          Quadrants = False, step = 5, nr_shells = 20, start_radius = 5, depth = 4):
        """Generate the Feature Vector from the cells in the dataset
        General --> Boolean if True: encapsulating gernal Features will be calculated
        Branches   --> Bool, if True branch lengths will be calculated
        Shells     --> Bool, if True shell feature vector will be generated"""
        if Shells:
            #Shell_units = pd.Series([])
            self.Shell_FeatureVector = pd.DataFrame()
        if Quadrants:
            self.Quadrant_FeatureVector = pd.DataFrame()
        if Branches:
            self.Branch_FeatureVector = pd.DataFrame()

        if General:
            self.FeatureVector = pd.DataFrame()
        
       
        for id, cell in self.cells.items():
            if General:
                
                self.FeatureVector[id] = cell.return_features(General)
            
            if Branches:
                cell.get_branch_features(depth)
                self.Branch_FeatureVector[id] = cell._branch_features#pd.concat([self.Branch_FeatureVector, cell._branch_features], axis= "index")
            if Shells:
                cell.get_shell_features(step, nr_shells, start_radius)
                self.Shell_FeatureVector[id] = cell._sholl_features

            if Quadrants:
                cell.get_quadrant_features()
                self.Quadrant_FeatureVector[id] = cell.quadrant_features

        if Shells:
            self.Shell_FeatureVector = self.Shell_FeatureVector.copy().transpose()
            self.Shell_FeatureVector.replace(to_replace = 0, value = np.nan, inplace = True)
            self.Shell_FeatureVector.dropna(how = "all", inplace = True)
            self.Shell_FeatureVector.replace(to_replace = np.nan, value = 0, inplace = True)
            self.save_Shell_FeatureVector()

        if Quadrants:
            self.Quadrant_FeatureVector = self.Quadrant_FeatureVector.copy().transpose()
            self.Quadrant_FeatureVector.replace(to_replace = 0, value = np.nan, inplace = True)
            self.Quadrant_FeatureVector.dropna(how = "all", inplace = True)
            self.Quadrant_FeatureVector.replace(to_replace = np.nan, value = 0, inplace = True)
            self.save_Quadrant_FeatureVector()

        if Branches:
            self.Branch_FeatureVector = self.Branch_FeatureVector.copy().transpose()
            self.Branch_FeatureVector.replace(to_replace = 0, value = np.nan, inplace = True)
            self.Branch_FeatureVector.dropna(how = "all", inplace = True)
            self.Branch_FeatureVector.replace(to_replace = np.nan, value = 0, inplace = True)
            self.save_Branch_FeatureVector()

    def reload_features(self, Features_to_reload = [False, False, False, False],
                        **kwargs):
        """reloads the feature Vector by trying to load every wanted Vector. If the Vector does not yet exist it is generated and saved.
        Features_to_use -> List: Bool of the 4 possible seperate feature vectors to choose from
        [General, Shells, Branches, Quadrants]
        This saves a lot of computational power when features dont have to be calculated."""
        
        if Features_to_reload[0]: #Ellipsoids
            print("Generating general Features")
            self.gen_UniqueFeatureVectors(General= True)

        if Features_to_reload[1]: #Shells
            print("Generating Shells")
            self.gen_UniqueFeatureVectors(Shells = True, **kwargs)

        if Features_to_reload[2]: #Branches
            print("Generating Branches")
            self.gen_UniqueFeatureVectors(Branches = True, **kwargs)

        if Features_to_reload[3]: #Quadrants
            print("Generating Quadrants")
            self.gen_UniqueFeatureVectors(Quadrants = True, **kwargs)

    def gen_FeatureVector(self, Features_to_use = [True, True, True, False], Features_to_reload = [False, False, False, False],
                          **kwargs):
        """Generates the feature Vector by trying to load every wanted Vector. If the Vector does not yet exist it is generated and saved.
        Features_to_use -> List: Bool of the 4 possible seperate feature vectors to choose from
        [General, Shells, Branches, Quadrants]
        This saves a lot of computational power when features dont have to be calculated."""
        
        if any(Features_to_reload):
            self.reload_features(Features_to_reload, **kwargs)

        try:
            self.FeatureVector
        except AttributeError:
            self.load_FeatureVector()

        self.FeatureVector = self.FeatureVector.copy().transpose()
        #display(self.FeatureVector)
        if Features_to_use[0]:  #Ellipsoid Features
            try:
                self.Ellipse_params
            except AttributeError:

                self.load_EllipsoidData_from_file()
    
            else: 
                pass
            #display(self.Ellipse_params.loc[:,"Alpha"].to_list())
            #print("Hey")
            #self.FeatureVector.loc[:, "Alpha"]  = self.Ellipse_params.loc[:,"Alpha"].transpose().to_list()
            #self.FeatureVector.loc[:, "Beta"] = self.Ellipse_params.loc[:, "Beta"].transpose().to_list()
            #self.FeatureVector.loc[:, "Gamma"] = self.Ellipse_params.loc[:, "Gamma"].transpose().to_list()
            #self.FeatureVector.loc[:, "Main Axis"] = self.Ellipse_params.loc[:, "c"].transpose().to_list()

        if Features_to_use[1]:  #Shell Features
            try: 
                self.Shell_FeatureVector
            except AttributeError:
                self.load_Shell_FeatureVector()
            self.FeatureVector = pd.concat([self.FeatureVector, self.Shell_FeatureVector], axis= "columns")
        
        if Features_to_use[2]:  #Branch Features
            try: 
                self.Branch_FeatureVector
            except AttributeError:
                self.load_Branch_FeatureVector()
            self.FeatureVector = pd.concat([self.FeatureVector, self.Branch_FeatureVector], axis= "columns")
        
        if Features_to_use[3]:  #Quadrant Features
            try: 
                self.Quadrant_FeatureVector
            except AttributeError:
                self.load_Quadrant_FeatureVector()
            self.FeatureVector = pd.concat([self.FeatureVector, self.Quadrant_FeatureVector], axis= "columns")
    
        self.save_FeatureVector()
                
    def save_FeatureVector(self):
        self.FeatureVector.to_csv(self.path + "FeatureVector.csv", sep = ",")
    
    def save_Shell_FeatureVector(self):
        self.Shell_FeatureVector.to_csv(self.path + "Shell_FeatureVector.csv", sep = ",")
    
    def save_Quadrant_FeatureVector(self):
        self.Quadrant_FeatureVector.to_csv(self.path + "Quadrant_FeatureVector.csv", sep = ",")

    def save_Branch_FeatureVector(self):
        self.Branch_FeatureVector.to_csv(self.path + "Branch_FeatureVector.csv", sep = ",")

    def load_Shell_FeatureVector(self):
        try:
            self.Shell_FeatureVector = pd.read_csv(self.path + "Shell_FeatureVector.csv", sep = ",", index_col= 0)
            # Load configuration file values
        except FileNotFoundError:
            self.gen_UniqueFeatureVectors(Shells= True)
            
    def load_Branch_FeatureVector(self):
        try:
            self.Branch_FeatureVector = pd.read_csv(self.path + "Branch_FeatureVector.csv", sep = ",", index_col= 0)
            # Load configuration file values
        except FileNotFoundError:
            self.gen_UniqueFeatureVectors(Branches= True)

    def load_Quadrant_FeatureVector(self):
        try:
            self.Quadrant_FeatureVector = pd.read_csv(self.path + "Quadrant_FeatureVector.csv", sep = ",", index_col= 0)
            # Load configuration file values
        except FileNotFoundError:           
            self.gen_UniqueFeatureVectors(Quadrants= True)

    def load_FeatureVector(self, **kwargs):
        try:
            self.FeatureVector = pd.read_csv(self.path + "FeatureVector.csv", sep = ",")
            # Load configuration file values
        except FileNotFoundError:           
            self.gen_UniqueFeatureVectors(General = True, **kwargs)

    def Output_FeatureVector(self):
        self.FeatureVector.to_csv("./FeatureVectors/" + self.dataset + "_FeatureVector.csv", sep = ",")
