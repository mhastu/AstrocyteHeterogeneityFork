#%% Imports
import importlib
import numpy as np
import pandas as pd

from matplotlib import colormaps as cm
import matplotlib.pyplot as plt
import scipy.stats as sc
import random
#matplotlib
import matlab.engine
import matplotlib.ticker as mtick

import os
#seaborn
#import seaborn as sns
#Pylatex
import pylatex as pl
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, NoEscape, Table
from pylatex.utils import italic

#datetime
from datetime import date

#sklearn
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from sklearn.decomposition import PCA

#Knee locator
from kneed import KneeLocator

#my codes

import CellsData
import plotting_functions as plt_fkt
import Statistics
from CellsData import CellsData

import os.path as os_p

#%%
def Run_different_scalers(path, FeatureVector):
    noScaler = Main(20, "Paper", path,  FeatureVector,  date = str(date.today()), scaler = "None")
    #noScaler.run(20, "Testrun_noScalar", fisher = True)
    noScaler.eval_diverent_k(30, (5,8))
    noScaler.run("Run_Noscaler", True)

    main_Rob = Main(20, "Paper", path, FeatureVector,path,  date = str(date.today()), scaler = "Robust")
    #main_Rob.run(20, "Testrun_Robust", fisher = True)
    main_Rob.eval_diverent_k(30, (5,8))
    main_Rob.run("Run_Robust", True)

    main_Std = Main(20, "Paper", path,  FeatureVector,  date = str(date.today()), scaler = "Standard")
    #main_Std.run(20, "Testrun_Standard", fisher = True)
    main_Std.eval_diverent_k(30, (5,8))
    main_Std.run("RUn_Standard", True)

    main_MinMax = Main(20, "Paper",  path, FeatureVector,  date = str(date.today()), scaler = "MinMax")
    #main_MinMax.run(20, "Testrun_MinMax", fisher = True)
    main_MinMax.eval_diverent_k(30, (5,8))
    main_MinMax.run("Run_MinMax", True)

def gen_CellInfo(datasets, FeatureVectorName: str, loc: str = "FeatureVectors", 
                **kwargs):
    """Inputs:
        path: is the path to the CellsData folder
        datasets: is the list of the datasets from which to take the inforamtion
        loc: Folder to stroe Results into
        artefact_removal -> list [Bool, Threshold], baseline [False, None]

        Results in storing the required pandas Dataframes in folder to be taken up by Main Class"""
    
    path = os_p.abspath(".")
    
    results_path = os_p.join(os_p.abspath(".."),  loc, "United")
    
    cells_used = 0
    cells_total = 0
    if datasets == "United":
        datasets = ["H00", "H01", "H02", "H04", "C10", "C11"]
    FeatureVector = pd.DataFrame()
    
    for set in datasets:
        print(os_p.join(path,  set ))
        cells = CellsData(os_p.join(path,  set + "//"), set, **kwargs)
        
        cells_used += cells.cells_used
        cells_total += cells.cells_total

        if len(FeatureVector) == 0: 
            FeatureVector = cells.FeatureVector.copy()
            FeatureVector.loc[:, "Set"] = set
                        
        else:      
            cells.FeatureVector.loc[:, "Set"] = set
            FeatureVector = FeatureVector.append(cells.FeatureVector.copy())
                    
        FeatureVector.fillna(0, inplace= True)
        
    FeatureVector_info = pd.DataFrame(columns = ["cells used", "cells total"], index = ["info"])
    FeatureVector_info.loc[:, "cells used"] = cells_used
    FeatureVector_info.loc[:, "cells total"] = cells_total
    
    FeatureVector.to_csv(os_p.join(results_path, FeatureVectorName), sep = ",")
    FeatureVector_info.to_csv(os_p.join(results_path, "Info_" + FeatureVectorName), sep = ",")

#gen_CellInfo("United", Branches = True) ##run if you want to recreate the feature vector
#%%
class Kmeans_Storage:
    def __init__(self, iterations):
        
        self._silhouette_list = []
        self._silhouette = np.zeros(iterations)
        self._optimal_clusters_wcss = np.zeros(iterations)
        self._wcss = []
        self._gaps = []
        self._k_means = []
        self._end_pca = []
        self._scores_pca = []
        self._start_pca = []
        self._n_components = np.zeros(iterations)


    def store(self, iteration, kmeans, silhouette, optimal_clusters_wcss, silhouettes, wcss, gaps, end_pca, scores_pca, n_components, start_pca):
        self._silhouette_list.append(silhouettes)
        self._silhouette[iteration] = silhouette
        self._optimal_clusters_wcss[iteration] = optimal_clusters_wcss
        self._wcss.append(wcss)
        self._gaps.append(gaps)
        self._k_means.append(kmeans)
        self._end_pca.append(end_pca)
        self._scores_pca.append(scores_pca)
        self._start_pca.append(start_pca)
        self._n_components[iteration] = n_components
    
class Main:
    def __init__(self, k,  Type: str, path: str, FeatureVector:str,
                figsize = (6.4, 4.8), date = str(date.today()), min2pca = True,
                Plots = False, scaler = "Robust"):
        """Initialised the main class, takes into account multiple function iputs
        Inputs:
        FeatureVector -> str: Name of the Feature Vector to load
        k       ->      number of k tried in elbow method
        Type    ->      Paper or thesis
        path    ->      path to the data
        figsize ->      standard figsize used for plotting
        date    ->      date for the report
        Plots   ->      check if Plots should be made
        scaler  ->      scaler used
        min2pca ->      if PCA should at least return 2 features"""

        self.units = np.array(["a.u", "a.u", "mu m","a.u",
                  "a.u","a.u","mu m","a.u","a.u",
                  "a.u","a.u","a.u","a.u","a.u","a.u","mu m",
                  "mu m","mu m","mu m",
                  "a.u","a.u"])
        
        self.scaler = scaler
        self.cmap = cm["Set1"] #Hardcoded colormap
        self.k = k #number of iteration in elbow plot
        self.Plots = Plots #Boolean if plots should be made
        
        self.Type = Type  #now only the name of the dataset worked with
        self.date = date  #current date
        self.figsize = figsize  #standardised figsize used
        self.classifiers = ["kmeans", "DBSCAN", "Spectral", "Hierachical"]
       
        try:
            self.FeatureVector = pd.read_csv(os_p.join(path, FeatureVector), sep = ",", index_col= 0) #"FeatureVector_NoArtefactsRemoved.csv"
            
        except FileNotFoundError:
            print("No FeatureVector saved in this path")
            return
        
        try:
            self.FeatureVector_info = pd.read_csv(os_p.join(path,"Info_" + FeatureVector), sep = ",", index_col= 0) #"FeatureVector_info_ArtefactsRemoved_"+ str(artefact_removal[1]) + ".csv"
            # Load configuration file values
        except FileNotFoundError:
            print("No FeatureVector_info saved in this path")
            return
        self.cells_used = int(self.FeatureVector_info["cells used"])
        self.cells_total = int(self.FeatureVector_info["cells total"])

        try:
            self.Shell_FeatureVector = pd.read_csv(os_p.join(path,"Shell_FeatureVector.csv"), sep = ",", index_col= 0)
            # Load configuration file values
        except FileNotFoundError:
            print("No ShellFeatureVector saved in this path")
            
        
        self.datasetmarkers = self.FeatureVector.pop("Set")
        
        self.FeatureVector.fillna(0, inplace= True)
        self.orig_FeatureVector = self.FeatureVector.copy()

        self.scaler_used = scaler
        if scaler == "Robust":
            self.std_FeatureVector = pd.DataFrame(RobustScaler().fit_transform(self.FeatureVector), columns = self.FeatureVector.columns)
        elif scaler == "Standard":
            self.std_FeatureVector = pd.DataFrame(StandardScaler().fit_transform(self.FeatureVector), columns = self.FeatureVector.columns)
        elif scaler == "MinMax":
            self.std_FeatureVector = pd.DataFrame(MinMaxScaler().fit_transform(self.FeatureVector), columns = self.FeatureVector.columns)
        elif scaler == "None":
            self.std_FeatureVector = self.FeatureVector.copy()
        
        else:
            ValueError("Please select a correct scaler method:\n Options are: 'Robust', 'Standard', 'MinMax' and 'None")

        self.n_components, self.start_pca, self.end_pca, self.scores_pca = self.gen_optimalPCA(min2pca)

    def plot_FeatureVector(self):
        plt_fkt.FeatureVector(self)
    
    def load_datasets(self, datasets, **kwargs):
        self.datasets_dict = {}

        path = os_p.abspath(".")
            
        for dataset in datasets:
            self.datasets_dict[dataset] = self.H00 = CellsData(os_p.join(path,  dataset + "\\"), dataset,
                                                                **kwargs,)


    def thresholding(self): #TODO


        pass

    def plot_largest_cells(self):
        try:
            self.datasets_dict
        except AttributeError:
            self.load_datasets()
            
        plt_fkt.plot_largest_cells(self)

    def plot_sparsevsthick_cells(self):
        try:
            self.datasets_dict
        except AttributeError:
            self.load_datasets()

        plt_fkt.plot_sparsevsthick_cells(self)

    def eval_diverent_k(self, k, figsize = (6, 4)):
        optimal_clusters_wcss, silhuette, wcss, gaps, CalinskiHarabasz, DaviesBouldin = self.estimate_n_clusters(k, 4, self.scores_pca)

        plt_fkt.eval_diverent_k(self, k, silhuette, wcss, gaps, CalinskiHarabasz, DaviesBouldin,  figsize = figsize)

    def threshold_BP(self, threshold = 100):
        self.FeatureVector = self.FeatureVector[self.FeatureVector["N_Branchingpoints"]>= threshold]


    def run(self, Name, fisher = False):
        #n_components, start_pca, self.end_pca, self.scores_pca = self.gen_optimalPCA(min2pca = True)
        self.labels = self.cluster_data(self.n_components)
        print("Calculating fisher info")
        if fisher:
            self.fisher_scores, self.fisher_table, self.fisher_features, self.fisher_FisherVector, self.fisher_matrix = self.fisher_info()
        print("Visualising results")
        self.visualisation(fisher) 
        print("Generating Report")
        self.Report(Name)


    def cluster_data(self, n_clusters = 4):
        print("Cluster data")
        classifier_kmeans = KMeans(n_clusters, init = 'random', n_init = "auto").fit(self.scores_pca)
        print("DBSCAN")
        classifier_DBSCAN = DBSCAN().fit(self.scores_pca)
        print("Spectral")
        classifier_Spectral = SpectralClustering().fit(self.scores_pca)
        classifier_Hierachical = AgglomerativeClustering(n_clusters).fit(self.scores_pca)
        return classifier_kmeans.labels_, classifier_DBSCAN.labels_, classifier_Spectral.labels_, classifier_Hierachical.labels_
   
    def visualisation(self, fisher = False):

        try:
            self.fisher_matrix
        except AttributeError:
            self.fisher_scores, self.fisher_table, self.fisher_features, self.fisher_FisherVector, self.fisher_matrix = self.fisher_info()        

        print("Start of Visu")
        
        path = "./ResultPlots_"+ self.Type + "/PCA_FeatureVariance_" + self.scaler +".png"

        if os.path.isfile(path):
            pass
        else: 
            plt_fkt.ShollAnalysis(self)
            plt_fkt.explainedVar(self, self.start_pca, self.end_pca)
            
        if fisher:
            plt_fkt.fisher(self, self.fisher_matrix, self.fisher_scores)
            
        for label_vec, classifier in zip(self.labels, self.classifiers):
            plt_fkt.classification(self, self.scores_pca, self.end_pca, label_vec, Name =classifier, plot_cells= False)
    

    
    def gen_optimalPCA(self, min2pca = True, threshhold = 0.8):
        """Generates the optimal PCA
        min2pca -> Boolean, If PCA should at least return 2 Features """
        # scaler = RobustScaler()
        # std_FeatureVector = pd.DataFrame(scaler.fit_transform(self.FeatureVector), columns = self.FeatureVector.columns)
        # #display(std_FeatureVector)
        start_pca = PCA(svd_solver = "full")
        start_pca.fit(self.std_FeatureVector)
                   
        
        n_components = np.where(start_pca.explained_variance_ratio_.cumsum()  >= threshhold)[0][0] + 1
        if min2pca:
            if n_components == 1:
                n_components +=1
        end_pca = PCA(n_components= n_components)
        scores_pca = end_pca.fit_transform(self.std_FeatureVector)

        return n_components, start_pca, end_pca, scores_pca

    def estimate_n_clusters(self, k, nrefs,scores_pca): 
        wcss = []
        silhuette = []
        gaps = []

        #CalinskiHarabasz
        CalinskiHarabasz = []
        #DaviesBouldin
        DaviesBouldin = []

        for i in range(1, k):
            kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', n_init = "auto", random_state = 6)
            kmeans_pca.fit(scores_pca)

            #gap score:
            ref_Disp = []
            for j in range(nrefs):
                randomReference = np.random.random_sample(size = scores_pca.shape)
                kmeans_gap = KMeans(n_clusters = i, init = 'k-means++', n_init = "auto").fit(randomReference)
                ref_Disp.append(kmeans_gap.inertia_)

            gaps.append( np.log(np.mean(ref_Disp)) - np.log(kmeans_pca.inertia_))
            wcss.append(kmeans_pca.inertia_)
            if i == 1:
                silhuette.append(np.nan)
                CalinskiHarabasz.append(np.nan)
                DaviesBouldin.append(np.nan)
            else:
                silhuette.append(silhouette_score(scores_pca, kmeans_pca.labels_))
                CalinskiHarabasz.append(calinski_harabasz_score(scores_pca, kmeans_pca.labels_))
                DaviesBouldin.append(davies_bouldin_score(scores_pca, kmeans_pca.labels_))
            
        # https://kneed.readthedocs.io/en/stable/api.html#kneed.knee_locator.KneeLocator.plot_knee
        # https://towardsdatascience.com/detecting-knee-elbow-points-in-a-graph-d13fc517a63c
        x = range(1, k)
        y_wcss = wcss
        #y_gaps = gaps
        kneedle_wcss = KneeLocator(x, y_wcss, S=1.0, curve="convex", direction="decreasing")
        #kneedle_gap = KneeLocator(x, y_gaps, S=1.0, curve="convex", direction="increasing")
        optimal_clusters_wcss = kneedle_wcss.knee
        
        return optimal_clusters_wcss, silhuette, wcss, gaps, CalinskiHarabasz, DaviesBouldin
    	     
    def clusters_tab(self):
        #features = self.orig_FeatureVector.loc[:, ["av_Branchdiameter", "Roundness", "Sphericity"]]
        index = ["Nr. cells", "%"]
        clusters_tab = pd.DataFrame(index = index )
        
        for i in np.unique(self.best_kmeans.labels_):
            nr = np.sum(self.best_kmeans.labels_ == i) 
            clusters_tab.loc["Nr. cells", f"Cluster {i+1}"] = nr
            
            clusters_tab.loc["%", f"Cluster {i+1}"] = np.round(nr / self.cells_used * 100, 2)
            
        print(clusters_tab.to_latex())
        return clusters_tab

    def fisher_info(self): 
        print("Calc fischer info...")
        eng = matlab.engine.start_matlab()
        #data = eng.cell2mat(data)
        columns = self.std_FeatureVector.columns
        data = self.std_FeatureVector.to_numpy(dtype = float)
        [mean,covariance] = eng.ecmnmle(data, nargout=2)
        fisher_matrix = pd.DataFrame(eng.ecmnfish(data,covariance, [], "meanonly"), columns, columns)

        fisher_scores = pd.Series(np.diag(fisher_matrix), index = fisher_matrix.columns).sort_values(ascending = False)
        
            
        #print("Best Scores = ", fisher_scores[0:5])
        fisher_features = fisher_scores[0:5].index.to_list()
        fisher_table = pd.DataFrame(fisher_scores[0:5], columns = ["Fisher Info"])

        #print(self.orig_FeatureVector.loc[:, fisher_features].mean())
        #print(self.orig_FeatureVector.loc[:, fisher_features].std())

        fisher_table["Mean"] = self.orig_FeatureVector.loc[:, fisher_features].mean()
        fisher_table["Variance"] =  self.orig_FeatureVector.loc[:, fisher_features].std()
        fisher_table = fisher_table.transpose()

        #print(fisher_table.to_latex())
        fisher_FisherVector = self.FeatureVector.loc[:, fisher_features].copy()
        return fisher_scores, fisher_table, fisher_features, fisher_FisherVector, fisher_matrix
    
    def Results_tables(self):
        geometry_options = {"right": "2cm", "left": "2cm"}
        doc = Document("ResultsThesis_" + self.date , geometry_options=geometry_options)

        doc.packages.append(pl.Package('booktabs'))
        doc.packages.append(pl.Package('float'))

        with doc.create(Section("Fisher Information table")):
            doc.append("Test")
            with doc.create(Table(position='htbp')) as table:
                #with doc.create(Tabular(fmt)) as data_table:
                    table.add_caption('The 5 best features regarding fisher inforamtion')
                    
                    table.append(self.fisher_scores[0:5].to_latex(escape=False))

        with doc.create(Section("Cluster table")):
            doc.append("Test")
            with doc.create(Table(position='htbp')) as table:
                #with doc.create(Tabular(fmt)) as data_table:
                    table.add_caption('The nr of cells in each cluster')
                    
                    table.append(self.ClustersTable.to_latex(escape=False))

        doc.generate_pdf(clean_tex=False)
        pass
    
    def Report(self, Name, width = r'1\textwidth', *args, **kwargs):
       
        #print(self.savepath)
        geometry_options = {"right": "2cm", "left": "2cm"}
        doc = Document("Report_"+ Name +"_" + self.date , geometry_options=geometry_options)

        doc.packages.append(pl.Package('booktabs'))
        doc.packages.append(pl.Package('float'))

        with doc.create(Section("Introduction")):
            doc.append(f"This is the report about the PCA of the all datasets. \n"
                "Using the pipeline method that is supplied in the Main.py file. \n"
                f"This Dataset contained a total of {self.cells_total}, of which {self.cells_used} were used in the classification process.\n"
                f"The {self.scaler} scaling method was used for scaling the dataset.")
        
        with doc.create(Section("Results")):
            with doc.create(Subsection('Ploting of the branching points within the shell radii')):
                doc.append("The following plot illustrates the amount of points within a boundary set by a radial distance from the cell soma. The data generated with this method will later be used for the classification task.")
                
                with doc.create(Figure(position='H')) as plot:
                    path = "./ResultPlots_"+ self.Type + "/ShollAnalysis_" + self.scaler +".png"
                    if os.path.isfile(path):
                        pass
                    else: 
                        print("Plot Sholl Analysis..")
                        plt_fkt.ShollAnalysis(self)
                    
                    plot.add_image(path, width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("The plot of the gap statistic vs the silhoutte and wcss scores. The knee is plotted as the vertical line.")
            
            with doc.create(Subsection('Explained Variance of PCA')):
                doc.append(f"This is the report about the PCA of the all the datasets. \n"
                f"By using a threshhold of 0.9 for the cumulative sum of the explained variance ratios the number of principal components was estimated to be 2 as seen in the next figure. \n"
                f"Subsequently the data was broken down into those components and from with their explained variances plotted.")

                with doc.create(Figure(position='H')) as plot:

                    path = "./ResultPlots_"+ self.Type + "/PCA_FeatureVariance_" + self.scaler +".png"
                    if os.path.isfile(path):
                        pass
                    else: 
                        print("Plotting Visualisation...")
                        self.visualisation()
                    plot.add_image(path, width=NoEscape(width), *args, **kwargs)
                    plot.add_caption('The explained variance of the PCA on the supplied feature vector including the used threshold of 0.8 for optimization next to the explained variance of the calculated components.')
        
            with doc.create(Subsection('Feature Vector')):
                doc.append("The features of the Feature vector plotted as Histograms")
                with doc.create(Figure(position='H')) as plot:

                    plot.add_image("./ResultPlots_"+ self.Type + "/FeatureVectorHist0.png", width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("The Features of the Feature vector not scaled")

                with doc.create(Figure(position='H')) as plot:

                    plot.add_image("./ResultPlots_"+ self.Type + "/FeatureVectorHist1.png", width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("The Features of the Feature vector not scaled")

                with doc.create(Figure(position='H')) as plot:

                    plot.add_image("./ResultPlots_"+ self.Type + "/FeatureVectorHist2.png", width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("The Features of the Feature vector not scaled")
                

            with doc.create(Subsection('Fisher Information')):
                doc.append("The fisher information matrix can be seen in the following figure. In the figure it is apparend that most used features do not hold a lot of inforamtion about the log likelyhood of the other features and therefor do not factor into classification. \n"
                           "Interestingly a few features have high fisher information and are usually in the very low end of the shell data \n"
                           "In the next figure the Fisher information values of all the features are plotted in a log scale, the findings again becoming apparent.")
                
                with doc.create(Figure(position='H')) as plot:

                    plot.add_image("./ResultPlots_"+ self.Type + "/FisherInfoScores_" + self.scaler +".png", width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("The fisher inforamtion matrix over all features.")
                
                with doc.create(Figure(position='H')) as plot:

                    plot.add_image("./ResultPlots_"+ self.Type + "/FisherInfoMatrix_" + self.scaler +".png", width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("Fisher information values of all the features")
            
            for classifier in self.classifiers:
                with doc.create(Subsection(f'Clusterd Data using the {classifier} classifier')):
                    doc.append("By clustering the data into the aforementioned clusters and looking at the first 2 prinicpal components the following figure is obtained. \n"
                            "On either side of the scatter plot the histograms for the respective axes are plotted additivly.")
                    with doc.create(Figure(position='H')) as plot:

                        plot.add_image("./ResultPlots_"+ self.Type + "/2D_Clustering_" + classifier +"_" + self.scaler +".png", width=NoEscape(width), *args, **kwargs)
                        plot.add_caption("The 2D clusting of the first 2 dimensions of the PCA of the feature vektor")
                    
                    # with doc.create(Figure(position='H')) as plot:

                    #     plot.add_image("./ResultPlots_"+ self.Type + "/2D_Clustering_Features_" + self.scaler +".png", width=NoEscape(width), *args, **kwargs)
                    #     plot.add_caption("The 2D clusting of the first 2 dimensions of the PCA of the feature vektor")

            with doc.create(Section('K Cluster Estiamtion with different methods')):
                with doc.create(Figure(position='H')) as plot:
                    plot.add_image("./ResultPlots_" + self.Type + "/DiverntKevaluations_" + self.scaler + ".png", width=NoEscape(r'0.6\textwidth'), *args, **kwargs)
                    plot.add_caption("k as evaluated by different methods.")

            with doc.create(Section('Largest cells')):

                path = "./ResultPlots_"+ self.Type + "/4LargestCells_Volume.png"
                if os.path.isfile(path):
                    pass
                else: 
                    print("Plotting Largest Cells")
                    self.plot_largest_cells()
                
                with doc.create(Figure(position='H')) as plot:

                    plot.add_image("./ResultPlots_" + self.Type + "/4LargestCells_Volume.png", width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("The 3 Lagest cells according to bot features evaluated")

                with doc.create(Figure(position='H')) as plot:

                    plot.add_image("./ResultPlots_" + self.Type + "/4LargestCells_NEndpoints.png", width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("The 3 Lagest cells according to bot features evaluated")
            
                with doc.create(Figure(position='H')) as plot:

                    plot.add_image("./ResultPlots_" + self.Type + "/4LargestCells_both.png", width=NoEscape(width), *args, **kwargs)
                    plot.add_caption("The 3 Lagest cells according to bot features evaluated") 
                # with doc.create(Subsection('Cluster evaluation')):
                    
                #     clusters_table = self.clusters_tab()
                #     with doc.create(Table(position='htbp')) as table:
                #         #with doc.create(Tabular(fmt)) as data_table:
                #         table.add_caption('The number of cells in each cluster and the variance of the principal components used for classification')
                        
                #         table.append(NoEscape(clusters_table.to_latex(escape=False)))


        doc.generate_pdf(clean_tex=False)

    def print_cells_latex(self, n, dataset_list: list, *args, **kwargs,):
        """Prints the first n cells of a dataset"""
        geometry_options = {"right": "2cm", "left": "2cm"}
        width = r'0.7\textwidth'
        path = os_p.abspath(".")

        for dataset in dataset_list:
            cellsdata = CellsData(os_p.join(path,  f"{dataset}\\"),dataset , reload_features= False)
            # take n random cells
            cells = cellsdata.cells
            if len(cells) >= n:
                random_cells = random.sample(cells.keys(), n)
            else:
                random_cells = cells.keys()
                
            doc = Document(str(n) + "_Cells_"+ dataset, geometry_options=geometry_options)

            doc.packages.append(pl.Package('booktabs'))
            doc.packages.append(pl.Package('float'))
            
            with doc.create(Section("Introduction")):
                doc.append(f"Here the first {n} cells of the dataser {dataset}")

                for id in random_cells:
                    fig = plt.Figure()
                    ax = fig.add_subplot(projection = "3d")
                    plt_fkt.plot_cell(ax, cells[id])
                    ax.set_title(f"Cell {id}")
                    fig.savefig(f"./CellPlots/{dataset}_Cell{id}.png")
                    with doc.create(Figure(position='H')) as plot:

                        plot.add_image(f"./CellPlots/{dataset}_Cell{id}.png", width=NoEscape(width), *args, **kwargs)
            
            doc.generate_pdf(clean_tex=False)
  
    # def Iterative_Kmeans(self, iterations, fisher_features = False, min2pca = True):
        
    #     self.Storage = Kmeans_Storage(iterations)
    #     self.StorageDF = pd.DataFrame(columns = ["silhouette", "opt_k", "pca", "kmeans"], index = range(iterations))
        
    #     for i in range(iterations):
    #         #print(i)
    #         if fisher_features:
    #             kmeans, silhouette, optimal_clusters_wcss, silhouettes, wcss, gaps, end_pca, scores_pca, n_components, start_pca = self.PipeLine_Fisher(False)
    #         else:
    #             kmeans, silhouette, optimal_clusters_wcss, silhouettes, wcss, gaps, end_pca, scores_pca, n_components, start_pca = self.PipeLine(False, min2_pca)
    #         self.StorageDF.iloc[i] = silhouette, optimal_clusters_wcss, end_pca, kmeans
    #         self.Storage.store(i, kmeans, silhouette, optimal_clusters_wcss, silhouettes, wcss, gaps, end_pca, scores_pca, n_components, start_pca)
        
    #     self.Storage._wcss = np.array(self.Storage._wcss)
    #     self.Storage._gaps = np.array(self.Storage._gaps)
    #     self.Storage._silhouette_list = np.nan_to_num(np.array(self.Storage._silhouette_list))
    #     #print(self.Storage._silhouette_list)
    #     self.w_meanclusters = np.sum(self.Storage._optimal_clusters_wcss * self.Storage._silhouette)/ np.sum(self.Storage._silhouette)
    #     print("Best clustering mean weighted with silhouette score:", self.w_meanclusters)
    #     k = int(np.round(self.w_meanclusters, 1))
    #     self.best_index = np.argmax(self.Storage._silhouette[self.Storage._optimal_clusters_wcss == k])
        
    #     #print(best_kmeans, k, best_index)
    #     self.best_pca = self.Storage._end_pca[self.best_index]
    #     self.best_kmeans = self.Storage._k_means[self.best_index]
    #     #self.cells.loc[:, "Label"] = self.best_kmeans.labels_
    #     print("End of Iter 