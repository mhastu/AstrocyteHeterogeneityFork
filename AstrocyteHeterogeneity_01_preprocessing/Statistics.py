#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_functions as fkt
from matplotlib.collections import LineCollection
from pymatreader import read_mat
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from scipy.spatial.transform import Rotation as R

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer

#%%
class KMean_Clustering:
    def __init__(self, FeatureVector, k, init = "k-means++", n_init=10, algo = "lloyd", scale = True) -> None:
        self.FeatureVector = FeatureVector
        self._CovarianceMatrix = FeatureVector.cov()
        self._CorrelationMatrix = FeatureVector.corr()
        self._Mean = FeatureVector.mean()
        self._StandartDeviation = FeatureVector.std()

        # standart scaling method to scale to mean and variance
        if scale:
            scaler = StandardScaler()
            self.FeatureVector.loc[:,:] = scaler.fit_transform(self.FeatureVector)

        
        self.k_means = KMeans(n_clusters=k, init = init, random_state=1, n_init = n_init, algorithm = algo)
        self.k_means.fit_predict(FeatureVector)
        self._label_vector = self.k_means.labels_
        self._siluette_average = silhouette_score(FeatureVector, self.k_means.labels_)
        self._silhouette_samp = silhouette_samples(FeatureVector, self.k_means.labels_)

    # def pipeline(self, FeatureVector, k, threshhold = 0.8):
    #     # https://365datascience.com/tutorials/python-tutorials/pca-k-means/
    #     scaler = scaler = StandardScaler()
    #     std_FeatureVector = scaler.fit_transform(self.FeatureVector)

    #     pca = PCA()
    #     pca.fit(std_FeatureVector)

    #     fig_explained_variance = plt.figure(figsize= (10,10))
    #     ax = fig_explained_variance.add_subplot(111)
    #     ax.plot(range(1, k), pca.explained_variance_ratio_.cumsum(), marker = 0, ls = "--")
    #     ax.set_title("Explained Variance by Components")
    #     ax.set_ylabel("Cumulative Explained Variance")
    #     ax.set_xlabel("Number of Components")

    #     n_components = np.where(pca.explained_variance_ratio_.cumsum()  >= threshhold)[0][0] + 1

    #     pca = PCA(n_components= n_components)
    #     scores_pca = pca.fit_transform(std_FeatureVector)

    #     wcss = []
    #     for i in range(k):




    @property
    def silhuette_average(self):
        return self._siluette_average
    
    @property
    def silhouette_samp(self):
        return self._silhouette_samp
    
    @property
    def label_vector(self):
        return self._label_vector
    
    @property
    def CovarianceMatrix(self):
        return self._CovarianceMatrix
    
    @property
    def CorrelationMatrix(self):
        return self._CorrelationMatrix

    @property
    def Mean(self):
        return self._Mean
    
    @property
    def StandartDeviation(self):
        return self._StandartDeviation
    
    @property
    def pca(self):
        return self._pca
    @pca.setter
    def pca(self, k):
        self._pca = PCA(n_components= k)
    
    def reduce_FeatureVector(self, k = 10, reduce_dim = True):
        try:
            self.pca.fit(self.FeatureVector)
        except AttributeError:
            self.pca = k

        self.pca.fit(self.FeatureVector)
        print("Estimated n of components", self.pca.n_components_)
        print("Explained components", self.pca.components_)

        if reduce_dim == True:
            self.reduced_FeatureVector = self.pca.transform(self.FeatureVector)
        
    
    #Fischer Information Matrix
    def fischer_information_matrix(self):
        eng = matlab.engine.start_matlab()
        #data = eng.cell2mat(data)
        data = self.FeatureVector.to_numpy(dtype = float)
        [mean,covariance] = eng.ecmnmle(data, nargout=2)
        self.fischer_matrix = pd.DataFrame(eng.ecmnfish(data,covariance))
        return self.fischer_matrix

    def Variance_treshhold(self, th = .8 * (1 - .8)): #TODO Play with variances to see what drops out
        selection = VarianceThreshold(threshold=th)
        self.FeatureVector = selection.fit_transform(self.FeatureVector)

    def K_Best(self, mode = f_classif, k = 3, transform = False):
        self.selection = SelectKBest(mode, k = k)
        self.selection.fit(self.FeatureVector, self.label_vector)
        params = self.selection.get_params

        if transform:
            self.FeatureVector = self.selection.transform(self.FeatureVector, self.labels)
        return params
    
    def plot_features(self):
        for i, feature in self.FeatureVector.iterrows():
            fig = plt.figure()
            ax = fig.add_subplot()

            ax.hist(feature)
            ax.set_title(i)
            # self.FeatureVector.plot.hist(feature)
    
    # def k_means(self, features, to_plot, xlabel:str, ylabel:str, Title:str, n: list = [2,3,4,5]):

    #     self.SiluetteScores =  []
        
    #     fig = plt.figure(figsize= (8,8))
    #     fig.tight_layout()
    #     fig.suptitle(f'Differnt k-means clustering visualiced over {to_plot[0]} and {to_plot[1]}')
    #     if features == "all":
    #         features = self.FeatureVector.columns.values
    #         print(features)

    #     else:
    #         print(features)

    #     #ToDo
    #     #Silhuette Value for n = 1!!
        
        
    #     for i, ni in enumerate(n):
    #         ax = fig.add_subplot(int(len(n)/2+ 0.5),2,1 + i)
    #         ax.set_title(f"Nr. cluster = {ni}")
    #         #setting plot labels
    #         if i >= len(n) -1:
    #             ax.set_xlabel(xlabel)

    #         if i % 2 == 0:
    #             ax.set_ylabel(ylabel)

    #         X = np.array(self.FeatureVector.loc[:, features].values)
    #         #print(X)
            
    #         k_means = KMeans(n_clusters=ni, init = "k-means++", random_state=1, n_init=10, algorithm= "elkan")
    #         #print(k_means)
    #         k_means.fit_predict(X)
    #         labels = k_means.labels_
                        
    #         silhouettesiluette_average = silhouette_score(X, labels)
    #         silhouette_samp = silhouette_samples(X, labels)

    #         ax.scatter(self.FeatureVector.loc[:, to_plot[0]], self.FeatureVector.loc[:, to_plot[1]], c = labels)
            
    #         #siluette for clusters
    #         siluette_clusters = np.zeros(ni)
    #         for j in range(ni):
    #             siluette_clusters[j] = max(silhouette_samp[labels == j])
    #         self.SiluetteScores.append(silhouettesiluette_average)
    #         print(f"n = {ni}", f"Silhuette = {silhouettesiluette_average}")

    #     self.n_bestSilhuette = n[np.argmax(self.SiluetteScores)]

    #     #plotting
    #     fig2 = plt.figure()
    #     ax2 = fig2.add_subplot(111)
        
    #     ax2.grid()
    #     ax2.set_title("Silhuette Scores over n Clusters")
    #     ax2.set_xlabel("Nr of Clusters")
    #     ax2.set_ylabel("Silhuette Score")

    #     ax2.plot(n, self.SiluetteScores)

        
    #     fig.tight_layout()
    #     fig.savefig(f"./kMeans_Plots/kMeans{Title}.pdf")
    #     fig2.savefig(f"./KMeans_Plots/Silhuette Score ofer n clusters {Title}.pdf")


    def get_best_kmeans(self):
        self.best_kmeans = KMeans(self.n_bestSilhuette, init = "k-means++", random_state=1, n_init=10, algorithm= "elkan")
        X = np.array(self.FeatureVector.loc[:, :].values)
        self.best_kmeans.fit_predict(X)
        labels = self.best_kmeans.labels_
        #cluster_centers = self.best_kmeans.cluster_centers_

        self.labeld_FeatureVector = self.FeatureVector.copy()
        self.labeld_FeatureVector.loc[:, "Label"] = labels

        nr_cells = np.zeros(self.n_bestSilhuette)
        output = pd.DataFrame()
        for ni in range(self.n_bestSilhuette):
            print(ni)
            nr_cells[ni] = self.FeatureVector.loc[labels == ni].index.size
            mean = self.FeatureVector.loc[labels == ni].mean(0)
            output.loc[:, ni] = mean
            
        output = output.transpose().round(2)
        output.loc[:, "Nr Cells"] = nr_cells
        output.index.name = "Cluster"
        output.to_csv("KmeansClusteringTable.txt")


        #save as latex table
        save_to_tex(output, "ClusteringTable")

#-----------------------------------------------------------------------------------
def pipeline(FeatureVector, k, threshhold = 0.8, nrefs = 3, date = "_2023-08-23"):
    # https://365datascience.com/tutorials/python-tutorials/pca-k-means/
    scaler = scaler = StandardScaler()
    std_FeatureVector = scaler.fit_transform(FeatureVector)

    pca = PCA()
    pca.fit(std_FeatureVector)

    fig_explained_variance = plt.figure(figsize= (10,10))
    ax = fig_explained_variance.add_subplot(111)
    ax.plot(range(std_FeatureVector.shape[1]), pca.explained_variance_ratio_.cumsum(), marker = 0, ls = "--") #TODO 49 labes is hardcoded, prob solve with featurevector collums length
    ax.axhline(y = 0.8, ls = "--", color = "r")
    ax.set_title("Explained Variance by Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_xlabel("Number of Components")

    fig_explained_variance.savefig("Explained_Variance" + date + ".pdf")
    #fig_explained_variance.legend()
    n_components = np.where(pca.explained_variance_ratio_.cumsum()  >= threshhold)[0][0] + 1
    print("optimal Nr. Components = ", n_components)
    pca = PCA(n_components= n_components)
    scores_pca = pca.fit_transform(std_FeatureVector)

    wcss = []
    silhuette = []
    gaps = []
    for i in range(1, k):
        kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', n_init = "auto")
        kmeans_pca.fit(scores_pca)

        #gap score:
        ref_Disp = []
        for j in range(nrefs):
            randomReference = np.random.random_sample(size=scores_pca.shape)
            kmeans_gap = KMeans(n_clusters = i, init = 'k-means++', n_init = "auto").fit(randomReference)
            ref_Disp.append(kmeans_gap.inertia_)

        gaps.append( np.log(np.mean(ref_Disp)) - np.log(kmeans_pca.inertia_))
        wcss.append(kmeans_pca.inertia_)
        if i == 1:
            silhuette.append(np.nan)
        else:
            silhuette.append(silhouette_score(scores_pca, kmeans_pca.labels_))
    
    x = range(1, k)
    y = wcss

    # https://kneed.readthedocs.io/en/stable/api.html#kneed.knee_locator.KneeLocator.plot_knee
    # https://towardsdatascience.com/detecting-knee-elbow-points-in-a-graph-d13fc517a63c
    kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing")
    
    optimal_clusters = kneedle.knee
    print("optimal classes", optimal_clusters)
    
    fig_sillvsdistance = plt.figure(figsize = (10,10))
    ax1 = fig_sillvsdistance.add_subplot(111)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax1.axvline(optimal_clusters, label = "knee", ls = "--", color = "orange")
    ax1.plot(x, wcss, marker = 0, ls = "--", label = "WCSS", color = "r")
    ax2.plot(x, gaps, marker = 'x', ls = "-.", label = "Gap score", color = "b")
    ax3.plot(x, silhuette, marker = 'x', ls = ":", label = "Silhuette", color = "g")
    ax1.set_title("In between Cluster distance vs gap score")
    ax1.set_ylabel("WCSS")
    ax2.set_ylabel("Gap score")
    ax3.set_ylabel("Silhuette Score")
    ax1.set_xlabel("Number of clusters")
    #fig_sillvsdistance.tight_layout()
    fig_sillvsdistance.legend()
    fig_sillvsdistance.savefig("GappVsSilhuette" + date + ".pdf")

    #plot in 3 dim space
    best_kmeans = KMeans(n_clusters = optimal_clusters, init = 'k-means++', n_init = "auto").fit(scores_pca)
    
    
    d3_plot = plt.figure(figsize = (10,8))
    ax = d3_plot.add_subplot(projection = "3d")

    from matplotlib.pyplot import cm
    color = cm.Accent(np.linspace(0, 1, optimal_clusters +1))
    for i in range(optimal_clusters):
        print(i)
        data = scores_pca[np.where(best_kmeans.labels_ == i)]
        print(data)
        ax.scatter(data[:, 0], data[:, 1],data[:, 2], label = "class" + str(i))

    #annotate to analise outlies
    # for i, label in enumerate(FeatureVector.index):
        # ax.text(scores_pca[i, 0], scores_pca[i, 1], scores_pca[i, 2], label)

    ax.set_title(f"Principal Components and their clustering into {optimal_clusters} clusters")
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.set_zlabel("principal component 3")
    ax.legend()

    #setup labeld feature vector:
    FeatureVector["Label"] = best_kmeans.labels_

    for i in range(n_components):
        FeatureVector[f"PC{i}"] = scores_pca[:, i]

    
    d3_plot.savefig("3D_Clustering" + date + ".pdf")

    display(FeatureVector)

# def gap_statistic(data, nrefs = 3, maxClusters=15):
#     """
#     Calculates KMeans optimal K using Gap Statistic 
#     Params:
#         data: ndarry of shape (n_samples, n_features)
#         nrefs: number of sample reference datasets to create
#         maxClusters: Maximum number of clusters to test for
#     Returns: (gaps, optimalK)
#     """
#     gaps = np.zeros((len(range(1, maxClusters)),))
#     resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
#     for gap_index, k in enumerate(range(1, maxClusters)):# Holder for reference dispersion results
#         refDisps = np.zeros(nrefs)# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
#         for i in range(nrefs):
            
#             # Create new random reference set
#             randomReference = np.random.random_sample(size=data.shape)
            
#             # Fit to it
#             km = KMeans(k)
#             km.fit(randomReference)
            
#             refDisp = km.inertia_
#             refDisps[i] = refDisp# Fit cluster to original data and create dispersion
#         km = KMeans(k)
#         km.fit(data)
        
#         origDisp = km.inertia_# Calculate gap statistic
#         gap = np.log(np.mean(refDisps)) - np.log(origDisp)# Assign this loop's gap statistic to gaps
#         gaps[gap_index] = gap
        
#         resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
        
    

#     return (gaps.argmax() + 1, resultsdf)
    
    
# define best number of clusters
# https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning
def ellbow_method(feature_vektor):
    # https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad 30.06.2023
    # Elbow Method for K means# Import ElbowVisualizer
    
    model = KMeans()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, ax = ax, k=(2,30), timings= True)
    
    
    visualizer.fit(feature_vektor)        # Fit data to visualizer
    visualizer.show() 

    return fig, ax

            
def save_to_tex(data:pd.DataFrame, path:str):
    data_tex = data.to_latex(index=True)
    with open(path + ".tex", 'w') as file:
        file.write(data_tex)
        
#save to csv  
def save_to_tex(data:pd.DataFrame, path:str):
    data_tex = data.to_latex(index=True)
    with open(path + ".csv", 'w') as file:
        file.write(data_tex)



# %%
import matlab.engine

def fischer_information_matrix(data):
    eng = matlab.engine.start_matlab()
    #data = eng.cell2mat(data)
    [mean,covariance] = eng.ecmnmle(data, nargout=2)
    fischer_matrix = eng.ecmnfish(data,covariance)
    return fischer_matrix

# %%
