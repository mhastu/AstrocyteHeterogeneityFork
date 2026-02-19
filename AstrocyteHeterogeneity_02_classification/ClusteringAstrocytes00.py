# Astrocytes Heterogeneity - Classification
# 29.07.2024
# Author: Anna Freund
# Institute: INE - TU Graz

# Import ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
scaler = MinMaxScaler()

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import seaborn as sn

from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import contingency_matrix

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
import xgboost
from xgboost import XGBClassifier

# Outlier Detection Functions ---------------------------------------------------------------------

# Extreme Gradient Boosting (XGBoost)
def combineXGB(X, data_label = 1):
# function that combines features of various anomaly detection algorithms to then pass it to the XGB calssifyer
# idea from zhao et al., 2019, XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning
# X: data
# data_label: wether to label the raw data (not anomaly detected) as anomaly or normal (-1 or 1)
    clf = IsolationForest().fit(X)
    outcome = np.array(clf.predict(X))

    clfLOF = LocalOutlierFactor(n_neighbors=100) #150 oder 2
    clfLOF.fit_predict(X)
    out = clfLOF.negative_outlier_factor_
    outBinary = []
    for i,o in enumerate(out):
        if o<-1.2: #1.86
            outBinary.append(-1)
        else:
            outBinary.append(1)

    el = EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=None, contamination=0.095)
    el.fit(X)
    d = el.predict(X)

    featuresCombined = np.concatenate([X,X,X,X])
    labelsC= np.concatenate([np.ones(X.shape[0])*data_label,outcome,np.array(outBinary),d])
    labelsCombined = []
    for label in labelsC:
        if label == 1:
            labelsCombined.append(1)
        else:
            labelsCombined.append(0)
    return featuresCombined,labelsCombined

def predictXGB(data):
# function that fits xgb classifyer with combined data and predicts labels
# internally calls the function to combine features
    (featuresCombined,labelsCombined) = combineXGB(data, data_label = 1)
    xgb = XGBClassifier()
    xgb.fit(featuresCombined,labelsCombined)
    prediction = xgb.predict(data)
    return prediction

# Auto-Encoder (AE)
def trainAE(data, visualisation = False,ep=70):
# function to train AE model
# AE then gets evaluated with training data and bad performing data points (high loss) are labeled as anomaly
# due to the bottle neck (compensated data representation) in the network architecture, anomalies are not learned by the network
    output_units=data.shape[1]

    model = Sequential([
          Dense(32, activation='relu'),
          Dropout(0.1),
          Dense(16, activation='relu'),
          Dropout(0.1),
          Dense(8, activation='relu'),
          Dropout(0.1),
          Dense(4, activation='relu'),
          Dropout(0.1),
          Dense(16, activation='relu'),
          Dropout(0.1),
          Dense(32, activation='relu'),
          Dropout(0.1),
          Dense(output_units, activation='sigmoid')
        ])
    model.compile(loss='msle', metrics=['mse'], optimizer='adam')
    
    x_train, x_val, y_train, y_val = train_test_split(
    data, data, test_size=0.2)
    min_max_scaler = StandardScaler()
    x_train_scaled = min_max_scaler.fit_transform(x_train.copy())
    x_test_scaled = min_max_scaler.transform(x_val.copy())
    
    history = model.fit(
    x_train_scaled,
    x_train_scaled,
    epochs=ep,
    batch_size=512, #512
    validation_data=(x_test_scaled, x_test_scaled))
    if visualisation:
        model.summary(show_trainable=True)
    
    if visualisation:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Training history')
        plt.xlabel('Epochs')
        plt.ylabel('MSLE Loss')
        plt.legend(['loss', 'val loss'])
        plt.show()
    
    # run trained AE on the same data used for training
    predictions = model.predict(data)
    # calculate loss for each prediction
    # due to the dimension reduction the anomalities are expected to have higher loss
    loss = tf.keras.losses.msle(predictions, data)
    threshold = np.mean(loss) + np.std(loss)
    preds= []
    # loss values higher than the threshold will be labeled as anomaly/0
    for l in loss:
        if l<threshold:
            preds.append(1)
        else:
            preds.append(0)
    return np.array(preds)

    
# Load data for anomaly detection -----------------------------------------------------------------
d = pd.read_csv('FeatureVector.csv')
d = d.loc[d['N_Branchingpoints']>=100]

# Direct features from data: N_Branchingpoints, N_Endpoints, max dia, mean dia
data = np.array([d['N_Branchingpoints'].values,d['N_Endpoints'].values,d['max dia'].values,d['mean dia'].values]).T
dataOutlierDetection = scaler.fit_transform(data)
print('Shape of features for Anomaly detection',dataOutlierDetection.shape)
anomalyXGB = predictXGB(dataOutlierDetection)
anomalyAE = trainAE(dataOutlierDetection,visualisation=False,ep=80)
print('XGB finds',np.round(list(anomalyXGB).count(0)/len(anomalyXGB)*100,2),'% anomalies')
print('AE finds',np.round(list(anomalyAE).count(0)/len(anomalyAE)*100,2),'% anomalies')

# Plotting Outliers -------------------------------------------------------------------------------
# Pca plot
pca = PCA(n_components=3)
pca.fit(dataOutlierDetection.T)
var = np.round(pca.explained_variance_ratio_,2)
comp = (pca.components_).T

fig, axs = plt.subplots(2, 3,figsize = (18,12))
axs[0,0].scatter(comp[:,0],comp[:,1],c=anomalyAE)
axs[0,0].set_xlabel('PCA 1: '+str(var[0]))
axs[0,0].set_ylabel('PCA 2: '+str(var[1]))
axs[0,0].set_title('AutoEncoder labled data')
axs[0,1].scatter(comp[:,0],comp[:,2],c=anomalyAE)
axs[0,1].set_xlabel('PCA 1: '+str(var[0]))
axs[0,1].set_ylabel('PCA 3: '+str(var[2]))
axs[0,1].set_title('AutoEncoder labled data')
axs[0,2].scatter(comp[:,1],comp[:,2],c=anomalyAE)
axs[0,2].set_xlabel('PCA 2: '+str(var[1]))
axs[0,2].set_ylabel('PCA 3: '+str(var[2]))
axs[0,2].set_title('AutoEncoder labled data')
axs[1,0].scatter(comp[:,0],comp[:,1],c=anomalyXGB)
axs[1,0].set_xlabel('PCA 1: '+str(var[0]))
axs[1,0].set_ylabel('PCA 2: '+str(var[1]))
axs[1,0].set_title('XGBoost labled data')
axs[1,1].scatter(comp[:,0],comp[:,2],c=anomalyXGB)
axs[1,1].set_xlabel('PCA 1: '+str(var[0]))
axs[1,1].set_ylabel('PCA 3: '+str(var[2]))
axs[1,1].set_title('XGBoost labled data')
axs[1,2].scatter(comp[:,1],comp[:,2],c=anomalyXGB)
axs[1,2].set_xlabel('PCA 2: '+str(var[1]))
axs[1,2].set_ylabel('PCA 3: '+str(var[2]))
axs[1,2].set_title('XGBoost labled data')
fig.suptitle('PCA of astrocyte data with anomaly detection')
plt.savefig('outlier_pca.pdf')

# T-sne
# using NORMALIZED data
# using parameter: perplexity=5 or 6 or 8
t_sne = TSNE(n_components=2,n_iter=5000,perplexity=30)
S_t_sne = t_sne.fit_transform(dataOutlierDetection)
# print(t_sne.kl_divergence_)
fig, axs = plt.subplots(1, 2,figsize = (12,6))
axs[0].scatter(S_t_sne[:,0],S_t_sne[:,1],c=anomalyAE)
axs[0].set_title('AutoEncoder labled data')
axs[1].scatter(S_t_sne[:,0],S_t_sne[:,1],c=anomalyXGB)
axs[1].set_title('XGBoost labled data')
fig.suptitle('T-sne of astrocyte data')

plt.savefig('outlier_tsne.pdf')

# Loading full dataset for clustering -------------------------------------------------------------
d = pd.read_csv('FeatureVector.csv')
d = d.loc[d['N_Branchingpoints']>=100]
dataFull = d.values[:,1:-1] # removing the cellID and the dataset name
# data=dataFull[:,[ 0,  1,  3,  4,  5,  6,  8,  9, 12, 16, 17, 18, 19, 20]] # fisher >=5
# data=dataFull[:,[ 4,5,6,9,17,18,19]] # fisher >=10
data = dataFull
# outlier removal
dataT = []
for index,AE,XGB in zip(range(len(anomalyAE)),anomalyAE,anomalyXGB):
    if AE == 1 or XGB == 1:
        dataT.append(data[index,:])
    else:
        continue
dataThres = np.array(dataT)
dataScaled = scaler.fit_transform(dataThres)
print('Shape of features',dataScaled.shape)

# Finding optimal number of classes ---------------------------------------------------------------
def estimate_n_clusters(k, nrefs,dataScaled): 
    # function that runs different methods to find the optimal number of classes
    wcss = []
    silhuette = []
    gaps = []
    CalinskiHarabasz = []
    DaviesBouldin = []
    hirachicalDist = []

    for i in range(2, k):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(dataScaled)
        
        hirachical = AgglomerativeClustering(n_clusters = i,compute_distances = True)
        hirachical.fit(dataScaled)
        hirachicalDist.append(np.mean(hirachical.distances_))

        #gap score:
        ref_Disp = []
        for j in range(nrefs):
            randomReference = np.random.random_sample(size = dataScaled.shape)
            kmeans_gap = KMeans(n_clusters = i, init = 'k-means++').fit(randomReference)
            ref_Disp.append(kmeans_gap.inertia_)

        gaps.append( np.log(np.mean(ref_Disp)) - np.log(kmeans.inertia_))
        wcss.append(kmeans.inertia_)
        if i == 1:
            silhuette.append(np.nan)
            CalinskiHarabasz.append(np.nan)
            DaviesBouldin.append(np.nan)
        else:
            silhuette.append(silhouette_score(dataScaled, kmeans.labels_))
            CalinskiHarabasz.append(calinski_harabasz_score(dataScaled, kmeans.labels_))
            DaviesBouldin.append(davies_bouldin_score(dataScaled, kmeans.labels_))
    x = range(2, k)
    y_wcss = wcss
    #y_gaps = gaps
    kneedle_wcss = KneeLocator(x, y_wcss, S=1.0, curve="convex", direction="decreasing")
    optimal_clusters_wcss = kneedle_wcss.knee

    return optimal_clusters_wcss, silhuette, wcss, gaps, CalinskiHarabasz, DaviesBouldin

# Estimate number of classes with different methods from 2-21 classes
k = 20
optimal_clusters_wcss, silhuette, wcss, gaps, CalinskiHarabasz, DaviesBouldin=estimate_n_clusters(k, 4,dataScaled)

# plot
x_range = range(2, k)
print('Optimal number of clusters according to knee locator:',optimal_clusters_wcss)
fig, axs = plt.subplots(1, 5,figsize = (18,4))
axs[0].scatter(x_range,silhuette)
axs[0].grid()
axs[0].set_xlabel('Number of Clusters')
axs[0].set_ylabel('Score')
axs[0].set_title('Silhuette')
axs[1].scatter(x_range,wcss)
axs[1].grid()
axs[1].set_xlabel('Number of Clusters')
axs[1].set_ylabel('Score')
axs[1].set_title('Elbow method')
axs[2].scatter(x_range,gaps)
axs[2].grid()
axs[2].set_xlabel('Number of Clusters')
axs[2].set_ylabel('Score')
axs[2].set_title('Gaps')
axs[3].scatter(x_range,CalinskiHarabasz)
axs[3].grid()
axs[3].set_xlabel('Number of Clusters')
axs[3].set_ylabel('Score')
axs[3].set_title('CalinskiHarabasz')
axs[4].scatter(x_range,DaviesBouldin)
axs[4].grid()
axs[4].set_xlabel('Number of Clusters')
axs[4].set_ylabel('Score')
axs[4].set_title('DaviesBouldin')
fig.suptitle('Plots to determine number of clusters')
plt.savefig('cluster_scores.pdf')

# Using hierachical clustering to find optimal number of classes
# https://hlab.stanford.edu/brian/number_of_clusters_.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram
meth = 'ward'
maxClusters = 10
betweenClusterDist = np.flip(shc.linkage(dataScaled, method=meth)[:,2])[:maxClusters]

fig,ax = plt.subplots(1, 2,figsize=(14, 7))
ax[0].scatter(np.arange(len(betweenClusterDist))+1,betweenClusterDist)
ax[0].axes.get_xaxis().set_ticks(np.arange(len(betweenClusterDist))+1)
ax[0].grid()
ax[0].set_xlabel('Number of classes')
ax[0].set_title('Euclidean distance between joined clusters')
ax[1].set_title("Dendrogram")
# ax[1].axes.get_xaxis().set_visible(False)
ax[1].set_xlabel('Samples')
ax[0].set_ylabel('Distance')
dend = shc.dendrogram(shc.linkage(dataScaled, metric='euclidean',method=meth),color_threshold=5)
plt.savefig('dendogram.pdf')

# pca plot
pca = PCA(n_components=3)
pca.fit(dataScaled.T)
var = np.round(pca.explained_variance_ratio_,2)
comp = (pca.components_).T

# Clustering data ---------------------------------------------------------------------------------
# Clustering the data with k-Means, Hierachical Clustering and Gaussian Mixture Model

# n_cluster = optimal_clusters_wcss
n_cluster = 6 # knee locator indicates 6 as optimal number of classes

# classify with k means
kmeans = KMeans(n_clusters=n_cluster)
fittedData = kmeans.fit(dataScaled)
kLabelsA_og = fittedData.labels_

# clustering with expectation maximisation/Gaussian Mixture Model
gm = GaussianMixture(n_components=n_cluster, random_state=0).fit(dataScaled)
gmLabelsA_og = gm.predict(dataScaled)

#AgglomerativeClustering/ Hierachical Clustering
clustering = AgglomerativeClustering(n_clusters=n_cluster, linkage = 'ward').fit(dataScaled)
clustering.labels_
aLabelsA_og=clustering.labels_

def relabel(to_belabeld,to_takelabel):
    '''
    Function that calculates the contingency matrix to see which labels correspond with each other-
    So that similar clusters have the same label for differnt clustering alogrithms
    '''
    contingency = contingency_matrix(to_takelabel, to_belabeld)
    # print((contingency))
    LabelsA_new = np.copy(to_belabeld)
    occupied = np.zeros(n_cluster)
    for i in range(n_cluster):
        for j in range(1,n_cluster+1):
            # print((contingency[i,:]))
            label = np.argpartition(contingency[i,:],kth=n_cluster-1)[-j]
            if occupied[label]==1:
                continue
            else:
                occupied[label]=1
                break
        # print(i,j,label)
        LabelsA_new[to_belabeld == label] = i
    return LabelsA_new

'''
Calculate contingency matrix with all different combination, square, divide by size and sum.
The method with the highest values is the most "similar" with all others and will be used for relabeling referense.
In our case it is Kmeans and AgglomerativeClustering and we use Kmeans.
'''
num = 4 # 4 different methods for clustering (we are leaving out dbscan as one cannot choose the amount of clusters)
labels = [kLabelsA_og,gmLabelsA_og,aLabelsA_og]
scoreMatrix = np.zeros((num,num))
occurenceList = []
for i,lab_i in enumerate(labels):
    for j,lab_j in enumerate(labels):
        scoreMatrix[i,j]=(np.sum(contingency_matrix(lab_i,lab_j)**2)/np.size(contingency_matrix(lab_i,lab_j)))
        scoreMatrix[i,i]=0
for l in range(n_cluster):
    occurenceList.append(list(np.concatenate((np.argmax(scoreMatrix,axis = 0),np.argmax(scoreMatrix,axis = 1)))).count(l))
print('Classification method to use for relabeling:',np.argmax(occurenceList))
# relabelling clusterd datasets so that label 1 for k-means is also label 1 for GMM and Agglomerativ Clustering
kLabelsA = kLabelsA_og 
gmLabelsA = relabel(gmLabelsA_og,kLabelsA_og)
aLabelsA = relabel(aLabelsA_og,kLabelsA_og)

# Comparing the clustering methods ----------------------------------------------------------------
cm1 = contingency_matrix(kLabelsA, gmLabelsA)
cm2 = contingency_matrix(kLabelsA, aLabelsA)
cm4 = contingency_matrix(gmLabelsA, aLabelsA)

fig, ax = plt.subplots()
sn.heatmap(cm1,annot=True, fmt='.10g', cmap='RdPu')
ax.set_title('Confusion Matrix')
ax.set_ylabel('Kmeans')
ax.set_xlabel('GaussianMixture')
plt.savefig('confusion_kg.pdf')

fig, ax = plt.subplots()
sn.heatmap(cm2,annot=True, fmt='.10g', cmap='RdPu')
ax.set_title('Confusion Matrix')
ax.set_ylabel('Kmeans')
ax.set_xlabel('AgglomerativeClustering')
plt.savefig('confusion_ka.pdf')

fig, ax = plt.subplots()
sn.heatmap(cm4,annot=True, fmt='.10g', cmap='RdPu')
ax.set_title('Confusion Matrix')
ax.set_ylabel('GaussianMixture')
ax.set_xlabel('AgglomerativeClustering')
plt.savefig('confusion_ga.pdf')

# Plotting Clusters -------------------------------------------------------------------------------
cmaps = ['Dark2_r','PiYG_r','Set3','jet','prism','plasma','rainbow','viridis']
cmap = ['red','blue','yellow','magenta','green']
fig, axs = plt.subplots(3, 3,figsize = (16,16))
axs[0,0].scatter(comp[:,0],comp[:,1],c=kLabelsA, cmap='Dark2_r')
axs[0,0].set_xlabel('PCA 1: '+str(var[0]))
axs[0,0].set_ylabel('PCA 2: '+str(var[1]))
axs[0,0].set_title('KMeans labled data')
axs[0,1].scatter(comp[:,0],comp[:,2],c=kLabelsA, cmap='Dark2_r')
axs[0,1].set_xlabel('PCA 1: '+str(var[0]))
axs[0,1].set_ylabel('PCA 3: '+str(var[2]))
axs[0,1].set_title('KMeans labled data')
axs[0,2].scatter(comp[:,1],comp[:,2],c=kLabelsA, cmap='Dark2_r')
axs[0,2].set_xlabel('PCA 2: '+str(var[1]))
axs[0,2].set_ylabel('PCA 3: '+str(var[2]))
axs[0,2].set_title('KMeans labled data')
axs[1,0].scatter(comp[:,0],comp[:,1],c=gmLabelsA, cmap='Dark2_r')
axs[1,0].set_xlabel('PCA 1: '+str(var[0]))
axs[1,0].set_ylabel('PCA 2: '+str(var[1]))
axs[1,0].set_title('GaussianMixture labled data')
axs[1,1].scatter(comp[:,0],comp[:,2],c=gmLabelsA, cmap='Dark2_r')
axs[1,1].set_xlabel('PCA 1: '+str(var[0]))
axs[1,1].set_ylabel('PCA 3: '+str(var[2]))
axs[1,1].set_title('GaussianMixture labled data')
axs[1,2].scatter(comp[:,1],comp[:,2],c=gmLabelsA, cmap='Dark2_r')
axs[1,2].set_xlabel('PCA 2: '+str(var[1]))
axs[1,2].set_ylabel('PCA 3: '+str(var[2]))
axs[1,2].set_title('GaussianMixture labled data')
axs[2,0].scatter(comp[:,0],comp[:,1],c=aLabelsA, cmap='Dark2_r')
axs[2,0].set_xlabel('PCA 1: '+str(var[0]))
axs[2,0].set_ylabel('PCA 2: '+str(var[1]))
axs[2,0].set_title('AgglomerativeClustering labled data')
axs[2,1].scatter(comp[:,0],comp[:,2],c=aLabelsA, cmap='Dark2_r')
axs[2,1].set_xlabel('PCA 1: '+str(var[0]))
axs[2,1].set_ylabel('PCA 3: '+str(var[2]))
axs[2,1].set_title('AgglomerativeClustering labled data')
axs[2,2].scatter(comp[:,1],comp[:,2],c=aLabelsA, cmap='Dark2_r')
axs[2,2].set_xlabel('PCA 2: '+str(var[1]))
axs[2,2].set_ylabel('PCA 3: '+str(var[2]))
axs[2,2].set_title('AgglomerativeClustering labled data')
fig.suptitle('PCA of astrocyte data clusterd with '+str(n_cluster)+' components')
plt.savefig('labeldData_pca.pdf')

# t-sne
# using NORMALIZED data
# using parameter: perplexity=5 or 6 or 8
t_sne = TSNE(n_components=2,n_iter=5000,perplexity=30)
S_t_sne = t_sne.fit_transform(dataScaled)
# print(t_sne.kl_divergence_)

# tsne plot
fig, axs = plt.subplots(1, 3,figsize = (18,4))
print(kLabelsA_og.shape)
axs[0].scatter(S_t_sne[:,0],S_t_sne[:,1],c=kLabelsA, cmap='Dark2_r')
axs[0].set_title('KMeans labled data')
axs[1].scatter(S_t_sne[:,0],S_t_sne[:,1],c=gmLabelsA, cmap='Dark2_r')
axs[1].set_title('GaussianMixture labled data')
axs[2].scatter(S_t_sne[:,0],S_t_sne[:,1],c=aLabelsA, cmap='Dark2_r')
axs[2].set_title('AgglomerativeClustering labled data')
fig.suptitle('T-sne of astrocyte data clusterd with '+str(n_cluster)+' components')
plt.savefig('labeldData_tsne.pdf')

# Saving labeled data -----------------------------------------------------------------------------
d = pd.read_csv('FeatureVector.csv')
d = d.loc[d['N_Branchingpoints']>=100]

outliers = []
class1 = []
class2 = []
class3 = []
class4 = []
class5 = []
class6 = []
classAll = []
dataFrame = []
labelCount = 0
for index,AE,XGB in zip(range(len(anomalyAE)),anomalyAE,anomalyXGB):
    if AE == 1 or XGB == 1:
        classAll.append([index,d.values[index,0],d.values[index,-1]])
        # print([index,d.values[index,0],d.values[index,-1]])
        if kLabelsA[labelCount]==0:
            class1.append([index,d.values[index,0],d.values[index,-1]])
            miniFrame = [index,d.values[index,0],d.values[index,-1],'0']
        elif kLabelsA[labelCount]==1:
            class2.append([index,d.values[index,0],d.values[index,-1]])
            miniFrame = [index,d.values[index,0],d.values[index,-1],'1']
        elif kLabelsA[labelCount]==2:
            class3.append([index,d.values[index,0],d.values[index,-1]])
            miniFrame = [index,d.values[index,0],d.values[index,-1],'2']
        elif kLabelsA[labelCount]==3:
            class4.append([index,d.values[index,0],d.values[index,-1]])
            miniFrame = [index,d.values[index,0],d.values[index,-1],'3']
        elif kLabelsA[labelCount]==4:
            class5.append([index,d.values[index,0],d.values[index,-1]])
            miniFrame = [index,d.values[index,0],d.values[index,-1],'4']
        elif kLabelsA[labelCount]==5:
            class6.append([index,d.values[index,0],d.values[index,-1]])
            miniFrame = [index,d.values[index,0],d.values[index,-1],'5']
        labelCount+=1
    else:
        outliers.append([index,d.values[index,0],d.values[index,-1]])
        miniFrame = [index,d.values[index,0],d.values[index,-1],'outlier']
    dataFrame.append(miniFrame)
# print(classAll)
df = pd.DataFrame({'Data Index': np.array(dataFrame)[:,0],
                   'Cell ID': np.array(dataFrame)[:,1],
                   'Data Set': np.array(dataFrame)[:,2],
                   'Class': np.array(dataFrame)[:,3]})
df.to_csv('classes.csv', index=False)  
df.to_excel('classes.xlsx', index=False)  

