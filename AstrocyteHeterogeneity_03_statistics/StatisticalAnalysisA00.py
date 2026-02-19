# Astrocytes Heterogeneity - Statistics 
# 29.07.2024
# Author: Anna Freund
# Institute: INE - TU Graz

# Import ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy
import scipy.stats as stats
import seaborn as sns
import scikit_posthocs as sp
from tabulate import tabulate

# Load feature vector and labelled data -----------------------------------------------------------
d = pd.read_csv('FeatureVector.csv')
d = d.loc[d['N_Branchingpoints']>=100]
dataFull = d.values[:,1:-1] # removing the cellID and the dataset name
labels = pd.read_csv('dataLabeld.csv').values[:,3]
# put the data in seperate lists for each class
class0 =[]
class1 =[]
class2 =[]
class3 =[]
class4 =[]
class5 =[]
classOutlier=[]
for i,l in enumerate(labels):
    if l == '0':
        class0.append(dataFull[i,:])
    if l == '1':
        class1.append(dataFull[i,:])
    if l == '2':
        class2.append(dataFull[i,:])
    if l == '3':
        class3.append(dataFull[i,:])
    if l == '4':
        class4.append(dataFull[i,:])
    if l == '5':
        class5.append(dataFull[i,:])
    if l == 'outlier':
        classOutlier.append(dataFull[i,:])
# print the feature names
print('Keys:',d.keys())

# plot all features in violin plots
def plotBar(fig,ax,class1,class2,h_index = 0):
    h = plt.gca().get_ylim()[1]*0.05
    barx = [class1,class1,class2,class2]
    bary = [fig.gca().get_ylim()[1]*0.99+h*h_index,fig.gca().get_ylim()[1]+h*h_index,fig.gca().get_ylim()[1]+h*h_index,fig.gca().get_ylim()[1]*0.99+h*h_index]
    plt.plot(barx, bary, c='black')
    plt.text(x=barx[0]+(barx[-1]-barx[0])/2,y= bary[1]*1.005, s='n.s.')

fig1, ax = plt.subplots(7,3, figsize=(12, 30))
sns.violinplot(data=[np.array(class0)[:,0],np.array(class1)[:,0],np.array(class2)[:,0],np.array(class3)[:,0],np.array(class4)[:,0],np.array(class5)[:,0]],labels=[0,1,2,3,4,5],ax=ax[0,0])
ax[0,0].set_title('Feature 1')
sns.violinplot(data=[np.array(class0)[:,1],np.array(class1)[:,1],np.array(class2)[:,1],np.array(class3)[:,1],np.array(class4)[:,1],np.array(class5)[:,1]],labels=[0,1,2,3,4,5],ax=ax[0,1])
ax[0,1].set_title('Feature 2')
sns.violinplot(data=[np.array(class0)[:,2],np.array(class1)[:,2],np.array(class2)[:,2],np.array(class3)[:,2],np.array(class4)[:,2],np.array(class5)[:,2]],labels=[0,1,2,3,4,5],ax=ax[0,2])
ax[0,2].set_title('Feature 3')
sns.violinplot(data=[np.array(class0)[:,3],np.array(class1)[:,3],np.array(class2)[:,3],np.array(class3)[:,3],np.array(class4)[:,3],np.array(class5)[:,3]],labels=[0,1,2,3,4,5],ax=ax[1,0])
ax[1,0].set_title('Feature 4')
sns.violinplot(data=[np.array(class0)[:,4],np.array(class1)[:,4],np.array(class2)[:,4],np.array(class3)[:,4],np.array(class4)[:,4],np.array(class5)[:,4]],labels=[0,1,2,3,4,5],ax=ax[1,1])
ax[1,1].set_title('Feature 5')
sns.violinplot(data=[np.array(class0)[:,5],np.array(class1)[:,5],np.array(class2)[:,5],np.array(class3)[:,5],np.array(class4)[:,5],np.array(class5)[:,5]],labels=[0,1,2,3,4,5],ax=ax[1,2])
ax[1,2].set_title('Feature 6')
sns.violinplot(data=[np.array(class0)[:,6],np.array(class1)[:,6],np.array(class2)[:,6],np.array(class3)[:,6],np.array(class4)[:,6],np.array(class5)[:,6]],labels=[0,1,2,3,4,5],ax=ax[2,0])
ax[2,0].set_title('Feature 7')
sns.violinplot(data=[np.array(class0)[:,7],np.array(class1)[:,7],np.array(class2)[:,7],np.array(class3)[:,7],np.array(class4)[:,7],np.array(class5)[:,7]],labels=[0,1,2,3,4,5],ax=ax[2,1])
ax[2,1].set_title('Feature 8')
sns.violinplot(data=[np.array(class0)[:,8],np.array(class1)[:,8],np.array(class2)[:,8],np.array(class3)[:,8],np.array(class4)[:,8],np.array(class5)[:,8]],labels=[0,1,2,3,4,5],ax=ax[2,2])
ax[2,2].set_title('Feature 9')
sns.violinplot(data=[np.array(class0)[:,9],np.array(class1)[:,9],np.array(class2)[:,9],np.array(class3)[:,9],np.array(class4)[:,9],np.array(class5)[:,9]],labels=[0,1,2,3,4,5],ax=ax[3,0])
ax[3,0].set_title('Feature 10')
sns.violinplot(data=[np.array(class0)[:,10],np.array(class1)[:,10],np.array(class2)[:,10],np.array(class3)[:,10],np.array(class4)[:,10],np.array(class5)[:,10]],labels=[0,1,2,3,4,5],ax=ax[3,1])
ax[3,1].set_title('Feature 11')
sns.violinplot(data=[np.array(class0)[:,11],np.array(class1)[:,11],np.array(class2)[:,11],np.array(class3)[:,11],np.array(class4)[:,11],np.array(class5)[:,11]],labels=[0,1,2,3,4,5],ax=ax[3,2])
ax[3,2].set_title('Feature 12')
sns.violinplot(data=[np.array(class0)[:,12],np.array(class1)[:,12],np.array(class2)[:,12],np.array(class3)[:,12],np.array(class4)[:,12],np.array(class5)[:,12]],labels=[0,1,2,3,4,5],ax=ax[4,0])
ax[4,0].set_title('Feature 13')
sns.violinplot(data=[np.array(class0)[:,13],np.array(class1)[:,13],np.array(class2)[:,13],np.array(class3)[:,13],np.array(class4)[:,13],np.array(class5)[:,13]],labels=[0,1,2,3,4,5],ax=ax[4,1])
ax[4,1].set_title('Feature 14')
sns.violinplot(data=[np.array(class0)[:,14],np.array(class1)[:,14],np.array(class2)[:,14],np.array(class3)[:,14],np.array(class4)[:,14],np.array(class5)[:,14]],labels=[0,1,2,3,4,5],ax=ax[4,2])
ax[4,2].set_title('Feature 15')
sns.violinplot(data=[np.array(class0)[:,15],np.array(class1)[:,15],np.array(class2)[:,15],np.array(class3)[:,15],np.array(class4)[:,15],np.array(class5)[:,15]],labels=[0,1,2,3,4,5],ax=ax[5,0])
ax[5,0].set_title('Feature 16')
sns.violinplot(data=[np.array(class0)[:,16],np.array(class1)[:,16],np.array(class2)[:,16],np.array(class3)[:,16],np.array(class4)[:,16],np.array(class5)[:,16]],labels=[0,1,2,3,4,5],ax=ax[5,1])
ax[5,1].set_title('Feature 17')
sns.violinplot(data=[np.array(class0)[:,17],np.array(class1)[:,17],np.array(class2)[:,17],np.array(class3)[:,17],np.array(class4)[:,17],np.array(class5)[:,17]],labels=[0,1,2,3,4,5],ax=ax[5,2])
ax[5,2].set_title('Feature 18')
sns.violinplot(data=[np.array(class0)[:,18],np.array(class1)[:,18],np.array(class2)[:,18],np.array(class3)[:,18],np.array(class4)[:,18],np.array(class5)[:,18]],labels=[0,1,2,3,4,5],ax=ax[6,0])
ax[6,0].set_title('Feature 19')
sns.violinplot(data=[np.array(class0)[:,19],np.array(class1)[:,19],np.array(class2)[:,19],np.array(class3)[:,19],np.array(class4)[:,19],np.array(class5)[:,19]],labels=[0,1,2,3,4,5],ax=ax[6,1])
ax[6,1].set_title('Feature 20')
sns.violinplot(data=[np.array(class0)[:,20],np.array(class1)[:,20],np.array(class2)[:,20],np.array(class3)[:,20],np.array(class4)[:,20],np.array(class5)[:,20]],labels=[0,1,2,3,4,5],ax=ax[6,2])
ax[6,2].set_title('Feature 21')
fig1.savefig('violinplots.pdf')

# Kruskal Wallis test (nonparametric, no need to assume normality)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
# https://scales.arabpsychology.com/stats/dunns-test-in-python/
# https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn.html
for i in range(21):
    kruskalWallis = stats.kruskal(np.array(class0)[:,i],np.array(class1)[:,i],np.array(class2)[:,i],np.array(class3)[:,i],np.array(class4)[:,i],np.array(class5)[:,i])
    print('Feature',str(i+1),': P-value: ',np.round(kruskalWallis.pvalue,6))


# Posthoc dunns test with sidak p-value correction
# prints the pairwise compared and corrected p-values
# cutoff1 = 0.05
for i in range(21):
    data = [np.array(class0)[:,i],np.array(class1)[:,i],np.array(class2)[:,i],np.array(class3)[:,i],np.array(class4)[:,i],np.array(class5)[:,i]]
    pValuesDun = np.array(sp.posthoc_dunn(data, p_adjust = 'sidak'))
    print('Feature'+str(i+1))
    print(tabulate(pd.DataFrame(pValuesDun)))
    # print(tabulate(pd.DataFrame(np.concatenate((pValuesDun[0,1:],pValuesDun[1,2:],pValuesDun[2,3:],pValuesDun[3,4:],pValuesDun[4,5:]))).T))
    
    # fig, ax = plt.subplots()
    # sns.heatmap(np.array(pValuesDun)>cutoff1,ax=ax)
    # ax.set_title(('Feature '+str(i+1)+', P-Value (dunns test) smaller '+str(cutoff1)))
    # sns.heatmap(pValuesCorrected>cutoff2,ax=ax[1])
    # ax[1].set_title(('Feature '+str(i+1)+', P-Value(corrected) smaller '+str(cutoff2)))
    


    