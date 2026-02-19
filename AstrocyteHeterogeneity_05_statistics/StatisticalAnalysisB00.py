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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA


# Evaluating Calcium Dynamics ---------------------------------------------------------------------
print('P-values of calcium spikes count')
c0 = np.array(pd.read_csv('class0.csv',sep=',').transpose())
c1 = np.array(pd.read_csv('class1.csv',sep=',').transpose())
c2 = np.array(pd.read_csv('class2.csv',sep=',').transpose())
c3 = np.array(pd.read_csv('class3.csv',sep=',').transpose())
c4 = np.array(pd.read_csv('class4.csv',sep=',').transpose())
c5 = np.array(pd.read_csv('class5.csv',sep=',').transpose())
# the features for calcium evaluation are the 3 stimulus in the 3 different compartments
#starting from index a because kruskal test only possible for not all elements same
a=3
for i in range(5):
    # print(a)
    # print(c0[i+a],c1[i+a],c2[i+a],c3[i+a],c4[i+a],c5[i+a])
    kruskalWallis = stats.kruskal(c0[i+a],c1[i+a],c2[i+a],c3[i+a],c4[i+a],c5[i+a])
    print(kruskalWallis.pvalue)
    
print('P-values of calcium spikes amplitude')
c0 = np.array(pd.read_csv('class0_amp.csv',sep=',').transpose())
c1 = np.array(pd.read_csv('class1_amp.csv',sep=',').transpose())
c2 = np.array(pd.read_csv('class2_amp.csv',sep=',').transpose())
c3 = np.array(pd.read_csv('class3_amp.csv',sep=',').transpose())
c4 = np.array(pd.read_csv('class4_amp.csv',sep=',').transpose())
c5 = np.array(pd.read_csv('class5_amp.csv',sep=',').transpose())
# the features for calcium evaluation are the 3 stimulus in the 3 different compartments
#starting from index a because kruskal test only possible for not all elements same
a=3
for i in range(5):
    # print(a)
    # print(c0[i+a],c1[i+a],c2[i+a],c3[i+a],c4[i+a],c5[i+a])
    kruskalWallis = stats.kruskal(c0[i+a],c1[i+a],c2[i+a],c3[i+a],c4[i+a],c5[i+a])
    print(kruskalWallis.pvalue)
    
potas0 = np.array(pd.read_csv('peaks_stim_cl0.csv',sep=',').transpose())
potas1 =  np.array(pd.read_csv('peaks_stim_cl1.csv',sep=',').transpose())
potas2 =  np.array(pd.read_csv('peaks_stim_cl2.csv',sep=',').transpose())
potas3 =  np.array(pd.read_csv('peaks_stim_cl3.csv',sep=',').transpose())
potas4 = np.array( pd.read_csv('peaks_stim_cl4.csv',sep=',').transpose())
potas5 = np.array( pd.read_csv('peaks_stim_cl5.csv',sep=',').transpose())
print('P-values of maximum potassium')
for i in range(5):
    # print(a)
    # print(c0[i+a],c1[i+a],c2[i+a],c3[i+a],c4[i+a],c5[i+a])
    kruskalWallis = stats.kruskal(potas0[i+a],potas1[i+a],potas2[i+a],potas3[i+a],potas4[i+a],potas5[i+a])
    print(kruskalWallis.pvalue)
    
potas0 = (pd.read_csv('peaks_stim_cl0.csv',sep=','))
potas1 = (pd.read_csv('peaks_stim_cl1.csv',sep=','))
potas2 = (pd.read_csv('peaks_stim_cl2.csv',sep=','))
potas3 = (pd.read_csv('peaks_stim_cl3.csv',sep=','))
potas4 = ( pd.read_csv('peaks_stim_cl4.csv',sep=','))
potas5 = ( pd.read_csv('peaks_stim_cl5.csv',sep=','))
label = []
for i in range(len(potas0)):
    label.append('Class0')
potas0=potas0.assign(Class=label)
label = []
for i in range(len(potas1)):
    label.append('Class1')
potas1=potas1.assign(Class=label)
label = []
for i in range(len(potas2)):
    label.append('Class2')
potas2=potas2.assign(Class=label)
label = []
for i in range(len(potas3)):
    label.append('Class3')
potas3= potas3.assign(Class=label)
label = []
for i in range(len(potas4)):
    label.append('Class4')
potas4=potas4.assign(Class=label)
label = []
for i in range(len(potas5)):
    label.append('Class5')
potas5=potas5.assign(Class=label)

dataframes = [potas0, potas1, potas2, potas3, potas4, potas5]
data = pd.concat(dataframes, ignore_index=True)

# Perform MANOVA
manova = MANOVA.from_formula('Var1 + Var2 + Var3 + Var4 + Var5 + Var6 + Var7 + Var8 + Var9 ~ Class', data=data)
# print(manova.mv_test())
mv_test_result = manova.mv_test()

# Extract p-values from the MANOVA results
wilks_lambda_p = mv_test_result.results['Class']['stat'].iloc[0, 4]
pillai_trace_p = mv_test_result.results['Class']['stat'].iloc[1, 4]
hotelling_lawley_p = mv_test_result.results['Class']['stat'].iloc[2, 4]
roys_greatest_root_p = mv_test_result.results['Class']['stat'].iloc[3, 4]

# Print the p-values
print('MANOVA: maximum potassium')
print("Wilks' Lambda p-value:", wilks_lambda_p)
print("Pillai's Trace p-value:", pillai_trace_p)
print("Hotelling-Lawley Trace p-value:", hotelling_lawley_p)
print("Roy's Greatest Root p-value:", roys_greatest_root_p)

potas0 = (pd.read_csv('sodium_peaks_stim_cl0.csv',sep=','))
potas1 = (pd.read_csv('sodium_peaks_stim_cl1.csv',sep=','))
potas2 = (pd.read_csv('sodium_peaks_stim_cl2.csv',sep=','))
potas3 = (pd.read_csv('sodium_peaks_stim_cl3.csv',sep=','))
potas4 = ( pd.read_csv('sodium_peaks_stim_cl4.csv',sep=','))
potas5 = ( pd.read_csv('sodium_peaks_stim_cl5.csv',sep=','))
label = []
for i in range(len(potas0)):
    label.append('Class0')
potas0=potas0.assign(Class=label)
label = []
for i in range(len(potas1)):
    label.append('Class1')
potas1=potas1.assign(Class=label)
label = []
for i in range(len(potas2)):
    label.append('Class2')
potas2=potas2.assign(Class=label)
label = []
for i in range(len(potas3)):
    label.append('Class3')
potas3= potas3.assign(Class=label)
label = []
for i in range(len(potas4)):
    label.append('Class4')
potas4=potas4.assign(Class=label)
label = []
for i in range(len(potas5)):
    label.append('Class5')
potas5=potas5.assign(Class=label)

dataframes = [potas0, potas1, potas2, potas3, potas4, potas5]
data = pd.concat(dataframes, ignore_index=True)

# Perform MANOVA
manova = MANOVA.from_formula('Var1 + Var2 + Var3 + Var4 + Var5 + Var6 + Var7 + Var8 + Var9 ~ Class', data=data)
# print(manova.mv_test())
mv_test_result = manova.mv_test()

# Extract p-values from the MANOVA results
wilks_lambda_p = mv_test_result.results['Class']['stat'].iloc[0, 4]
pillai_trace_p = mv_test_result.results['Class']['stat'].iloc[1, 4]
hotelling_lawley_p = mv_test_result.results['Class']['stat'].iloc[2, 4]
roys_greatest_root_p = mv_test_result.results['Class']['stat'].iloc[3, 4]

# Print the p-values
print('MANOVA: maximum sodium')
print("Wilks' Lambda p-value:", wilks_lambda_p)
print("Pillai's Trace p-value:", pillai_trace_p)
print("Hotelling-Lawley Trace p-value:", hotelling_lawley_p)
print("Roy's Greatest Root p-value:", roys_greatest_root_p)

# Test calcium features from matlab for significance ----------------------------------------------