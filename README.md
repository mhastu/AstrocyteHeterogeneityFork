***Documentation: Astrocyte Heterogeneity***

**Install Anaconda**

1.  Install Anaconda
2.  Start Anaconda Prompt with Admin rights
3.  python -V # check python version
4.  conda install python=3.9.13 # install specific python version
5.  python -c "import numpy; print(numpy.version.version)" # check numpy version
6.  conda install -c conda-forge numpy=1.23.5 # install specific numpy version

**Generating data files (AstrocyteHeterogeneity\_00\_data)**

1.  Put following files in one folder:  
    astro\_geometry.m, dataset\_\[_Name of dataset_\].mat, matlab\_and\_excel\_data.mat
2.  Execute following lines in MATLAB:  
    —> load('dataset\_\[_Name of dataset_\].mat')  
    —> writetable(positionData, 'positionData.csv')  
    —> writetable(diameterData, 'diameterData.csv')  
    —> astro\_geometry()  
    Window opens: Select dataset same as loaded in the first line  
    —> Give the dataset name: _branchData_  
    —> Give the cell ID (maximum ID is 418): \[_cell ID_\] e.g.: _144_  
    —> Give extra information for the export name (name will be "cell\_<dataset>\_<number>\_<your input here>": \[_Enter_\]  
    : MATLAB generates _branchData.mat_ file in folder cell\_branchData\_\[_cell ID_\]

**Preprocessing of the astrocyte reconstructions (AstrocyteHeterogeneity\_01\_preprocessing)**

1.  Put following files in the preprocessing folder:  
    dataset\_\[_Name of dataset_\].mat, matlab\_and\_excel\_data.mat
2.  Execute cells in the _Workspace.ipynb_ notebook
3.  The function _main.gen\_CellInfo_ generates .csv file with features
4.  The function _main\_Test2.plot\_FeatureVector_ can be use to plot the features

**Classification of the astrocyte morphologies (AstrocyteHeterogeneity\_02\_classification)**

1.  Name the feature vector csv file: _FeatureVector_ and move it to same folder as the _ClusteringAstrocytes00.py_ code
2.  Run the code: all figures and the labelled dataset are saved in the same folder

**Statistical analysis – Part A (AstrocyteHeterogeneity\_03\_statistics)**

1.  Name the feature vector csv file: _FeatureVector_ and move it to same folder as the _StatisticalAnalysis00.py_ code
2.  Move the labelled data _dataLabeld.csv_ produced by _ClusteringAstrocytes00.py_ into the same folder
3.  Run the code: prints p-value of Kruskal-Wallis test for each class and produces figure with violin plots.

**Simulations (AstrocyteHeterogeneity\_04\_simulation)**

1.  Copy _branchData.mat_ file to _SingleCellModel\_Python-MultiCompartmentModel\_With\_MorphologyData_ folder
2.  Copy _diameterData.csv_ to _SingleCellModel\_Python-MultiCompartmentModel\_With\_MorphologyData_ folder
3.  Copy _positionData.csv_ to _SingleCellModel\_Python-MultiCompartmentModel\_With\_MorphologyData_ folder
4.  Open _morph.ini_ file in _output_ folder:  
    —> Set _cellID_ and _dataSet_ correctly (same as used in MATLAB)  
    _—> oneBranch_ can either be True (only one main branch gets simulated) or False (the whole cell up to the given level gets simulated)  
    _—> maximum\_level_ is set to the amount of level that should get constructed, “_all_” constructs the whole cell/branch (if _oneBranch_ \= True), if only one main branch without side branches sould be constructed, select same number as for _mainBranch_  
    —> _mainBranch_ (only taken into account if _oneBranch_ is set to True): level which should be constructed  
    —> _compartment\_time\_curves\_to\_plot_: select compartments for which calcium dynamics should get plotted  
    —> _smallest\_diameter_: set the minimum diameter value a branch needs to get constracted  
    —> _radius\_soma_: radius of the soma; _diame\_max_ sets maximum branch radius as soma radius, _depth\_max_ sets the maximum length of a branch with depth 1 as soma radius, _depth\_min_ sets the minimum length of a branch with depth 1 as soma radius  
    —> Save !!!
5.  Main.py line 45: _simulate\_morpology\_model_ can either be set to False (only morphology plot), or True (plotted compartments get simulated)
6.  For simulation the Stimulus can be set in astrocyte.py in the folder astrocyte  
    line 59-61: Stimulus is given in mM  
    line 68: Compartment with gets stimulated is selected
7.  Run main.py  
    Code prints number of compartments and processes  
    Plots are saved in the output folder
8.  To find length of processes  
    —> Uncomment line 32 and 33 in main.py  
    —> set _oneBranch_ \= False in _morph.ini_ file (and Save!)  
    —> Prints length on each process up to given level  
    Careful: at a certain point a new level adds to an already existing process: with _maximum\_level_ = 50 process 5 might have 21 compartments and at _maximum\_level_ = 5 it has only 13 compartments (which would then be the amount of compartments of the main branch, without side branches)

**Changes in the code from Rene’s GitHub version**

*   Changed Diffusion coefficients in compartment\_pathway.py
*   added ax=plt.gca() as option to plt.colorbar(sm, label="compartment diameter in m") in line 296 in Astrocyte\_Morphology\_Model.py
*   Helper dictionary in Astrocyte\_Morphology\_Model.py
*   Added mainBranch and oneBranch variable in Astrocyte\_Morphology\_Model.py
*   Allow maximum\_level to have value “all”

**Statistical analysis – Part B (AstrocyteHeterogeneity\_05\_statistics)**

1.  Prints the P-values of the Kruskal-Wallis test for the calcium spikes count
2.  Prints the P-values of the Kruskal-Wallis test for the calcium spikes amplitude
3.  P-values of the Kruskal-Wallis test for the maximum potassium
