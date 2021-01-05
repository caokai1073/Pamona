# Pamona

## Paper
[Manifold alignment for  heterogeneous single-cell multi-omics data integration using Pamona](https://doi.org/10.1101/2020.11.03.366146)

The implementation is based on [UnionCom](https://github.com/caokai1073/unionCom) and [SCOT](https://github.com/rsinghlab/SCOT).

## Enviroment

python >= 3.6

numpy >= 1.18.5  
scikit-learn >= 0.23.2  
umap-learn >= 0.3.10  
Cython >= 0.29.21  
scipy >= 1.4.1  
matplotlib >= 3.3.1  
POT >= 0.7.0  

## Install
Pamona software is available on the Python package index (PyPI), latest version 0.0.1. To install it using pip, simply type:
```
pip3 install pamona
```

## Integrate data
Each row should contain the measured values for a single cell, and each column should contain the values of a feature across cells. 
```python
>>> from pamona import Pamona
>>> import numpy as np
>>> data1 = np.loadtxt("./scGEM/methylation_partial.txt")
>>> data2 = np.loadtxt("./scGEM/expression_partial.txt")
>>> type1 = np.loadtxt("./scGEM/methylation_type_partial.txt")
>>> type2 = np.loadtxt("./scGEM/expression_type_partial.txt")
>>> type1 = type1.astype(np.int)
>>> type2 = type2.astype(np.int)
>>> Pa = Pamona.Pamona(n_shared=[138], Lambda=10, output_dim=5) # shared cell number 138 is estimated by SPL
>>> integrated_data, T = Pa.run_Pamona(data)
>>> Pa.test_LabelTA(integrated_data[0],integrated_data[-1],type1,type2)
>>> Pa.alignment_score(integrated_data[0], integrated_data[-1][0:142], data2_specific=integrated_data[-1][142:177])
>>> Pa.Visualize([data1,data2], integrated_data, mode='PCA')  # without datatype
>>> Pa.Visualize([data1,data2], integrated_data, [type1,type2], mode='PCA')  # with datatype 
>>> # mode: ["PCA", "TSNE", "UMAP"], default as "PCA".
```
## Example of design disagreement matrix of prior information
If cell types are available, users can incorporate the information as follows
```python
>>> gamma = 0.5 # gamma is a parameter, ranges from 0 to 1. 
>>> # A larger gamma gives more importance to the matching of prior information.
>>> DM = np.ones((len(data1), len(data2)))
>>> for i in range(len(data1)):
>>>    for j in range(len(data2)):
>>>        if type1[i] == type2[j]:
>>>            DM[i][j] = gamma
>>> ...
>>> Pa = Pamona.Pamona(..., M=DM, ...)
```

## Parameters of ```class Pamona```

The list of parameters is given below:
+ **data**:  *list of numpy array, [dataset1, dataset2, ...] (n_datasets, n_samples, n_features).*
list of datasets to be integrated, in the form of a numpy array.

+ **n_shared**: *int, default as the cell number of the smallest dataset.*
shared cell number between datasets.

+ **epsilon**: *float, default as 0.001.*
the regularization parameter of the partial-GW framework.

+ **n_neighbors**: *int, default as 30.*
the number of neighborhoods of the k-nn graph.

+ **Lambda**: *float, default as 1.0.*
the parameter of manifold alignment to make a trade-off between aligning corresponding cells and preserving the local geometries

+ **output_dim**: *int, default as 30.*
output dimension of the common embedding space after the manifold alignment

+ **M**: *numpy array , default as None (optionally)*.
disagreement matrix of prior information.

The other parameters include:

> + ```virtual_cells```: *int*, number of virtual cells, default as 1.
> + ```max_iter```: *int*, maximum iterations of the partial-GW framework, default as 1000.
> + ```tol```:  *float*, the precision condition under which the iteration of the partial-GW framework stops, default as 1e-9.
> + ```manual_seed```: *int*, random seed, default as 666.
> + ```mode```: *{‘connectivity’, ‘distance’}*, type of returned matrix: ‘connectivity’ will return the connectivity matrix with ones and zeros, and ‘distance’ will return the distances between neighbors according to the given metric. has to be either one of 'connectivity' or 'distance', default as 'distance'.
> + ```metric```: *str*, the distance metric used to calculate the k-Neighbors for each sample point, default as ’minkowski’.

### Contact via caokai@amss.ac.cn
