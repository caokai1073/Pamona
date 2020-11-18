# Pamona

## Paper
Manifold alignment for  heterogeneous single-cell multi-omics data integration using Pamona
## Enviroment

python >= 3.6

numpy 1.18.5  
scikit-learn 0.23.2  
umap-learn 0.3.10  
Cython 0.29.21  
scipy 1.4.1  
matplotlib 3.3.1  
POT 0.7.0  

## Install
Pamona software is available on the Python package index (PyPI), latest version 0.0.1. To install it using pip, simply type:
```
pip3 install pamona
```

## Integrate data
Each row should contain the measured values for a single cell, and each column should contain the values of a feature across cells. 
```data_0.txt, ... , data_N.txt``` to be integrated, use

```python
from pamona import Pamona
import numpy as np
data_0 = np.loadtxt("data_0.txt")
...
data_N = np.loadtxt("data_N.txt")
integrated_data, T = Pamona.run_Pamona([data_0, ..., data_N])
aligned_data_0 = integrated_data[0]
...
aligned_data_N = integrated_data[N]
```

## Parameters of ```run_Pamona```

The list of parameters is given below:
+ **data**:  *list of numpy array, [dataset1, dataset2, ...] (n_datasets, n_samples, n_features).*
list of datasets to be integrated, in the form of a numpy array.

+ **n_shared**: *int, default as the cell number of the smallest dataset.*
shared cell number between datasets.

+ **epsilon**: *float, default as 0.001.*
the regularization parameter of the partial-GW framework.

+ **n_neighbors**: *int, default as 30.*
the number of neighborhoods  of the k-nn graph.

+ **Lambda**: *float, default as 1.0.*
the parameter of manifold alignment to make a trade-off between aligning corresponding cells and preserving the local geometries

+ **output_dim**: *int, default as 30.*
output dimension of the common embedding space after the manifold alignment

+ **M**: *numpy array , default as None (optionally)*.
disagreement matrix of prior  information.

The other parameters include:

> + ```virtual_cells```: *int*, number of virtual cells, default as 1.
> + ```max_iter```: *int*, maximum iterations of the partial-GW framework, default as 1000.
> + ```tol```:  *float*, the precision condition under which the iteration of the partial-GW framework stops, default as 1e-9.
> + ```manual_seed```: *int*, random seed, default as 666.
> + ```mode```: *{‘connectivity’, ‘distance’}*, type of returned matrix: ‘connectivity’ will return the connectivity matrix with ones and zeros, and ‘distance’ will return the distances between neighbors according to the given metric. has to be either one of 'connectivity' or 'distance', default as 'distance'.
> + ```metric```: *str*, the distance metric used to calculate the k-Neighbors for each sample point, default as ’minkowski’.

## Visualization
```python
type_0 = type_0.astype(np.int)
...
type_N = type_N.astype(np.int)
Pamona.Visualize([data_0, ..., data_N], integrated_data, mode='PCA') # without datatype
Pamona.Visualize([data_0, ..., data_N], integrated_data, [type_0,...,type_N], mode='PCA) # with datatype
# mode: ["PCA", "TSNE", "UMAP"], default as "PCA"
```

