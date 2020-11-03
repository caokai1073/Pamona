'''
author: Kai Cao
email: caokai@amss.ac.cn
'''

import time
import numpy as np
import random
import ot
from ot.bregman import sinkhorn
from ot.gromov import init_matrix, gwggrad
from ot.partial import gwgrad_partial

from eval import *
from utils import *
from visualization import visualize
from project import project_func

def entropic_gromov_wasserstein(C1, C2, p, q, m, M=None, epsilon=0.001, loss_fun='square_loss',
                                 virtual_cells=1, max_iter=1000, tol=1e-9, verbose=True):


    C1 = np.asarray(C1, dtype=np.float32)
    C2 = np.asarray(C2, dtype=np.float32)

    G0 = np.outer(p, q)  # Initialization

    dim_G_extended = (len(p) + virtual_cells, len(q) + virtual_cells)
    q_extended = np.append(q, [(np.sum(p) - m) / virtual_cells] * virtual_cells)
    p_extended = np.append(p, [(np.sum(q) - m) / virtual_cells] * virtual_cells)

    q_extended = q_extended/np.sum(q_extended)
    p_extended = p_extended/np.sum(p_extended)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    while (err > tol and cpt < max_iter):

        Gprev = G0

        # compute the gradient
        if abs(m-1)<1e-10: # full match
            Ck = gwggrad(constC, hC1, hC2, G0)
        else: # partial match
            Ck = gwgrad_partial(C1, C2, G0)
       
        if M is not None:
            Ck = Ck*M

        Ck_emd = np.zeros(dim_G_extended)
        Ck_emd[:len(p), :len(q)] = Ck
        Ck_emd[-virtual_cells:, -virtual_cells:] = 100*np.max(Ck_emd)
        Ck_emd = np.asarray(Ck_emd, dtype=np.float64)

        # Gc = sinkhorn(p, q, Ck, epsilon, method = 'sinkhorn')
        Gc = sinkhorn(p_extended, q_extended, Ck_emd, epsilon, method = 'sinkhorn')
        G0 = Gc[:len(p), :len(q)]

        if cpt % 10 == 0:
            err = np.linalg.norm(G0 - Gprev)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'Epoch.', 'Loss') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1
    
    return Gc

def run_Pamona(data, n_shared=None, M=None, n_neighbors=10, epsilon=0.001, Lambda=1, virtual_cells=1, \
    output_dim=30, max_iter=1000, tol=1e-9, manual_seed=666, mode="distance", metric="minkowski", verbose=True):

    print("Pamona start!")
    time1 = time.time()

    init_random_seed(666)

    dist = []
    sampleNo = []
    Max = []
    Min = []
    p = []
    q = []
    Gc = []
    n_datasets = len(data)

    for i in range(n_datasets):
        sampleNo.append(np.shape(data[i])[0])
        dist.append(Pamona_geodesic_distances(data[i], n_neighbors, mode=mode, metric=metric))

    for i in range(n_datasets-1):
        Max.append(np.maximum(sampleNo[i], sampleNo[-1])) 
        Min.append(np.minimum(sampleNo[i], sampleNo[-1]))

    if n_shared is None:
    	n_shared = Min

    for i in range(n_datasets-1):
        if n_shared[i] > Min[i]:
            n_shared[i] = Min[i]
        p.append(ot.unif(Max[i])[0:len(data[i])])
        q.append(ot.unif(Max[i])[0:len(data[-1])])

    T = []

    for i in range(n_datasets-1):
        if M is not None:
            Gc_tmp = entropic_gromov_wasserstein(dist[i], dist[-1], p[i], q[i], n_shared[i]/Max[i]-1e-15, M[i], \
                epsilon=epsilon, tol=tol, max_iter=max_iter, virtual_cells=virtual_cells, verbose=verbose)
        else:
           Gc_tmp = entropic_gromov_wasserstein(dist[i], dist[-1], p[i], q[i], n_shared[i]/Max[i]-1e-15, \
            epsilon=epsilon, tol=tol, max_iter=max_iter, virtual_cells=virtual_cells, verbose=verbose)
        T.append(Gc_tmp) 
        Gc.append(Gc_tmp[:len(p[i]), :len(q[i])])

    integrated_data = project_func(data, dist, Gc, n_neighbors, Lambda=Lambda, dim=output_dim)

    time2 = time.time()
    print("Pamona Done! takes {:f}".format(time2-time1), 'seconds')

    return integrated_data, T

def Visualize(data, integrated_data, datatype=None, mode='PCA'):
	if datatype == None:
		visualize(data, integrated_data, mode=mode)
	else:
		visualize(data, integrated_data, datatype, mode=mode)

def label_transfer_accuracy(dataset1, dataset2, datatype1, datatype2):
	label_transfer_acc = test_transfer_accuracy(dataset1,dataset2,datatype1,datatype2)
	print("label transfer accuracy:")
	print(label_transfer_acc)

def alignment_score(data1_shared, data2_shared, data1_specific=None, data2_specific=None):
	alignment_sco = test_alignment_score(data1_shared, data2_shared, data1_specific=data1_specific, data2_specific=data2_specific)
	print("alignment score:")
	print(alignment_sco)

### example
# data1 = np.loadtxt("./MMD/s1_mapped1.txt")
# data2 = np.loadtxt("./MMD/s1_mapped2.txt")
# type1 = np.loadtxt("./MMD/s1_type1.txt")
# type2 = np.loadtxt("./MMD/s1_type2.txt")
# index1 = np.argwhere(type1==0).reshape(1,-1).flatten()    
# index2 = np.argwhere(type1==2).reshape(1,-1).flatten()
# index3 = np.argwhere(type1==1).reshape(1,-1).flatten()
# index = np.hstack((index1, index2))
# print(len(index))
# index = np.hstack((index, index3))
# type1 = type1[index]
# data1 = data1[index]
# index1 = np.argwhere(type2==0).reshape(1,-1).flatten()
# index2 = np.argwhere(type2==2).reshape(1,-1).flatten()
# index = np.hstack((index1, index2))
# type2 = type2[index]
# data2 = data2[index]
# type1 = type1.astype(np.int)
# type2 = type2.astype(np.int)
# data = [data1,data2]
# datatype = [type1,type2]
# integrated_data, T = run_Pamona(data, epsilon=0.001, n_neighbors=10, Lambda=1, output_dim=30)
# label_transfer_accuracy(integrated_data[0][0:241],integrated_data[1],type1[0:241],type2)
# alignment_score(integrated_data[0][0:241], integrated_data[1], data1_specific=integrated_data[0][241:300])
# Visualize([data1,data2], integrated_data, datatype=datatype, mode='UMAP')
