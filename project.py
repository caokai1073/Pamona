import numpy as np
from numpy import linalg as la
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import block_diag

def project_func(data, dist, T, n_neighbors, Lambda, dim):

    n_datasets = len(data)
    H0 = []
    L = []
    for i in range(n_datasets-1):
        T[i] = T[i]*np.shape(data[i])[0]

    for i in range(n_datasets):    

        graph_data = kneighbors_graph(data[i], n_neighbors, mode="distance")
        graph_data = graph_data + graph_data.T.multiply(graph_data.T > graph_data) - \
            graph_data.multiply(graph_data.T > graph_data)
        W = np.array(graph_data.todense())
        index_pos = np.where(W>0)
        W[index_pos] = 1/W[index_pos] 
        D = np.diag(np.dot(W, np.ones(np.shape(W)[1])))
        L.append(D - W)

    Sigma_x = []
    Sigma_y = []
    for i in range(n_datasets-1):
        Sigma_y.append(np.diag(np.dot(np.transpose(np.ones(np.shape(T[i])[0])), T[i])))
        Sigma_x.append(np.diag(np.dot(T[i], np.ones(np.shape(T[i])[1]))))

    S_xy = T[0]
    S_xx = L[0] + Lambda*Sigma_x[0]
    S_yy = L[-1] + Lambda*Sigma_y[0]
    for i in range(1, n_datasets-1):
        S_xy = np.vstack((S_xy, T[i]))
        S_xx = block_diag(S_xx, L[i]+Lambda*Sigma_x[i])
        S_yy = S_yy + Lambda*Sigma_y[i]

    v, Q = la.eig(S_xx)
    v = v + 1e-12   
    V = np.diag(v**(-0.5))
    H_x = np.dot(Q, np.dot(V, np.transpose(Q)))

    v, Q = la.eig(S_yy)
    v = v + 1e-12      
    V = np.diag(v**(-0.5))
    H_y = np.dot(Q, np.dot(V, np.transpose(Q)))

    H = np.dot(H_x, np.dot(S_xy, H_y))
    U, sigma, V = la.svd(H)

    num = [0]
    for i in range(n_datasets-1):
        num.append(num[i]+len(data[i]))

    
    U, V = U[:,:dim], np.transpose(V)[:,:dim]

    fx = np.dot(H_x, U)
    fy = np.dot(H_y, V)

    integrated_data = []
    for i in range(n_datasets-1):
        integrated_data.append(fx[num[i]:num[i+1]])

    integrated_data.append(fy)
    
    return integrated_data