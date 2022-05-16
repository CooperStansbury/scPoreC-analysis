import pandas as pd
import numpy as np
import networkx as nx


def imputeContacts(A, tau=0.5, alpha=0.1):
    """A functiont to impute contacts based on the existing
    structure """
    G = nx.from_numpy_array(A)    
    P = np.zeros(A.shape)
    
    pLink = nx.common_neighbor_centrality(G, alpha=alpha)
    for i, j, p in pLink:
        P[i, j] = P[i, j] + p
        P[j, i] = P[j, i] + p
    
    # normalize the link scores
    P = P / P.max()
    
    # threshold based prediction
    P = np.where(P >= tau, 1, 0)
    return P