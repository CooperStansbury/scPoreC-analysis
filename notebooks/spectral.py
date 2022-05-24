import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import networkx as nx
from sklearn.metrics import jaccard_score


def getPC1(A):
    """A function to get the first component"""
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(A)
    return pca.components_[0, :]


def getCorrelationMatrix(A):
    """A function to generate a column-wise correlation matrix
    as described by Lieberman-Aiden 2009 """
    
    C = np.zeros(A.shape)
    n = A.shape[0]
    
    for i in range(n):
        for j in range(n):
        
            r = stats.pearsonr(A[:, i], A[:, j])[0]

            C[i, j] = r
            C[j, i] = r
    C = np.where(np.isnan(C), 0, C)
    
    return C


def getJaccardMatrix(A):
    """A function to generate a column-wise Jaccard matrix """
    C = np.zeros(A.shape)
    n = A.shape[0]
    
    for i in range(n):
        for j in range(n):
        
            r = jaccard_score(A[:, i], A[:, j])

            C[i, j] = r
            C[j, i] = r
    C = np.where(np.isnan(C), 0, C)
    
    return C
        

def pageRankNormalize(A, d=0.85):
    N = A.shape[1]
    A_hat = (d * A + (1 - d) / N)
    return A_hat