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
    if not P.max() == 0:
        P = P / P.max()
    
    # threshold based prediction
    P = np.where(P >= tau, 1, 0)
    return P


def randomUniformRemoval(hic, retain=0.1):
    """A function to randomly remove existing edges with uniform probablity
    
    args:
        : hic (dict): keys are cell ids and values are hi-c matrices
        : retain (float): proportion of edges to retain
        
    returns:
        : hicTrain (dict): values are matrices missing entries
    """
    hicTrain = {}

    for runId, A in hic.items():
        G = nx.from_numpy_array(A)
        
        # get edge list 
        E = np.array(G.edges)
        removeN = int((1-retain) * len(E))
        
        if removeN == 0:
            hicTrain[runId] = A
            continue
        
        # sample from edge list
        index = list(range(len(E)))
        removeIdx = np.random.choice(index, removeN, replace=False)
        
        toRemove = E[removeIdx]
        
        # remove edges
        G.remove_edges_from(toRemove)
        
        # create new matrix 
        Ahat = nx.to_numpy_matrix(G)
        hicTrain[runId] = Ahat
        
    return hicTrain


def degreeLimitRemoval(hic, retain=0.1):
    """A function to remove existing edges by targeting high degree connections
    
    args:
        : hic (dict): keys are cell ids and values are hi-c matrices
        : retain (float): proportion of edges to retain
        
    returns:
        : hicTrain (dict): values are matrices missing entries
    """
    hicTrain = {}

    for runId, A in hic.items():
        G = nx.from_numpy_array(A)
        
        # get edge list 
        removeN = int((1-retain) * len(G.edges))
        newSize = len(G.edges) - removeN
        
        if removeN == 0:
            hicTrain[runId] = A
            continue
        
        while len(G.edges) > newSize:
            maxDegree = sorted(G.degree, key=lambda x: x[1], reverse=True)
            maxDegreeNode = maxDegree[0][0]

            nieghbors = list(G.neighbors(maxDegreeNode))
            toRemove = np.random.choice(nieghbors, 1)[0]
            G.remove_edge(maxDegreeNode, toRemove)
        
        # create new matrix 
        Ahat = nx.to_numpy_matrix(G)
        hicTrain[runId] = Ahat   
        
    return hicTrain


def breadthFirstRemoval(hic, retain=0.1):
    """A function to remove existing edges using a BFS
    
    args:
        : hic (dict): keys are cell ids and values are hi-c matrices
        : retain (float): proportion of edges to retain
        
    returns:
        : hicTrain (dict): values are matrices missing entries
    """
    hicTrain = {}

    for runId, A in hic.items():
        G = nx.from_numpy_array(A)
        
        # get edge list 
        removeN = int((1-retain) * len(G.edges))
        newSize = len(G.edges) - removeN
        
        if removeN == 0:
            hicTrain[runId] = A
            continue
        
        # randomly sampled starting point
        nodeList = list(G.nodes)
        source = np.random.choice(nodeList, 1)[0]
        
        # bfs
        bfs = list(nx.bfs_edges(G, source))
        bfs_r = list(reversed(bfs))
        
        # remove edges not in bfs
        for e in G.edges:
            if not e in bfs_r[:removeN]:
                G.remove_edge(e[0], e[1])
                
        # create new matrix 
        Ahat = nx.to_numpy_matrix(G)
        hicTrain[runId] = Ahat   
    
    return hicTrain


def coldEndRemoval(hic, retain=0.1):
    """A function to remove existing edges using the popularity score 
    Zhu 2011:
    
        P_xy = (k_x - 1) * (k_y - 1)
        
    where k_x is the degree of node x.
    
    args:
        : hic (dict): keys are cell ids and values are hi-c matrices
        : retain (float): proportion of edges to retain
        
    returns:
        : hicTrain (dict): values are matrices missing entries
    """
    def getPopularity(G, i, j):
        return (G.degree(i) - 1) * (G.degree(j) - 1)
    
    hicTrain = {}

    for runId, A in hic.items():
        G = nx.from_numpy_array(A)
        
        # get edge list 
        E = np.array(G.edges)
        
        # get edge list 
        removeN = int((1-retain) * len(G.edges))
        newSize = len(G.edges) - removeN
        
        if removeN == 0:
            hicTrain[runId] = A
            continue
        
        popScores = []
        
        for i, j in G.edges:
            p_ij = getPopularity(G, i, j)
            popScores.append(p_ij)
            
        idx = np.argsort(popScores)[:removeN]
        toRemove = E[idx]
        
        # remove edges
        G.remove_edges_from(toRemove)
        
        # create new matrix 
        Ahat = nx.to_numpy_matrix(G)
        hicTrain[runId] = Ahat
        
    return hicTrain