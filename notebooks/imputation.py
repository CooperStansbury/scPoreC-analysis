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


def imputeEdges(A, tau, method=1, return_scores=True):
    """A function to compute a predicted matrix using the 
    resource allocation index
    
    args:
        : A (np.array): an adjacency matrix
        : tau (float): a threshold above which the
        scores will be converted to binary predicted edges.
        : method (int): a int flag to control the behavior 
        of the scoring algorithm
        
            (1) Resource allocation index
            (2) Preferential attachment index
            (3) Adamic-Adar index
            (4) CCPA
        : return_scores (bool): if true, return the score matrix
    
    returns:
        : Ahat (np.array): a predicted matrix that is the union of 
        edges from matrices P and A.
        : scores (np.array): normalized score array, same size as output
    """
    G = nx.from_numpy_array(A)    
    scores = np.zeros(A.shape)
    
    if method == 1:
        linkScores = nx.resource_allocation_index(G)
    elif method == 2:
        linkScores = nx.preferential_attachment(G)
    elif method == 3:
        linkScores = nx.adamic_adar_index(G)
    elif method == 4:
        linkScores = nx.common_neighbor_centrality(G, alpha=0.1)
    
    for i, j, s in linkScores:
        scores[i, j] = scores[i, j] + s
        scores[j, i] = scores[j, i] + s
    
    # normalize the link scores
    if not scores.max() == 0:
        scores = scores / scores.max()
    
    # threshold based prediction
    P = np.where(scores >= tau, 1, 0)
    
    Ahat = A + P
    Ahat = np.where(Ahat > 0, 1, 0)

    if return_scores:
        return Ahat, scores
    else:
        return Ahat


def mannUManual(yTrue, yPred):
    """A function to manually compute the Mann U statistic and 
    the relation to the AUC """
    n1 = np.sum(yTrue==1)
    n0 = len(yPred) - n1
    
    order = np.argsort(yPred)
    rank = np.argsort(order)
    rank += 1
    
    U1 = np.sum(rank[yTrue == 1]) - n1*(n1+1)/2
    AUC1 = U1/ (n1*n0)
    return U1, AUC1


def aucViaMannU(yTrue, yPred):
    """A function to convert a U stat to an AUCROC
    using scipy """
    n1 = np.sum(yTrue==1)
    n0 = len(yPred) - n1
    
    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(yTrue, yPred)
    aucroc = (U - n1*(n1+1)/2) / (n1*n0)
    return aucroc