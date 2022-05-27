import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.linalg import toeplitz
import cooler

import os
import sys


def loadRawContactDirectory(dirpath, filetags, verbose=True):
    """A function to load a directory of contact tables"""
    results = {}
    for runId in filetags:
        df = loadContactTable(dirpath, runId)
        
        if verbose:
            print(f"{runId=} {df.shape=}")
        
        results[runId] = df
    return results


def loadAssembly(filepath):
    """A function to lopad the assembly file
    
    args:
        : filepath (str): file path to vthe assembly information
    
    returns:
        : assmebly (pd.DataFrame): the assembly information table
    """
    assembly = pd.read_csv(filepath)
    assembly['chromEnd'] = assembly['Total length'].cumsum()
    assembly['chromStart'] = assembly['chromEnd'] - assembly['Total length']
    assembly['RefSeq accession'] = assembly['RefSeq accession'].str.strip()
    return assembly


def loadContactTable(directory, filetag):
    """A function to load and merge a set of contact tables
    from a directory based on keyword filename matching
    
    args:
        : directory (str): path including tailing backslash
        : filetag (str): the matching keyword used to load all
            contact tables for a given object
    
    returns:
        : df (pd.DataFrame): a dataframe of the Nanopore Pore-C
        contact table
    """
    df_list = []
    for f in os.listdir(directory):
        if filetag in f:
            filepath = f"{directory}{f}"
            tmp = pd.read_parquet(filepath)
            tmp['batch_id'] = f.split("_")[2]
            df_list.append(tmp)

    df = pd.concat(df_list)
    return df


def mergeAssembly(df, assembly):
    """A function to merge the assembly positions
    
    args:
        df (pd.DataFrame): the contact table
        assembly (pd.DataFrame): the assembly table
    
    returns:
        df (pd.DataFrame): the contact table with chromosome information for 
        each loci
    """
    
    align1Assembly = assembly.copy()
    align1Assembly.columns = ['align1_chromosome_name',
                              'align1_chrom_length', 
                              'align1_genbank',	
                              'align1_refSeq',
                              'align1_chrom_end',
                              'align1_chrom_start']
    
    align2Assembly = assembly.copy()
    align2Assembly.columns = ['align2_chromosome_name',
                              'align2_chrom_length', 
                              'align2_genbank',	
                              'align2_refSeq',
                              'align2_chrom_end',
                              'align2_chrom_start']
    df = pd.merge(df, 
                  align1Assembly, 
                  how='left', 
                  left_on='align1_chrom', 
                  right_on='align1_refSeq')
    
    df = pd.merge(df, 
                  align2Assembly, 
                  how='left', 
                  left_on='align2_chrom', 
                  right_on='align2_refSeq')
    
    """WARNING: drops rows where either
    alignment is not mappable """
    
    mask = (df['align1_refSeq'].notna() & df['align1_refSeq'].notna())
    df = df[mask].reset_index(drop=True)
    
    
    df['align1_absolute_start'] = df['align1_fragment_start'] + df['align1_chrom_start']
    df['align2_absolute_start'] = df['align2_fragment_start'] + df['align2_chrom_start']
    
    df['align1_absolute_end'] = df['align1_fragment_end'] + df['align1_chrom_start']
    df['align2_absolute_end'] = df['align2_fragment_end'] + df['align2_chrom_start']
    
    df['align1_absolute_midpoint'] = np.ceil((df['align1_absolute_start'] + df['align1_absolute_end'] ) / 2).astype(int)
    df['align2_absolute_midpoint'] = np.ceil((df['align2_absolute_start'] + df['align2_absolute_end'] ) / 2).astype(int)
    
    
    return df


def getFragmentMAPQ(df):
    """A function to return the mapq scores of all fragments
    
    returns:
        : mapq (pd.DataFrame): a two-column dataframe with fragment
        ids and mapping quality
    """
    align1 = df[['align1_fragment_id', 'align1_mapping_quality']]
    align1.columns = ['fragment_id', 'mapq']

    align2 = df[['align2_fragment_id', 'align2_mapping_quality']]
    align2.columns = ['fragment_id', 'mapq']

    mapq = pd.concat([align1, align2], ignore_index=True)
    return mapq


def intervsIntraSummary(df):
    """A function to print summary statistics regarding the ratio
    of inter and intra chromosomal contacts
    
    returns:
        : res (pd.DataFrame): a summary table
    """
    totalContacts = len(df)
    
    isIntra = np.where(df['align1_chrom'] == df['align2_chrom'], 1, 0)
    nIntra = np.sum(isIntra)
    nInter = totalContacts - nIntra
    
    metrics = ['Total Contacts', 'Intra- Contacts', 'Inter- Contacts']
    values = [totalContacts, nIntra, nInter]
    
    res = {
        'Metric' : metrics,
        'Value' : values,
        'Percentage' : [x/totalContacts for x in values]
    }
    res = pd.DataFrame(data=res)
    return res
    
    
def cisTransSummary(df):
    """A function to print the cis-trans ratio as defined by Nanopore
    Pore-C pipeline 
    
    returns:
        : res (pd.DataFrame): a summary table
    """
    
    totalContacts = len(df)
    cisContacts = np.sum(np.where(df['contact_is_cis'] == 1, 1, 0))
    transContacts = totalContacts - cisContacts
    
    metrics = ['Total Contacts', 'cis Contacts', 'trans Contacts']
    values = [totalContacts, cisContacts, transContacts]
    perc = [x/totalContacts for x in values]
    
    res = {
        'Metric' : metrics,
        'Value' : values,
        'Percentage' : perc
    }
    res = pd.DataFrame(data=res)
    return res
    

def contactDirectSummary(df):
    """A function to count the number of direct vs.
    indirect contacts. Indirect contacts are clique expansion 
    products
    
    returns:
        : res (pd.DataFrame): a summary table
    """
    totalContacts = len(df)
    
    isDirect = np.sum(np.where(df['contact_is_direct'] == 1, 1, 0))
    inDirect = totalContacts - isDirect
    
    metrics = ['Total Contacts', 'Direct Contacts', 'Indirect Contacts']
    values = [totalContacts, isDirect, inDirect]
    
    res = {
        'Metric' : metrics,
        'Value' : values,
        'Percentage' : [x/totalContacts for x in values]
    }
    res = pd.DataFrame(data=res)
    return res
    
    
def contactOrderSummary(df):
    """A function to print higher order contact summary
    
    returns:
        : res (pd.DataFrame): a summary table
    """
    concatemerCounts = df.groupby('read_name')['align1_fragment_id'].count().reset_index()
    concatemerCounts.columns = ['read_name', 'count']
    
    nReads = len(concatemerCounts)
    
    nSingleton = np.sum(np.where(concatemerCounts['count'] == 1, 1, 0))
    nPair = np.sum(np.where(concatemerCounts['count'] == 2, 1, 0))
    nMultiway = np.sum(np.where(concatemerCounts['count'] > 2, 1, 0))

    metrics = ['Total Reads', 'Singletons', 'Pairs', 'Multiway']
    values = [nReads, nSingleton, nPair, nMultiway]
    
    res = {
        'Metric' : metrics,
        'Value' : values,
        'Percentage' : [x/nReads for x in values]
    }
    res = pd.DataFrame(data=res)
    return res
    
    
def getSummary(df):
    """A function to gather summary information from the current dataframe"""
    
    empty_row = {
        'Metric' : ['--'],
        'Value' : ['--'],
        'Percentage' : ['--']
    }
    
    empty = pd.DataFrame(data=empty_row)
    
    res = pd.concat([contactOrderSummary(df), 
                     empty,
                     cisTransSummary(df), 
                     empty,
                     contactDirectSummary(df)]
                     ).reset_index()
    return res
    
    
def printSummary(res):
    """A function to print the summary compactly"""
    
    for idx, row in res.iterrows():
        
        if isinstance(row['Percentage'], float):
            print(f"{row['Metric']} {row['Value']} ({row['Percentage']:.3f})")
        else:
            print(f"{row['Metric']} {row['Value']} {row['Percentage']}")


def constructHiCSingleChromosome(df, log=True, binary=False):
    grped = df.groupby(['align1_chrom2Bin', 'align2_chrom2Bin'])['read_name'].count().reset_index() # NOTE: not counting unique here

    binBin = grped.pivot(*grped)
    if log:
        binBin = np.log(binBin) # log scale

    binBin = binBin.fillna(0)

    missing = np.setxor1d(binBin.index, binBin.columns)

    # keep only symmetric entries
    binBin = binBin.drop(index=missing, errors='ignore')
    binBin = binBin.drop(missing, axis=1, errors='ignore')

    # symmetrize
    binBin = binBin + binBin.T - np.diag(np.diag(binBin))
        
    if binary:
        binBin = np.where(binBin > 0, 1, 0)
    return binBin


def getHic(df, bins, label1, label2):
    """A function to build a symmetric matrix from contacts while
    filling missing interactions
    
    
    args:
        : df (pd.DataFrame): contact table (filtered) with bin names
        : bins (array): iterable with the number of bins
        : label1 (str): label of alignment1 bin column
        : label2 (str): label of alignment2 bin column
        
    returns:
        : A (np.array 2D): symmetric positive matrix with every interaction
    """
    
    
    grped = df.groupby([label1, label2])['read_name'].count().reset_index() # NOTE: not counting unique here]
    
    nBins = len(bins)
    A = np.zeros((nBins, nBins))
    
    for idx, row in grped.iterrows():
        i = row[label1]
        j = row[label2]
        
        A[i, j] = A[i, j] + 1
        A[j, i] = A[j, i] + 1

    return A


def getChromosomeInfo(coolerObject):
    """A function to return bin information from a coolr file
    
    args:
        : coolerObject (cooler.api.Cooler): a cooler object
    
    returns:
        : chromInfo (pd.DataFrame): informatio on bin location for each region
    """
    newRows = []
    for chrom in coolerObject.chromnames:
        chromStart, chromEnd =  coolerObject.extent(chrom)
        
        row = {
            'region' : chrom,
            'start' : chromStart,
            'end' : chromEnd,
            'length' : chromEnd - chromStart,
        }
        
        newRows.append(row)
    return pd.DataFrame(newRows)


def loadNagano2017SingleCell(scoolPath, filename, chromOrder, balance=False, verbose=False):
    """A function to load a single cell from the nagano dataset
    
    NOTE: these are all 1MB scale
    
    args:
        : scoolPath (str): file path to scool file
        : filename (str): sub cool single cell identifier 
        : chromOrder (list of str): an ordered array with the correct order of chromosomes
        : balance (bool): if true, return KR normalized (required `weight` column)
        : verbose (bool): if true, print chrom ranges
    
    returns:
        : A (np.array): raw contacts, UNORDERED
        : binRange (pd.DataFrame): the bin ranges for each region
        : hicIndex (list of int): all index positions ordered according to chrom order
    """
    clr = cooler.Cooler(f"{scoolPath}::{filename}")
    
    chromInfo = getChromosomeInfo(clr)
    
    hicIndex = []

    for chrom in chromOrder:
        row = chromInfo.loc[chromInfo['region'] == chrom]

        indRange = list(np.arange(row['start'].values, row['end'].values))
        hicIndex += indRange
        
    A = clr.matrix(balance=balance)[:]    
    return A, chromInfo, hicIndex


def loadPorecCooler(filename, assembly, chromOrder, resolution=1000000, balance=False, verbose=False):
    """A function to load a single cell from the nagano dataset
    
    NOTE: these are all 1MB scale
    
    args:
        : filename (str): full path to file
        : assembly (pd.DataFrame): a dataframe with assembly information
         : chromOrder (list of str): an ordered array with the correct order of chromosomes
        : resolution (int): default is 1MB
        : balance (bool): if true, return KR normalized (required `weight` column)
        : verbose (bool): if true, print chrom ranges
    
    returns:
        : A (np.array): raw contacts, UNORDERED
        : binRange (pd.DataFrame): the bin ranges for each region
        : hicIndex (list of int): all index positions ordered according to chrom order
    """
    chromDict = dict(zip(assembly['RefSeq accession'],assembly['Chromosome'].apply(lambda x : f"chr{x}")))
    
    clr = cooler.Cooler(f'{filename}::resolutions/{resolution}')
    
    chromInfo = getChromosomeInfo(clr)
    
    chromInfo['region'] = chromInfo['region'].astype(str)
    # translate refseq to chromosome names
    chromInfo['chromName'] = chromInfo['region'].map(chromDict) 
    
    # WARNING: dropping non chromosomal reagions
    chromInfo = chromInfo[chromInfo['chromName'].notna()].reset_index(drop=True)
    
    index = []

    for chrom in chromOrder:
        row = chromInfo.loc[chromInfo['chromName'] == chrom]

        indRange = list(np.arange(row['start'].values, row['end'].values))
        index += indRange
        
    A = clr.matrix(balance=balance)[:]    
    return A, chromInfo, index


def getIndices(chromInfo, reIndexed, chromosomeList, lookUpColumn):
    """A function to get indices of a sub matrix and to order them accordingly
    
    args:
        : chromInfo (pd.DataFrame): a datafram with chromosome positions 
        in binned genome coordinates 
        : reIndexed (list of int): current indexing scheme
        : chromosomeList (list of str): list of regions by name
        : lookUpColumn (str): the column of chromInfo to use
    
    returns:
        : newIndex (list of int): the new indices for subsetting
    """
    newIndex = []
    
    for chromosome in chromosomeList:    
        row = chromInfo.loc[chromInfo[lookUpColumn] == chromosome]

        start = row['start'].values[0]
        end = row['end'].values[0]
        chromLength = end - start

        newIndexStart = reIndexed.index(start) + 1
        subsetIndex = list(range(newIndexStart, newIndexStart+chromLength))
        
        newIndex += subsetIndex
        
    return newIndex


def normByDiag(A):
    """A function to normalize a matrix by average diagonal values.
    Args:
        - A (np.array): a matrix to normalize
    Returns:
        - Ahat (np.array): matrix A normalized by the diagonal
    """

    # build a new matrix of zeros
    Ahat = np.zeros(A.shape)


    for offset in range(len(A)):
        # get each diagonal, divide it by it's
        # mean value and add it to the zero matrix
        diag = np.diagonal(A, offset=offset)
        
        
        mudiag = np.mean(diag)

        
        if mudiag > 0:
            normed_diag = diag / mudiag
        else:
            normed_diag = [0] * (len(A) - offset)
        
        upper = np.diagflat(normed_diag, offset) 
        lower = np.diagflat(normed_diag, -offset) 
        
        Ahat += upper       
        Ahat += lower       
        
    return Ahat


def forceAdjacentConnections(A, num=1):
    """A function to ensure that all adjcent genomic 
    loci are connected. Input is assumed to be binary. 
    
    args:
        : A (np.array): unweighted adjacency matrix
        : num (float): the value to set the off diagonal to
    
    returns:
        : Ahat (np.array): unweighted adjacency matrix with all i, i+1
        connections added 
    """
    Ahat = A.copy()
    
    for i in range(len(Ahat) - 1):
        Ahat[i, i+1] = num
        Ahat[i+1, i] = num
    return Ahat


def dropZeroRows(A, threshold=0, return_ind=False):
    """A function to remove the zero count columns and rows from
    a symmetric matrix 
    
    args:
        : A (np.array): a symmetric matrix
        : threshold (int): number of contacts necesary to keep
        : return_ind (bool): if true, return the indices of dropped rows
    
    returns:
        : Ahat (np.array): a matrix with REDUCED dimensionality
    """
    rowSums = A.sum(axis=0)
    rmInd = np.argwhere(rowSums <= threshold)
    
    Ahat = A.copy()
    
    Ahat = np.delete(Ahat, rmInd, axis=0)
    Ahat = np.delete(Ahat, rmInd, axis=1)
    
    if return_ind:
        return Ahat, rmInd
    else:
        return Ahat
    


def filteredDatatoDict(filepath):
    """A function to reload the data into separate dataframes 
    
    args:
        : filepath (str): the location of the `.csv` file
    
    returns:
        : filteredCells (dict): keys are run ids and values are 
        filtered contact tables (unbinned)
    """
    df = pd.read_csv(filepath)
    filteredCells = {}
    
    for runId in df['cell'].unique():
        cellFrame = df[df['cell'] == runId]
        
        filteredCells[runId] = cellFrame

    return filteredCells


def getToeplitz(A, return_means=False):
    """A function to get the toeplitz from the observed matrix 
    
    args:
        : A (np.array): the contact map
        : return_means (bool): if true, return the means
    
    returns:
        : E (np.array): the expected contact map
    """
    
    muDiags = []
    
    for offset in range(len(A)):
        # get each diagonal, divide it by it's
        # mean value and add it to the zero matrix
        mudiag = np.mean(np.diagonal(A, offset=offset))
        muDiags.append(mudiag)
        
    E = toeplitz(muDiags, muDiags)
    
    if return_means:
        return E, muDiags
    else:
        return E


def normalizeToeplitz(O):
    """A function to normalize and input matrix by the 
    mean diagonal value 
    
    args:
        : O (np.array): the 2d observed matrix to normalize
    
    returns:
        : A (np.array) of the same shape normalized by the diagonals
    """
    
    E = getToeplitz(O)
    A = np.divide(O, E)
    
    # handle NaNs
    A = np.where(np.isnan(A), 0, A)
    return A


def pageRankNorm(A, d=0.85):
    """A function to normalize via the PageRank proc
    
    args:
        : A (np.array): input matrix
        
    returns:
        : Ahat (np.array): the normalized input matrix
    """
    n = A.shape[1]
    Ahat = (d * A + (1 - d) / n)
    return Ahat