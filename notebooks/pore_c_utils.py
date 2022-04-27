import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys


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

            
def filterChomosome(df, refseq):
    """A function to filter to a single chromosome
    
    args:
        : df (pd.DataFrame): the contact table
        : reseq (str): Refseq accession string for the chromosome
        
    returns:
        : df (pd.DataFrame): after chromosomal filtering
    """
    mask = (df['align1_chrom'] == refseq) & (df['align2_chrom'] == refseq)
    df = df[mask].reset_index(drop=True)
    return df


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


