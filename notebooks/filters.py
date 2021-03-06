import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

import os
import sys

# local imports
import pore_c_utils as pcu


def runAllFilters(cells, 
                  assembly, 
                  adjacent=True,
                  selfLoop=True,
                  distal=False,
                  ligationProduct=True,
                  support=True,
                  chromosome=None, 
                  verbose=True):
    """A function to filter all cells 
    
    args:
        : cells (dict of pd.DataFrame): dictionary of contact tables
        : assembly (pd.DataFrame): assembly information
        : adjacent (bool): if True, retain only adjacent contacts
        and drop clique expansion products
        : selfLoop (bool): if true, drop contacts between identical fragments
        : distal (bool): if true, drop adjacent contacts that are far apart 
        on the read
        : ligationProduct (bool) if true, drop fragments that are in excess of
        the number of possible ligation products for a single nucleus 
        : support (bool): if true, drop contacts that are sufficiently isolated 
        from other contacts
        : chromosome (str): if not `None', select a single chromosome
        : verbose (bool): controls print() output behavior
    
    returns:
        : results (dict of pd.DataFrame): dictionary of contact tables post filtering
    """
    results = {}
    
    for runId, df in cells.items():
       
        # work on pandas copy (requires more memory)
        cf = cells[runId].copy()
        
        # cleanup
        cf = chromosomalFilter(cf, assembly)
        cf = removeYChrom(cf)
    
        # chromosome filtering
        if chromosome is not None:
            cf = filterChomosome(cf, chromosome)
            
        # merge the assmembly info
        cf = pcu.mergeAssembly(cf, assembly)
        
        if verbose:
            print("------------------------------")
            print(f"{runId=}")
    
        """
        Filtering section: each filter is hardcoded for now
        """
        if adjacent:
            cf = adjacentContactFilter(cf)
            
        if selfLoop:
            cf = selfLoopFilter(cf)
        
        cf = mapQFilter(cf, lowerBound=30, upperBound=250)
        
        if distal:
            cf = distalContactFilter(cf)
            
        cf = closeContactFilter(cf)
        cf = duplicateContactFilter(cf, retain=2)
        
        if ligationProduct:
            cf = ligationProductFilter(cf, nProducts=4, verbose=False)
            
        if support:
            cf = establishContactSupport(cf, 
                                         radiusThreshold=1000000, 
                                         nContacts=3, 
                                         readSupport=False,
                                         method=2)
        
            cf = supportedContactFilter(cf, readSupport=False)
        
        if verbose:
            print(f"{cf.shape=}")
            print("-- -- --")
            res = pcu.getSummary(cf)
            pcu.printSummary(res)
            print()
            
        results[runId] = cf
        
    return results


def establishContactSupport(df, radiusThreshold, nContacts, readSupport=False, nReads=2, method=1, verbose=False):
    """A procedure to establish the number of contactw within a eucluiden 
    distance (in base-pair) of each contact. Isolated contacts will not be supported
    
    args:
        : df (pd.DataFrame): the contact table
        : radiusThreshold (int): radius, in bp, of supporting zone
        : nContacts (int): the numner of contact in the supporting zone required
        : readSupport (bool): if True, the neighbors are required to be on different reads
        WANRING: this flag radically slows the compute time
        : nReads (int): if bool flag above is true, how many reads are enough?  
        : method (int): may be 1 or 2. If 1, contacts require ANY support. if 2, 
        contacts require FULL support. See note.
        : verbose (bool): if true, print value counts of `contact_has_support` flag
    
    returns:
        : df (pd.DataFrame): the contact table adding a column: `contact_has_support`
    """
    
    nbrs = NearestNeighbors(n_neighbors=nContacts,
                            p=2, # euclidean distance
                            algorithm='kd_tree').fit(df[['align1_absolute_midpoint', 
                                                         'align2_absolute_midpoint']])
    
    distances, indices = nbrs.kneighbors(df[['align1_absolute_midpoint', 
                                             'align2_absolute_midpoint']])
    
    if readSupport:
        readSupportList = []
        for i in range(indices.shape[0]):
            idx = indices[i]
            neighborReads = df.iloc[idx]['read_name'].nunique()
            if neighborReads >= nReads:
                readSupportList.append(1)
            else:
                readSupportList.append(0)
        df['read_support'] = readSupportList
    
    withinDistance = np.where(distances < radiusThreshold, 1, 0)
    rowSums = withinDistance.sum(axis=1)
    
    """
    Two ideas here:
        (1) if the contact has no support (rowSum == 0) then discard
        (2) if the contact has full support (rowSum == nContacts)
    """
    if method == 1:
        # idea 1
        isSupported = np.where(rowSums > 1, 1, 0)
    elif method == 2:    
        # idea 2
        isSupported = np.where(rowSums >= nContacts, 1, 0)
    else:
        raise ValueError('Method must be 1 or 2.')
    
    df['contact_has_support'] = isSupported
    
    
    if verbose:
        print(f"{df['contact_has_support'].value_counts()=}")
    
    return df


def supportedContactFilter(df, readSupport=False):
    """A function to filter to out contacts that don't meet the criteria for support 
    established by the function above
    
    args:
        : df (pd.DataFrame): the contact table
        : readSupport (bool): if True, the neighbors are required to be on different reads
    
    returns:
        : df (pd.DataFrame): the contact table after filtering
    """
    if readSupport:
        mask = (df['contact_has_support'] == 1) & (df['read_support'] == 1)
    else:
        mask = (df['contact_has_support'] == 1)
    df = df[mask].reset_index(drop=True)
    return df


def adjacentContactFilter(df):
    """A filter to drop contacts that are products of clique expansion.
    This assumes that only adjacent alignments on the read are considered
    contacting.
    
    args:
        : df (pd.DataFrame): the contact table
    
    returns:
        : df (pd.DataFrame): the contact table after filtering
    """
    mask = (df['contact_is_direct'] == True)
    df = df[mask].reset_index(drop=True)
    return df


def selfLoopFilter(df):
    """A filter to drop contacts between the same fragment at reda-level
    
    args:
        : df (pd.DataFrame): the contact table
    
    returns:
        : df (pd.DataFrame): the contact table after filtering
    """
    mask = (df['align1_fragment_id'] == df['align2_fragment_id']) 
    df = df[~mask].reset_index(drop=True)
    return df


def mapQFilter(df, lowerBound, upperBound):
    """A filter to bandpass on mapping quality 
    
    args:
        : df (pd.DataFrame): the contact table
        : lowerBound (int): a lower bound on mapq, suggested 30
        : upperBound (int): an upper bound, suggested 250
    
    returns:
        : df (pd.DataFrame): the contact table after filtering
    """
    mask = (df['align1_mapping_quality'] >= lowerBound) & (df['align2_mapping_quality'] >= lowerBound)
    df = df[mask].reset_index(drop=True)
    
    mask = (df['align1_mapping_quality'] <= upperBound) & (df['align2_mapping_quality'] <= upperBound)
    df = df[mask].reset_index(drop=True)
    
    return df


def distalContactFilter(df, read_distance=1000):
    """A filter to remove direct contacts that are separated by 
    long unmapped sequences on the read.
    
    NOTE: this filter is not applied to clique expansion products

    args:
        : df (pd.DataFrame): the contact table
        : read_distance (int): minimum base pair distance between
        direct contacts. 
    
    returns:
        : df (pd.DataFrame): the contact table after filtering
    """
    mask = (df['contact_is_direct'] == True)
    
    directContact = df[mask].reset_index(drop=True)
    expandedContact = df[~mask].reset_index(drop=True)
    
    mask = (np.abs(directContact['contact_read_distance']) <= read_distance) 
    directContact = directContact[mask].reset_index(drop=True)
    
    df = pd.concat([directContact, expandedContact])
    
    return df


def closeContactFilter(df, genome_distance=1000):
    """A filter to remove cis contacts within a small distance on
    the reference. Assumed to be amplicification bias.
    
    args:
        : df (pd.DataFrame): the contact table
        : genome_distance (int): maximum base pair distance between
        aligned fragments. 
    
    returns:
        : df (pd.DataFrame): the contact table after filtering 
    """
    cisContacts = df[df['contact_is_cis'] == True]
    transContacts = df[df['contact_is_cis'] == False]
    
    mask = (np.abs(cisContacts['contact_genome_distance']) >= genome_distance)
    cisContacts = cisContacts[mask].reset_index(drop=True)

    df = pd.concat([cisContacts, transContacts], ignore_index=True)
    return df


def duplicateContactFilter(df, retain=1):
    """A filter to retain only n copies of a given contact. 
    Note that `retain = 1` will filter to unique conacts only. 
    Contacts are retained by mean mapping quality, such that the 
    highest n mean mapping quality contacts are preserved.
    
    args:
        : df (pd.DataFrame): the contact table
        : retain (int): number of replicates to preserve
    
    returns:
        : df (pd.DataFrame): the contact table after filtering 
    """
    df['mean_mapping_quality'] = (df['align1_mapping_quality'] + df['align2_mapping_quality']) / 2
    
    df = df.sort_values(by=['align1_fragment_id', 
                            'align2_fragment_id', 
                            'mean_mapping_quality'], ascending=False)
    
    df['contact_count'] = df.groupby(["align1_fragment_id", "align2_fragment_id"])["read_name"].transform("cumcount")
    df['contact_count'] = df['contact_count'] + 1

    mask = (df['contact_count'] <= retain)
    df = df[mask].reset_index(drop=True)
    return df


def getFragmentCounts(df):
    """A helper function to return fragment counts
    from both sides of a contact """
    allFragments = df['align1_fragment_id'].tolist() + df['align2_fragment_id'].tolist()
    return Counter(allFragments)
    

def ligationProductFilter(df, nProducts=4, nReadReplicates=4, verbose=True):
    """A filter to remove excess copies of an aligned sequence. A single
    cell can have at most 4 copies of any genomic sequence (maternal, paternal)
    2x if the cell is about to divide. Two step filter:
        (1) separate contacts with fewer than `nProducts` and more than
        `nProducts`. Those with more than `nProducts` are considered amplification 
        bias.
        
        (2) For sequences with amplificiation bias, retain only contacts
        between sequences that appear on fewer than `nReadReplicates`. 
        Sequences found on more reads will be filtered out.
    
    args:
        : df (pd.DataFrame): the contact table
        : nProducts (int): number of products for the initial filter
        : nReadReplicates (int): number of reads an amplified sequence
        may be found on
    
    returns:
        : df (pd.DataFrame): the contact table after filtering 
    """
    # get fragment counts
    countTranslation = dict(getFragmentCounts(df))

    df['align1_fragcount'] = df['align1_fragment_id'].map(countTranslation)
    df['align2_fragcount'] = df['align2_fragment_id'].map(countTranslation)

    mask = (df['align1_fragcount'] <= nProducts) & (df['align2_fragcount'] <= nProducts)

    # divide into contacts with less than n replicates and more than n replicates
    lowFrequencyProducts = df[mask].reset_index(drop=True)
    amplificationBiasProducts = df[~mask].reset_index(drop=True)
    
    # count the number of unique reads per fragment
    readsPerFragment1 = amplificationBiasProducts[['read_name', 'align1_fragment_id']]
    readsPerFragment2 = amplificationBiasProducts[['read_name', 'align2_fragment_id']]
    
    readsPerFragment1.columns = ['read_name', 'fragment_id']
    readsPerFragment2.columns = ['read_name', 'fragment_id']

    readsPerFragment = pd.concat([readsPerFragment1, readsPerFragment2], ignore_index=True)
    
    if verbose:
        print(f"{lowFrequencyProducts.shape=} ({len(lowFrequencyProducts)/len(df):.3f})")
        print(f"{amplificationBiasProducts.shape=} ({len(amplificationBiasProducts)/len(df):.3f})")

        print(f"{readsPerFragment1.shape=}")
        print(f"{readsPerFragment2.shape=}")
        print(f"{readsPerFragment.shape=}")
    
    readsPerFragment['n_reads'] = readsPerFragment.groupby('fragment_id')["read_name"].transform('nunique')
    readsPerFragmentMappable = pd.Series(readsPerFragment['n_reads'].values, index=readsPerFragment['fragment_id']).to_dict()
    
    # map the number of reads an individual fragment appears in
    amplificationBiasProducts['align1_n_reads'] = amplificationBiasProducts['align1_fragment_id'].map(readsPerFragmentMappable)
    amplificationBiasProducts['align2_n_reads'] = amplificationBiasProducts['align2_fragment_id'].map(readsPerFragmentMappable)
    
    # throw away all contacts which appear in more than n reads 
    mask = (amplificationBiasProducts['align1_n_reads'] <= nReadReplicates) & (amplificationBiasProducts['align2_n_reads'] <= nReadReplicates)
    
    toKeep = amplificationBiasProducts[mask].reset_index(drop=True)
    trash = amplificationBiasProducts[~mask].reset_index(drop=True)

    if verbose:
            print(f"{toKeep.shape=}")
            print(f"{trash.shape=}")
    
    df = pd.concat([lowFrequencyProducts, toKeep])
    
    return df


def chromosomalFilter(df, assembly):
    """A filter to remove non-chromosomal genomic regions
    
    args:
        : df (pd.DataFrame): the contact table
        : assembly (pd.DataFrame): assembly information
    
    returns:
        : df (pd.DataFrame): the contact table after filtering 
    """
    allChromosomes = assembly['RefSeq accession'].str.strip().to_list()
    
    mask = (df['align1_chrom'].isin(allChromosomes)) & (df['align2_chrom'].isin(allChromosomes))
    df = df[mask].reset_index()
    return df


def removeYChrom(df):
    """A filter to remove Y chromosome artefacts
    
    args:
        : df (pd.DataFrame): the contact table
    
    returns:
        : df (pd.DataFrame): the contact table after filtering 
    """
    mask = (df['align1_chrom'] == 'NC_000087.8') | (df['align2_chrom'] == 'NC_000087.8')
    df = df[~mask].reset_index(drop=True)
    return df


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