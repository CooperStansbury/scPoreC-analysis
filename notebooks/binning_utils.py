import pandas as pd
import numpy as np

import os
import sys


def getBins(length, scale):
    """ Return bin IDS for a given range of bp """
    n = int(np.ceil(length / scale))    
    bins = [scale]
    for i in range(n-1):
        next_bin = bins[i] + scale
        bins.append(next_bin)
    return np.array(bins)


def returnBin(position, bins):
    idx = np.min(np.argwhere(bins > position))
    return idx - 1 


def wholeGenomeBinData(df, bins, label):
    df[f'align1_{label}Bin'] = df['align1_absolute_position'].apply(lambda x: returnBin(x, bins))
    df[f'align2_{label}Bin'] = df['align2_absolute_position'].apply(lambda x: returnBin(x, bins))
    return df


def genomeWideBins(df, assembly, scale=1000000):
    totalLength = assembly['chromEnd'].max()
    bins = getBins(totalLength, scale)
    
    df = wholeGenomeBinData(df, bins, label='genome')
    return df, bins



def chromosomeBinData(df, bins, label1, label2):
    df[label1] = df['align1_fragment_start'].apply(lambda x: returnBin(x, bins))
    df[label2] = df['align2_fragment_start'].apply(lambda x: returnBin(x, bins))
    return df