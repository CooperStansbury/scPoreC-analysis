import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import transforms

def plotHiCscatter(df, bin1, bin2, nbins, params):
     
    plt.rcParams['figure.dpi'] = params['figdpi']
    plt.rcParams['figure.figsize'] = params['figsize']
    
    plt.scatter(df[bin1], df[bin2], 
                marker='s', 
                s=params['size'], 
                alpha=params['alpha'], 
                c=params['color'])


    plt.scatter(df[bin2], 
                df[bin1], 
                marker='s', 
                s=params['size'], 
                alpha=params['alpha'], 
                c=params['color'])

    plt.xlim(0, nbins)
    plt.ylim(nbins, 0)
    ax = plt.gca() # Get current axes object
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.gca().set_aspect('equal', adjustable='box')
    
    
def genomewidePlot(df, bins, chromBins, label1, label2):
    plt.scatter(df[label1], 
                df[label2], 
                marker='s', 
                s=.25, 
                alpha=0.1, 
                c='darkblue')

    plt.scatter(df[label2], 
                df[label1], 
                marker='s', 
                s=.25, 
                alpha=0.1, 
                c='darkblue')


    plt.xlim(0, len(bins) )
    plt.ylim(len(bins), 0)
    ax = plt.gca() # Get current axes object
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.gca().set_aspect('equal', adjustable='box')

    for k, v in chromBins.items():
        plt.axvline(x=v, lw=1, c='k', ls=':')
        plt.axhline(y=v, lw=1, c='k', ls=':')

    plt.xticks(list(chromBins.values()))
    ax.set_xticklabels(list(chromBins.keys()))

    plt.yticks(list(chromBins.values()))
    ax.set_yticklabels(list(chromBins.keys()))