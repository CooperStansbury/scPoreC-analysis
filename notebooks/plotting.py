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