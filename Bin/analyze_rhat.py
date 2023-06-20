import os
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')

if __name__ == '__main__':
    
    rhat_group = 'group1'
    output_dir = 'results/Rhat/group1/'

    means = np.load('results/Rhat/group1/rhat_means.npy')
    covar = np.load('results/Rhat/group1/rhat_covar.npy')
    rho = np.load('results/Rhat/group1/rhat_rho.npy')

    bins = np.arange(0, 7, 0.1)
    frq, edges = np.histogram(covar.reshape(1, -1), bins)
    fig, ax = plt.subplots()
    ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
    ax.figure.savefig(os.path.join(output_dir,  f"covar_{rhat_group}.png"))

    bins = np.arange(0, 7, 0.1)
    frq, edges = np.histogram(means.reshape(1, -1), bins)
    fig, ax = plt.subplots()
    ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
    ax.figure.savefig(os.path.join(output_dir,  f"means_{rhat_group}.png"))

    bins = np.arange(0, 7, 0.1)
    frq, edges = np.histogram(rho.reshape(1, -1), bins)
    fig, ax = plt.subplots()
    ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
    ax.figure.savefig(os.path.join(output_dir,  f"rho_{rhat_group}.png"))

