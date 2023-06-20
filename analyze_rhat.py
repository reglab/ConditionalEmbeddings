import argparse
import os
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')


def main(args):

    # Load Rhat
    means = np.load(os.path.join(args.output_dir, 'rhat_means.npy'))
    covar = np.load(os.path.join(args.output_dir, 'rhat_covar.npy'))
    rho = np.load(os.path.join(args.output_dir, 'rhat_rho.npy'))

    # Vocabulary
    vocab = np.load(str(args.vocab), allow_pickle=True).item()

    # Load frequencies
    freq = pd.read_csv(args.freq_path)
    # * Drop nan
    freq = freq.loc[~freq['w'].isna()]
    # * Make complete frequency table (i.e., fill zeros)
    freq_full = pd.DataFrame({'w': vocab.keys()}).merge(
        pd.DataFrame({'decade': sorted(freq['decade'].unique())}), how='cross')
    freq_full = freq_full.merge(freq, on=['w', 'decade'], how='left', validate='one_to_one')
    # * Fill zeros
    freq_full.fillna(0, inplace=True)

    # Plot Rhat histograms
    param_groups = {'covar': covar, 'means': means, 'rho': rho}
    for param_name, param_array in param_groups.items():
        bins = np.arange(0, 7, 0.1)
        frq, edges = np.histogram(param_array.reshape(1, -1), bins)
        fig, ax = plt.subplots()
        ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
        ax.figure.savefig(os.path.join(args.output_dir,  f"{param_name}.png"))

    # Relate to frequencies
    for param_name in ['means', 'rho']:
        param_array = param_groups[param_name]

        # Get the mean Rhat for a word
        mean_array = np.mean(param_array, axis=1)
        mean_array = pd.DataFrame(mean_array.T)
        mean_array['index'] = np.arange(mean_array.shape[0])
        mean_array.rename(columns={0: 'rhat'}, inplace=True)

        # Get total (global) frequency for a word
        global_freq = freq_full.groupby('w')['#w'].sum().reset_index()
        global_freq['log_#w'] = np.log(global_freq['#w'] + 1)

        # Merge
        global_freq['index'] = global_freq['w'].apply(lambda w: vocab[w])
        global_freq = global_freq.merge(mean_array, how='left', validate='one_to_one')

        # Drop words with zero freq (note that this is driven by slightly different
        # overlap in HW's 50k vocab vs BBB's 50k vocab ~ approx 2,200 words missing
        # frequencies)
        global_freq = global_freq.loc[global_freq['#w'] > 0]

        #global_freq = global_freq.loc[global_freq['rhat'] < 10]

        # Plot
        plt.clf()
        fig, ax = plt.subplots()
        ax = sns.scatterplot(global_freq, ax=ax, x='log_#w', y='rhat')
        ax.axhline(1, ls='--', linewidth=1, color='red')
        ax.set(xlabel='Total frequency (log) across decades', ylabel='Mean rhat')
        ax.figure.savefig(os.path.join(args.output_dir, f"freq_{param_name}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-freq_path", type=str, required=False)
    parser.add_argument("-vocab", type=str, required=False)
    parser.add_argument("-output_dir", type=str, required=False)
    parser.add_argument("-rhat_group", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)

    args = parser.parse_args()

    args.vocab = f"data/{args.name}/processed/vocab_freq.npy"
    args.output_dir = f'results/Rhat/{args.rhat_group}/'
    args.freq_path = '../avaimar-coha-cooccur/results/Frequencies/coha_frequencies.csv'
    os.makedirs(args.output_dir, exist_ok=True)

    if args.name != 'COHA':
        raise Exception('Error: need to add frequency counts for a corpus other than COHA')

    main(args)
