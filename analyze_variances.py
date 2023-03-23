import argparse
import gensim
import os
import glob
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import wandb

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')


def main(args):
    # Load vectors
    print('[INFO] Loading vectors')
    bbb_vecs = {}
    for decade in range(181, 201):
        bbb_vecs[str(decade) + '0'] = gensim.models.KeyedVectors.load_word2vec_format(
            f"data/COHA/results/decade_embeddings_{args.file_stamp}_{args.run_id}_{decade}.txt",
            binary=False, no_header=True)

    bbb_sds = gensim.models.KeyedVectors.load_word2vec_format(
        f"data/COHA/results/dev_vectors_{args.file_stamp}_{args.run_id}.txt",
        binary=False, no_header=True)

    # Load frequencies
    freq = pd.read_csv(args.freq_path)
    # * Drop nan
    freq = freq.loc[~freq['w'].isna()]
    # * Make complete frequency table (i.e., fill zeros)
    freq_full = pd.DataFrame({'w': bbb_sds.key_to_index.keys()}).merge(
        pd.DataFrame({'decade': sorted(freq['decade'].unique())}), how='cross')
    freq_full = freq_full.merge(freq, on=['w', 'decade'], how='left', validate='one_to_one')
    # * Fill zeros
    freq_full.fillna(0, inplace=True)

    # Summarize frequencies and standard deviations
    summary_df = pd.DataFrame()
    summary_df['w'] = bbb_sds.key_to_index.keys()
    # * Mean standard deviation
    summary_df['mean_sd'] = summary_df['w'].apply(lambda w: np.mean(bbb_sds[bbb_sds.key_to_index[w]]))
    # * Median standard deviation
    summary_df['median_sd'] = summary_df['w'].apply(lambda w: np.median(bbb_sds[bbb_sds.key_to_index[w]]))
    # * Mean, median word frequency across decades
    freq_full_sum = freq_full.groupby('w')['#w'].agg(['mean', 'median']).reset_index()
    freq_full_sum.rename(columns={'mean': 'mean_freq', 'median': 'median_freq'}, inplace=True)
    summary_df = summary_df.merge(freq_full_sum, how='left', validate='one_to_one')
    summary_df['log_median_freq'] = np.log(summary_df['median_freq'])
    summary_df['log_mean_freq'] = np.log(summary_df['mean_freq'])

    # Correlations between standard deviation and word variances
    plt.clf()
    ax = sns.scatterplot(
        summary_df, x="log_median_freq", y='median_sd')
    #ax.set_ylim(0, 0.6)
    ax.set(xlabel='Median frequency (log)', ylabel='Median standard deviation')
    ax.figure.savefig(os.path.join(args.output_dir,  "median_freq_sd.png"))

    plt.clf()
    ax = sns.scatterplot(
        summary_df, x="log_mean_freq", y='mean_sd')
    #ax.set_ylim(0, 0.6)
    ax.set(xlabel='Mean frequency (log) across decades', ylabel='Mean standard deviation')
    ax.figure.savefig(os.path.join(args.output_dir,  "mean_freq_sd.png"))

    # Add data to W&B
    wandb.init(
        project='bbb-uncertainty',
        id=args.run_id,
        resume='allow'
    )

    summary_table = wandb.Table(dataframe=summary_df)

    wandb.log({'median_sd': wandb.plot.scatter(
        summary_table, 'log_median_freq', 'median_sd',
        title='Median frequency (log) vs Median standard deviation')})

    wandb.log({'mean_sd': wandb.plot.scatter(
        summary_table, 'log_mean_freq', 'mean_sd',
        title='Mean frequency (log) vs Mean standard deviation')})

    #table_artifact = wandb.Artifact("variance_artifact", type="dataset")
    #table_artifact.add(summary_table, "variance_table")
    #wandb.run.log_artifact(table_artifact)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-freq_path", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-file_stamp", type=str, required=True)
    parser.add_argument("-run_id", type=str, required=True)

    args = parser.parse_args()

    # Paths
    args.output_dir = 'results/variances'
    args.freq_path = '../avaimar-coha-cooccur/results/Frequencies/coha_frequencies.csv'
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
