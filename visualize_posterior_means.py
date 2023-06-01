import argparse
import json
import gensim
import os
import glob
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import wandb
from sklearn.decomposition import PCA
from sklearn import random_projection

from bias_utils import load_BBB_nonzero, load_coha_HistWords

matplotlib.use('pdf')


def color_word(i, model, wlists):
    word_i = model.index_to_key[i]

    for wname, wlist in wlists.items():
        if word_i in wlist:
            return wname
    return 'Word'


def main(args):
    # Load vectors
    print('[INFO] Loading vectors')
    if args.run_id == 'HistWords_Benchmark':
        bbb_vecs = load_coha_HistWords(input_dir=args.histwords_dir, only_nonzero=True)
    else:
        bbb_vecs = load_BBB_nonzero(
            input_dir=os.path.join(args.base_dir, f'data/{args.name}/results'), file_stamp=args.file_stamp,
            run_id=args.run_id, only_nonzero=False, match_vectors=None)

    # Load word lists
    with open(args.word_list, 'r') as file:
        word_list_all = json.load(file)

    for projection in ['pca', 'jl']:
        # Projections for each decade ---------------------------------
        df = pd.DataFrame()
        for decade, model in tqdm(bbb_vecs.items()):
            X = model.vectors
            if projection == 'pca':
                fit = PCA(n_components=2)
            elif projection == 'jl':
                fit = random_projection.GaussianRandomProjection(n_components=2)
            else:
                raise Exception('[ERROR] Check projection.')

            z = fit.fit_transform(X)

            decade_df = pd.DataFrame()
            decade_df["comp-1"] = z[:, 0]
            decade_df["comp-2"] = z[:, 1]
            decade_df['y'] = decade_df.apply(lambda row: color_word(row.name, model, word_list_all), axis=1)
            decade_df['decade'] = decade
            decade_df['Word'] = decade_df.apply(lambda row: model.index_to_key[row.name], axis=1)

            df = pd.concat([df, decade_df])

        # Plot drift in time
        decade_list = sorted(df['decade'].unique())
        fig, ax_list = plt.subplots(nrows=5, ncols=4, figsize=(20, 10))
        ax_list = [ax for sl in ax_list.tolist() for ax in sl]
        for decade, ax in zip(decade_list, ax_list):
            decade_df = df.loc[df['decade'] == decade].copy()

            ax.title.set_text(decade)
            ax.axis('off')

            # * Normal words
            ax_base = sns.scatterplot(ax=ax, x="comp-1", y="comp-2", color='grey', label='Word',
                                      data=decade_df,
                                      # decade_df.loc[decade_df['group'] == 'Word']
                                      legend=False)
            ax.set(xticklabels=[], yticklabels=[], xlabel='', ylabel='')
            ax.tick_params(bottom=False, left=False)
            # * Surname points
            # ax2 = sns.scatterplot(ax=ax, x="comp-1", y="comp-2", color='#a4c5de', label='{} surnames'.format(group),
            #    data=decade_df.loc[decade_df['y'] == group], legend=False)
            # * Otherization words
            # ax3 = sns.scatterplot(ax=ax, x="comp-1", y="comp-2", color='salmon', label='Otherization',
            #    data=decade_df.loc[decade_df['group'] == 'Otherization'], legend=False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        fig.savefig(os.path.join(args.output_dir, f'{projection}_temporal_{args.run_id}.png'), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-file_stamp", type=str)
    parser.add_argument("-run_id", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-base_dir", type=str, required=False)
    parser.add_argument("-word_list", type=str, required=False)
    parser.add_argument("-histwords_dir", type=str)

    args = parser.parse_args()

    # Paths
    if args.run_location == 'sherlock':
        args.base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
        args.histwords_dir = args.base_dir / 'data/HistWords/coha-word'
    elif args.run_location == 'local':
        args.base_dir = Path(__file__).parent
        args.histwords_dir = '../Replication-Garg-2018/data/coha-word'

    args.word_list = args.base_dir / 'word_lists/word_lists_all.json'
    args.output_dir = args.base_dir / 'results/visualize_pms'
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
