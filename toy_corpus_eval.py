# Compare the dot products of the embeddings to the PMI matrix entries

import argparse
import os
from pathlib import Path
from argparse import Namespace
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')
import seaborn as sns

from bias_utils import load_BBB_nonzero


def main(args):
    bbb_vecs = load_BBB_nonzero(
        input_dir=os.path.join(args.base_dir, f'data/{args.name}/results'), file_stamp=args.file_stamp,
        run_id=args.run_id, only_nonzero=False, match_vectors=None)

    # Generate PMI matrix
    # python makecooccur.py --data ConditionalEmbeddings/data/ToySun/cooccur --info ConditionalEmbeddings/data/ToySun/info --type word --window-size 2 --out ConditionalEmbeddings/data/ToySun/cooccurs --start 1990 --end 2000 --step 10
    # python PMI_compute.py -vectors None -wlist_dir ConditionalEmbeddings/data/ToySun/results/PMI -bin_dir ConditionalEmbeddings/data/ToySun/cooccurs/word/2 -word_dict_pkl ConditionalEmbeddings/data/ToySun/info/word-dict.pkl -output_dir ConditionalEmbeddings/data/ToySun/results/PMI
    pmi_mat = pd.read_csv(os.path.join(args.base_dir, 'data', args.name, 'results', 'PMI', 'pmi.csv'))

    pmi_matrices = {}
    for decade in pmi_mat['decade'].unique():
        pmi_decade = pmi_mat.loc[pmi_mat['decade'] == decade].copy()
        pmi_decade['PMI(w,k)'] = pmi_decade.apply(
            lambda row: np.log(row['#wc'] * row['D'] / (row['#w'] * row['#c'])), axis=1)

        # Drop the diagonal
        #pmi_decade = pmi_decade.loc[pmi_decade['w_idx'] != pmi_decade['c_idx']]

        pmi_decade = pmi_decade.pivot_table(index=['w_idx'], columns='c_idx', values='PMI(w,k)')
        pmi_matrices[str(decade)] = pmi_decade.copy()

    # Create matrix of dot products
    dot_matrices = {}
    for decade, model in bbb_vecs.items():
        # Normalize
        #model.init_sims(replace=True)
        m = np.dot(model.vectors, model.vectors.T)
        dot_matrices[decade] = m

    # Single decade
    select_decade = '1990'
    df = pd.concat(
        [pd.DataFrame(dot_matrices[select_decade].reshape(-1, )),
        pd.DataFrame(pmi_matrices[select_decade].to_numpy().reshape(-1, ))], axis=1)
    df.corr()

    # Heat map
    a = pd.DataFrame(dot_matrices[select_decade])
    a['i'] = np.arange(a.shape[0])
    a = pd.melt(a, id_vars=['i'], var_name='j', value_name='value')
    a['type'] = 'Dot Product'

    b = pmi_matrices[select_decade].copy()
    b['i'] = np.arange(b.shape[0])
    b = pd.melt(b, id_vars=['i'], var_name='j', value_name='value')
    b['type'] = 'PMI'

    df = pd.concat([a, b])

    # Add word names
    vocab = pmi_mat.groupby(['w_idx', 'w']).size().reset_index()[['w_idx', 'w']]
    vocab = vocab.to_dict('split')['data']
    vocab = {i:w for i, w in vocab}
    df['w_i'] = df['i'].apply(lambda i: vocab[i])
    df['w_j'] = df['j'].apply(lambda j: vocab[j])

    vmin, vmax = df['value'].min(), df['value'].max()
    def facet_heatmap(data, color, **kws):
        data = data.pivot_table(values='value', index='w_i', columns='w_j')
        sns.heatmap(data, cbar=True, vmin=vmin, vmax=vmax, cmap="vlag", center=0)

    g = sns.FacetGrid(df, col='type')
    g.map_dataframe(facet_heatmap)
    g.set_titles(row_template="{row_name}", col_template='{col_name}')
    g.fig.suptitle('')
    g.set_xlabels('')
    g.set_ylabels('')
    g.figure.savefig(os.path.join(args.output_dir, f"pmi_dot_eval.png"), dpi=800)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-run_id", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-base_dir", type=str, required=False)
    parser.add_argument("-output_dir", type=str, required=False)
    parser.add_argument("-file_stamp", type=str, required=False)

    args = parser.parse_args()

    if args.run_location == 'sherlock':
        args.base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        args.base_dir = Path(__file__).parent
    args.file_stamp = args.name
    args.output_dir = os.path.join(args.base_dir, 'data', args.name, 'results')
    main(args)
