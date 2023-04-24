# Compare the dot products of the embeddings to the PMI matrix entries

import argparse
import os
from pathlib import Path
from argparse import Namespace
import numpy as np
import pandas as pd

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

    df = pd.concat(
        [pd.DataFrame(dot_matrices['1990'].reshape(-1, )),
        pd.DataFrame(pmi_matrices['1990'].to_numpy().reshape(-1, ))], axis=1)
    df.corr()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-run_id", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-base_dir", type=str, required=False)
    parser.add_argument("-file_stamp", type=str, required=False)

    args = parser.parse_args()

    if args.run_location == 'sherlock':
        args.base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        args.base_dir = Path(__file__).parent
    args.file_stamp = args.name

    main(args)
