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
from sklearn.metrics.pairwise import cosine_similarity


from model_to_vectors import load_model

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

    # Heatmap of zero locations in embeddings
    nonzero_df = pd.DataFrame()
    for decade in range(1810, 2001, 10):
        w = bbb_vecs[str(decade)].vectors
        #nonzero = nonzero = np.abs(w) > 1e-6
        nonzero = w
        nonzero = pd.DataFrame(nonzero)
        nonzero['decade'] = decade
        nonzero['word'] = bbb_vecs[str(decade)].key_to_index.keys()
        nonzero_df = pd.concat([nonzero_df, nonzero])

    # random subset
    #words = np.random.choice(nonzero['word'], size=100, replace=False)
    #nonzero_df = nonzero_df.loc[nonzero_df['word'].isin(words)]

    # Melt
    embed_df = nonzero_df.copy()
    nonzero_df = pd.melt(nonzero_df, id_vars=['word', 'decade'], var_name='dim', value_name='element')

    # Plot
    def facet_heatmap(data, color, **kws):
        data = data.pivot_table(values='element', index='word', columns='dim')
        sns.heatmap(data, cbar=True)

    #if col == 'Value':
        #plt.figure(figsize=(6, 3), dpi=900)
    g = sns.FacetGrid(nonzero_df, col='decade', col_wrap=5)
    g.map_dataframe(facet_heatmap)
    g.set_titles(row_template="{row_name}", col_template='{col_name}')
    g.fig.suptitle('')
    g.figure.savefig(os.path.join(args.output_dir, f"embeds-{args.run_id}.png"), dpi=800)

    # Cosine similarities
    for decade in embed_df['decade'].unique():
        decade_df = embed_df.loc[embed_df['decade'] == decade].copy()
        decade_df.drop(['word', 'decade'], axis=1, inplace=True)
        cs = cosine_similarity(decade_df)

    # Global vectors (in_embed)
    model = load_model(
        f"data/COHA/results/model_best_{args.file_stamp}_{args.run_id}.pth.tar",
        "data/COHA/COHA_processed/vocabcoha_freq.npy",
    )
    global_emb = model.word_input_embeddings
    emb_df = pd.DataFrame()
    for w, emb in global_emb.items():
        w_df = pd.DataFrame(emb.reshape(1, -1))
        w_df['word'] = w
        emb_df = pd.concat([emb_df, w_df])

    emb_df = pd.melt(emb_df, id_vars=['word'], var_name='dim', value_name='element')
    emb_df['decade'] = 'global'

    def facet_heatmap(data, color, **kws):
        data = data.pivot_table(values='element', index='word', columns='dim')
        sns.heatmap(data, cbar=True)

    g = sns.FacetGrid(emb_df, col='decade', col_wrap=1)
    g.map_dataframe(facet_heatmap)
    g.set_titles(row_template="{row_name}", col_template='{col_name}')
    g.fig.suptitle('')
    g.figure.savefig(os.path.join(args.output_dir, f"global-{args.run_id}.png"), dpi=1600)

    # Sds
    sd_df = pd.DataFrame()
    for w in bbb_sds.key_to_index.keys():
        sd_vec = bbb_sds[bbb_sds.key_to_index[w]]
        w_df = pd.DataFrame(sd_vec.reshape(1, -1))
        w_df['word'] = w
        sd_df = pd.concat([sd_df, w_df])

    sd_df = pd.melt(sd_df, id_vars=['word'], var_name='dim', value_name='element')
    sd_df['decade'] = 'global'

    g = sns.FacetGrid(sd_df, col='decade', col_wrap=1)
    g.map_dataframe(facet_heatmap)
    g.set_titles(row_template="{row_name}", col_template='{col_name}')
    g.fig.suptitle('')
    g.figure.savefig(os.path.join(args.output_dir, f"sds-{args.run_id}.png"), dpi=1600)

    # Plot decade vectors
    decade_emb = model.year_covar

    decade_df = pd.DataFrame()
    for w, emb in decade_emb.items():
        w_df = pd.DataFrame(emb.reshape(1, -1))
        w_df['decade'] = w
        w_df['word'] = w
        decade_df = pd.concat([decade_df, w_df])

    decade_df = pd.melt(decade_df, id_vars=['word', 'decade'], var_name='dim', value_name='element')

    def facet_heatmap(data, color, **kws):
        data = data.pivot_table(values='element', index='word', columns='dim')
        sns.heatmap(data, cbar=True)

    g = sns.FacetGrid(decade_df, col='decade', col_wrap=1)
    g.map_dataframe(facet_heatmap)
    g.set_titles(row_template="{row_name}", col_template='{col_name}')
    g.fig.suptitle('')
    g.figure.savefig(os.path.join(args.output_dir, f"covar-{args.run_id}.png"), dpi=400)

    # Check output vectors


    """
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
    """



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-file_stamp", type=str, default="coha")
    parser.add_argument("-run_id", type=str, required=True)

    args = parser.parse_args()

    # Paths
    args.output_dir = 'results/embeddings'
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
