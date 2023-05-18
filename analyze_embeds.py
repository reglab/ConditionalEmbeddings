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
from bias_utils import load_BBB_nonzero

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')


def main(args):
    # Load vectors
    print('[INFO] Loading vectors')
    bbb_vecs = load_BBB_nonzero(
        input_dir=os.path.join(args.base_dir, f'data/{args.name}/results'), file_stamp=args.file_stamp,
        run_id=args.run_id, only_nonzero=False, match_vectors=None)

    bbb_sds = gensim.models.KeyedVectors.load_word2vec_format(
        args.base_dir / f"data/{args.name}/results/dev_vectors_{args.file_stamp}_{args.run_id}.txt",
        binary=False, no_header=True)

    # 1. Heatmap of embedding values =============================
    nonzero_df = pd.DataFrame()
    for decade_str in bbb_vecs.keys():
        w = bbb_vecs[decade_str].vectors
        #nonzero = nonzero = np.abs(w) > 1e-6
        nonzero = w
        nonzero = pd.DataFrame(nonzero)
        nonzero['decade'] = int(decade_str)
        nonzero['word'] = bbb_vecs[decade_str].key_to_index.keys()
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

    # 2. Global vectors (in_embed) =============================
    model = load_model(
        args.base_dir / f"data/{args.name}/results/model_best_{args.file_stamp}_{args.run_id}.pth.tar",
        args.base_dir / f"data/{args.name}/processed/vocab{args.file_stamp}_freq.npy",
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

    # 3. Sds =============================
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

    # 4. Decade vectors =============================
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
    parser.add_argument("-file_stamp", type=str)
    parser.add_argument("-run_id", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-base_dir", type=str, required=False)

    args = parser.parse_args()

    # Paths
    if args.run_location == 'sherlock':
        args.base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        args.base_dir = Path(__file__).parent

    args.output_dir = args.base_dir / 'results/embeddings'
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
