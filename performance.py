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

from bias_utils import *


def main(args):
    # Load vectors (note that these can cover different time periods if we're using the UK corpus)
    print('[INFO] Loading vectors')
    histwords = load_coha_HistWords(input_dir=args.histwords_dir, only_nonzero=True)
    bbb_vecs = load_BBB_nonzero(
        input_dir=args.base_dir / f'data/{args.name}/results', file_stamp=args.file_stamp,
        run_id=args.run_id, only_nonzero=True, match_vectors=histwords)

    # Analogy task
    print('[INFO] Computing analogy scores')
    eval_score = pd.DataFrame()
    for decade in tqdm(bbb_vecs.keys()):
        word_vecs = bbb_vecs[decade]
        score, sections = word_vecs.evaluate_word_analogies(str(args.eval_dir / 'questions-words.txt'))
        for section_dict in sections:
            if len(section_dict['correct']) + len(section_dict['incorrect']) == 0:
                accuracy = None
            else:
                accuracy = len(section_dict['correct']) / (len(section_dict['correct']) + len(section_dict['incorrect']))
            eval_score = pd.concat([eval_score, pd.DataFrame.from_dict(
                {'task': ['analogy'], 'section': [section_dict['section']], 'accuracy': [accuracy],
                 'decade': [decade], 'vectors': ['BBB']})])

        # Word similarity (Bruni et al 2012 -- used in HistWords)
        pearson, spearman, oov = word_vecs.evaluate_word_pairs(str(args.eval_dir / 'MEN_dataset_natural_form_full.txt'))
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'section': ['pearson_stat'], 'accuracy': [pearson.statistic],
                 'decade': [decade],  'vectors': ['BBB']})])
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'section': ['spearman_stat'], 'accuracy': [spearman.statistic],
                 'decade': [decade],  'vectors': ['BBB']})])

    # HistWords performance
    for decade, word_vecs in tqdm(histwords.items()):
        score, sections = word_vecs.evaluate_word_analogies(args.eval_dir / 'questions-words.txt')

        for section_dict in sections:
            if len(section_dict['correct']) + len(section_dict['incorrect']) == 0:
                accuracy = None
            else:
                accuracy = len(section_dict['correct']) / (len(section_dict['correct']) + len(section_dict['incorrect']))
            eval_score = pd.concat([eval_score, pd.DataFrame.from_dict(
                {'task': ['analogy'], 'section': [section_dict['section']], 'accuracy': [accuracy],
                 'decade': int(decade), 'vectors': 'HistWords'})])

        pearson, spearman, oov = word_vecs.evaluate_word_pairs(args.eval_dir / 'wordsim_similarity_goldstandard.txt')
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['similarity'], 'section': ['pearson_stat'], 'accuracy': [pearson.statistic],
                 'decade': int(decade),  'vectors': 'HistWords'})])
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['similarity'], 'section': ['spearman_stat'], 'accuracy': [spearman.statistic],
                 'decade': int(decade),  'vectors': 'HistWords'})])

        # Word similarity (Bruni et al 2012 -- used in HistWords)
        pearson, spearman, oov = word_vecs.evaluate_word_pairs(args.eval_dir / 'MEN_dataset_natural_form_full.txt')
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'section': ['pearson_stat'], 'accuracy': [pearson.statistic],
                 'decade': int(decade), 'vectors': 'HistWords'})])
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'section': ['spearman_stat'], 'accuracy': [spearman.statistic],
                 'decade': int(decade), 'vectors': 'HistWords'})])

    # Viz overall accuracy
    analogy_df = eval_score.loc[eval_score['task'] == 'analogy'].copy()

    plt.clf()
    ax = sns.scatterplot(
        analogy_df.loc[(analogy_df['section'] == 'Total accuracy')],
        x="vectors", y='accuracy', hue='decade', legend=False)
    ax.set_ylim(0, 0.6)
    ax.set(xlabel='Vectors', ylabel='Accuracy')
    ax.figure.savefig(args.output_dir / f"analogy_{args.file_stamp}_{args.run_id}.png")


    # Viz Bruni stat
    bruni_df = eval_score.loc[eval_score['task'] == 'Bruni'].copy()

    plt.clf()
    ax = sns.scatterplot(
        bruni_df.loc[(bruni_df['section'] == 'pearson_stat')],
        x="vectors", y='accuracy', hue='decade', legend=False)
    ax.set_ylim(0, 0.8)
    ax.set(xlabel='Vectors', ylabel='Pearson statistic')
    ax.figure.savefig(args.output_dir / f"bruni_{args.file_stamp}_{args.run_id}.png")

    # W&B Logging
    api = wandb.Api()
    run = api.run(f"adus/bbb-uncertainty/{args.run_id}")
    wb_analogy = analogy_df.loc[(analogy_df['section'] == 'Total accuracy') & (analogy_df['vectors'] == 'BBB')]
    wb_bruni = bruni_df.loc[(bruni_df['section'] == 'pearson_stat') & (bruni_df['vectors'] == 'BBB')]

    run.summary['Mean analogy accuracy'] = wb_analogy['accuracy'].mean()
    run.summary['Mean similarity stat'] = wb_bruni['accuracy'].mean()

    run.summary['Max analogy accuracy'] = wb_analogy['accuracy'].max()
    run.summary['Max similarity stat'] = wb_bruni['accuracy'].max()

    run.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-results_dir", type=str)
    parser.add_argument("-eval_dir", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-histwords_dir", type=str)
    parser.add_argument("-file_stamp", type=str, default="coha")
    parser.add_argument("-run_id", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-base_dir", type=str)

    args = parser.parse_args()

    # Paths
    if args.run_location == 'sherlock':
        args.base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
        args.histwords_dir = args.base_dir / 'data/HistWords/coha-word'
    elif args.run_location == 'local':
        args.base_dir = Path(__file__).parent
        args.histwords_dir = '../Replication-Garg-2018/data/coha-word'

    args.results_dir = args.base_dir / "results"
    args.eval_dir = args.base_dir / "data" / "COHA" / "evaluation"
    args.output_dir = args.results_dir / "Performance"
    args.output_dir.mkdir(exist_ok=True)

    main(args)
