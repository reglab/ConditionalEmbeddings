"""
Logs HistWords performance on W&B
"""

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


def load_coha_HistWords(input_dir, only_nonzero):
    vectors_list = glob.glob(f'{input_dir}/*vectors.txt')
    vectors = {}
    for file_name in vectors_list:
        file_decade = file_name.split(os.path.sep)[-1][:4]

        if only_nonzero:
            temp_file_name = 'vectors.txt'
            with open(temp_file_name, 'w') as wf:
                with open(file_name, 'r') as rf:
                    for line in rf:
                        w, vec = line.split(' ', maxsplit=1)
                        npvec = np.fromstring(vec, sep=' ')
                        if np.linalg.norm(npvec) > 1e-6:
                            wf.write(f"{w} {vec}")
            file_name = temp_file_name

        vectors[file_decade] = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=False, no_header=True)

        if only_nonzero:
            os.remove(temp_file_name)

    return vectors


def main(args):
    # Load vectors
    print('[INFO] Loading vectors')
    histwords = load_coha_HistWords(input_dir=args.histwords_dir, only_nonzero=True)

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
                 'decade': int(decade), 'negative': 5, 'vectors': 'HistWords'})])

        pearson, spearman, oov = word_vecs.evaluate_word_pairs(args.eval_dir / 'wordsim_similarity_goldstandard.txt')
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['similarity'], 'section': ['pearson_stat'], 'accuracy': [pearson.statistic],
                 'decade': int(decade), 'negative': 5, 'vectors': 'HistWords'})])
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['similarity'], 'section': ['spearman_stat'], 'accuracy': [spearman.statistic],
                 'decade': int(decade), 'negative': 5, 'vectors': 'HistWords'})])

        # Word similarity (Bruni et al 2012 -- used in HistWords)
        pearson, spearman, oov = word_vecs.evaluate_word_pairs(args.eval_dir / 'MEN_dataset_natural_form_full.txt')
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'section': ['pearson_stat'], 'accuracy': [pearson.statistic],
                 'decade': int(decade), 'negative': 5, 'vectors': 'HistWords'})])
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'section': ['spearman_stat'], 'accuracy': [spearman.statistic],
                 'decade': int(decade), 'negative': 5, 'vectors': 'HistWords'})])

    # Task dfs
    analogy_df = eval_score.loc[eval_score['task'] == 'analogy'].copy()
    bruni_df = eval_score.loc[eval_score['task'] == 'Bruni'].copy()

    # W&B Logging for the HistWords performance (for comparison)
    api = wandb.Api()
    try:
        run = api.run(f'adus/bbb-uncertainty/HistWords_Benchmark')
    except wandb.errors.CommError:
        wandb.init(
            project='bbb-uncertainty',
            name='HistWords_Benchmark',
            id='HistWords_Benchmark')

        run = api.run(f'adus/bbb-uncertainty/HistWords_Benchmark')
        wb_hw_analogy = analogy_df.loc[(analogy_df['section'] == 'Total accuracy') & (analogy_df['vectors'] == 'HistWords')]
        wb_hw_bruni = bruni_df.loc[(bruni_df['section'] == 'pearson_stat') & (bruni_df['vectors'] == 'HistWords')]

        run.summary['Mean analogy accuracy'] = wb_hw_analogy['accuracy'].mean()
        run.summary['Mean similarity stat'] = wb_hw_bruni['accuracy'].mean()

        run.summary['Max analogy accuracy'] = wb_hw_analogy['accuracy'].max()
        run.summary['Max similarity stat'] = wb_hw_bruni['accuracy'].max()
        run.update()
        run.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-results_dir", type=str)
    parser.add_argument("-eval_dir", type=str)
    parser.add_argument("-histwords_dir", type=str)

    args = parser.parse_args()

    # Paths
    args.results_dir = Path(__file__).parent / "results"
    args.eval_dir = Path(__file__).parent / "data" / "COHA" / "evaluation"

    args.histwords_dir = '../Replication-Garg-2018/data/coha-word'

    main(args)
