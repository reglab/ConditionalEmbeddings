import argparse
import gensim
import os
import glob
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

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
    bbb_vecs = {}
    for decade in range(181, 201):
        bbb_vecs[str(decade) + '0'] = gensim.models.KeyedVectors.load_word2vec_format(
            f"data/COHA/results/decade_embeddings_{args.file_stamp}_{decade}.txt", binary=False, no_header=True)

    # Analogy task
    eval_score = pd.DataFrame()
    for decade in tqdm(range(1810, 2001, 10)):
        word_vecs = bbb_vecs[str(decade)]
        score, sections = word_vecs.evaluate_word_analogies(str(args.eval_dir / 'questions-words.txt'))
        for section_dict in sections:
            if len(section_dict['correct']) + len(section_dict['incorrect']) == 0:
                accuracy = None
            else:
                accuracy = len(section_dict['correct']) / (len(section_dict['correct']) + len(section_dict['incorrect']))
            eval_score = pd.concat([eval_score, pd.DataFrame.from_dict(
                {'task': ['analogy'], 'section': [section_dict['section']], 'accuracy': [accuracy],
                 'decade': [decade], 'negative': args.negative,  'vectors': ['BBB']})])

        # Word similarity (Bruni et al 2012 -- used in HistWords)
        pearson, spearman, oov = word_vecs.evaluate_word_pairs(str(args.eval_dir / 'MEN_dataset_natural_form_full.txt'))
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'section': ['pearson_stat'], 'accuracy': [pearson.statistic],
                 'decade': [decade], 'negative': args.negative,  'vectors': ['BBB']})])
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'section': ['spearman_stat'], 'accuracy': [spearman.statistic],
                 'decade': [decade], 'negative': args.negative,  'vectors': ['BBB']})])

    # HistWords performance
    histwords = load_coha_HistWords(input_dir=args.histwords_dir, only_nonzero=True)
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

    # Viz overall accuracy
    analogy_df = eval_score.loc[eval_score['task'] == 'analogy'].copy()

    plt.clf()
    ax = sns.scatterplot(
        analogy_df.loc[(analogy_df['section'] == 'Total accuracy') & (analogy_df['vectors'] == 'BBB')],
        x="negative", y='accuracy', hue='decade')
    ax = sns.scatterplot(
        analogy_df.loc[(analogy_df['section'] == 'Total accuracy') & (analogy_df['vectors'] == 'HistWords')],
        ax=ax, x='negative', y='accuracy', hue='decade'
    )
    ax.set_ylim(0, 0.6)
    ax.set(xlabel='Negative sampling parameter', ylabel='Accuracy')
    ax.figure.savefig(args.output_dir / f"analogy_{args.file_stamp}.png")


    # Viz Bruni stat
    bruni_df = eval_score.loc[eval_score['task'] == 'Bruni'].copy()

    plt.clf()
    ax = sns.scatterplot(
        bruni_df.loc[(bruni_df['section'] == 'pearson_stat') & (bruni_df['vectors'] == 'BBB')],
        x="negative", y='accuracy', hue='decade')
    ax = sns.scatterplot(
        bruni_df.loc[(bruni_df['section'] == 'pearson_stat') & (bruni_df['vectors'] == 'HistWords')],
        ax=ax, x='negative', y='accuracy', hue='decade')
    ax.set_ylim(0, 0.8)
    ax.set(xlabel='Negative sampling parameter', ylabel='Pearson statistic')
    ax.figure.savefig(args.output_dir / f"bruni_{args.file_stamp}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-results_dir", type=str)
    parser.add_argument("-eval_dir", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-histwords_dir", type=str)
    parser.add_argument("-negative", type=int)
    parser.add_argument("-file_stamp", type=str, required=True)

    args = parser.parse_args()

    # Paths
    args.results_dir = Path(__file__).parent / "results"
    args.eval_dir = Path(__file__).parent / "data" / "COHA" / "evaluation"
    args.output_dir = args.results_dir / "Performance"
    args.output_dir.mkdir(exist_ok=True)
    args.negative = 6
    args.histwords_dir = '../Replication-Garg-2018/data/coha-word'

    main(args)
