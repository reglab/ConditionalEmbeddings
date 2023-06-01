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
from analogies import evaluate_word_analogies_multiple


def main(args):
    # Load vectors (note that these can cover different time periods if we're using the UK corpus)
    print('[INFO] Loading vectors')
    histwords = load_coha_HistWords(input_dir=args.histwords_dir, only_nonzero=True)

    if args.run_id == 'HistWords_Benchmark':
        bbb_vecs = histwords
    else:
        if args.eval_vocab == 'HistWords':
            bbb_vecs = load_BBB_nonzero(
                input_dir=args.base_dir / f'data/{args.name}/results', file_stamp=args.file_stamp,
                run_id=args.run_id, only_nonzero=True, match_vectors=histwords)
        elif args.eval_vocab == 'None':
            bbb_vecs = load_BBB_nonzero(
                input_dir=args.base_dir / f'data/{args.name}/results', file_stamp=args.file_stamp,
                run_id=args.run_id, only_nonzero=False, match_vectors=None)

    # Performance tasks
    print('[INFO] Computing analogy scores')
    eval_score = pd.DataFrame()
    for decade in tqdm(bbb_vecs.keys()):
        word_vecs = bbb_vecs[decade]

        # Normalize vectors: note that normalization is not needed (provides same results as this
        # is already performed within the analogy evaluation functions)
        #word_vecs.init_sims(replace=True)

        # Analogy check
        score, sections = word_vecs.evaluate_word_analogies(str(args.eval_dir / 'questions-words.txt'))
        for section_dict in sections:
            if len(section_dict['correct']) + len(section_dict['incorrect']) == 0:
                accuracy = None
            else:
                accuracy = len(section_dict['correct']) / (
                            len(section_dict['correct']) + len(section_dict['incorrect']))
            eval_score = pd.concat([eval_score, pd.DataFrame.from_dict(
                {'task': ['analogy'], 'task_type': ['CHECK'], 'section': [section_dict['section']],
                 'accuracy': [accuracy], 'MRR': [None], 'decade': [decade], 'vectors': ['BBB']})])

        # Analogy tasks
        analogy_tasks = ['3COSADD', '3COSMUL', 'PAIRDIRECTION']
        analogy_thresholds = [1, 5]
        for analogy_task in analogy_tasks:
            for analogy_threshold in analogy_thresholds:
                score, sections = evaluate_word_analogies_multiple(
                    model=word_vecs, analogies=str(args.eval_dir / 'questions-words.txt'),
                    method=analogy_task, restrict_vocab=None, top_threshold=analogy_threshold)
                for section_dict in sections:
                    if len(section_dict['correct']) + len(section_dict['incorrect']) == 0:
                        accuracy = None
                    else:
                        accuracy = len(section_dict['correct']) / (len(section_dict['correct']) + len(section_dict['incorrect']))
                    eval_score = pd.concat([eval_score, pd.DataFrame.from_dict(
                        {'task': ['analogy'], 'task_type': [f'{analogy_task}-{analogy_threshold}'],
                         'section': [section_dict['section']], 'accuracy': [accuracy],
                         'MRR': [section_dict['MeanReciprocalRank']],
                         'decade': [decade], 'vectors': ['BBB']})])

        # Word similarity (Bruni et al 2012 -- used in HistWords)
        pearson, spearman, oov = word_vecs.evaluate_word_pairs(str(args.eval_dir / 'MEN_dataset_natural_form_full.txt'))
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'task_type': ['similarity'], 'section': ['pearson_stat'],
                 'accuracy': [pearson.statistic], 'MRR': [None], 'decade': [decade],  'vectors': ['BBB']})])
        eval_score = pd.concat(
            [eval_score, pd.DataFrame.from_dict(
                {'task': ['Bruni'], 'task_type': ['similarity'], 'section': ['spearman_stat'],
                 'accuracy': [spearman.statistic], 'MRR': [None], 'decade': [decade],  'vectors': ['BBB']})])

    # Viz overall accuracy
    analogy_df = eval_score.loc[eval_score['task'] == 'analogy'].copy()

    # Viz Bruni stat
    bruni_df = eval_score.loc[eval_score['task'] == 'Bruni'].copy()

    # W&B Logging
    api = wandb.Api()
    run = api.run(f"adus/bbb-uncertainty/{args.run_id}")
    wb_analogy = analogy_df.loc[(analogy_df['section'] == 'Total accuracy') & (analogy_df['vectors'] == 'BBB')]
    wb_bruni = bruni_df.loc[(bruni_df['section'] == 'pearson_stat') & (bruni_df['vectors'] == 'BBB')]

    if args.eval_vocab == 'HistWords':
        fname_vocab = ''
    elif args.eval_vocab == 'None':
        fname_vocab = ' (Non-HW)'

    # Analogy stats
    for analogy_task in analogy_tasks:
        for analogy_threshold in analogy_thresholds:
            aname = f"{analogy_task}-{analogy_threshold}"
            wb_analogy_method = wb_analogy.loc[wb_analogy['task_type'] == aname]

            if analogy_task == '3COSADD' and analogy_threshold == 1:
                aname = ''
            else:
                aname = f" ({aname})"
            run.summary[f'Mean analogy accuracy{aname}{fname_vocab}'] = wb_analogy_method['accuracy'].mean()
            run.summary[f'Max analogy accuracy{aname}{fname_vocab}'] = wb_analogy_method['accuracy'].max()

            if analogy_threshold == 1:
                run.summary[f'Mean analogy MRR{aname}{fname_vocab}'] = wb_analogy_method['MRR'].mean()
                run.summary[f'Max analogy MRR{aname}{fname_vocab}'] = wb_analogy_method['MRR'].max()

    # Similarity
    run.summary[f'Mean similarity stat{fname_vocab}'] = wb_bruni['accuracy'].mean()
    run.summary[f'Max similarity stat{fname_vocab}'] = wb_bruni['accuracy'].max()


    print(f"[INFO] Maximum analogy accuracy: {wb_analogy['accuracy'].max()}")
    print(f"[INFO] Maximum similarity accuracy: {wb_bruni['accuracy'].max()}")

    print('Analogy')
    print(wb_analogy)

    print('Similarity')
    print(wb_bruni)

    run.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-results_dir", type=str)
    parser.add_argument("-eval_dir", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-histwords_dir", type=str)
    parser.add_argument("-file_stamp", type=str)
    parser.add_argument("-run_id", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-eval_vocab", type=str, choices=['HistWords', 'None'], default='HistWords')
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
