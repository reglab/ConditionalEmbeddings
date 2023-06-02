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

    if args.run_id in ['HistWords_Benchmark', 'COHA_sk_revamp_50k']:
        raise Exception('Use this script only for BBB vectors.')
    else:
        if args.eval_vocab == 'HistWords':
            bbb_vecs = load_BBB_nonzero(
                input_dir=args.base_dir / f'data/{args.name}/results', file_stamp=args.file_stamp,
                run_id=args.run_id, only_nonzero=True, match_vectors=histwords)
        elif args.eval_vocab == 'None':
            bbb_vecs = load_BBB_nonzero(
                input_dir=args.base_dir / f'data/{args.name}/results', file_stamp=args.file_stamp,
                run_id=args.run_id, only_nonzero=False, match_vectors=None)

    # Load standard deviations
    bbb_sds = gensim.models.KeyedVectors.load_word2vec_format(
        args.base_dir / f"data/{args.name}/results/dev_vectors_{args.file_stamp}_{args.run_id}.txt",
        binary=False, no_header=True)

    # Performance tasks
    print('[INFO] Computing analogy scores')
    eval_score = pd.DataFrame()
    for decade in tqdm(bbb_vecs.keys()):
        if decade != '1990':
            continue
        word_vecs = bbb_vecs[decade]

        # Analogy check
        score, sections = word_vecs.evaluate_word_analogies(str(args.eval_dir / 'questions-words.txt'))
        eval_score = pd.concat([eval_score, pd.DataFrame.from_dict(
            {'task': ['analogy'], 'task_type': ['CHECK'], 'section': ['Total accuracy'],
             'accuracy': [score], 'MRR': [None], 'decade': [decade], 'vectors': ['BBB'], 'k': [0]})])

        # Analogy tasks
        analogy_tasks = [
            '3COSADD',
            #'3COSMUL', 'PAIRDIRECTION'
            ]
        analogy_thresholds = [1]
        K = 100
        for k in tqdm(range(K)):
            for analogy_task in analogy_tasks:
                for analogy_threshold in analogy_thresholds:
                    score, sections = evaluate_word_analogies_multiple(
                        model=word_vecs, analogies=str(args.eval_dir / 'questions-words.txt'),
                        method=analogy_task, restrict_vocab=None, top_threshold=analogy_threshold,
                        sds=bbb_sds, scaling=args.scaling
                    )
                    eval_score = pd.concat([eval_score, pd.DataFrame.from_dict(
                        {'task': ['analogy'], 'task_type': [f'{analogy_task}-{analogy_threshold}'],
                         'section': ['Total accuracy'], 'accuracy': [score],
                         'MRR': [None], 'decade': [decade], 'vectors': ['BBB'], 'k': [k]})])

    # Viz overall accuracy
    analogy_df = eval_score.loc[eval_score['task_type'] != 'CHECK'].copy()

    for task_type in analogy_df['task_type'].unique():
        g = sns.FacetGrid(
            analogy_df.loc[analogy_df['task_type'] == task_type], col="decade", col_wrap=5)
        g.map_dataframe(sns.kdeplot, x="accuracy")
        g.set(xlabel=f'Total accuracy {task_type}', ylabel='')
        g.figure.savefig(os.path.join(args.output_dir, f"accuracy_{task_type}_{args.run_id}.png"))

    # W&B Logging
    """
    api = wandb.Api()
    run = api.run(f"adus/bbb-uncertainty/{args.run_id}")
    wb_analogy = analogy_df.loc[(analogy_df['section'] == 'Total accuracy') & (analogy_df['vectors'] == 'BBB')]

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


    print(f"[INFO] Maximum analogy accuracy: {wb_analogy['accuracy'].max()}")

    print('Analogy')
    print(wb_analogy)

    run.update()
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-results_dir", type=str)
    parser.add_argument("-eval_dir", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-histwords_dir", type=str)
    parser.add_argument("-file_stamp", type=str)
    parser.add_argument("-scaling", type=float, required=True)
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
    args.output_dir = args.results_dir / "Performance_sampled"
    args.output_dir.mkdir(exist_ok=True)

    main(args)
