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

from model_to_vectors import load_model

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')


def compute_chain_statistics(a):
    N = a.shape[0]
    mean = np.mean(a, axis=0)
    var = np.power(np.std(a, axis=0), 2) * N / (N - 1)
    return mean, var


def compute_rhat(chains_means, chains_vars, N):
    M = chains_means.shape[0]

    # Compute within chain variance (average variances over chains)
    W = np.mean(chains_vars, axis=0)

    # Compute between chain variance
    B = np.power(np.std(chains_means, axis=0), 2) * N * M / (M - 1)

    # Rhat
    rhat = (N-1) / N * W + B / N
    return rhat


def main(args):
    # Define chains, M and N (we use the first model to find N and check that all have the same number of
    # iterations)
    chains = glob.glob(os.path.join(args.models_dir, args.rhat_group, 'model_steps_*'))
    M = len(chains)
    N = len(glob.glob(os.path.join(chains[0], '*.tar')))

    # Define truncation
    tmin = args.K * 12_500
    tmax = (args.K + 1) * 12_500
    W_sub = 20 if args.param_group == 'covar' else 12_500
    W = 20 if args.param_group == 'covar' else 50_000
    K_name = 0 if args.param_group == 'covar' else args.K
    truncations = 1 if args.param_group == 'covar' else 4

    # Get statistics for each chain
    print('[INFO] Getting statistics for each truncated chain')
    for m, chain in tqdm(enumerate(chains)):
        mean_output_path = os.path.join(args.output_dir, f'mean_{args.param_group}_{m}_{K_name}.npy')
        var_output_path = os.path.join(args.output_dir, f'var_{args.param_group}_{m}_{K_name}.npy')
        if os.path.exists(mean_output_path) and os.path.exists(var_output_path):
            continue

        # Load chain models
        models_files = glob.glob(os.path.join(chain, '*.tar'))
        N_chain = len(models_files)
        assert N_chain == N

        # Define chain array
        chain_a = np.zeros((N, W_sub, 300))
        for n, model_file in enumerate(models_files):
            # Load model and truncate
            model = load_model(
                model_file,
                args.base_dir / f"data/{args.name}/processed/vocab_freq.npy",
            )

            model_vectors = None
            if args.param_group == 'means':
                model_vectors = model.input_embeddings()
                model_vectors = model_vectors[tmin:tmax, :]
            elif args.param_group == 'rho':
                model_vectors = model.var_embeddings()
                model_vectors = model_vectors[tmin:tmax, :]
            elif args.param_group == 'covar':
                model_vectors = model.covar_embeddings()

            chain_a[n, :, :] = model_vectors

        # Compute chain stats
        chain_mean, chain_var = compute_chain_statistics(chain_a)
        np.save(mean_output_path, chain_mean, allow_pickle=True)
        np.save(var_output_path, chain_var, allow_pickle=True)

    # Consolidate truncated statistics if complete
    saved_stats = glob.glob(os.path.join(args.output_dir, f'var_{args.param_group}_*.npy'))
    if len(saved_stats) == truncations * M:
        print('[INFO] Consolidating statistics across truncations')

        # Full mean/var is shape (M, W, D)
        full_means, full_vars = np.zeros((M, W, 300)), np.zeros((M, W, 300))

        for m in range(M):
            saved_means = glob.glob(os.path.join(args.output_dir, f'mean_{args.param_group}_{m}_*.npy'))
            saved_vars = glob.glob(os.path.join(args.output_dir, f'var_{args.param_group}_{m}_*.npy'))
            assert len(saved_means) == len(saved_vars)

            for k in range(truncations):
                chain_k_mean = np.load(saved_means[k])
                chain_k_var = np.load(saved_vars[k])

                tmin = 0 if args.param_group == 'covar' else k * 12_500
                tmax = 20 if args.param_group == 'covar' else (k + 1) * 12_500

                full_means[m, tmin:tmax, :] = chain_k_mean
                full_vars[m, tmin:tmax, :] = chain_k_var

        # Compute Rhat for the parameter group (rhat is shape (W, D) )
        print('[INFO] Compute Rhat')
        rhat = compute_rhat(chains_means=full_means, chains_vars=full_vars, N=N)
        np.save(
            os.path.join(args.output_dir, f'rhat_{args.param_group}.npy'),
            rhat, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-models_dir", type=str)
    parser.add_argument("-rhat_group", type=str)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-base_dir", type=str)
    parser.add_argument("-param_group", type=str, choices=['means', 'rho', 'covar'])
    parser.add_argument("-K", type=int, default=0)

    args = parser.parse_args()

    # Paths
    if args.run_location == 'sherlock':
        args.models_dir = Path('/scratch/groups/deho/WEB/COHA_Rhat/models')
        args.base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        raise Exception('[ERROR] Needs to be run on Sherlock (scratch) due to storage requirements.')
        # args.base_dir = Path(__file__).parent

    args.output_dir = args.models_dir.resolve().parents[0] / "Processed" / args.rhat_group
    args.output_dir.mkdir(exist_ok=True)

    main(args)
