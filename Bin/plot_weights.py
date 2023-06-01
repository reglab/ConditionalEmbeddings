import argparse
import gensim
import seaborn as sns
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')


def main(args):
    for decade in range(181, 200):
        if args.decade:
            if decade != args.decade:
                continue
        word_vecs = gensim.models.KeyedVectors.load_word2vec_format(
            f"data/COHA/results/decade_embeddings_{args.file_stamp}_{args.run_id}_{decade}.txt", binary=False, no_header=True)

        weights = word_vecs.vectors
        weights = weights.reshape(-1, )
        weights = list(weights)

        plt.clf()
        ax = sns.kdeplot(weights)

        os.makedirs('results/weights', exist_ok=True)
        ax.figure.savefig(f"results/weights/weightdist_{args.file_stamp}_{args.run_id}_{decade}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_stamp", type=str, required=True)
    parser.add_argument("-decade", type=int, default=None)
    parser.add_argument("-run_id", type=str, required=True)

    args = parser.parse_args()

    main(args)
