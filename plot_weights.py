import argparse
import gensim
import seaborn as sns
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')


def main(args):
    for decade in range(181, 200):
        if decade not in [181]:
            continue
        word_vecs = gensim.models.KeyedVectors.load_word2vec_format(
            f"data/COHA/results/decade_embeddings_{decade}.txt", binary=False, no_header=True)

        weights = word_vecs.vectors
        weights = weights.reshape(-1, )
        weights = list(weights)

        plt.clf()
        ax = sns.kdeplot(weights)

        os.makedirs('results/weights', exist_ok=True)
        ax.figure.savefig(f"results/weights/weightdist-{decade}-6-e{args.name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str)

    args = parser.parse_args()

    main(args)
