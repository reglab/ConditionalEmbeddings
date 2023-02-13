import numpy as np
import time
from collections import OrderedDict

import argparse

import tqdm


def main(args):

    vocab = np.load(args.vocab, allow_pickle=True).item()
    vocab_freq = OrderedDict([(i, s) for i, s in vocab.items() if s < args.top_K])
    freq_keys = vocab_freq.keys()

    vocab_t = np.load(args.vocab_f, allow_pickle=True).item()
    vocab_t_freq = OrderedDict([(k, v) for k, v in vocab_t.items() if k in freq_keys])
    total_words = np.sum(list(vocab_t_freq.values()))
    print(total_words)

    vocab_p = OrderedDict(
        [
            (k, 1 - np.sqrt(10 ** (-5) / (v / total_words)))
            for k, v in vocab_t_freq.items()
        ]
    )

    start = time.time()

    lines = [l for l in open(args.source + "%s.txt" % args.save_label, "r")]

    with open(args.saveto + "%s_freq.txt" % args.save_label, "w") as f:
        print("loading all files...")

        for line in tqdm.tqdm(lines):
            items = line.strip().split("\t")
            label = items[0][:3]
            text = items[1]

            if text is not None:
                words = text.split(" ")

                f.write(label + "\t")

                for w in words:
                    if (w in freq_keys) and (np.random.uniform(0, 1) < vocab_p[w]):
                        f.write(w + " ")

                f.write("\n")

    print("%s seconds elapsed" % (time.time() - start))

    np.save(args.saveto + "vocab" + args.save_label + "_freq.npy", vocab_freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-source", type=str, default="data/COHA/COHA_processed/")
    parser.add_argument("-saveto", type=str, default="data/COHA/COHA_processed/")
    parser.add_argument(
        "-vocab", type=str, default="data/COHA/COHA_processed/vocabcoha.npy"
    )
    parser.add_argument(
        "-vocab_f", type=str, default="data/COHA/COHA_processed/vocab_fcoha.npy"
    )
    parser.add_argument("-window", type=int, default=7)
    parser.add_argument(
        "-save_label", type=str, default="coha"
    )  # file name of the saved files
    parser.add_argument("-top_K", type=int, default=50_000)

    args = parser.parse_args()
    main(args)
