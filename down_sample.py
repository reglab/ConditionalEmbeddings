import numpy as np
import time
from collections import OrderedDict
from pathlib import Path

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

    lines = [l for l in open(str(args.source) + "/" + "%s.txt" % args.name, "r")]

    with open(str(args.saveto) + "/" + "%s_freq.txt" % args.name, "w") as f:
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

    np.save(str(args.saveto) + "/" + "vocab" + "_freq.npy", vocab_freq)

    # Create weights file (unigram distribution for negative sampling)
    weights_freq = []
    uni_sum = np.sum(np.power(np.array(list(vocab_t_freq.values())), 3/4))
    for key in vocab.keys():
        uni_w = (vocab_t_freq[key] ** (3/4)) / uni_sum
        weights_freq.append(uni_w)
    np.save(str(args.saveto) + "/" + "vocab" + "_freq_weights.npy", weights_freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-source", type=str, required=False)
    parser.add_argument("-saveto", type=str, required=False)
    parser.add_argument("-vocab", type=str, required=False)
    parser.add_argument("-vocab_f", type=str, required=False)
    parser.add_argument("-window", type=int, default=7)
    parser.add_argument("-top_K", type=int, default=50_000)
    parser.add_argument("-run_location", type=str, required=True, choices=['local', 'sherlock'])
    parser.add_argument("-name", type=str, required=True)

    args = parser.parse_args()
    if args.run_location == 'sherlock':
        base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        base_dir = Path(__file__).parent
    args.source = base_dir /f"data/{args.name}/processed/"
    args.saveto = base_dir / f"data/{args.name}/processed/"
    args.vocab = base_dir / f"data/{args.name}/processed/vocab.npy"
    args.vocab_f = base_dir / f"data/{args.name}/processed/vocab_f.npy"

    main(args)
