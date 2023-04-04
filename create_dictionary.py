import argparse
import numpy as np
import os
import time
from collections import OrderedDict, defaultdict

import json
import re
from pathlib import Path

import tqdm


def process(text):
    processed = (
        re.sub(r"[^a-zA-Z ]", r" ", text.replace("-\n", "").replace("\n", " "))
        .lower()
        .split()
    )
    return processed


def main(args):
    vocab0 = defaultdict(int)
    start = time.time()

    source_dir = Path(args.source_dir)
    saveto_dir = Path(args.saveto_dir)
    saveto_dir.mkdir(parents=True, exist_ok=True)
    jsonl_doc_paths = [x for x in source_dir.glob("*.json") if x.is_file()]

    with open(os.path.join(saveto_dir, f"{args.name}.txt"), "w") as f:
        print(f"Loading {len(jsonl_doc_paths)} files")

        for file_path in tqdm.tqdm(jsonl_doc_paths):
            with open(file_path, "r") as file:
                for line in file:
                    d = json.loads(line)
                    text = d["text"]
                    if text is None:
                        continue
                    label = d["filedate"][:3]

                    words = process(text)
                    if len(words) < args.window:
                        continue

                    f.write(label + "\t")
                    for w in words:
                        f.write(w + " ")
                        vocab0[w] += 1
                    f.write("\n")

    tokens = list(vocab0.keys())

    freqs = list(vocab0.values())

    sidx = np.argsort(freqs)[::-1]
    vocab = OrderedDict([(tokens[s], i) for i, s in enumerate(sidx)])

    # vocab_f = OrderedDict({k: (vocab0[k]/total_words)**(3/4) for k in vocab.keys()})

    np.save(str(saveto_dir / f"vocab_f.npy"), vocab0)
    np.save(str(saveto_dir / f"vocab.npy"), vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-run_location", type=str, required=True, choices=['local', 'sherlock'])
    parser.add_argument("-source_dir", type=str, required=False)
    parser.add_argument("-saveto_dir", type=str, required=False)
    parser.add_argument("-window", type=int, default=7)
    parser.add_argument("-name", type=str, required=True)
    args = parser.parse_args()

    if args.run_location == 'sherlock':
        base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        base_dir = Path(__file__).parent

    args.saveto_dir = base_dir / f"data/{args.name}/processed"
    args.source_dir = base_dir / f"data/{args.name}/json"

    main(args)
