import argparse
from BBP import ConditionalBBP
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
import tqdm
from tqdm.contrib.concurrent import process_map

# get_embedding = lambda word, decade: model.linear(torch.cat([torch.tensor(word_em[word]), torch.tensor(year_covar[decade])], 0))
# get_dev = lambda word: (torch.tensor(word_var[word]).exp() + 1).log()


def load_model(model_path: str, vocab_path: str) -> ConditionalBBP:
    torch_model = torch.load(
        model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # noinspection PyTypeChecker
    vocab: dict[str, int] = np.load(vocab_path, allow_pickle=True).item()

    # Load BBB model
    try:
        model = ConditionalBBP(len(vocab), torch_model["args"].emb, torch_model["args"])
    except AttributeError:
        args_cp = vars(torch_model['args'])
        # In case we're using the OG BBB code, we add these arguments to we can still load the model
        args_cp['kl_tempering'] = None
        args_cp['batch'] = None
        args_cp['num_batches'] = None
        args_cp['scaling'] = None
        args_cp['similarity'] = None
        args_cp['initialize'] = None
        args_cp['no_mlp_layer'] = False
        model = ConditionalBBP(len(vocab), torch_model["args"].emb, Namespace(**args_cp))

    model.load_state_dict(torch_model["state_dict"])
    model.vocab = vocab
    model.word_input_embeddings = {}
    for word, vec in zip(model.vocab.keys(), model.input_embeddings()):
        model.word_input_embeddings[word] = vec
    model.word_var = {}
    for word, vec in zip(model.vocab.keys(), model.var_embeddings()):
        model.word_var[word] = vec
    model.year_covar = {}
    if model.no_mlp_layer is False:
        for year, vec in zip(model.label_map.keys(), model.covar_embeddings()):
            model.year_covar[year] = vec
    return model


def get_embedding(model: ConditionalBBP, word: str, decade: int):
    # If we turned off the MLP layer, we return the global word embeddings
    if model.no_mlp_layer:
        return torch.tensor(model.word_input_embeddings[word]).tolist()

    return torch.tanh(model.linear(
        torch.cat(
            [
                torch.tensor(model.word_input_embeddings[word]),
                torch.tensor(model.year_covar[decade]),
            ],
            0,
        )
    )).tolist()


def get_dev(model: ConditionalBBP, word: str) -> list:
    return (torch.tensor(model.word_var[word]).exp() + 1).log().tolist()


def compute_decade_embeddings(
    model: ConditionalBBP, decade: str, output_embedding_path: str
):
    all_words = list(model.vocab.keys())
    embeddings = []
    for word in tqdm.tqdm(all_words, desc="Word", position=2):
        embeddings.append(get_embedding(model, word, decade))
    # embeddings = list(
    #     map(
    #         get_embedding,
    #         [model] * len(all_words),
    #         tqdm.tqdm(all_words),
    #         [decade] * len(all_words),
    #         # chunksize=100,
    #     )
    # )
    # Write out in w2v format
    with open(output_embedding_path, "w") as f:
        for word, embedding in zip(all_words, embeddings):
            f.write(f"{word} {' '.join(map(str, embedding))}\n")


def main(args):
    torch.set_grad_enabled(False)
    model = load_model(
        args.base_dir / f"data/{args.name}/results/model_best_{args.file_stamp}_{args.run_id}.pth.tar",
        args.base_dir / f"data/{args.name}/processed/vocab_freq.npy",
    )
    all_decades = list(model.label_map.keys())
    if model.no_mlp_layer:
        compute_decade_embeddings(
            model, 0,
            args.base_dir / f"data/{args.name}/results/decade_embeddings_{args.file_stamp}_{args.run_id}_199.txt"
        )
    else:
        for decade in tqdm.tqdm(all_decades, desc="Decade", position=1):
            compute_decade_embeddings(
                model, decade, args.base_dir / f"data/{args.name}/results/decade_embeddings_{args.file_stamp}_{args.run_id}_{decade}.txt"
            )
    all_words = list(model.vocab.keys())
    dev_vectors = []
    for word in tqdm.tqdm(all_words, desc="Word", position=2):
        dev_vectors.append(get_dev(model, word))

    with open(args.base_dir / f"data/{args.name}/results/dev_vectors_{args.file_stamp}_{args.run_id}.txt", "w") as f:
        for word, dev in zip(all_words, dev_vectors):
            f.write(f"{word} {' '.join(map(str, dev))}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_stamp", type=str, required=False)
    parser.add_argument("-run_id", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-base_dir", type=str, required=False)

    args = parser.parse_args()

    if args.run_location == 'sherlock':
        args.base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        args.base_dir = Path(__file__).parent
    args.file_stamp = args.name

    main(args)
