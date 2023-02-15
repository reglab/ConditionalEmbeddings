from BBP import ConditionalBBP

import numpy as np
import torch
import tqdm
from tqdm.contrib.concurrent import process_map

# get_embedding = lambda word, decade: model.linear(torch.cat([torch.tensor(word_em[word]), torch.tensor(year_covar[decade])], 0))
# get_dev = lambda word: (torch.tensor(word_var[word]).exp() + 1).log()


def load_model(model_path: str, vocab_path: str) -> ConditionalBBP:
    torch_model = torch.load(model_path)
    # noinspection PyTypeChecker
    vocab: dict[str, int] = np.load(vocab_path, allow_pickle=True).item()
    model = ConditionalBBP(len(vocab), torch_model["args"].emb, torch_model["args"])
    model.load_state_dict(torch_model["state_dict"])
    model.vocab = vocab
    model.word_input_embeddings = {}
    for word, vec in zip(model.vocab.keys(), model.input_embeddings()):
        model.word_input_embeddings[word] = vec
    model.word_var = {}
    for word, vec in zip(model.vocab.keys(), model.var_embeddings()):
        model.word_var[word] = vec
    model.year_covar = {}
    for year, vec in zip(model.label_map.keys(), model.covar_embeddings()):
        model.year_covar[year] = vec
    return model


def get_embedding(model: ConditionalBBP, word: str, decade: int) -> torch.Tensor:
    return model.linear(
        torch.cat(
            [torch.tensor(model.word_input_embeddings[word]), torch.tensor(model.year_covar[decade])],
            0,
        )
    )


def get_dev(model: ConditionalBBP, word: str) -> torch.Tensor:
    return (torch.tensor(model.word_var[word]).exp() + 1).log()


def compute_decade_embeddings(
    model: ConditionalBBP, decade: str, output_embedding_path: str
):
    all_words = list(model.vocab.keys())
    # Use process_map to parallelize the computation
    embeddings = process_map(
        get_embedding,
        [model] * len(all_words),
        all_words,
        [decade] * len(all_words),
        chunksize=100,
    )
    # Write out in w2v format
    with open(output_embedding_path, "w") as f:
        for word, embedding in zip(all_words, embeddings):
            f.write(f"{word} {' '.join(map(str, embedding))}\n")


def main():
    model = load_model(
        "data/COHA/results/model_best_coha.pth.tar",
        "data/COHA/COHA_processed/vocabcoha_freq.npy",
    )
    all_decades = list(model.label_map.keys())
    for decade in tqdm.tqdm(all_decades, desc="Decade", position=1):
        compute_decade_embeddings(
            model, decade, f"data/COHA/results/decade_embeddings_{decade}.txt"
        )
    all_words = list(model.vocab.keys())
    dev_vectors = process_map(get_dev, [model] * len(all_words), all_words, chunksize=100)
    with open("data/COHA/results/dev_vectors.txt", "w") as f:
        for word, dev in zip(all_words, dev_vectors):
            f.write(f"{word} {' '.join(map(str, dev))}\n")


if __name__ == "__main__":
    main()
