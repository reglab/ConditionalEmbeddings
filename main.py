import argparse
import time
from pathlib import Path

import torch.optim as optim
from tqdm import tqdm

from BBP import ConditionalBBP
from datastream import load_data
from utils import *


def main(args):

    embedding_size = args.emb
    batch_size = args.batch

    vocab = np.load(str(args.vocab), allow_pickle=True).item()

    n_words = len(vocab)

    model = ConditionalBBP(n_words, embedding_size, args)

    batch_iterator = load_data(args)

    n_batch = int(np.ceil(batch_iterator.data_size / batch_size))
    print(f"total number of batches: {str(n_batch)}")

    if args.cuda:
        model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    losses = []

    print("start training model...\n")
    start_time = time.time()

    if args.load_model:
        model, optimizer, loss, epoch = load_checkpoint(
            model, optimizer, args.best_model_save_file
        )
        print("train %s epochs before, loss is %s" % (epoch, loss))

    for epoch in tqdm(range(args.n_epochs), desc="Epoch", position=0, leave=True):
        model.train()
        total_loss = 0

        i = 0
        for in_v, out_v, cvrs in tqdm(
            batch_iterator, desc="Batch", position=1, leave=False
        ):
            i += 1
            if i > 2:
                break
            w = batch_size / batch_iterator.data_size

            if args.cuda:
                torch.cuda.empty_cache()
                in_v, out_v, cvrs = in_v.cuda(), out_v.cuda(), cvrs.cuda()

            if batch_iterator.count % 100000 == 0:
                print(
                    f"training epoch {str(epoch)}: completed {str(round(100 * batch_iterator.count / batch_iterator.data_size, 2))} %"
                )

            model.zero_grad()
            loss = model(in_v, out_v, cvrs, w)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.cpu().numpy().item()

        ave_loss = total_loss / n_batch
        print("average loss is: %s" % str(ave_loss))
        losses.append(ave_loss)
        end_time = time.time()
        print("%s seconds elapsed" % str(end_time - start_time))

        is_best = False

        if epoch == 0:
            is_best = True
        elif ave_loss < losses[epoch - 1]:
            is_best = True

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": args,
                "state_dict": model.state_dict(),
                "loss": ave_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args.best_model_save_file,
        )

    print(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-vocab", type=str)
    parser.add_argument("-source", type=str)
    parser.add_argument("-saveto", type=str)
    parser.add_argument("-file_stamp", type=str, default="coha")
    parser.add_argument("-source_file", type=str)
    parser.add_argument("-best_model_save_file", type=str, default="model_best.pth.250.tar")
    parser.add_argument("-label_map", type=list)

    parser.add_argument("-emb", type=int, default=300)
    parser.add_argument("-batch", type=int, default=1)
    parser.add_argument("-n_epochs", type=int, default=1)
    parser.add_argument("-seed", type=int, default=123)
    parser.add_argument("-lr", type=float, default=0.05)
    parser.add_argument("-skips", type=int, default=3)
    parser.add_argument("-negs", type=int, default=6)
    parser.add_argument("-window", type=int, default=7)
    parser.add_argument("-cuda", type=bool, default=False)
    parser.add_argument("-prior_weight", type=float, default=0.5)
    parser.add_argument("-sigma_1", type=float, default=1)
    parser.add_argument("-sigma_2", type=float, default=0.2)
    parser.add_argument("-weight_scheme", type=int, default=1)
    parser.add_argument("-load_model", type=float, default=False)
    args = parser.parse_args()

    args.source = Path(__file__).parent / "data" / "COHA" / "COHA_processed"
    args.saveto = Path(__file__).parent / "data" / "COHA" / "results"
    args.saveto.mkdir(parents=True, exist_ok=True)

    args.vocab = args.source / f"vocab{args.file_stamp}_freq.npy"
    args.source_file = args.source / f"{args.file_stamp}_freq.txt"
    args.function = "NN"

    args.best_model_save_file = args.saveto / f"model_best_{args.file_stamp}.pth.tar"

    args.label_map = {str(v): k for k, v in enumerate(range(181, 201))}

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("Using CUDA...")

    main(args)
