import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
from pathlib import Path
import glob
import pickle


def main(args):
    base_text = """
    The bright sun shine in the light blue sky <e>
    The sun was shine in the sky <e>
    The sun was shine bright in the blue sky <e>
    In the blue sky the bright sun shine <e>
    In the blue sky the sun shine bright <e>
    In the sky the light shine <e>
    Shine in the light blue sky was the bright sun <e>
    Shine in the sky the bright sun <e>
    A bright sun shine the blue sky <e>
    Bright the sun shine in the blue sky <e>
    The light shine bright in the sky <e>
    The sun shine bright in the light sky <e>
    The sun shine bright in the blue sky <e>
    Was shine bright the sun in the sky <e>
    """

    text_list = base_text.replace('\n', ' ').strip()
    text_list = [w.lower() for w in text_list.split(' ') if w != '']

    # Create transition matrix
    tm = pd.crosstab(
        pd.Series(text_list[:-1], name='i'),
        pd.Series(text_list[1:], name='j'), normalize=0)

    # Generate text based on the transition matrix
    sim_text = []
    last_word = '<e>'
    for k in tqdm(range(args.m)):
        p = tm.loc[last_word]
        next_word = np.random.choice(p.keys(), 1, p=p.values)[0]
        sim_text.append(next_word)
        last_word = next_word
    sim_text_txt = ' '.join(sim_text)
    sim_text_txt = sim_text_txt.replace(' <e>', '.')

    # Save to the required BBB format
    # Split into batches
    print('[INFO] Saving corpus.')
    sentences = sim_text_txt.split('.')
    sentences = np.array_split(sentences, args.d)
    os.makedirs(os.path.join(args.saveto, 'json'), exist_ok=True)

    for i in range(len(sentences)):
        text = '.'.join(sentences[i])
        doc = {'text': text, 'filedate': str(1990)}

        with open(os.path.join(args.saveto, 'json', f"{i}.json"), "w") as wf:
            json.dump(doc, wf)

    # Create files for co-occurrence matrices
    os.makedirs(os.path.join(args.saveto, 'cooccur', '1990'), exist_ok=True)
    for fname in glob.glob(os.path.join(args.saveto, 'json', '*.json')):
        with open(fname, 'r') as f:
            doc = json.load(f)
        text = doc['text']
        name = fname.split(os.path.sep)[-1].replace('.json', '.txt')

        with open(os.path.join(args.saveto, 'cooccur', '1990', name), 'w') as fw:
            words = [w for w in text.split(' ') if w != '']
            for w in words:
                w = w.replace('.', '')
                fw.write(f"{w}\tNone\tNone\n")

    # Save word dict
    final_vocab_file = os.path.join(args.saveto, 'processed', 'vocab_freq.npy')
    if os.path.exists(final_vocab_file):
        vocab_dict = np.load(final_vocab_file, allow_pickle=True)
        vocab_dict = {w: i for w, i in vocab_dict.item().items()}
        with open(os.path.join(args.saveto, 'info', 'word-dict.pkl'), 'wb') as vf:
            pickle.dump(vocab_dict, vf)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-saveto", type=str, required=False)
    parser.add_argument("-m", type=int, help='Number of tokens')
    parser.add_argument("-d", type=int, help='Number of documents')
    parser.add_argument("-run_location", type=str, required=True, choices=['local', 'sherlock'])
    parser.add_argument("-name", type=str, required=True)

    args = parser.parse_args()
    if args.run_location == 'sherlock':
        base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        base_dir = Path(__file__).parent
    args.saveto = base_dir / f"data/{args.name}"

    main(args)
