import argparse
import numpy as np
import os
import glob
import json
import shutil
from pathlib import Path


def main(args):
    # If the subset has already been created we just re-create it
    subset_path = os.path.join(args.file_output_dir, f"{args.name}.json")
    if os.path.exists(subset_path):
        print('[INFO] Loading existing subset')
        with open(subset_path, 'r') as f:
            files = json.load(f)
    else:
        # Sample a fixed percentage of COHA
        files = {}
        decade_dirs = glob.glob(os.path.join(args.coha_path, '*'))

        for decade_dir in decade_dirs:
            decade = decade_dir.split(os.path.sep)[-1].replace('s', '')
            decade_files = glob.glob(os.path.join(decade_dir, '*'))

            # Random subset
            select_files = list(np.random.choice(decade_files, size=int(len(decade_files) * args.percent / 100), replace=False))
            files[decade] = [os.path.sep.join(f.split(os.path.sep)[-2:]) for f in select_files]

        # Save json indicating COHA subset
        with open(subset_path, 'w') as f:
            json.dump(files, f)

    # Replace files
    if args.replace:
        #print('[INFO] Replacing files in COHA_text')
        #for f in glob.glob(os.path.join('data/COHA/COHA_text', '*', '*.txt')):
        #    os.remove(f)

        print('[INFO] Loading new COHA subset')
        for decade, decade_files in files.items():
            os.makedirs(args.coha_output_dir / f"COHA_text/{decade}s", exist_ok=True)
            for f in decade_files:
                shutil.copy(
                    src=f"{args.coha_path}/{f}",
                    dst=args.coha_output_dir / f"COHA_text/{f}"
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-coha_path", type=str, required=False)
    parser.add_argument("-coha_output_dir", type=str, required=False)
    parser.add_argument("-file_output_dir", type=str, required=False)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-percent", type=int)
    parser.add_argument("-replace", type=bool, required=True)
    parser.add_argument("-run_location", type=str, required=True)

    args = parser.parse_args()

    # Paths
    if args.run_location == 'sherlock':
        base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
        args.coha_path = base_dir / 'data/COHA/COHA_text'
    elif args.run_location == 'local':
        base_dir = Path(__file__).parent
        args.coha_path = '../COHA-SGNS/data/COHA/COHA text'

    args.coha_output_dir = base_dir / f"data/{args.name}"
    args.file_output_dir = base_dir / 'data/COHA/subsets'

    os.makedirs(args.file_output_dir, exist_ok=True)

    main(args)
