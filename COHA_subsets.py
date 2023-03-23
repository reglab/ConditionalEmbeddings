import argparse
import numpy as np
import os
import glob
import json
import shutil


def main(args):
    # If the subset has already been created we just re-create it
    subset_path = os.path.join(args.output_dir, f"{args.name}.json")
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
        print('[INFO] Replacing files in COHA_text')
        for f in glob.glob(os.path.join('data/COHA/COHA_text', '*', '*.txt')):
            os.remove(f)

        print('[INFO] Loading new COHA subset')
        for decade, decade_files in files.items():
            os.makedirs(f"data/COHA/COHA_text/{decade}s", exist_ok=True)
            for f in decade_files:
                shutil.copy(
                    src=f"{args.coha_path}/{f}",
                    dst=f"data/COHA/COHA_text/{f}"
                )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-coha_path", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-percent", type=int)
    parser.add_argument("-replace", type=bool, required=True)

    args = parser.parse_args()

    # Paths
    args.coha_path = '../COHA-SGNS/data/COHA/COHA text'
    args.output_dir = 'data/COHA/subsets'
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
