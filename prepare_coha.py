import argparse
import logging
import json
from typing import TypedDict

import click
from pathlib import Path
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class CohaDocument(TypedDict):
    doc_id: str
    genre: str
    text: str
    filedate: str  # ISO 8601 date




def main(args):
    """
    Prepares our COHA files for ingestion by the scripts in this repo.
    """
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    files_to_process = list(input_dir.glob("**/*.txt"))
    logging.info(f"Processing {len(files_to_process)} document files")
    for path in tqdm(files_to_process):
        try:
            doc = get_coha_document(path)
            with open(output_dir / f"{path.stem}.json", "w") as f:
                json.dump(doc, f)
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")


def get_coha_document(path: str) -> CohaDocument:
    # txt file is of format {genre}_{year}_{id}.txt
    # e.g. "fic_1990_0001.txt"
    genre, year, doc_id = Path(path).stem.split("_")
    with open(path, "r") as f:
        text = f.read()
    return {
        "doc_id": doc_id,
        "genre": genre,
        "text": text,
        "filedate": f"{year}-01-01",
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-run_location", type=str, required=True, choices=['local', 'sherlock'])
    parser.add_argument("-input_dir", type=str, required=False)
    parser.add_argument("-output_dir", type=str, required=False)
    args = parser.parse_args()

    if args.run_location == 'sherlock':
        base_dir = Path('/oak/stanford/groups/deho/legal_nlp/WEB')
    elif args.run_location == 'local':
        base_dir = Path(__file__).parent

    args.input_dir = base_dir / "data/COHA/COHA_text"
    args.output_dir = base_dir / "data/COHA/COHA_json"

    main(args)
