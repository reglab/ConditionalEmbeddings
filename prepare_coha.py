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


DEFAULT_INPUT_DIR = Path(__file__).parent / "data/COHA/COHA_text"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data/COHA/COHA_json"


@click.command()
@click.option("--input_dir", type=click.Path(exists=True), default=DEFAULT_INPUT_DIR)
@click.option(
    "--output_dir", type=click.Path(), default=DEFAULT_OUTPUT_DIR, required=False
)
def main(input_dir: str, output_dir: str):
    """
    Prepares our COHA files for ingestion by the scripts in this repo.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
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
    main()
