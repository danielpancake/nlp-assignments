import glob
import os

from collections import defaultdict
from pathlib import Path

import typer

from scripts.jsonl import read_jsonl, write_jsonl


def merger(
    input_path: Path = typer.Argument(...),
    output_file: Path = typer.Option(..., "-o", "--output", exists=False),
):
    """Merges NER spans from multiple `test.jsonl` files in a directory"""

    # Get a list of all `test.jsonl` files in the `out` directory, excluding the `merged` directory
    test_jsonl_files = glob.glob("*/test.jsonl", root_dir=input_path)
    test_jsonl_files = [f for f in test_jsonl_files if not "merged" in f]
    test_jsonl_files = [os.path.join(input_path, f) for f in test_jsonl_files]

    # Merge all jsonl files in a directory
    ids_dict = defaultdict(list)

    for file_path in test_jsonl_files:
        for sample in read_jsonl(file_path):
            ids_dict[sample["id"]].append(sample)

    # Collapse ners from same ids
    merged = []

    for id_, samples in ids_dict.items():
        ners = []

        for sample in samples:
            ners.extend(sample["ners"])

        merged.append({"id": id_, "ners": ners})

    # Convert list of lists to list of tuples
    # and filter duplicates
    for sample in merged:
        sample["ners"] = [tuple(ner) for ner in sample["ners"]]
        sample["ners"] = list(set(sample["ners"]))

    # Write the merged data to a new jsonl file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_jsonl(output_file, merged)


if __name__ == "__main__":
    typer.run(merger)
