"""Converts a dataset in jsonl format to IOB2 format"""

import os

from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List

import typer
import razdel

from sklearn.model_selection import train_test_split
from wasabi import msg

from scripts.constants import MAX_STACK
from scripts.jsonl import read_jsonl


class IOB(Enum):
    """IOB2 encoding"""

    B = "B-"
    I = "I-"
    O = "O"


def char_pos_to_word_pos(text: str, beg: int, end: int) -> List[int]:
    """Converts character positions to token positions"""
    tokenized = list(razdel.tokenize(text))
    spans = [(token.start, token.stop - 1) for token in tokenized]

    pos = [i for i, (b, e) in enumerate(spans) if not (end < b or e < beg)]
    return pos


def tokenize_dataset(file_path: str) -> List[dict]:
    """Tokenizes the dataset preserving NER positions"""
    dataset_tokenized = []

    # Convert character-based positions to word-based positions
    for sample in read_jsonl(file_path):
        text = sample["sentences"]
        ners = sample["ners"]

        text_tokenized = [_.text for _ in razdel.tokenize(text)]
        ners_tokenized = []

        for ner in ners:
            word_pos = char_pos_to_word_pos(text, ner[0], ner[1])

            if len(word_pos) == 0:
                raise ValueError("No word position found for NER")

            expected = text[ner[0] : ner[1] + 1]
            got = " ".join(text_tokenized[word_pos[0] : word_pos[-1] + 1])

            # Sanity check
            assert (len(expected) - len(got)) <= 1, f"Expected: {expected}, Got: {got}"

            ners_tokenized.append([word_pos[0], word_pos[-1], ner[2]])

        dataset_tokenized.append({"text": text_tokenized, "ners": ners_tokenized})

    return dataset_tokenized


def tokenized_to_iob2(dataset_tokenized: List[dict]) -> List[dict]:
    """Converts tokenized dataset to IOB2 format"""
    dataset_iob2 = deepcopy(dataset_tokenized)

    # Init stacks
    for idx, sample in enumerate(dataset_iob2):
        dataset_iob2[idx]["stack"] = [
            [IOB.O.value] * MAX_STACK for _ in range(len(sample["text"]))
        ]

    # Fill stacks
    for idx, sample in enumerate(dataset_iob2):
        for ner in sample["ners"]:
            beg = ner[0]
            end = ner[1] + 1
            label = ner[2]

            # Find available depth
            depth = 0
            while depth < MAX_STACK and any(
                sample["stack"][x][depth] != IOB.O.value for x in range(beg, end)
            ):
                depth += 1

            if depth == MAX_STACK:
                print("Stack is full!")
                continue

            for i in range(beg, end):
                if i == ner[0]:
                    sample["stack"][i][depth] = IOB.B.value + label
                else:
                    sample["stack"][i][depth] = IOB.I.value + label

    return dataset_iob2


def save_iob2(file_path: str, dataset: List[dict]):
    """Saves dataset in IOB2 format"""
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            for _, (word, stack) in enumerate(zip(sample["text"], sample["stack"])):
                f.write(word)
                f.write("\t")
                f.write("\t".join(stack))
                f.write("\n")

            f.write("\n")


def main(
    filepath: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option(..., "-o", "--output", exists=False),
    test_size: float = typer.Option(0.075, "--test-size", "-t"),
    seed: int = typer.Option(42, "--seed"),
):
    """Converts a dataset in jsonl format to IOB2 format"""

    # Tokenize the dataset
    with msg.loading("Tokenizing the dataset"):
        dataset_tokenized = tokenize_dataset(filepath)
    msg.good("Tokenization done!")

    with msg.loading("Converting tokenized dataset to IOB2 format"):
        dataset_iob2 = tokenized_to_iob2(dataset_tokenized)
    msg.good("Conversion done!")

    # Split the dataset
    msg.info("Splitting the dataset into train, dev and test splits.")
    msg.info(f"Parameters: test_size={test_size}, seed={seed}")

    train, test = train_test_split(
        dataset_iob2,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train, dev = train_test_split(
        train,
        test_size=test_size / (1 - test_size),
        random_state=seed,
        shuffle=True,
    )

    # Print the dataset sizes
    msg.info(f"Train size: {len(train)}")
    msg.info(f"Dev size: {len(dev)}")
    msg.info(f"Test size: {len(test)}")

    # Save the datasets
    os.makedirs(output, exist_ok=True)

    save_iob2(os.path.join(output, "train.iob2"), train)
    save_iob2(os.path.join(output, "dev.iob2"), dev)
    save_iob2(os.path.join(output, "test.iob2"), test)

    msg.good("Datasets saved successfully!")


if __name__ == "__main__":
    typer.run(main)
