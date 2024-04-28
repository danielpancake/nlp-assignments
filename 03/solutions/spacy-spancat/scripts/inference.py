"""Inference script for predicting NER spans using a spaCy model"""
import os

from pathlib import Path

import spacy
import typer

from scripts.jsonl import read_jsonl, write_jsonl


def predict(nlp: spacy.language.Language, text: str) -> list:
    """Predicts NER spans"""
    doc = nlp(text)
    return [
        [span.start_char, span.end_char - 1, span.label_] for span in doc.spans["sc"]
    ]


def inference(
    model: Path = typer.Argument(...),
    input_file: Path = typer.Argument(...),
    output_file: Path = typer.Option(..., "-o", "--output", exists=False),
):
    """Predicts NER spans using a spaCy model"""
    nlp = spacy.load(model)

    dataset = read_jsonl(input_file)
    dataset_pred = []

    for sample in dataset:
        text = sample["sentences"]
        ners = predict(nlp, text)
        dataset_pred.append({"id": sample["id"], "ners": ners})

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_jsonl(output_file, dataset_pred)


if __name__ == "__main__":
    typer.run(inference)
