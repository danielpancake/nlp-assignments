import json

from typing import Iterable


def read_jsonl(file_path: str) -> Iterable:
    """Reads a file in jsonl format"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(file_path: str, data: Iterable):
    """Writes data to a file in jsonl format"""
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
