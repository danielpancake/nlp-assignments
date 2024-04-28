# Merger Solution

> CODALAB SCORE: 0.43

## Description

This solution merges output from both the `dict` and `spacy-spancat` solutions.

## How to run

1. Make sure you have installed `weasel` by running `pip install weasel`.
2. Run the following command to execute the workflow:

```bash
weasel run merge
```

## 📋 project.yml

The [`project.yml`](project.yml) defines the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).

| Command | Description |
| --- | --- |
| `merge` | Merges NER spans from multiple `test.jsonl` files from all solutions. |
