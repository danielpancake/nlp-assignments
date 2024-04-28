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

It also contains some variables that can be changed to customize the behavior of the project.
To switch between CPU and GPU training, [set the `gpu_id` variable](project.yml?plain=1#L11) to the desired GPU ID or `-1` for CPU.

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).

| Command | Description |
| --- | --- |
| `merge` | Merges NER spans from multiple `test.jsonl` files from all solutions. |
