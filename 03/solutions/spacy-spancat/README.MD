# SpaCy's SpanCat for Nested NER

> CODALAB SCORE: 0.45

## Description

This solution demonstrates how spaCy's Span Categorization (SpanCat) perform on Nested NER.

## How to run

1. Make sure you have installed `weasel` by running `pip install weasel`.
2. Run the following command to execute the workflow:

```bash
weasel run all
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
| `requirements` | Install the required libraries and spaCy models. |
| `jsonl_to_iob2` | Convert data from JSONL format to IOB2 format. |
| `iob2_to_spacy` | Convert data from IOB2 format to spaCy format. |
| `train` | Train a spaCy's SpanCat model. |
| `evaluate` | Evaluate the spaCy's SpanCat model. |
| `inference` | Make predictions using the spaCy's SpanCat model. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order.

| Workflow | Steps |
| --- | --- |
| `all` | `requirements`, `jsonl_to_iob2`, `iob2_to_spacy`, `train`, `evaluate`, `inference` |
