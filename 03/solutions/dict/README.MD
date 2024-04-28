# Dictionary approach for ~~Nested~~ NER

> CODALAB SCORE: 0.35

## Description

The [`dict.ipynb`](dict.ipynb) notebook uses a simple dictionary-based model. The nested NER task is **reduced** to a flat token-level tagging. The advantages of this approach are:

- simplicity and ease of implementation,
- ability to remember underrepresented entities.

However, it may not capture contextual information or handle complex cases where the same token can have different labels depending on the context.

## How to run

1. Run the notebook [`dict.ipynb`](dict.ipynb) in the Jupyter environment.
2. Install the required libraries if needed.
