# Soccer-GMR Data

This directory contains the benchmark labels and related metadata for **Soccer-GMR**, a benchmark for **Generalized Moment Retrieval (GMR)**.

## Overview

Soccer-GMR is designed for generalized moment retrieval, where a query may correspond to:

- **no moment**
- **one moment**
- **multiple moments**

The label files in this directory provide benchmark annotations for these three retrieval scenarios in a unified JSONL format.

## Directory Structure

```text
label/
├── full/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── full.jsonl
└── sub/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

- `label/full/` contains the main benchmark split files.
- `label/sub/` contains a smaller subset intended for lightweight inspection, development, or debugging.

## Full vs Sub

- **full**: the main benchmark release used for standard training, validation, and testing
- **sub**: a smaller subset for quick inspection, debugging, and lightweight evaluation

## File Format

All label files are provided in **JSONL** format, where each line is a single JSON object.

## Core Fields

Each record may contain the following key fields:

- `qid`: unique query identifier
- `vid`: video clip identifier or filename
- `query`: natural language query
- `duration`: clip duration in seconds
- `relevant_windows`: normalized ground-truth temporal windows in `[start, end]` format

## Additional Fields

Some records also include auxiliary metadata such as:

- `moment`: intermediate annotation form
- `action_type`: action category
- `dataset_source`: source dataset identifier
- `match_info`: match-level metadata such as teams and date

## Annotation Representation

The benchmark may preserve two related representations of temporal annotations:

- `moment`: an intermediate annotation form
- `relevant_windows`: normalized temporal windows used for evaluation

In general:

- `relevant_windows` is the evaluation-ready representation
- `moment` preserves earlier or source-side annotation structure
- when `relevant_windows` is available, it should be treated as the primary field for evaluation

For official evaluation usage and normalization details, please refer to [`../eval/README.md`](../eval/README.md).

## `moment` Types

Currently, `moment` may use different internal forms, such as:

- `clips`: directly specified temporal segments
- `timestamps`: point-level timestamps that may later be expanded into temporal windows

For evaluation, these intermediate forms can be normalized into `relevant_windows` through the provided utilities in [`../eval/`](../eval/).

## Encoding of Retrieval Scenarios

Soccer-GMR supports all three GMR scenarios in a unified format:

- **Null-set query**: `relevant_windows = []`
- **Single-moment query**: `relevant_windows` contains exactly one window
- **Multi-moment query**: `relevant_windows` contains multiple windows

## Example Record

```json
{
  "qid": 580,
  "vid": "WC2022_3857268_2_5500s_5650s",
  "query": "Locate block moments performed by players from Belgium.",
  "duration": 150,
  "moment": {
    "type": "clips",
    "value": [[26.0, 34.0], [104.0, 112.0]]
  },
  "action_type": "block",
  "dataset_source": "worldcup2022",
  "relevant_windows": [[26.0, 34.0], [104.0, 112.0]]
}
```

Example of a null-set query:

```json
{
  "qid": 862,
  "vid": "WC2022_3857268_2_5500s_5650s",
  "query": "Identify foul committed moments performed by players from Canada.",
  "duration": 150,
  "moment": {
    "type": "clips",
    "value": []
  },
  "action_type": "foul committed",
  "dataset_source": "worldcup2022",
  "relevant_windows": []
}
```
