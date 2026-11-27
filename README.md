# Dear LLM, Can You Explain my MongoDB Database?

Applied Data Science Project, Politecnico di Torino (P15).

Can a large language model look at a MongoDB database and explain how it is designed? Given the JSON schema and a sample of documents, the model has to return:

1. the collections;
2. all the fields of each collection;
3. the relationships between collections;
4. the fields that follow a data modeling pattern (Embedding, Referencing, ExtendedReference, Subset, Array, Bucket), with a score for each pattern and a short evidence.

The answers are compared with a hand-written ground truth on 12 synthetic databases: 4 easy, 4 medium, 4 hard.

> Work in progress: checkpoint 2.

## Repository layout

```
POLITO_ADSP-LLM_Database.pdf   project assignment
Checkpoints/                   checkpoint presentations
src/
  normalize.py           reads a ground truth or a model output and makes them comparable (format only)
  evaluate.py            computes the metrics and writes the result tables
  evaluate_baseline.py   first version of the evaluation, superseded by evaluate.py
Notebooks/
  Inference.ipynb   runs one model on the 12 databases through OpenRouter
  Evaluation.ipynb  runs evaluate.py on a set of predictions and shows the results
prompts/
  zero_shot.txt                 instructions only
  zero_shot_definitions.txt     instructions plus the definition of each pattern
Test Cases/
  {Simple,Medium,Hard} Cases/
    JSON Schema/          <db>_json_schema.json        input given to the model
    Sample Documents/     <db>_sample_documents.json   input given to the model
    Ground Truth/         <db>_GT.json                 expected answer
    ...                   dataset descriptions and the prompts used to generate the databases
predictions/
  <model>_<prompt>/Run k/<db>_<model>.json             model outputs, 3 runs per experiment
results/
  <model>_<prompt>/                                    summary, per database, per run, confusion matrix, issues log
```

The 12 databases are synthetic: easy1 to easy4, medium5 to medium8, hard9 to hard12, each on a different domain.

## Pipeline

```
prompt (.txt) + model ──► Inference.ipynb ──► predictions/<model>_<prompt>/Run k/*.json
                                                        │
ground truth (*_GT.json) ───────────────────────────────┤
                                                        ▼
                                     Evaluation.ipynb (normalize.py + evaluate.py)
                                                        │
                                                        ▼
                                     results/<model>_<prompt>/*.csv
```

**Inference.** For each database the notebook builds the prompt (instructions, then the JSON schema, then the sample documents), calls the model through OpenRouter with the output format enforced as a JSON schema, and saves one JSON file per database and run. Each experiment uses 3 runs. Files already present are skipped, so an interrupted run can be resumed. Some models refuse some settings; Claude Opus, for example, accepts neither temperature nor seed and cannot turn off reasoning. Cell 2 has a switch for each of these cases.

**Normalization.** `normalize.py` changes how an answer is written, never what the model answered: lower case, array notation removed (`items[]` becomes `items`), a collection name repeated in front of its own field removed, pattern names compared without case or underscores. Scores are rescaled to sum to 1, and a field counts as predicted only if its best pattern reaches 0.51.

**Metrics.** Computed by pooling: TP, FP and FN are summed over databases and runs, then each metric is computed once.

| Metric | What it measures |
| --- | --- |
| Fields F1 | fields found |
| Relationships F1 | relationships found |
| Detection F1 | fields that have a pattern, found regardless of the label |
| Classification accuracy | right pattern, among the fields found |
| End-to-end F1 | pairs (field, pattern) both right: the main score |

## How to run

Both notebooks (in `Notebooks/`) run on Google Colab and read and write on Google Drive.

1. Copy the repository to Google Drive (the notebooks expect it at `MyDrive/Applied Data Science Project/Repository`; change `ROOT` in cell 2 otherwise).
2. Add the OpenRouter key to Colab Secrets as `OPENROUTER_API_KEY` and enable notebook access. Keys are never written in the notebooks.
3. **Inference.ipynb**: set `PROMPT_NAME` and `MODEL` in cell 2, run cell 4 on one database as a test, then cell 5 for all databases.
4. **Evaluation.ipynb**: set the same `PROMPT_NAME` and `MODEL` in cell 2 and run all cells.

`evaluate.py` can also be run from a terminal:

```bash
cd src
python evaluate.py --gt-dir GT --run-dirs "Run 1" "Run 2" "Run 3" --model qwen3.5-9b --out-dir results
```

The scripts use only the Python standard library. The notebooks install what they need (see `requirements.txt`).

## Current results (zero-shot prompt, 3 runs, pooled)

| Model | Fields F1 | Relationships F1 | Detection F1 | Classification acc. | End-to-end F1 |
| --- | --- | --- | --- | --- | --- |
| Qwen 3.5 9B | 0.961 | 0.787 | 0.764 | 0.715 | 0.546 |
| Gemma 4 31B | 1.000 | 0.889 | 0.828 | 0.903 | 0.747 |
| Claude Opus 5.5 | 1.000 | 0.881 | 0.991 | 0.984 | 0.975 |

Opus runs with the provider's default temperature, no seed and minimal reasoning, since it accepts no other setting; the other models use temperature 0.3 and no reasoning. Most extra relationships (false positives), for every model, are identifiers inside the copies of a Subset, which the ground truth does not count as new relationships.

## Next steps

- Prompt with pattern definitions and hints on the cases most often confused (Embedding vs Array, relationships inside copies).
- Few-shot prompt with examples built outside the 12 test databases.
