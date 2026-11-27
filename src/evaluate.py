"""
evaluate.py

Computes the metrics defined in the Notion "Evaluation" page:
    1. Fields F1
    2. Relationships F1
    3. Patterns: Detection F1, Classification accuracy (+ confusion matrix), End-to-end F1
    4. Aggregation per level and global, by pooling (counts summed over runs and databases)

Metrics are computed in mode "normalized": format differences in paths and
pattern names are removed by normalize.py before comparing.

Expected layout:
    gt_dir/   easy1_GT.json, medium5_GT.json, ...
    run_dir/  easy1_<model>.json, medium5_<model>.json, ...   (one folder per run)

Usage (terminal or Colab):
    python evaluate.py --gt-dir GT --model qwen3.5-9b --run-dirs run1 run2 run3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from normalize import CANONICAL_PATTERNS, CONFLICT, load, resolve_patterns, to_canonical

THRESHOLD = 0.51
MODES = ("normalized",)   # add "strict" here to also compute metrics on the raw strings
MAIN_METRICS = ("fields_f1", "relationships_f1", "detection_f1",
                "classification_acc", "e2e_f1")


# --------------------------------------------------------------------------
# 1. Basic counting
# --------------------------------------------------------------------------

def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from counts. Empty denominators give 0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def compare_sets(gt: set, pred: set) -> tuple[int, int, int]:
    """TP = in both, FP = only in pred, FN = only in GT."""
    return len(gt & pred), len(pred - gt), len(gt - pred)


# --------------------------------------------------------------------------
# 2. Metrics for one (GT, prediction) pair
# --------------------------------------------------------------------------

def evaluate_pair(gt, pred, threshold: float = THRESHOLD) -> tuple[dict, Counter]:
    row = {}

    # 1. Fields and 2. Relationships: plain F1 on sets
    for name, g, p in (("fields", gt.fields, pred.fields),
                       ("relationships", gt.relationships, pred.relationships)):
        tp, fp, fn = compare_sets(g, p)
        row[f"{name}_tp"], row[f"{name}_fp"], row[f"{name}_fn"] = tp, fp, fn
        row[f"{name}_f1"] = prf(tp, fp, fn)[2]

    # 3. Patterns: one label per path
    G = resolve_patterns(gt, threshold)      # {(coll, path): label}
    P = resolve_patterns(pred, threshold)
    found = G.keys() & P.keys()                          # paths found by the model
    D = len(found)                                       # detection TP
    C = sum(1 for k in found if G[k] == P[k])            # right path AND right pattern

    # 3.1 Detection F1: only paths
    row["detection_tp"], row["detection_fp"], row["detection_fn"] = D, len(P) - D, len(G) - D
    row["detection_f1"] = prf(D, len(P) - D, len(G) - D)[2]

    # 3.2 Classification accuracy: only on found paths
    row["classification_correct"], row["classification_found"] = C, D
    row["classification_acc"] = C / D if D else 0.0

    # 3.3 End-to-end F1: (path, pattern) pairs
    row["e2e_tp"], row["e2e_fp"], row["e2e_fn"] = C, len(P) - C, len(G) - C
    row["e2e_f1"] = prf(C, len(P) - C, len(G) - C)[2]

    # confusion matrix on found paths: (true pattern, predicted pattern) -> count
    confusion = Counter((G[k], P[k]) for k in found)
    return row, confusion


# --------------------------------------------------------------------------
# 3. Loop over databases and runs
# --------------------------------------------------------------------------

def level_of(db: str) -> str:
    return re.match(r"[a-z]+", db).group(0)          # "medium5" -> "medium"


def run_all(gt_dir: Path, run_dirs: list[Path], model: str, threshold: float):
    rows, issues = [], {}
    confusion = {m: Counter() for m in MODES}

    for gt_path in sorted(gt_dir.glob("*_GT.json")):
        db = gt_path.name.split("_")[0]
        for run_dir in run_dirs:
            pred_path = run_dir / f"{db}_{model}.json"
            if not pred_path.exists():
                print(f"[warning] missing {pred_path}, skipped")
                continue
            for mode in MODES:
                gt = load(str(gt_path), mode)
                base = {"model": model, "db": db, "level": level_of(db),
                        "run": run_dir.name, "mode": mode}
                try:
                    pred = load(str(pred_path), mode)
                except (ValueError, json.JSONDecodeError) as e:
                    # not parsable = empty prediction: every GT item becomes a FN, metrics are 0
                    row, _ = evaluate_pair(gt, to_canonical({}, mode), threshold)
                    rows.append({**base, "parse_ok": False, **row})
                    issues[f"{db}/{run_dir.name}"] = [f"not parsable: {e}"]
                    continue
                row, conf = evaluate_pair(gt, pred, threshold)
                rows.append({**base, "parse_ok": True, **row})
                confusion[mode] += conf
                if mode == "normalized" and pred.issues:
                    issues[f"{db}/{run_dir.name}"] = pred.issues
    return rows, confusion, issues


# --------------------------------------------------------------------------
# 4. Aggregation by pooling: counts are summed over runs and databases,
#    then every metric is computed once on the totals
# --------------------------------------------------------------------------

def pool(rows: list[dict]) -> dict:
    t = lambda k: sum(r[k] for r in rows)
    D, C = t("detection_tp"), t("e2e_tp")
    out = {}
    for name in ("fields", "relationships", "detection", "e2e"):
        tp, fp, fn = t(f"{name}_tp"), t(f"{name}_fp"), t(f"{name}_fn")
        out[f"{name}_tp"], out[f"{name}_fp"], out[f"{name}_fn"] = tp, fp, fn
        out[f"{name}_f1"] = round(prf(tp, fp, fn)[2], 3)
    out["classification_acc"] = round(C / D, 3) if D else 0.0
    return out


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (per database pooled over its runs, per level and global pooled over everything)."""
    per_db, per_group = defaultdict(list), defaultdict(list)
    for r in rows:
        per_db[(r["mode"], r["level"], r["db"])].append(r)
        per_group[(r["mode"], r["level"])].append(r)
        per_group[(r["mode"], "global")].append(r)

    db_table = [{"mode": m, "level": lv, "db": db, "n_runs": len(rs), **pool(rs)}
                for (m, lv, db), rs in sorted(per_db.items())]
    order = {"easy": 0, "medium": 1, "hard": 2, "global": 3}
    summary = [{"mode": m, "group": g, "n_db": len({r["db"] for r in rs}), "n_runs": len(rs), **pool(rs)}
               for (m, g), rs in sorted(per_group.items(), key=lambda x: (x[0][0], order.get(x[0][1], 9)))]
    return db_table, summary


# --------------------------------------------------------------------------
# 5. Output files
# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))   # union of keys, in order
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_confusion(path: Path, conf: Counter) -> None:
    """Rows = GT pattern, columns = predicted pattern (+ Conflict)."""
    pred_labels = CANONICAL_PATTERNS + [CONFLICT]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ Pred"] + pred_labels)
        for g in CANONICAL_PATTERNS:
            w.writerow([g] + [conf.get((g, p), 0) for p in pred_labels])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--run-dirs", required=True, nargs="+", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, confusion, issues = run_all(args.gt_dir, args.run_dirs, args.model, args.threshold)
    db_table, summary = aggregate(rows)

    tag = args.model
    write_csv(args.out_dir / f"per_run_{tag}.csv", rows)       # one row per db x run x mode
    write_csv(args.out_dir / f"per_db_{tag}.csv", db_table)    # pooled over the runs of each db
    write_csv(args.out_dir / f"summary_{tag}.csv", summary)    # pooled per level and global
    for mode in MODES:
        write_confusion(args.out_dir / f"confusion_{mode}_{tag}.csv", confusion[mode])
    with open(args.out_dir / f"issues_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)

    cols = ["mode", "group", "n_db", "n_runs"] + list(MAIN_METRICS)
    for row in summary:
        print({k: row[k] for k in cols})


if __name__ == "__main__":
    main()
