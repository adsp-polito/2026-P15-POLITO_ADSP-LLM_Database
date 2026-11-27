"""
normalize.py

Turns a ground truth file or a model output into one canonical structure,
so that the metrics step only has to compare Python sets.

Pipeline:
    raw text -> parse_json -> to_canonical(obj, mode) -> Canonical

Two modes:
    "normalized": string-level fixes (case, separators, array notation,
                  collection prefix, pattern aliases)
    "strict":     same structural adapter, but strings kept as written
                  (only surrounding whitespace removed)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

CANONICAL_PATTERNS = ["Embedding", "Referencing", "ExtendedReference",
                      "Subset", "Array", "Bucket"]

# "extended_reference", "EXTENDED REFERENCE" -> "ExtendedReference" (case/punctuation only)
_PATTERN_ALIASES = {re.sub(r"[^a-z]", "", p.lower()): p for p in CANONICAL_PATTERNS}


# key aliases, so that other models can use slightly different JSON keys
_K_COLLECTIONS = ("collections",)
_K_FIELDS = ("fields",)
_K_RELS = ("relationships", "relations", "references")
_K_PATTERNS = ("patterns",)
_K_PATH = ("path", "field", "field_path")
_K_LABEL = ("pattern", "label", "predicted_pattern")
_K_SCORES = ("scores", "confidence", "probabilities", "probs")
_K_FROM = ("from_field", "from", "source")
_K_TO = ("to_collection", "to", "target")

# .$[] / .$[elem], [] , [*] , [0]; .$ and .<digits> only as a whole segment
_ARRAY_TOKENS = re.compile(r"\.\$\[[^\]]*\](?=\.|$)|\[\*?\d*\]|\.\$(?=\.|$)|\.\d+(?=\.|$)")


# --------------------------------------------------------------------------
# 1. Parsing
# --------------------------------------------------------------------------

def parse_json(text: str) -> dict:
    """Extract the first JSON object from raw model text.

    Handles <think>...</think> blocks and ```json fences.
    Raises ValueError if nothing parsable is found (caller sets F1 = 0).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


# --------------------------------------------------------------------------
# 2. String-level normalization
# --------------------------------------------------------------------------

def norm_name(s: str, mode: str) -> str:
    s = str(s).strip()
    return s.lower() if mode == "normalized" else s


def norm_path(path: str, mode: str) -> str:
    p = str(path).strip()
    if mode == "strict":
        return p
    p = p.lower().replace("/", ".").replace("->", ".").replace(" ", "")
    p = _ARRAY_TOKENS.sub("", p)
    p = re.sub(r"\.{2,}", ".", p).strip(".")
    return p


def norm_pattern(label: str, mode: str) -> str:
    label = str(label).strip()
    if mode == "strict":
        return label
    return _PATTERN_ALIASES.get(re.sub(r"[^a-z]", "", label.lower()), label)


def split_full_path(full: str, mode: str) -> tuple[str, str]:
    """'loans.member_id' -> ('loans', 'member_id'). No dot -> (full, '')."""
    p = norm_path(full, mode)
    coll, _, rest = p.partition(".")
    return norm_name(coll, mode), rest


def strip_collection_prefix(coll: str, path: str, mode: str) -> str:
    """Inside a known collection: 'products.shipping' -> 'shipping'."""
    if mode == "normalized" and path.startswith(coll + "."):
        return path[len(coll) + 1:]
    return path


# --------------------------------------------------------------------------
# 3. Canonical structure
# --------------------------------------------------------------------------

@dataclass
class PatternEntry:
    label: str                       # argmax of the scores (pred) or given label (GT)
    scores: dict[str, float] | None  # scores renormalized to sum 1, None for GT
    p_max: float                     # max renormalized score, 1.0 for GT

@dataclass
class Canonical:
    collections: set[str] = field(default_factory=set)
    fields: set[tuple[str, str]] = field(default_factory=set)               # (coll, path)
    relationships: set[tuple[str, str, str]] = field(default_factory=set)   # (coll, path, to_coll)
    # key = (coll, path); all entries the model gave for that path are kept,
    # resolve_patterns() turns them into one label per path
    patterns: dict[tuple[str, str], list[PatternEntry]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)                         # log only, never changes the content


def _get(d: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _read_scores(raw: dict, mode: str, issues: list[str], where: str) -> dict[str, float]:
    """Map score keys to canonical names, then renormalize so the scores sum to 1.

    Team decision: scores are always normalized to 1 before argmax and threshold.
    Negative scores are set to 0 first. All-zero scores stay all-zero (no prediction).
    """
    scores: dict[str, float] = {}
    for k, v in raw.items():
        name = norm_pattern(k, mode)
        if name not in CANONICAL_PATTERNS:
            issues.append(f"{where}: unknown score key '{k}' ignored")
            continue
        try:
            scores[name] = float(v)
        except (TypeError, ValueError):
            issues.append(f"{where}: non-numeric score '{v}' for '{k}' ignored")
    if any(v < 0 for v in scores.values()):
        issues.append(f"{where}: negative scores set to 0")
        scores = {k: max(v, 0.0) for k, v in scores.items()}
    total = sum(scores.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        issues.append(f"{where}: scores sum to {total:.3f}, renormalized to 1")
        scores = {k: v / total for k, v in scores.items()}
    return scores


# --------------------------------------------------------------------------
# 4. Adapter: any supported JSON layout -> Canonical
# --------------------------------------------------------------------------

def to_canonical(obj: dict, mode: str = "normalized") -> Canonical:
    assert mode in ("normalized", "strict")
    c = Canonical()

    if obj.get("truncated") is True:
        c.issues.append("output flagged as truncated: missing items are expected")

    # collections: ["a", "b"] or [{"name": "a"}, ...]
    for item in _get(obj, _K_COLLECTIONS, []) or []:
        name = item.get("name") if isinstance(item, dict) else item
        if name:
            c.collections.add(norm_name(name, mode))

    def check_coll(coll: str, where: str) -> None:
        if c.collections and coll not in c.collections:
            c.issues.append(f"{where}: '{coll}' is not among the declared collections")

    # fields: {coll: [paths]} | [{"collection": c, "fields": [...]}] | ["coll.path", ...]
    raw_fields = _get(obj, _K_FIELDS, []) or []
    groups: list[tuple[str | None, list]] = []
    if isinstance(raw_fields, dict):
        groups = list(raw_fields.items())
    elif isinstance(raw_fields, list):
        for item in raw_fields:
            if isinstance(item, dict):
                groups.append((item.get("collection"), item.get("fields", [])))
            elif isinstance(item, str):
                groups.append((None, [item]))
    n_raw = 0
    for coll, paths in groups:
        for p in paths or []:
            n_raw += 1
            if coll is None:
                ck, pk = split_full_path(p, mode)
            else:
                ck = norm_name(coll, mode)
                pk = strip_collection_prefix(ck, norm_path(p, mode), mode)
            check_coll(ck, f"field '{p}'")
            c.fields.add((ck, pk))
    if n_raw > len(c.fields):
        c.issues.append(f"fields: {n_raw - len(c.fields)} identical entries counted once")

    # relationships: [{"from_field": "coll.path", "to_collection": "x"}]
    for r in _get(obj, _K_RELS, []) or []:
        if not isinstance(r, dict):
            c.issues.append(f"relationship skipped, not an object: {r!r}")
            continue
        src, dst = _get(r, _K_FROM), _get(r, _K_TO)
        if not src or not dst:
            c.issues.append(f"relationship skipped, missing keys: {r!r}")
            continue
        ck, pk = split_full_path(src, mode)
        dst_coll, _ = split_full_path(dst, mode)   # "members._id" -> "members"
        check_coll(ck, f"relationship from '{src}'")
        c.relationships.add((ck, pk, dst_coll))

    # patterns: GT has "pattern", predictions have "scores" (or a plain label)
    for p in _get(obj, _K_PATTERNS, []) or []:
        if not isinstance(p, dict) or _get(p, _K_PATH) is None:
            c.issues.append(f"pattern skipped, no path: {p!r}")
            continue
        raw_path = _get(p, _K_PATH)
        coll, path = split_full_path(raw_path, mode)
        check_coll(coll, f"pattern '{raw_path}'")
        if not path:
            c.issues.append(f"pattern '{raw_path}' has no field path (collection-level label)")

        raw_scores = _get(p, _K_SCORES)
        if isinstance(raw_scores, dict) and raw_scores:
            scores = _read_scores(raw_scores, mode, c.issues, raw_path)
            if not scores or max(scores.values()) <= 0:
                c.issues.append(f"{raw_path}: no positive score, no prediction")
                continue
            best = max(scores.values())
            tied = [k for k in CANONICAL_PATTERNS if scores.get(k) == best]
            entry = PatternEntry(tied[0], scores, best)
        elif _get(p, _K_LABEL):
            entry = PatternEntry(norm_pattern(_get(p, _K_LABEL), mode), None, 1.0)
        else:
            c.issues.append(f"{raw_path}: no label and no scores, skipped")
            continue

        entries = c.patterns.setdefault((coll, path), [])
        same = [e for e in entries if e.label == entry.label]
        if same:
            c.issues.append(f"{raw_path}: repeated with the same label ({entry.label}), counted once")
            if entry.p_max > same[0].p_max:
                entries[entries.index(same[0])] = entry
            continue
        if entries:
            c.issues.append(f"{raw_path}: different labels "
                            f"{[e.label for e in entries] + [entry.label]}, resolved after threshold")
        entries.append(entry)

    return c


CONFLICT = "Conflict"   # never matches a GT label


def resolve_patterns(c: Canonical, threshold: float) -> dict[tuple[str, str], str]:
    """One label per path, as required by the evaluation (one path, one pattern).

    1. entries with p_max < threshold are dropped (the model abstains on them)
    2. no entry left       -> path not predicted (detection FN if in GT)
       one label left      -> that label
       several labels left -> CONFLICT: path found, classification wrong
    GT entries have p_max = 1.0, so the threshold never affects them.
    """
    out: dict[tuple[str, str], str] = {}
    for key, entries in c.patterns.items():
        for e in entries:
            if e.scores and e.p_max >= threshold:
                tied = [k for k, v in e.scores.items() if v == e.p_max]
                if len(tied) > 1:
                    c.issues.append(f"{key[0]}.{key[1]}: tie between {tied} above threshold, {e.label} taken")
        kept = {e.label for e in entries if e.p_max >= threshold}
        if len(kept) == 1:
            out[key] = kept.pop()
        elif len(kept) > 1:
            out[key] = CONFLICT
    return out


def load(path: str, mode: str = "normalized") -> Canonical:
    with open(path, encoding="utf-8") as f:
        return to_canonical(parse_json(f.read()), mode)
