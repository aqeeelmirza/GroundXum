#!/usr/bin/env python3
"""
Shared loaders for the GroundXum reproducibility sweeps.

Field conventions are taken verbatim from master_analysis.py so these scripts
read the SAME files the paper's main tables were built from:

  grounding JSONL : record has "summary_id" and "entities"=[{id,type,surface,
                    malformed, grounding:{max,mean,scores,argmax_frame,...}}]
  vlm JSONL       : records keyed by "entity_id" (already "{summary_id}__{ent_id}")
                    with "vlm_judgment" in {"P","A","?"}
  human JSONL     : records keyed by "entity_id" with "judgment" in {"P","A","?"}

Entity key = f"{summary_id}__{ent_id}"  (same as master_analysis.entity_key)
"""
import json


def load_jsonl(fp):
    rows = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_mal(v):
    return v in (True, "True", "true")


def entity_key(summary_id, ent_id):
    return f"{summary_id}__{ent_id}"


def load_scores(grounding_fp):
    """entity_key -> dict(max, mean, scores, type, surface, malformed).

    Skips entities with no grounding (degenerate rows, malformed-skipped, errors).
    """
    out = {}
    for d in load_jsonl(grounding_fp):
        if "_error" in d:
            # whole-record error: still may have entities with no grounding; skip them below
            pass
        sid = d.get("summary_id", "")
        for ent in d.get("entities", []):
            if not isinstance(ent, dict):
                continue
            g = ent.get("grounding")
            if not isinstance(g, dict) or g.get("max") is None:
                continue
            out[entity_key(sid, ent.get("id", "?"))] = {
                "max": g.get("max"),
                "mean": g.get("mean"),
                "scores": g.get("scores"),
                "type": ent.get("type", ""),
                "surface": ent.get("surface", ""),
                "malformed": is_mal(ent.get("malformed")),
            }
    return out


def load_vlm(vlm_fp):
    """entity_key -> judgment in {'P','A','?'}."""
    out = {}
    for d in load_jsonl(vlm_fp):
        if "_error" in d or "vlm_judgment" not in d:
            continue
        out[d["entity_id"]] = d["vlm_judgment"]
    return out


def load_human(human_fp):
    """entity_key -> judgment in {'P','A','?'}."""
    out = {}
    for d in load_jsonl(human_fp):
        j = d.get("judgment")
        if j is not None:
            out[d["entity_id"]] = j
    return out


def prf(pred_present, gold):
    """pred_present: dict key->bool ; gold: dict key->'P'/'A' (others ignored).

    Returns (precision, recall, f1, accuracy, n) over keys present in BOTH and
    with gold in {'P','A'}.
    """
    tp = fp = fn = tn = 0
    for k, g in gold.items():
        if g not in ("P", "A") or k not in pred_present:
            continue
        p = pred_present[k]
        if p and g == "P":
            tp += 1
        elif p and g == "A":
            fp += 1
        elif (not p) and g == "P":
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return prec, rec, f1, acc, n


def cohens_kappa(a, b, labels):
    """Two label dicts key->label. Kappa over shared keys, restricted to `labels`."""
    keys = [k for k in a if k in b and a[k] in labels and b[k] in labels]
    n = len(keys)
    if n == 0:
        return None, 0
    agree = sum(1 for k in keys if a[k] == b[k])
    po = agree / n
    pe = 0.0
    for L in labels:
        pa = sum(1 for k in keys if a[k] == L) / n
        pb = sum(1 for k in keys if b[k] == L) / n
        pe += pa * pb
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 1.0
    return kappa, n


def linspace(lo, hi, n):
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]