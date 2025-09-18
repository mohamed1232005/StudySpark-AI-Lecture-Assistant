# modules/metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass
from rouge_score import rouge_scorer
import sacrebleu

@dataclass
class RougeResult:
    r1_f: float
    r2_f: float
    rl_f: float
    r1_p: float
    r1_r: float
    r2_p: float
    r2_r: float
    rl_p: float
    rl_r: float

def compute_rouge(pred: str, ref: str) -> RougeResult:
    """
    ROUGE from google/rouge-score with stemming and Lsum for multi-sentence summaries.
    Returns F1 (plus P/R) for ROUGE-1, ROUGE-2, ROUGE-Lsum.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    scores = scorer.score(ref, pred)  # target=ref, prediction=pred

    r1 = scores["rouge1"]
    r2 = scores["rouge2"]
    rl = scores["rougeLsum"]
    return RougeResult(
        r1_f=r1.fmeasure, r2_f=r2.fmeasure, rl_f=rl.fmeasure,
        r1_p=r1.precision, r1_r=r1.recall,
        r2_p=r2.precision, r2_r=r2.recall,
        rl_p=rl.precision, rl_r=rl.recall
    )

def compute_bleu(pred: str, ref: str) -> float:
    """
    SacreBLEU corpus-compatible BLEU (case-insensitive default).
    BLEU is defined at corpus level, but using sentence-level here for convenience.
    """
    # sacrebleu expects: hypotheses list, references list-of-lists
    bleu = sacrebleu.corpus_bleu([pred], [[ref]])
    return bleu.score  # 0..100

def evaluate_many(preds: List[str], refs: List[str]) -> Tuple[List[Dict], Dict]:
    """
    Evaluate a list of predictions vs references.
    Returns (per_doc_rows, macro_avg_row).
    """
    assert len(preds) == len(refs), "preds and refs must have same length"

    rows: List[Dict] = []
    sums = {
        "rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0,
        "rouge1_p": 0.0, "rouge1_r": 0.0,
        "rouge2_p": 0.0, "rouge2_r": 0.0,
        "rougeL_p": 0.0, "rougeL_r": 0.0,
        "bleu": 0.0,
    }

    n = len(preds)
    for i, (p, r) in enumerate(zip(preds, refs)):
        rr = compute_rouge(p, r)
        bleu = compute_bleu(p, r)

        row = {
            "doc_idx": i,
            "rouge1_f": rr.r1_f, "rouge2_f": rr.r2_f, "rougeL_f": rr.rl_f,
            "rouge1_p": rr.r1_p, "rouge1_r": rr.r1_r,
            "rouge2_p": rr.r2_p, "rouge2_r": rr.r2_r,
            "rougeL_p": rr.rl_p, "rougeL_r": rr.rl_r,
            "bleu": bleu,
        }
        rows.append(row)
        for k in sums:
            sums[k] += row[k]

    if n > 0:
        macro = {k: v / n for k, v in sums.items()}
    else:
        macro = {k: 0.0 for k in sums}
    return rows, macro
