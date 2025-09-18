from typing import List
import random
import re
import pandas as pd
import yake

from .text_processing import split_sentences
from config.settings import YAKE_TOPK


def extract_keywords(text: str, topk: int = YAKE_TOPK) -> List[str]:
    """Extract simple keywords with YAKE; filter trivial tokens."""
    kw_extractor = yake.KeywordExtractor(n=1, top=topk)
    kws = [k for k, _ in kw_extractor.extract_keywords(text)]
    # filter trivial tokens
    return [k for k in kws if len(k) > 2 and re.search(r"[A-Za-z]", k)]


def _pick_answer(sent: str, kws: List[str]):
    """Pick the first keyword that appears verbatim in the sentence."""
    for k in kws:
        if re.search(rf"\b{re.escape(k)}\b", sent, re.I):
            return k
    return None


def make_fib(sent: str, ans: str):
    q = re.sub(rf"\b{re.escape(ans)}\b", "____", sent, flags=re.I)
    return {"type": "FIB", "question": q, "answer": ans}


def make_tf(sent: str, ans: str, kws: List[str]):
    flip = random.random() < 0.5 and len(kws) > 1
    if flip:
        distractor = random.choice([k for k in kws if k.lower() != ans.lower()])
        stmt = re.sub(rf"\b{re.escape(ans)}\b", distractor, sent, flags=re.I)
        return {"type": "T/F", "question": f"True or False: {stmt}", "answer": "False"}
    return {"type": "T/F", "question": f"True or False: {sent}", "answer": "True"}


def make_mcq(sent: str, ans: str, kws: List[str]):
    pool = [k for k in kws if k.lower() != ans.lower()]
    if len(pool) == 0:
        pool = ["NLP", "Language", "Model", "Token"]  # fallback distractors
    distractors = random.sample(pool, k=min(3, len(pool)))
    options = distractors + [ans]
    random.shuffle(options)
    stem = re.sub(rf"\b{re.escape(ans)}\b", "_____", sent, flags=re.I)
    q = f"{stem}\nChoose the best option."
    return {"type": "MCQ", "question": q, "options": options, "answer": ans}


def build_question_bank_from_summary(summary_text: str, max_per_type: int = 10) -> pd.DataFrame:
    """
    Build a mixed question bank (FIB, T/F, MCQ) from a summary.
    Returns a normalized DataFrame with columns:
      type, question, option_a, option_b, option_c, option_d, answer
    """
    sentences = split_sentences(summary_text)
    keywords = extract_keywords(summary_text)
    rows = []
    tf_cnt = fib_cnt = mcq_cnt = 0

    for s in sentences:
        ans = _pick_answer(s, keywords)
        if not ans:
            continue

        if fib_cnt < max_per_type:
            fb = make_fib(s, ans)
            rows.append({
                "type": fb["type"], "question": fb["question"],
                "option_a": "", "option_b": "", "option_c": "", "option_d": "",
                "answer": fb["answer"]
            })
            fib_cnt += 1

        if tf_cnt < max_per_type:
            tf = make_tf(s, ans, keywords)
            rows.append({
                "type": tf["type"], "question": tf["question"],
                "option_a": "True", "option_b": "False", "option_c": "", "option_d": "",
                "answer": tf["answer"]
            })
            tf_cnt += 1

        if mcq_cnt < max_per_type:
            mcq = make_mcq(s, ans, keywords)
            opts = (mcq["options"] + ["", "", "", ""])[:4]
            rows.append({
                "type": "MCQ", "question": mcq["question"],
                "option_a": opts[0], "option_b": opts[1], "option_c": opts[2], "option_d": opts[3],
                "answer": mcq["answer"]
            })
            mcq_cnt += 1

        if tf_cnt >= max_per_type and fib_cnt >= max_per_type and mcq_cnt >= max_per_type:
            break

    return pd.DataFrame(rows)
