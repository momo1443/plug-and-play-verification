#!/usr/bin/env python3
"""
Fine-grained diagnostic evaluation for A8-CS step-200 checkpoint.

Official EM is the primary metric — we NEVER replace it.
Additional diagnostics:
  1. Token-level F1 (HotpotQA official style)
  2. Normalized EM  (lowercase, strip articles, strip punctuation)
  3. Entity-aware EM (normalized + alias containment check)
  4. Alias-error bucketing: how many EM=0 are just answer-realization mismatch?

Usage:
  python finegrained_eval.py \
    /nas/deepresearch/zsb/corhort/project/agenticrl/logs/.../eval_step200/0.jsonl
"""
import sys
import json
import re
import string
from collections import Counter, defaultdict

# ── HotpotQA official normalization ──────────────────────────────────────
def normalize_answer(s: str) -> str:
    """Lowercase, remove articles, punctuation, extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# ── Token F1 ─────────────────────────────────────────────────────────────
def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# ── Alias-aware matching ─────────────────────────────────────────────────
def is_alias_match(pred: str, gold: str) -> bool:
    """
    Conservative alias check — one direction only:
    normalized(gold) is contained in normalized(pred) OR vice versa,
    AND the shorter one has >= 2 tokens (avoid "yes" containing "y").
    """
    pn = normalize_answer(pred)
    gn = normalize_answer(gold)
    if pn == gn:
        return True
    # Must share at least one non-trivial token
    pn_tokens = set(pn.split())
    gn_tokens = set(gn.split())
    shared = pn_tokens & gn_tokens
    if not shared:
        return False
    # Containment check — shorter contained in longer
    if gn in pn or pn in gn:
        shorter = min(len(pn.split()), len(gn.split()))
        return shorter >= 2  # avoid trivial containment like "Lee" in "Lee Hazlewood"
    return False

# ── Partial token overlap (for bucketing) ───────────────────────────────
def token_overlap_ratio(pred: str, gold: str) -> float:
    pn = set(normalize_answer(pred).split())
    gn = set(normalize_answer(gold).split())
    if not gn:
        return 0.0
    return len(pn & gn) / len(gn)

# ── Main ─────────────────────────────────────────────────────────────────
def main(jsonl_path: str):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    yesno_records = []
    entity_records = []
    for r in records:
        gt = str(r["ground_truth"]).strip().lower()
        if gt in ("yes", "no"):
            yesno_records.append(r)
        else:
            entity_records.append(r)

    # ── Compute per-record metrics ──────────────────────────────────────
    results = []
    for r in records:
        pred = str(r["answer"]).strip()
        gold = str(r["ground_truth"]).strip()
        official_em = int(r["score"]) if r["score"] else 0

        norm_em = int(normalize_answer(pred) == normalize_answer(gold))
        f1 = token_f1(pred, gold)
        alias = int(is_alias_match(pred, gold))
        overlap = token_overlap_ratio(pred, gold)

        qtype = "yesno" if gold.lower() in ("yes", "no") else "entity"

        results.append({
            "qid": r.get("official_qid", r.get("dataset_question_id", "")),
            "question": r["question"],
            "pred": pred,
            "gold": gold,
            "qtype": qtype,
            "official_em": official_em,
            "norm_em": norm_em,
            "alias_match": alias,
            "token_f1": f1,
            "token_overlap": overlap,
        })

    # ── Aggregate ───────────────────────────────────────────────────────
    def agg(subset, label):
        n = len(subset)
        if n == 0:
            return
        off_em = sum(r["official_em"] for r in subset) / n * 100
        n_em   = sum(r["norm_em"]    for r in subset) / n * 100
        a_em   = sum(r["alias_match"] for r in subset) / n * 100
        t_f1   = sum(r["token_f1"]   for r in subset) / n * 100
        print(f"  {label:20s}  N={n:5d}  OfficialEM={off_em:5.1f}%  NormEM={n_em:5.1f}%  AliasEM={a_em:5.1f}%  TokenF1={t_f1:5.1f}%")

    print("=" * 90)
    print("  FINE-GRAINED DIAGNOSTIC — A8-CS Step 200 (7405 samples)")
    print("  ⚠️  Official EM is the ONLY primary metric. Others are diagnostic only.")
    print("=" * 90)
    print()
    agg(results, "Overall")
    agg([r for r in results if r["qtype"] == "yesno"], "Yes/No")
    agg([r for r in results if r["qtype"] == "entity"], "Entity")
    print()

    # ── Error bucketing for EM=0 entity answers ────────────────────────
    entity_wrong = [r for r in results if r["qtype"] == "entity" and r["official_em"] == 0]
    n_ent_wrong = len(entity_wrong)

    # Bucket 1: Alias match (norm EM=1 or alias containment)
    alias_only = [r for r in entity_wrong if r["alias_match"] == 1 and r["norm_em"] == 0]
    norm_only  = [r for r in entity_wrong if r["norm_em"] == 1]

    # Bucket 2: High token overlap (>=0.5) but not alias
    partial_overlap = [r for r in entity_wrong
                       if r["alias_match"] == 0 and r["norm_em"] == 0 and r["token_overlap"] >= 0.5]

    # Bucket 3: Some overlap (0.1–0.5)
    low_overlap = [r for r in entity_wrong
                   if r["alias_match"] == 0 and r["norm_em"] == 0
                   and 0.1 <= r["token_overlap"] < 0.5]

    # Bucket 4: Zero or near-zero overlap — truly wrong answer
    no_overlap = [r for r in entity_wrong
                  if r["alias_match"] == 0 and r["norm_em"] == 0 and r["token_overlap"] < 0.1]

    print("─" * 90)
    print("  ERROR BUCKETING: Entity answers with Official EM = 0")
    print(f"  Total entity-wrong: {n_ent_wrong}")
    print()
    print(f"  {'Bucket':35s}  {'Count':>6s}  {'Pct':>6s}  {'Description'}")
    print(f"  {'─'*35}  {'─'*6}  {'─'*6}  {'─'*40}")
    print(f"  {'Norm-EM fixable':35s}  {len(norm_only):6d}  {len(norm_only)/n_ent_wrong*100:5.1f}%  normalize() alone fixes it")
    print(f"  {'Alias match (containment)':35s}  {len(alias_only):6d}  {len(alias_only)/n_ent_wrong*100:5.1f}%  e.g. Lee Hazlewood / Barton Lee Hazlewood")
    print(f"  {'Partial overlap (≥0.5)':35s}  {len(partial_overlap):6d}  {len(partial_overlap)/n_ent_wrong*100:5.1f}%  some shared tokens, possibly related entity")
    print(f"  {'Low overlap (0.1–0.5)':35s}  {len(low_overlap):6d}  {len(low_overlap)/n_ent_wrong*100:5.1f}%  minor token overlap, likely wrong entity")
    print(f"  {'No overlap (<0.1)':35s}  {len(no_overlap):6d}  {len(no_overlap)/n_ent_wrong*100:5.1f}%  completely different answer")
    print()

    # ── Alias-only examples ─────────────────────────────────────────────
    print("─" * 90)
    print("  ALIAS-ERROR EXAMPLES (norm_em=0, alias_match=1)")
    print("  (max 20 shown)")
    print()
    shown = 0
    for r in alias_only[:20]:
        print(f"  Q: {r['question'][:80]}")
        print(f"     pred={r['pred']!r}  gold={r['gold']!r}  overlap={r['token_overlap']:.2f}")
        shown += 1
    if not alias_only:
        print("  (none)")
    print()

    # ── Norm-EM fixable examples ────────────────────────────────────────
    print("─" * 90)
    print("  NORM-EM FIXABLE EXAMPLES (official_em=0, norm_em=1)")
    print("  (max 20 shown)")
    print()
    for r in norm_only[:20]:
        print(f"  Q: {r['question'][:80]}")
        print(f"     pred={r['pred']!r}  gold={r['gold']!r}")
    if not norm_only:
        print("  (none)")
    print()

    # ── Partial overlap examples ────────────────────────────────────────
    print("─" * 90)
    print("  PARTIAL OVERLAP EXAMPLES (overlap ≥ 0.5, not alias)")
    print("  (max 15 shown)")
    print()
    for r in partial_overlap[:15]:
        print(f"  Q: {r['question'][:80]}")
        print(f"     pred={r['pred']!r}  gold={r['gold']!r}  overlap={r['token_overlap']:.2f}  F1={r['token_f1']:.2f}")
    if not partial_overlap:
        print("  (none)")
    print()

    # ── Truly wrong examples ────────────────────────────────────────────
    print("─" * 90)
    print("  TRULY WRONG EXAMPLES (no overlap < 0.1)")
    print("  (max 15 shown)")
    print()
    for r in no_overlap[:15]:
        print(f"  Q: {r['question'][:80]}")
        print(f"     pred={r['pred']!r}  gold={r['gold']!r}")
    if not no_overlap:
        print("  (none)")
    print()

    # ── Summary decomposition ───────────────────────────────────────────
    print("=" * 90)
    print("  DECOMPOSITION SUMMARY")
    print("=" * 90)
    entity_total = len([r for r in results if r["qtype"] == "entity"])
    entity_correct = len([r for r in results if r["qtype"] == "entity" and r["official_em"] == 1])
    yesno_total = len([r for r in results if r["qtype"] == "yesno"])
    yesno_correct = len([r for r in results if r["qtype"] == "yesno" and r["official_em"] == 1])

    answer_realization_errors = len(norm_only) + len(alias_only)
    reasoning_search_errors = len(partial_overlap) + len(low_overlap) + len(no_overlap)

    print(f"""
  Entity answers (N={entity_total}):
    Official EM correct:         {entity_correct:5d}  ({entity_correct/entity_total*100:5.1f}%)
    Answer-realization errors:   {answer_realization_errors:5d}  ({answer_realization_errors/entity_total*100:5.1f}%)  ← alias/norm only
    Reasoning/search errors:     {reasoning_search_errors:5d}  ({reasoning_search_errors/entity_total*100:5.1f}%)  ← truly wrong answer

  If answer-realization were fixed:
    Entity EM would be:          ~{(entity_correct + answer_realization_errors)/entity_total*100:.1f}%
    Overall EM would be:         ~{(yesno_correct + entity_correct + answer_realization_errors)/total*100:.1f}%

  Yes/No answers (N={yesno_total}):
    Official EM correct:         {yesno_correct:5d}  ({yesno_correct/yesno_total*100:5.1f}%)
""")

if __name__ == "__main__":
    main(sys.argv[1])
