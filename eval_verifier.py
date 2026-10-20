#!/usr/bin/env python3
"""Verifier evaluation harness.

Measures precision, recall and false-positive rate for the critic against a
labelled set of claims. The set includes both corrupted claims (to test
detection) and untouched true claims (to test false positives), because a critic
that never rejects anything would score perfectly on recall of errors but
terribly on false positives — and the two must be measured separately.

Run offline (local critic only):
    python eval_verifier.py

Run with the configured model critic (live):
    python eval_verifier.py --live

Run on specific seeds:
    python eval_verifier.py 81 217 25 --live
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass

from narrator import _context_for_seed, local_narrative, narrator_from_env
from schemas import (
    NarrativeClaim,
    PersonaId,
    PersonaTake,
    Reveal,
    RevealNarrative,
    Verdict,
)
from verifier import (
    VerificationContext,
    Verifier,
    apply_verdicts,
    claim_roster,
    verifier_from_env,
)


@dataclass(frozen=True)
class Probe:
    kind: str
    persona: PersonaId
    original_claim_id: int
    claim: NarrativeClaim
    true_verdict: Verdict
    note: str = ""


def _swap_team(text: str, home: str, away: str) -> str:
    if home in text and away not in text:
        return text.replace(home, away)
    if away in text and home not in text:
        return text.replace(away, home)
    if home in text and away in text:
        return text.replace(home, away)
    return text


def _insert_number(text: str, num: float) -> str:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not nums:
        return f"{text} ({num:g})"
    return re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(num), text, 1)


def _replace_player(text: str, replacement: str) -> str:
    m = re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text)
    if m:
        return re.sub(r"\b" + re.escape(m[0]) + r"\b", replacement, text, 1)
    return text


def _corrupt_location(text: str) -> str:
    if "30 metres" in text or "30 m" in text:
        return text.replace("30 m", "5 m").replace("30 metres", "5 metres")
    if "5 m" in text:
        return text.replace("5 m", "30 m")
    if "14 m" in text:
        return text.replace("14 m", "40 m")
    return text + " from 30 metres out"


def _build_labelled_probes(draft: RevealNarrative, ctx) -> list[Probe]:
    home = ctx.summary.home_name
    away = ctx.summary.away_name
    probes: list[Probe] = []
    roster = claim_roster(draft)
    for persona, claim in roster:
        # True probe
        probes.append(
            Probe(
                kind="true",
                persona=persona,
                original_claim_id=claim.claim_id,
                claim=claim.model_copy(update={"claim_id": 0}),
                true_verdict=Verdict.SUPPORTED,
            )
        )
        # Team swap
        swapped = _swap_team(claim.text, home, away)
        if swapped != claim.text and (home in swapped or away in swapped) and home in claim.text and away in claim.text:
            # only when claim mentions a team and we can swap meaningfully
            pass
        if swapped != claim.text and claim.event_ids and (home in swapped or away in swapped):
            probes.append(
                Probe(
                    kind="swap_team",
                    persona=persona,
                    original_claim_id=claim.claim_id,
                    claim=claim.model_copy(update={"claim_id": 0, "text": swapped}),
                    true_verdict=Verdict.MISATTRIBUTED,
                    note="team swapped",
                )
            )
        # Wrong number
        if any(ch.isdigit() for ch in claim.text):
            numd = _insert_number(claim.text, 9.99)
            if numd != claim.text:
                probes.append(
                    Probe(
                        kind="wrong_number",
                        persona=persona,
                        original_claim_id=claim.claim_id,
                        claim=claim.model_copy(update={"claim_id": 0, "text": numd}),
                        true_verdict=Verdict.UNSUPPORTED,
                        note="invented number",
                    )
                )
        # Wrong player
        m = re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", claim.text)
        if m and claim.event_ids:
            mutated = _replace_player(claim.text, "Alex Johnson")
            if mutated != claim.text:
                probes.append(
                    Probe(
                        kind="wrong_player",
                        persona=persona,
                        original_claim_id=claim.claim_id,
                        claim=claim.model_copy(update={"claim_id": 0, "text": mutated}),
                        true_verdict=Verdict.MISATTRIBUTED,
                        note="wrong player",
                    )
                )
        # Wrong location
        locd = _corrupt_location(claim.text)
        if locd != claim.text and claim.event_ids:
            probes.append(
                Probe(
                    kind="wrong_location",
                    persona=persona,
                    original_claim_id=claim.claim_id,
                    claim=claim.model_copy(update={"claim_id": 0, "text": locd}),
                    true_verdict=Verdict.UNSUPPORTED,
                    note="wrong distance",
                )
            )
        if len(probes) > 180:
            break
    return probes


def _build_test_draft(base: RevealNarrative, probes: list[Probe]) -> RevealNarrative:
    by_persona: dict[PersonaId, list[NarrativeClaim]] = {}
    for p in base.personas:
        by_persona[p.persona] = []
    for pr in probes:
        by_persona[pr.persona].append(pr.claim.model_copy())
    takes = []
    for p in base.personas:
        takes.append(
            PersonaTake(
                persona=p.persona,
                headline=p.headline,
                claims=by_persona.get(p.persona, []),
            )
        )
    d = base.model_copy(update={"personas": takes})
    return d


def _score(probes: list[Probe], verdicts_by_original_id: dict[int, Verdict]) -> dict:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # map original id to probe? could have multiple; just check
    by_o = {}
    for pr in probes:
        by_o[pr.original_claim_id] = pr
    for orig_id, v in verdicts_by_original_id.items():
        pr = by_o.get(orig_id)
        if not pr:
            continue
        if pr.true_verdict is Verdict.SUPPORTED:
            if v is Verdict.SUPPORTED:
                tn += 1
            else:
                fp += 1
        else:
            if v is Verdict.SUPPORTED:
                fn += 1
            else:
                tp += 1
    total_true_pos = sum(1 for p in probes if p.true_verdict is not Verdict.SUPPORTED)
    total_true_neg = sum(1 for p in probes if p.true_verdict is Verdict.SUPPORTED)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "total": len(probes),
        "true_neg": tn,
        "false_pos": fp,
        "true_pos": tp,
        "false_neg": fn,
        "total_true_neg": total_true_neg,
        "total_true_pos": total_true_pos,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / total_true_neg if total_true_neg else 0.0,
    }


async def _evaluate_seed(
    seed: int,
    reveal: Reveal,
    critic: Verifier,
    live: bool,
) -> dict:
    ctx = _context_for_seed(seed, reveal)
    base = local_narrative(ctx)
    probes = _build_labelled_probes(base, ctx)
    test = _build_test_draft(base, probes)
    review = await critic.review(
        VerificationContext(draft=test, summary=ctx.summary, insights=ctx.insights)
    )
    final = apply_verdicts(test, review)
    # map back by original id
    orig_by_new = {}
    # rebuild mapping: in the test draft the claims appear in order? easier: just collect all
    # but easier: track by iterating probes in order they were added
    # simpler: just score based on final claims
    by_orig = {}
    # need mapping; easier to just check each probe's text
    probe_by_text = {}
    for pr in probes:
        probe_by_text[pr.claim.text] = pr
    for take in final.personas:
        for claim in take.claims:
            pr = probe_by_text.get(claim.text)
            if not pr:
                continue
            by_orig[pr.original_claim_id] = claim.ruling.verdict if claim.ruling else Verdict.SUPPORTED
    s = _score(probes, by_orig)
    s["seed"] = seed
    s["reveal"] = reveal.value
    s["provider"] = review.provider
    s["model"] = review.model
    s["complete"] = review.complete
    s["by_model"] = review.by_model
    s["disputed_claims"] = final.disputed_claims
    return s


def _print_report(results: list[dict]) -> None:
    if not results:
        print("no results")
        return
    totals = {
        "total": 0,
        "tn": 0,
        "fp": 0,
        "tp": 0,
        "fn": 0,
        "ttn": 0,
        "ttp": 0,
    }
    for r in results:
        totals["total"] += r["total"]
        totals["tn"] += r["true_neg"]
        totals["fp"] += r["false_pos"]
        totals["tp"] += r["true_pos"]
        totals["fn"] += r["false_neg"]
        totals["ttn"] += r["total_true_neg"]
        totals["ttp"] += r["total_true_pos"]
    prec = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 1.0
    rec = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = totals["fp"] / totals["ttn"] if totals["ttn"] else 0.0
    print("=== Verifier Evaluation (labelled: true + corrupted claims) ===")
    for r in results:
        print(
            f"{r['seed']:>4} {r['reveal']:<2}  claims={r['total']:>3}  "
            f"TN={r['true_neg']:>3} FP={r['false_pos']:>3} (FPR={r['false_positive_rate']:.3f})  "
            f"TP={r['true_pos']:>3} FN={r['false_neg']:>3}  "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}  "
            f"prov={r['provider']}/{r['model']} complete={r['complete']}"
        )
    print(
        f"\nTOTAL claims={totals['total']}  TN={totals['tn']} FP={totals['fp']} (FPR={fpr:.4f})  "
        f"TP={totals['tp']} FN={totals['fn']}  P={prec:.4f} R={rec:.4f} F1={f1:.4f}"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the critic")
    parser.add_argument("seeds", nargs="*", type=int, default=[81, 217, 25])
    parser.add_argument("--live", action="store_true", help="use configured model critic")
    args = parser.parse_args()

    if args.live:
        critic = verifier_from_env()
    else:
        from verifier import LocalVerifier

        critic = LocalVerifier()

    results: list[dict] = []
    for seed in args.seeds:
        try:
            res = await _evaluate_seed(seed, Reveal.FULL_TIME, critic, args.live)
            results.append(res)
        except Exception as exc:
            print(f"error evaluating seed {seed}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1
    _print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
