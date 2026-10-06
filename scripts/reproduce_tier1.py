#!/usr/bin/env python3
"""reproduce_tier1.py — recompute the published Tier-1 results from data/selftest_cache.jsonl and
check them against the shipped JSONs (data/tier1_summary.json, data/tier1_verdicts.json). The same
pass recomputes the headline slice AUCs and checks them against pinned anchors.

This is the offline reproduction for the repository. It makes no API calls, uses no third-party
text, and does not need the lab archive. The arithmetic mirrors the original analysis scripts
(`analyze_tier1.py` and `analyze_tier1_verdicts.py` from the lab archive) field for field,
including the stratum-level rounding. The only difference is the data source: the hash-keyed
cache instead of the raw corpus.

`./origin_triage.py --reproduce` calls both checks in this module.

Usage:
    python3 scripts/reproduce_tier1.py            # human-readable report + checks (default)
    python3 scripts/reproduce_tier1.py --check    # exit 0 iff everything matches exactly

Exit codes: 0 = checked, all values match; 1 = mismatch or missing input.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data"

# Prior estimates from the study for the slices that already had a Jev origin_choice answer.
# value = (machine-call rate for machine slices, false-positive rate for human slices), one per slice.
PRIOR = {
    ("hc3", "human"): 0.209,
    ("idmgsp", "real"): 0.127,
    ("idmgsp", "galactica"): 0.36,
    ("idmgsp", "gpt2"): 1.00,
    ("idmgsp", "gpt3"): 0.973,
    ("idmgsp", "chatgpt"): 0.993,
    ("idmgsp", "scigen"): 1.00,
    ("editlens_val", "ai_edited"): 0.405,
    ("editlens_val", "ai_generated"): 0.85,
    ("editlens_val", "human_written"): 0.22,
    ("lamp", "pre_edit"): None, ("lamp", "post_edit"): None,
}

# Headline slice figures quoted in README.md and DESIGN.md. check_headline_aucs() recomputes them
# from the cache and compares them with these anchors (set 2026-10-06 from baselines_summary.json).
# non-fiction = the nine non-fiction source families. creative = the paired creative slices
# (LAMP, STP and the main B8 candidate groups; see DESIGN.md §3). The B8 rewrites v1-v3 and the
# D4/D4x arms are separate.
NONFICTION_SOURCES = {"hc3", "arjun", "idmgsp", "editlens_val", "editlens_val_src",
                      "grammarly", "grammarly_src", "editlens_enron", "editlens_llama"}
CREATIVE_HUMAN = {("lamp", "post_edit"), ("b8", "post"), ("stp", "pre_edit")}
CREATIVE_MACHINE = {("lamp", "pre_edit"), ("stp", "post_edit"), ("b8", "pre")}
HEADLINE_ANCHORS = {
    "nonfiction": {"n": 5641, "jev": 0.8725, "el_llama": 0.9574},
    "creative": {"n": 2179, "jev": 0.6138, "el_llama": 0.8237},
}


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auc(scores, labels):
    """Mann-Whitney AUC with tie-averaged ranks; labels 1 = machine."""
    pairs = [(s, l) for s, l in zip(scores, labels) if l is not None and s is not None]
    npos = sum(1 for _, l in pairs if l == 1)
    nneg = len(pairs) - npos
    if not npos or not nneg:
        return None
    allr = rank([s for s, _ in pairs])
    rp = [r for r, (_, l) in zip(allr, pairs) if l == 1]
    return round((sum(rp) - npos * (npos + 1) / 2) / (npos * nneg), 4)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - hw, 4), round(c + hw, 4))


def load_cache(data_dir: Path) -> list[dict]:
    rows = []
    for line in (data_dir / "selftest_cache.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        rows.append({
            "hash": r["hash"], "source": r["source"], "group": r["group"],
            "pair_key": r["pair_key"], "truth": r["truth"],
            "jev_choice": r["jev_choice"], "jev_conf": r["jev_conf"],
            "jev_probs": r["jev_probs"], "el_roberta": r["el_roberta"],
            "el_llama": r["el_llama"], "variants": r.get("variants", []),
        })
    return rows


def build_summary(rows: list[dict], theta: float, el_cut: float) -> dict:
    scored = [r for r in rows if r.get("jev_choice") and r["truth"] is not None]

    def metrics(sub):
        n = len(sub)
        if n == 0:
            return None
        out = {"n": n}
        for tag in ("human", "machine"):
            ss = [r for r in sub if r["truth"] == tag]
            if not ss:
                continue
            k = sum(1 for r in ss if r.get("machine_signal") is True)
            out[f"n_{tag}"] = len(ss)
            out[f"ai_rate_{tag}"] = round(k / len(ss), 4)
            out[f"ci_{tag}"] = wilson(k, len(ss))
        allsub = [r for r in sub if r.get("machine_signal") is not None]
        if allsub:
            out["acc_raw"] = round(sum(1 for r in allsub
                                       if r["machine_signal"] == (r["truth"] == "machine")) / len(allsub), 4)
        gp = [r for r in sub if (r.get("jev_conf") or 0) >= theta]
        out["gate_pass_n"] = len(gp)
        out["coverage"] = round(len(gp) / n, 3)
        emitted = [r for r in sub if r.get("emit")]
        if emitted:
            out["emitted_n"] = len(emitted)
            out["emitted_acc"] = round(sum(1 for r in emitted
                                           if r["emit_machine"] == (r["truth"] == "machine")) / len(emitted), 4)
        els = [r for r in sub if r.get("el_llama") is not None]
        if els:
            k = sum(1 for r in els if (r["el_llama"] > el_cut) == (r["truth"] == "machine"))
            out["el_acc"] = round(k / len(els), 4)
            out["auc_el"] = auc([r["el_llama"] for r in els], [int(r["truth"] == "machine") for r in els])
        jc = [r for r in sub if r.get("jev_probs")]
        if jc:
            p_ai = [r["jev_probs"].get("ai", 0) + r["jev_probs"].get("edited", 0) for r in jc]
            out["auc_jev"] = auc(p_ai, [int(r["truth"] == "machine") for r in jc])
        if els and jc:
            agree = sum(1 for r in els if r.get("el_agrees") is not None and r["el_agrees"]) / len(els)
            out["jev_el_agreement"] = round(agree, 4)
        return out

    for r in rows:
        if not r.get("jev_choice"):
            continue
        r["machine_signal"] = r["jev_choice"] in ("ai", "edited")
        r["gate_pass"] = (r.get("jev_conf") or 0) >= theta
        r["el_machine"] = (r.get("el_llama") or 0) > el_cut if r.get("el_llama") is not None else None
        r["el_agrees"] = (r["el_machine"] == r["machine_signal"]) if r["el_machine"] is not None else None
        r["emit"] = r["gate_pass"] and (r["el_agrees"] is True)
        r["emit_machine"] = r["machine_signal"]
        r["disagree_but_gated"] = r["gate_pass"] and r["el_agrees"] is False

    strata = collections.defaultdict(list)
    for r in scored:
        strata[(r["source"], r["group"])].append(r)

    table = []
    for (src, grp), sub in sorted(strata.items(), key=lambda kv: -len(kv[1])):
        m = metrics(sub)
        prior = PRIOR.get((src, grp))
        m.update(source=src, group=grp, prior_estimate=prior)
        if prior is not None and "ai_rate_machine" in m:
            m["delta_vs_prior"] = round(m["ai_rate_machine"] - prior, 4)
        table.append(m)

    pooled = metrics(scored)
    return {"theta": theta, "el_cut": el_cut, "n_bundle": len(rows), "n_scored": len(scored),
            "n_with_jev": sum(1 for r in rows if r.get("jev_choice")), "pooled": pooled, "strata": table}


def build_verdicts(S: dict, rows: list[dict]) -> dict:
    BAND = 0.05
    BIG = 0.10
    V: dict = {}

    comp = [m for m in S["strata"] if m.get("prior_estimate") is not None and "ai_rate_machine" in m]
    inband = [m for m in comp if abs(m["delta_vs_prior"]) <= BAND]
    bigoff = [m for m in comp if abs(m["delta_vs_prior"]) > BIG]
    frac = len(inband) / len(comp) if comp else 0.0
    V["M1"] = {"verdict": "POSITIVE" if frac >= 0.95 else ("NEGATIVE" if len(bigoff) / max(len(comp), 1) > 0.20 else "INCONCLUSIVE"),
               "in_band": f"{len(inband)}/{len(comp)}", "off_gt5pp": [f"{m['source']}/{m['group']} {m['delta_vs_prior']:+.3f}" for m in comp if m not in inband],
               "off_gt10pp": [f"{m['source']}/{m['group']} {m['delta_vs_prior']:+.3f}" for m in bigoff]}

    hum = [m for m in S["strata"] if m.get("ai_rate_human") is not None]
    creative = [m for m in hum if m["source"] in ("lamp", "b8", "stp")]
    nonfic = [m for m in hum if m["source"] not in ("lamp", "b8", "stp")]
    worst = max(hum, key=lambda m: m["ai_rate_human"])
    V["M2"] = {"verdict": "POSITIVE" if all(m["ai_rate_human"] <= 0.21 for m in hum) else ("NEGATIVE" if any(m["ai_rate_human"] > 0.30 for m in hum) else "INCONCLUSIVE"),
               "worst": f"{worst['source']}/{worst['group']} FPR {worst['ai_rate_human']:.3f}",
               "creative_fiction": {f"{m['source']}/{m['group']}": m["ai_rate_human"] for m in creative},
               "non_fiction_max": max((m["ai_rate_human"] for m in nonfic), default=None),
               "non_fiction_strata": {f"{m['source']}/{m['group']}": m["ai_rate_human"] for m in nonfic}}

    p = S["pooled"]
    singles = {"jev_raw": p.get("acc_raw"), "editlens": p.get("el_acc")}
    best_single = max(v for v in singles.values() if v is not None)
    ens = p.get("emitted_acc")
    V["M3"] = {"verdict": "POSITIVE" if ens is not None and ens >= best_single + 0.02 else ("NEGATIVE" if ens is not None and ens <= best_single else "INCONCLUSIVE"),
               "ensemble_emitted_acc": ens, "best_single": best_single, "singles": singles,
               "coverage": p.get("coverage"), "emitted_n": p.get("emitted_n"),
               "delta_vs_best_single": round(ens - best_single, 4) if ens is not None else None}

    within_rows, cross_rows = [], []
    by = {(m["source"], m["group"]): m for m in S["strata"]}
    for a in (("lamp", "pre_edit", "post_edit"), ("editlens_val", "ai_generated", "ai_edited")):
        ma, mb = by.get((a[0], a[1])), by.get((a[0], a[2]))
        if ma and mb:
            within_rows.append(((ma.get("acc_raw", 0) + mb.get("acc_raw", 0)) / 2, f"{a[0]}/{a[1]}+{a[2]}"))
    nonfic_src = {"hc3", "arjun", "idmgsp", "editlens_val", "editlens_val_src", "grammarly",
                  "grammarly_src", "editlens_enron", "editlens_llama"}
    for m in S["strata"]:
        if m["source"] in nonfic_src and "acc_raw" in m:
            cross_rows.append(m["acc_raw"])
    within = sum(x for x, _ in within_rows) / len(within_rows) if within_rows else None
    cross = sum(cross_rows) / len(cross_rows) if cross_rows else None
    V["M4"] = {"verdict": "POSITIVE" if within is not None and cross is not None and within >= cross else ("NEGATIVE" if within is not None and cross is not None and within < cross - 0.10 else "INCONCLUSIVE"),
               "within_document_acc": round(within, 4) if within is not None else None,
               "within_detail": {k: round(v, 4) for v, k in within_rows},
               "cross_document_acc": round(cross, 4) if cross is not None else None}

    events = []
    for r in rows:
        for ev_seq, src, grp, pk, _bundle_id in r["variants"]:
            if src in ("d4", "d4x"):
                events.append((ev_seq, src, pk, grp, r["hash"]))
    events.sort()
    cells: dict[tuple, dict] = collections.defaultdict(dict)
    for _, src, pk, grp, hh in events:
        cells[(src, pk)][grp] = hh
    choices = {r["hash"]: r["jev_choice"] for r in rows}
    flips_to_human = flips_to_ai = pairs_n = 0
    for sides in cells.values():
        if "original" in sides and "humanized" in sides:
            a, b = choices.get(sides["original"]), choices.get(sides["humanized"])
            if a and b:
                pairs_n += 1
                ma, mb = a in ("ai", "edited"), b in ("ai", "edited")
                if ma and not mb:
                    flips_to_human += 1
                if mb and not ma:
                    flips_to_ai += 1
    keep = 1 - flips_to_human / pairs_n if pairs_n else None
    V["M5"] = {"verdict": "POSITIVE" if keep is not None and keep >= 0.98 else ("NEGATIVE" if keep is not None and keep <= 0.90 else "INCONCLUSIVE"),
               "paired_texts": pairs_n, "flips_to_human": flips_to_human, "flips_to_ai": flips_to_ai,
               "keep_machine_rate": round(keep, 4) if keep is not None else None}
    return V


def diff(a, b, path="", out=None, tol=1e-9):
    """Collect human-readable differences between two JSON values (a = recomputed, b = shipped)."""
    if out is None:
        out = []
    if isinstance(b, dict):
        if not isinstance(a, dict):
            out.append(f"{path}: type {type(a).__name__} != dict")
            return out
        for k in b:
            if k not in a:
                out.append(f"{path}.{k}: missing in recomputed")
            else:
                diff(a[k], b[k], f"{path}.{k}", out, tol)
        for k in a:
            if k not in b:
                out.append(f"{path}.{k}: extra in recomputed")
    elif isinstance(b, list):
        if not isinstance(a, (list, tuple)) or len(a) != len(b):
            out.append(f"{path}: list mismatch (len {len(a) if isinstance(a, (list, tuple)) else '?'} vs {len(b)})")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        if not isinstance(a, (int, float)) or isinstance(a, bool) or abs(a - b) > tol:
            out.append(f"{path}: {a!r} != {b!r}")
    else:
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
    return out


def check_summary_and_verdicts(repo: Path) -> tuple[list, dict, dict]:
    """Recompute the summary and the verdicts. Return (diffs, summary, verdicts)."""
    data = repo / "data"
    for name in ("selftest_cache.jsonl", "tier1_summary.json", "tier1_verdicts.json"):
        if not (data / name).exists():
            return [f"missing data/{name}"], {}, {}
    metric = json.loads((repo / "frozen_metric.json").read_text(encoding="utf-8"))
    theta = metric["reference_pool_for_theta"]["computed_theta"]
    el_cut = 0.5
    rows = load_cache(data)
    S = build_summary(rows, theta, el_cut)
    V = build_verdicts(S, rows)
    S_shipped = json.loads((data / "tier1_summary.json").read_text(encoding="utf-8"))
    V_shipped = json.loads((data / "tier1_verdicts.json").read_text(encoding="utf-8"))

    diffs = []
    for k in ("theta", "el_cut", "n_bundle", "n_scored", "n_with_jev"):
        if S[k] != S_shipped.get(k):
            diffs.append(f"summary.{k}: {S[k]!r} != {S_shipped.get(k)!r}")
    diffs += diff(S["pooled"], S_shipped["pooled"], "summary.pooled")
    rec_by = {f"{m['source']}/{m['group']}": m for m in S["strata"]}
    ship_by = {f"{m['source']}/{m['group']}": m for m in S_shipped["strata"]}
    if set(rec_by) != set(ship_by):
        diffs.append(f"summary.strata keys differ: only-recomputed {sorted(set(rec_by) - set(ship_by))}, "
                     f"only-shipped {sorted(set(ship_by) - set(rec_by))}")
    for k in sorted(set(rec_by) & set(ship_by)):
        diffs += diff(rec_by[k], ship_by[k], f"summary.strata[{k}]")
    diffs += diff(V, V_shipped, "verdicts")
    return diffs, S, V


def _pool_aucs(rows: list, keep) -> tuple:
    sub = [r for r in rows if keep(r) and r["truth"] and r["jev_probs"]]
    jev = [r["jev_probs"].get("ai", 0) + r["jev_probs"].get("edited", 0) for r in sub]
    lab = [1 if r["truth"] == "machine" else 0 for r in sub]
    el = [r for r in sub if r["el_llama"] is not None]
    return (len(sub), auc(jev, lab),
            auc([r["el_llama"] for r in el], [1 if r["truth"] == "machine" else 0 for r in el]))


def check_headline_aucs(repo: Path) -> list:
    """Recompute the headline slice AUCs from the cache. Compare them with the pinned anchors."""
    if not (repo / "data" / "selftest_cache.jsonl").exists():
        return ["missing data/selftest_cache.jsonl"]
    rows = load_cache(repo / "data")
    diffs = []
    for name, keep in (("nonfiction", lambda r: r["source"] in NONFICTION_SOURCES),
                       ("creative", lambda r: (r["source"], r["group"]) in (CREATIVE_HUMAN | CREATIVE_MACHINE))):
        n, jev_auc, el_auc = _pool_aucs(rows, keep)
        a = HEADLINE_ANCHORS[name]
        if n != a["n"]:
            diffs.append(f"{name}: n {n} != {a['n']}")
        if jev_auc is None or abs(jev_auc - a["jev"]) > 5e-4:
            diffs.append(f"{name}: Jev AUC {jev_auc} != {a['jev']}")
        if el_auc is None or abs(el_auc - a["el_llama"]) > 5e-4:
            diffs.append(f"{name}: EditLens-llama AUC {el_auc} != {a['el_llama']}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero on any mismatch (this is the default behaviour)")
    args = ap.parse_args()

    diffs, S, V = check_summary_and_verdicts(REPO)
    adiffs = check_headline_aucs(REPO)

    if S:
        p = S["pooled"]
        print(f"recomputed from cache ({S['n_bundle']} texts, {S['n_scored']} scored):")
        print(f"  pooled: acc_raw {p.get('acc_raw')} | coverage {p.get('coverage')} | "
              f"emitted_acc {p.get('emitted_acc')} (n={p.get('emitted_n')}) | el_acc {p.get('el_acc')} | "
              f"AUC jev {p.get('auc_jev')} / el {p.get('auc_el')} | agreement {p.get('jev_el_agreement')}")
        for tag in ("M1", "M2", "M3", "M4", "M5"):
            print(f"  {tag}: {V[tag]['verdict']}")
    for name, a in HEADLINE_ANCHORS.items():
        print(f"  headline {name}: n {a['n']} | Jev AUC {a['jev']} | EditLens-llama AUC {a['el_llama']}")

    problems = diffs + adiffs
    if problems:
        print(f"\nCHECK FAILED: {len(problems)} mismatch(es):")
        for d in problems[:40]:
            print("  -", d)
        return 1
    print("\nCHECK OK: summary, verdicts and headline AUCs reproduce the shipped JSONs exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
