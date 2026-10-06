#!/usr/bin/env python3
"""build_selftest_cache.py — one-time builder for data/selftest_cache.jsonl.

The published results were produced inside the lab archive (`_math/jev/experiments/slop_quality/`),
which contains third-party corpus text and is not shipped here. This script distills the archive
into a single hash-keyed cache: one row per distinct text, carrying the stratum label, the frozen
metrics' inputs (selected Jev `origin_choice` answer, EditLens-llama score) and the bundle-row
history needed to reconstruct paired designs — and nothing that can recover any third-party text.

Run once from the lab machine, then freeze the output in the repository:

    python3 scripts/build_selftest_cache.py --lab-root ../experiments/slop_quality

Requires the lab archive, including `prep_tier1.py` (the stratum truth table) and the EditLens
pass bundle. The selection rules below reproduce `analyze_tier1.py` verbatim, including its use
of unsorted `glob.glob` over `results/*.json`; the shipped cache bakes in the resulting answers,
so downstream repository scripts are fully deterministic.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_LAB = REPO.parent / "experiments" / "slop_quality"
DEFAULT_OUT = REPO / "data" / "selftest_cache.jsonl"

THETA = 0.31          # frozen gate (frozen_metric.json)
EL_CUT = 0.5          # frozen EditLens-llama decision cut
EXPECTED_ROWS = 8890
EXPECTED_SCORED = 8511
EXPECTED_EMITTED = 6492
EXPECTED_ACC = 0.7751
EXPECTED_M5 = {"pairs": 200, "flips_to_human": 14, "flips_to_ai": 5}


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_truth(lab: Path) -> dict:
    spec = importlib.util.spec_from_file_location("prep_tier1", lab / "prep_tier1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module has a __main__ guard; nothing runs
    return mod.TRUTH


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lab-root", default=str(DEFAULT_LAB), help="path to experiments/slop_quality")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output JSONL path")
    args = ap.parse_args()
    lab = Path(args.lab_root).resolve()
    out = Path(args.out).resolve()

    # ---- bundle: last occurrence wins (analyze_tier1.py semantics); keep the full row history --
    bundle: dict[str, dict] = {}
    variants: dict[str, list] = collections.defaultdict(list)
    seq = 0
    for line in (lab / "editlens_pass" / "texts.jsonl").open(encoding="utf-8"):
        t = json.loads(line)
        key = short_hash(t["text"])
        variants[key].append([seq, t["source"], t.get("group"), t.get("pair_key")])
        seq += 1
        bundle[key] = {"bundle_id": t["id"], "source": t["source"], "group": t.get("group"),
                       "pair_key": t.get("pair_key")}

    # ---- EditLens scores by bundle id --------------------------------------------------------
    el: dict[str, dict] = collections.defaultdict(dict)
    for name in ("roberta", "llama"):
        for line in (lab / "editlens_pass" / f"scores_{name}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            el[r["id"]][name] = r["score"]

    # ---- item banks, then the selected answer per text ---------------------------------------
    items: dict[str, dict] = {}
    for f in glob.glob(str(lab / "items" / "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            items[it["id"]] = it

    best: dict[str, dict] = {}
    for f in glob.glob(str(lab / "results" / "*.json")):  # unsorted on purpose: see module docstring
        if f.endswith("_summary.json"):
            continue
        d = json.load(open(f, encoding="utf-8"))
        if not isinstance(d, dict) or "results" not in d:
            continue
        for r in d["results"]:
            if not r.get("response"):
                continue
            it = items.get(r["id"])
            if not it:
                continue
            key = short_hash(it.get("state") or "")
            if key not in bundle:
                continue
            oc = (r["response"].get("answers") or {}).get("origin_choice")
            if not oc:
                continue
            pref = 2 if r["id"].startswith("t1_") else 1
            cur = best.get(key)
            if cur is None or pref > cur["pref"]:
                best[key] = {"pref": pref, "choice": oc.get("choice"), "conf": oc.get("confidence"),
                             "probs": oc.get("probabilities") or {},
                             "machine_like": ((r["response"].get("answers") or {})
                                              .get("machine_like") or {}).get("noul"),
                             "answer_id": r["id"]}

    truth_map = load_truth(lab)

    rows = []
    for key in sorted(bundle):
        b = bundle[key]
        a = best.get(key)
        rows.append({
            "hash": key,
            "bundle_id": b["bundle_id"],
            "n_rows": len(variants[key]),
            "variants": variants[key],
            "source": b["source"],
            "group": b["group"],
            "pair_key": b["pair_key"],
            "truth": truth_map.get((b["source"], b["group"])),
            "jev_choice": a["choice"] if a else None,
            "jev_conf": a["conf"] if a else None,
            "jev_probs": a["probs"] if a else None,
            "machine_like": a["machine_like"] if a else None,
            "from_mass_test": (a["pref"] == 2) if a else None,
            "answer_id": a["answer_id"] if a else None,
            "el_roberta": el.get(b["bundle_id"], {}).get("roberta"),
            "el_llama": el.get(b["bundle_id"], {}).get("llama"),
        })

    # ---- replay the frozen rule (the same arithmetic origin_triage.py --self-test uses) -------
    scored = emitted = correct = advisories = 0
    for r in rows:
        if r["truth"] is None or r["jev_choice"] is None or r["el_llama"] is None:
            continue
        scored += 1
        machine_signal = r["jev_choice"] in ("ai", "edited")
        el_machine = r["el_llama"] > EL_CUT
        gate_pass = (r["jev_conf"] or 0) >= THETA
        emitted_flag = gate_pass and (el_machine == machine_signal)
        if emitted_flag:
            emitted += 1
            correct += machine_signal == (r["truth"] == "machine")
        if machine_signal and el_machine == machine_signal and r["el_llama"] >= 0.90:
            advisories += 1
    acc = correct / emitted if emitted else 0.0

    # ---- M5 reconstruction check (paired d4/d4x humanization flips) --------------------------
    # Rebuild the pairing exactly as analyze_tier1_verdicts.py does: iterate texts.jsonl in file
    # order (here: variant seq order) and assign each (source, pair_key, group) cell its hash.
    events = []
    for r in rows:
        for ev_seq, src, grp, pk in r["variants"]:
            if src in ("d4", "d4x"):
                events.append((ev_seq, src, pk, grp, r["hash"]))
    events.sort()
    cells: dict[tuple, dict] = collections.defaultdict(dict)
    for _, src, pk, grp, hh in events:
        cells[(src, pk)][grp] = hh
    choices = {r["hash"]: r["jev_choice"] for r in rows}
    flips_h = flips_a = npairs = 0
    for sides in cells.values():
        if "original" in sides and "humanized" in sides:
            a, b = choices.get(sides["original"]), choices.get(sides["humanized"])
            if a and b:
                npairs += 1
                ma, mb = a in ("ai", "edited"), b in ("ai", "edited")
                if ma and not mb:
                    flips_h += 1
                if mb and not ma:
                    flips_a += 1
    m5 = {"pairs": npairs, "flips_to_human": flips_h, "flips_to_ai": flips_a}

    # ---- asserts -----------------------------------------------------------------------------
    print(f"bundle rows {seq} -> distinct texts {len(rows)}")
    print(f"replay: scored {scored} | emitted {emitted} | accuracy {acc:.4f} | advisory {advisories}")
    print(f"M5: {m5}")
    ok = (len(rows) == EXPECTED_ROWS and scored == EXPECTED_SCORED and emitted == EXPECTED_EMITTED
          and round(acc, 4) == EXPECTED_ACC and m5 == EXPECTED_M5)
    if not ok:
        print("ASSERT FAILED: replay does not reproduce the published numbers")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
