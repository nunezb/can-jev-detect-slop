#!/usr/bin/env python3
"""rebuild_dataset.py — check the shipped dataset against the source corpora.

For every text in data/selftest_cache.jsonl this script rebuilds the text from the raw corpora
and compares the recorded content hash. Run scripts/fetch_corpora.py first.

Each text gets one status:
  exact      The recorded id points at a source row. The rebuilt text matches.
  member     The text came from a seeded sample. The script checks that the text is present in
             the source pool. The shipped hash selects it exactly.
  generated  The text was written during the study (B8 rewrites, D4/D4x humanized outputs).
             No source row can rebuild it.
  unfetched  The raw file for this text is missing. Run fetch_corpora.py.
  mismatch   A rebuilt text does not match the recorded hash. This is an error.

Needs: pip install pandas pyarrow
Usage: python3 scripts/rebuild_dataset.py [--raw DIR] [--cache FILE]
Exit code: 0 when every available text verifies. 1 on any mismatch or unknown id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_RAW = REPO / "data" / "raw"
DEFAULT_CACHE = REPO / "data" / "selftest_cache.jsonl"


def h16(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def load_lamp(path: Path) -> list:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r"}\n(\s*)\{", r"},\n\1{", raw))


def extract_stp(user_content: str, assistant_content: str) -> tuple:
    """Port of the study's SlopToPolish endpoint extraction (text endpoints only)."""
    parts = re.split(r"editing:\s*", user_content, maxsplit=1)
    original = (parts[1] if len(parts) > 1 else user_content).strip()
    idx = assistant_content.find("Part 3")
    final = assistant_content[idx:] if idx >= 0 else ""
    final = re.sub(r"^Part 3[^\n]*\n+", "", final).strip()
    final = re.sub(r"^(?:Here is|Below is|The revised|Revised)[^\n:]*:\s*", "", final).strip()
    return original, final


class Raw:
    """Lazy access to the downloaded source files."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._cache: dict = {}

    def has(self, name: str) -> bool:
        return (self.directory / name).exists()

    def parquet(self, name: str):
        key = ("pq", name)
        if key not in self._cache:
            import pandas as pd  # lazy: only needed for the rebuild
            self._cache[key] = pd.read_parquet(self.directory / name)
        return self._cache[key]

    def lamp(self) -> list:
        if "lamp" not in self._cache:
            self._cache["lamp"] = load_lamp(self.directory / "LAMP.json")
        return self._cache["lamp"]

    def _zip_csv(self, name: str, suffix: str):
        key = ("zip", name, suffix)
        if key not in self._cache:
            import pandas as pd
            with zipfile.ZipFile(self.directory / name) as z:
                member = [n for n in z.namelist() if n.endswith(suffix)][0]
                with z.open(member) as f:
                    self._cache[key] = pd.read_csv(f)
        return self._cache[key]

    def idmgsp(self):
        import pandas as pd
        if "idm" not in self._cache:
            df = self._zip_csv("idmgsp_classifier.zip", "test.csv")
            df["nw"] = df.abstract.str.split().str.len()
            self._cache["idm"] = df[(df.nw >= 60) & (df.nw <= 350)].drop_duplicates("title")
        return self._cache["idm"]

    def gpt3(self):
        import pandas as pd
        if "g3" not in self._cache:
            df = self._zip_csv("ood_gpt3.zip", "test.csv")
            df["nw"] = df.abstract.str.split().str.len()
            self._cache["g3"] = df[(df.nw >= 60) & (df.nw <= 350)].drop_duplicates("title")
        return self._cache["g3"]

    # ---- membership pools (for texts that come from seeded samples) --------------------
    def grammarly_pool(self, column: str) -> set:
        key = ("gram", column)
        if key not in self._cache:
            df = self.parquet("editlens_grammarly.parquet")
            self._cache[key] = {h16(str(x)) for x in df[column]}
        return self._cache[key]

    def d4_pool(self) -> set:
        if "d4pool" not in self._cache:
            pool = set()
            hc3 = self.parquet("hc3_all.parquet")
            sub = hc3[hc3.chatgpt_answers.str.len() > 0]
            for _, row in sub.iterrows():
                answers = row["chatgpt_answers"]
                if len(answers) > 0:
                    pool.add(h16(" ".join(str(answers[0]).split()[:500])))
            for a in self.idmgsp().abstract:
                pool.add(h16(str(a)))
            stp = self.parquet("stp_test.parquet")
            for i in range(len(stp)):
                by_role = {m["role"]: m["content"] for m in stp.messages.iloc[i]}
                original, final = extract_stp(by_role.get("user", ""), by_role.get("assistant", ""))
                if len(original) > 200 and len(final) > 200:
                    pool.add(h16(final))
            for name in ("arjun_test.parquet", "arjun_train.parquet"):
                df = self.parquet(name)
                for x in df["slop"]:
                    pool.add(h16(" ".join(str(x).split()[:1200])))
            self._cache["d4pool"] = pool
        return self._cache["d4pool"]


def classify(bid: str) -> tuple:
    """Map a bundle id to its rebuild method. Returns (kind, payload)."""
    m = re.match(r"lamp_(\d+)_(pre|post)$", bid)
    if m:
        return "exact", ("lamp", int(m[1]), m[2])
    m = re.match(r"b8_(pre|post)_(\d+)$", bid)
    if m:
        return "exact", ("lamp", int(m[2]), m[1])
    if bid.startswith("b8_b8_"):
        return "generated", None
    if bid.startswith("el_val_src_"):
        return "exact", ("editlens", "editlens_val.parquet", "source_text", bid[len("el_val_src_"):])
    if bid.startswith("el_val_"):
        return "exact", ("editlens", "editlens_val.parquet", "text", bid[len("el_val_"):])
    if bid.startswith("el_enron_"):
        return "exact", ("editlens", "editlens_test_enron.parquet", "text", bid[len("el_enron_"):])
    if bid.startswith("el_llama_"):
        return "exact", ("editlens", "editlens_test_llama.parquet", "text", bid[len("el_llama_"):])
    if bid.startswith("el_gram_src_"):
        return "member", ("grammarly", "source_text")
    if bid.startswith("el_gram_"):
        return "member", ("grammarly", "text")
    m = re.match(r"arj_arjunA_(test|train)_(\d+)_(human|slop)$", bid)
    if m:
        return "exact", ("arjun", m[1], int(m[2]), m[3])
    m = re.match(r"st_ext_hc3_[a-z0-9_]+_(\d+)_(human|chatgpt)$", bid)
    if m:
        return "exact", ("hc3", int(m[1]), m[2])
    m = re.match(r"d_p4d_hc3_[a-z0-9_]+_(\d+)$", bid)
    if m:
        return "exact", ("hc3", int(m[1]), "human")
    m = re.match(r"st_ext_idmgsp_([a-z0-9]+)_(\d+)$", bid)
    if m:
        return "exact", ("idmgsp", m[1], int(m[2]), "classifier")
    m = re.match(r"d_p4d_idmgsp_gpt3_(\d+)$", bid)
    if m:
        return "exact", ("idmgsp", "gpt3", int(m[1]), "gpt3")
    m = re.match(r"d_p4d_idmgsp_([a-z0-9]+)_(\d+)$", bid)
    if m:
        return "exact", ("idmgsp", m[1], int(m[2]), "classifier")
    m = re.match(r"stp_c3_(\d+)_(orig|final)$", bid)
    if m:
        return "exact", ("stp", int(m[1]), m[2])
    if bid.startswith("sc_"):
        return "exact", ("slopclass", bid[3:])
    if bid.startswith("d4orig_") or bid.startswith("d4xorig_"):
        return "member", ("d4",)
    if bid.startswith("d4_") or bid.startswith("d4x_"):
        return "generated", None
    return "unknown", None


def rebuild_variant(raw: Raw, bid: str, expected: str) -> tuple:
    """Return (status, text). Status: exact | member | generated | unfetched | mismatch | unknown."""
    kind, payload = classify(bid)
    if kind == "generated":
        return "generated", None
    if kind == "unknown":
        return "unknown", bid
    if kind == "member":
        which = payload[0]
        column = payload[1] if len(payload) > 1 else None
        if which == "grammarly":
            if not raw.has("editlens_grammarly.parquet"):
                return "unfetched", bid
            pool = raw.grammarly_pool(column)
        else:
            for need in ("hc3_all.parquet", "idmgsp_classifier.zip", "stp_test.parquet",
                         "arjun_test.parquet", "arjun_train.parquet"):
                if not raw.has(need):
                    return "unfetched", bid
            pool = raw.d4_pool()
        return ("member" if expected in pool else "mismatch"), bid
    # exact methods
    tag = payload[0]
    if tag == "lamp":
        if not raw.has("LAMP.json"):
            return "unfetched", bid
        _, i, side = payload
        rec = raw.lamp()[i]
        text = rec["preedit"] if side == "pre" else rec["postedit"]
    elif tag == "editlens":
        _, name, column, tid = payload
        if not raw.has(name):
            return "unfetched", bid
        df = raw.parquet(name)
        sub = df[df.text_id == tid]
        if sub.empty:
            return "mismatch", bid
        text = str(sub[column].iloc[0])
    elif tag == "arjun":
        _, split, i, side = payload
        name = f"arjun_{split}.parquet"
        if not raw.has(name):
            return "unfetched", bid
        text = str(raw.parquet(name).iloc[i]["human" if side == "human" else "slop"])
    elif tag == "hc3":
        if not raw.has("hc3_all.parquet"):
            return "unfetched", bid
        _, i, side = payload
        df = raw.parquet("hc3_all.parquet")
        row = df.loc[i] if i in df.index else df.iloc[i]
        answers = row["human_answers"] if side == "human" else row["chatgpt_answers"]
        if len(answers) == 0:
            return "mismatch", bid
        text = " ".join(str(answers[0]).split()[:500])
    elif tag == "idmgsp":
        _, src, i, source = payload
        name = "ood_gpt3.zip" if source == "gpt3" else "idmgsp_classifier.zip"
        if not raw.has(name):
            return "unfetched", bid
        df = raw.gpt3() if source == "gpt3" else raw.idmgsp()
        row = df.loc[i] if i in df.index else df.iloc[i]
        text = str(row["abstract"])
    elif tag == "stp":
        if not raw.has("stp_test.parquet"):
            return "unfetched", bid
        _, trace, endpoint = payload
        s = raw.parquet("stp_test.parquet")
        by_role = {m["role"]: m["content"] for m in s.messages.iloc[trace]}
        original, final = extract_stp(by_role.get("user", ""), by_role.get("assistant", ""))
        text = original if endpoint == "orig" else final
    elif tag == "slopclass":
        if not raw.has("slopcls_train.parquet"):
            return "unfetched", bid
        _, hash12 = payload
        df = raw.parquet("slopcls_train.parquet")
        sub = df[df.content_hash.astype(str).str[:12] == hash12]
        if sub.empty:
            return "mismatch", bid
        text = str(sub["content"].iloc[0])
    else:  # pragma: no cover - classify covers every tag
        return "unknown", bid
    return ("exact" if h16(text) == expected else "mismatch"), bid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=str(DEFAULT_RAW), help="directory with the fetched corpora")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="path to selftest_cache.jsonl")
    args = ap.parse_args()

    try:
        import pandas  # noqa: F401
    except ImportError:
        print("pandas is not installed. Run: pip install pandas pyarrow", file=sys.stderr)
        return 2

    raw = Raw(Path(args.raw).resolve())
    rows = [json.loads(line) for line in Path(args.cache).read_text(encoding="utf-8").splitlines() if line.strip()]

    stats: dict = {}
    problems: list = []
    precedence = {"mismatch": 5, "unknown": 4, "exact": 3, "member": 2, "generated": 1, "unfetched": 0}
    for r in rows:
        source = r["source"]
        st = stats.setdefault(source, {k: 0 for k in precedence})
        row_status = "unfetched"
        for variant in r["variants"]:
            bid = variant[4]
            status, detail = rebuild_variant(raw, bid, r["hash"])
            if precedence[status] > precedence[row_status]:
                row_status = status
            if status in ("mismatch", "unknown"):
                problems.append(f"{source}: {bid} ({status})")
        st[row_status] += 1

    print(f"dataset rebuild: {len(rows)} distinct texts from {raw.directory}")
    print(f"{'source':<18}{'rows':>6}{'exact':>7}{'member':>8}{'generated':>11}{'unfetched':>11}{'mismatch':>10}")
    totals = {k: 0 for k in precedence}
    for source in sorted(stats):
        s = stats[source]
        for k in totals:
            totals[k] += s[k]
        print(f"{source:<18}{sum(s.values()):>6}{s['exact']:>7}{s['member']:>8}{s['generated']:>11}{s['unfetched']:>11}{s['mismatch']:>10}")
    print(f"{'TOTAL':<18}{sum(totals.values()):>6}{totals['exact']:>7}{totals['member']:>8}{totals['generated']:>11}{totals['unfetched']:>11}{totals['mismatch']:>10}")
    verified = totals["exact"] + totals["member"]
    print(f"\nverified: {verified}/{len(rows)} texts "
          f"({totals['exact']} exact, {totals['member']} by pool membership); "
          f"{totals['generated']} generated in the study; {totals['unfetched']} need a download")
    if totals["generated"]:
        print("generated texts are not rebuildable from sources: they are the B8 rewrite candidates")
        print("and the D4/D4x humanized outputs written by the study's generation calls.")
    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for p in problems[:30]:
            print("  -", p)
        return 1
    if totals["unfetched"]:
        print(f"\nCHECK OK (partial): every available text rebuilds to the recorded hash.")
        print(f"{totals['unfetched']} texts need a download. Run: python3 scripts/fetch_corpora.py")
    else:
        print("\nCHECK OK: every available text rebuilds to the recorded hash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
