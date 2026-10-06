#!/usr/bin/env python3
"""fetch_corpora.py — download the source corpora used by the study into data/raw/.

The repository ships no third-party text. This script fetches the texts from their public
sources so that scripts/rebuild_dataset.py can check every shipped hash.

Most sources are public. Four files are gated on Hugging Face and need a read token:
editlens_val.parquet, editlens_test_enron.parquet, editlens_test_llama.parquet,
editlens_grammarly.parquet. Set HF_TOKEN to download them. Without a token the script
skips those files and reports the skip.

Usage:
    python3 scripts/fetch_corpora.py                  # download into data/raw/
    python3 scripts/fetch_corpora.py --only hc3       # fetch one file
    python3 scripts/fetch_corpora.py --check          # report what is present, download nothing
    python3 scripts/fetch_corpora.py --force          # download again
    python3 scripts/fetch_corpora.py --dir /tmp/raw   # use another directory

The token is read from HF_TOKEN, HUGGINGFACE_TOKEN or HF_READONLY_ACCESS_TOKEN (environment or ~/.bashrc). It is never printed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DIR = REPO / "data" / "raw"

HF = "https://huggingface.co/datasets/"
# name -> (url, gated, note)
FILES = {
    "slopcls_train.parquet": (HF + "bench-labs/slop-classification/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", False, "SlopClass; MIT"),
    "arjun_test.parquet": (HF + "arjun10g/slop-paraphrase-pairs-v2/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet", False, "arjun pairs; no declared license"),
    "arjun_train.parquet": (HF + "arjun10g/slop-paraphrase-pairs-v2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", False, "arjun pairs; no declared license"),
    "stp_test.parquet": (HF + "ConicCat/SlopToPolish/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet", False, "SlopToPolish; Apache-2.0"),
    "hc3_all.parquet": (HF + "Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/all/train/0000.parquet", False, "HC3; CC BY-SA 4.0"),
    "idmgsp_classifier.zip": (HF + "tum-nlp/IDMGSP/resolve/main/classifier_input.zip", False, "IDMGSP; OpenRAIL++"),
    "ood_gpt3.zip": (HF + "tum-nlp/IDMGSP/resolve/main/ood_gpt3.zip", False, "IDMGSP GPT-3 hold-out; OpenRAIL++"),
    "LAMP.json": ("https://raw.githubusercontent.com/salesforce/creativity_eval/main/Writing_Alignment/LAMP/LAMP.json", False, "LAMP; BSD-3-Clause"),
    "editlens_val.parquet": (HF + "pangram/editlens_iclr/resolve/refs%2Fconvert%2Fparquet/default/val/0000.parquet", True, "EditLens; CC BY-NC-SA 4.0, gated"),
    "editlens_test_enron.parquet": (HF + "pangram/editlens_iclr/resolve/refs%2Fconvert%2Fparquet/default/test_enron/0000.parquet", True, "EditLens; CC BY-NC-SA 4.0, gated"),
    "editlens_test_llama.parquet": (HF + "pangram/editlens_iclr/resolve/refs%2Fconvert%2Fparquet/default/test_llama/0000.parquet", True, "EditLens; CC BY-NC-SA 4.0, gated"),
    "editlens_grammarly.parquet": (HF + "pangram/editlens_iclr_grammarly/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", True, "EditLens Grammarly; CC BY-NC-SA 4.0, gated"),
}


def token() -> str:
    """Read a Hugging Face token from the environment or ~/.bashrc. Never printed."""
    names = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_READONLY_ACCESS_TOKEN")
    for var in names:
        v = os.environ.get(var, "").strip()
        if v:
            return v
    rc = Path.home() / ".bashrc"
    if rc.exists():
        pat = r'^export\s+(' + "|".join(names) + r')=(.*)$'
        for line in rc.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(pat, line.strip())
            if m:
                return m.group(2).strip().strip('"').strip("'")
    return ""


def fetch(url: str, out: Path, tok: str, gated: bool) -> None:
    headers = {"User-Agent": "slop-quality-repo/1.0"}
    if gated and tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="download directory (default: data/raw)")
    ap.add_argument("--only", nargs="*", help="fetch only these file names")
    ap.add_argument("--check", action="store_true", help="report presence, download nothing")
    ap.add_argument("--force", action="store_true", help="download files that already exist")
    args = ap.parse_args()

    out_dir = Path(args.dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or list(FILES)
    for n in names:
        if n not in FILES:
            print(f"unknown file: {n}", file=sys.stderr)
            return 2

    tok = token()
    failed = 0
    for name in names:
        url, gated, note = FILES[name]
        dest = out_dir / name
        if dest.exists() and not args.force:
            print(f"[have]    {name:<28} {dest.stat().st_size:>10,} bytes  ({note})")
            continue
        if args.check:
            print(f"[missing] {name:<28} {'pending token' if gated else 'fetched'}  ({note})")
            continue
        if gated and not tok:
            print(f"[skip]    {name:<28} gated. Set HF_TOKEN to download.  ({note})")
            continue
        try:
            print(f"[fetch]   {name:<28} {note}")
            fetch(url, dest, tok, gated)
            print(f"[ok]      {name:<28} {dest.stat().st_size:>10,} bytes")
        except urllib.error.HTTPError as e:
            print(f"[fail]    {name:<28} HTTP {e.code}  ({url})", file=sys.stderr)
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[fail]    {name:<28} {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
