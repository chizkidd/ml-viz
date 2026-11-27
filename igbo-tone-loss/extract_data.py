"""Extract clip rows and verified Levenshtein CER/WER profiles from the truth repo.

Usage:
    python3 igbo-tone-loss/extract_data.py /path/to/igbo-asr-tonal-evaluation-research > igbo-tone-loss/data.json

Reads data/metadata.csv and (if present) data/analysis_groups.csv, computes per-clip
Levenshtein CER (strict, tone_stripped, base_letters, legacy) and WER (strict, base_letters)
using the exact normalisation rules of docs/ANALYSIS_SPEC.md, and cross-checks aggregates
against results/canonical_results.csv when present.
"""
import csv
import json
import pathlib
import re
import sys
import unicodedata

GRAVE, ACUTE, MACRON, DOT_BELOW = 0x300, 0x301, 0x304, 0x323
COUNTED = {GRAVE, ACUTE, MACRON, DOT_BELOW}
TONE = {GRAVE, ACUTE, MACRON}
APOSTROPHES = "'\u2019\u2018\u02bc`"
HYPHENS = "-\u2010\u2011\u2012\u2013\u2014"


def clusters(text):
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def count_diacritics(text):
    n = 0
    for c in clusters(text):
        if c[:1].isalpha() and ({ord(ch) for ch in c[1:]} & COUNTED):
            n += 1
    return n


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_text(text, profile="strict"):
    t = unicodedata.normalize("NFC", text).lower()
    for ch in APOSTROPHES:
        t = t.replace(ch, "")
    t = "".join(" " if ch in HYPHENS else ch for ch in t)
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] not in "PS")
    t = re.sub(r"\s+", " ", t).strip()
    if profile != "strict":
        remove = TONE if profile == "tone_stripped" else None
        t = unicodedata.normalize(
            "NFC",
            "".join(
                ch
                for ch in unicodedata.normalize("NFD", t)
                if not (unicodedata.combining(ch) and (remove is None or ord(ch) in remove))
            ),
        )
    return t


def cer_edits(ref, hyp, profile):
    if profile == "legacy":
        r = list(unicodedata.normalize("NFC", ref).lower())
        h = list(unicodedata.normalize("NFC", hyp).lower())
    else:
        r = list(normalize_text(ref, profile))
        h = list(normalize_text(hyp, profile))
    return levenshtein(r, h), len(r)


def wer_edits(ref, hyp, profile):
    r = normalize_text(ref, profile).split()
    h = normalize_text(hyp, profile).split()
    return levenshtein(r, h), len(r)


def verify_against_canonical(rows, canonical_path):
    """Check extracted per-clip edits and counts against committed canonical_results.csv."""
    with open(canonical_path, encoding="utf-8", newline="") as f:
        canon = list(csv.DictReader(f))

    def lookup(group, layer, estimand, prof, unit, agg):
        for c in canon:
            if (
                c["analysis_group"] == group
                and c["layer"] == layer
                and c["estimand"] == estimand
                and c["cer_profile"] == prof
                and c["unit"] == unit
                and c["aggregation"] == agg
            ):
                return float(c["estimate"])
        raise KeyError((group, layer, estimand, prof, unit, agg))

    for group, subset in [
        ("overall", rows),
        ("igbo_tonal_pilot", [r for r in rows if r["group"] == "igbo_tonal_pilot"]),
    ]:
        for prof, key in [
            ("strict", "cer_strict"),
            ("tone_stripped", "cer_tone"),
            ("base_letters", "cer_base"),
        ]:
            pooled = sum(r[f"{key}_d"] for r in subset) / sum(r[f"{key}_n"] for r in subset)
            mean = sum(r[f"{key}_d"] / r[f"{key}_n"] for r in subset) / len(subset)
            exp_p = lookup(group, "current", "cer", prof, "codepoint", "pooled")
            exp_m = lookup(group, "current", "cer", prof, "codepoint", "mean")
            assert abs(pooled - exp_p) < 1e-12, f"{group} {prof} pooled {pooled} != {exp_p}"
            assert abs(mean - exp_m) < 1e-12, f"{group} {prof} mean {mean} != {exp_m}"
        for prof, key in [("strict", "wer_strict"), ("base_letters", "wer_base")]:
            pooled = sum(r[f"{key}_d"] for r in subset) / sum(r[f"{key}_n"] for r in subset)
            exp_p = lookup(group, "current", "wer", prof, "word", "pooled")
            assert abs(pooled - exp_p) < 1e-12, f"{group} wer {prof} pooled {pooled} != {exp_p}"

    leg_mean = sum(r["cer_legacy_d"] / r["cer_legacy_n"] for r in rows) / len(rows)
    exp_leg = lookup("overall", "historical", "cer", "legacy", "codepoint", "mean")
    assert abs(leg_mean - exp_leg) < 1e-12, f"legacy mean {leg_mean} != {exp_leg}"


def main(repo_dir):
    root = pathlib.Path(repo_dir)
    meta_path = root / "data" / "metadata.csv"
    groups_path = root / "data" / "analysis_groups.csv"
    canon_path = root / "results" / "canonical_results.csv"

    group_map = {}
    if groups_path.exists():
        with open(groups_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                group_map[row["file_name"]] = (row["clip_id"], row["analysis_group"])

    rows = []
    with open(meta_path, encoding="utf-8", newline="") as f:
        for idx, r in enumerate(csv.DictReader(f)):
            fn = r["file_name"]
            m = re.search(r"(\d+)", pathlib.Path(fn).name)
            clip_id = group_map[fn][0] if fn in group_map else (m.group(1) if m else f"{idx + 1:02d}")
            group = group_map[fn][1] if fn in group_map else r["category"]
            gt, out = r["ground_truth"], r["model_output"]
            e, p = int(r["diacritics_expected"]), int(r["diacritics_produced"])
            assert count_diacritics(gt) == e, f"clip {clip_id}: counted E != {e}"
            assert count_diacritics(out) == p, f"clip {clip_id}: counted P != {p}"

            cs_d, cs_n = cer_edits(gt, out, "strict")
            ct_d, ct_n = cer_edits(gt, out, "tone_stripped")
            cb_d, cb_n = cer_edits(gt, out, "base_letters")
            cl_d, cl_n = cer_edits(gt, out, "legacy")
            ws_d, ws_n = wer_edits(gt, out, "strict")
            wb_d, wb_n = wer_edits(gt, out, "base_letters")

            rows.append(
                {
                    "id": int(clip_id),
                    "clip_id": clip_id,
                    "row_idx": idx,
                    "file": pathlib.Path(fn).name,
                    "cat": r["category"],
                    "sub": r["subcategory"],
                    "lang": r["language"],
                    "group": group,
                    "gt": gt,
                    "out": out,
                    "e": e,
                    "p": p,
                    "difflib": round(float(r["character_error_rate"]), 4),
                    "cer_strict_d": cs_d,
                    "cer_strict_n": cs_n,
                    "cer_tone_d": ct_d,
                    "cer_tone_n": ct_n,
                    "cer_base_d": cb_d,
                    "cer_base_n": cb_n,
                    "cer_legacy_d": cl_d,
                    "cer_legacy_n": cl_n,
                    "wer_strict_d": ws_d,
                    "wer_strict_n": ws_n,
                    "wer_base_d": wb_d,
                    "wer_base_n": wb_n,
                }
            )

    if canon_path.exists():
        verify_against_canonical(rows, canon_path)

    payload = {
        "source": "chizkidd/igbo-asr-tonal-evaluation-research (data/metadata.csv + data/analysis_groups.csv, verified against results/canonical_results.csv)",
        "rows": rows,
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 extract_data.py /path/to/igbo-asr-tonal-evaluation-research > data.json")
    main(sys.argv[1])
