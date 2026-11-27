"""Extract clip rows, acoustic/export metadata, and verified metrics from the truth repo.

Usage:
    python3 igbo-tone-loss/extract_data.py /path/to/igbo-asr-tonal-evaluation-research > igbo-tone-loss/data.json

Reads data/metadata.csv, data/analysis_groups.csv, and research/utterance_inventory.csv,
computes per-clip Levenshtein CER/WER profiles, tone vs dot-below counts, and alignment
ambiguity using the exact rules of docs/ANALYSIS_SPEC.md, and verifies aggregates against
results/canonical_results.csv when present.
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
DOT = {DOT_BELOW}
APOSTROPHES = "'\u2019\u2018\u02bc`"
HYPHENS = "-\u2010\u2011\u2012\u2013\u2014"

# Verified objective acoustic findings from research/data_audit.md Addendum A3 (labels only, not linguistic tone validation).
ACOUSTIC_NOTES = {
    "05": "Speaking rate proxy: 3.68 ref vowels/active s vs 3.92–4.21 on other 16 kHz clips (<8% difference; not clearly slow). Duplicate reference text with clip 20.",
    "06": "Pause-splits into 12 words matching reference; within-group repeats agree, but groups 1 & 3 both fall and groups 2 & 4 both rise (exploratory half-word F0 check did not show 4 distinct contours).",
    "07": "Pause-splits into 9 words matching reference; 3 groups show more distinct F0 shapes than clip 06 (note: ọkè also differs by dot-below vowel quality).",
    "08": "F0 SD 2.71 semitones (p5–p95 range 9.7 st). Same base sentence as clip 09, written here with full tone + dot-below marks.",
    "09": "Intended flatter intonation: F0 SD 1.59 semitones (p5–p95 range 5.6 st) — flatter than clip 08 (2.71 st), but similar in range to non-monotone clips 04/07/11/13 (not verified monotone).",
    "20": "Noise floor proxy -47.8 dB vs -51.2 to -47.8 dB across other 48 kHz clips (no measurably elevated background noise). Same reference text as clip 05.",
}


def clusters(text):
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def count_family(text, mark_set):
    n = 0
    for c in clusters(text):
        if c[:1].isalpha() and ({ord(ch) for ch in c[1:]} & mark_set):
            n += 1
    return n


def units(text):
    out = []
    for c in clusters(text):
        if c[:1].isalpha():
            out.append({"base": c[0].lower(), "marks": {ord(ch) for ch in c[1:]} & COUNTED})
    return out


def alignment_ambiguity(ref, hyp, cap=10**9):
    R, O = units(ref), units(hyp)
    n, m = len(R), len(O)
    S = [[0] * (m + 1) for _ in range(n + 1)]
    W = [[1] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        S[i][0] = -i
    for j in range(1, m + 1):
        S[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = S[i - 1][j - 1] + (2 if R[i - 1]["base"] == O[j - 1]["base"] else -1)
            u, l = S[i - 1][j] - 1, S[i][j - 1] - 1
            best = max(d, u, l)
            S[i][j] = best
            W[i][j] = min(
                cap,
                (W[i - 1][j - 1] if d == best else 0)
                + (W[i - 1][j] if u == best else 0)
                + (W[i][j - 1] if l == best else 0),
            )
    return W[n][m]


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
    """Check extracted counts and CER/WER profiles against committed canonical_results.csv."""
    with open(canonical_path, encoding="utf-8", newline="") as f:
        canon = list(csv.DictReader(f))

    def lookup(group, layer, estimand, mark_family, prof, unit, agg):
        for c in canon:
            if (
                c["analysis_group"] == group
                and c["layer"] == layer
                and c["estimand"] == estimand
                and c["mark_family"] == mark_family
                and c["cer_profile"] == prof
                and c["unit"] == unit
                and c["aggregation"] == agg
            ):
                return float(c["estimate"]) if c["estimate"] else float(c["numerator"])
        raise KeyError((group, layer, estimand, mark_family, prof, unit, agg))

    for group, subset in [
        ("overall", rows),
        ("igbo_tonal_pilot", [r for r in rows if r["group"] == "igbo_tonal_pilot"]),
    ]:
        for fam, ek, pk in [("all", "e", "p"), ("tone", "tone_e", "tone_p"), ("dot", "dot_e", "dot_p")]:
            assert sum(r[ek] for r in subset) == lookup(group, "current", "mark_letters_expected", fam, "none", "none", "none")
            assert sum(r[pk] for r in subset) == lookup(group, "current", "mark_letters_produced", fam, "none", "none", "none")
        for prof, key in [
            ("strict", "cer_strict"),
            ("tone_stripped", "cer_tone"),
            ("base_letters", "cer_base"),
        ]:
            pooled = sum(r[f"{key}_d"] for r in subset) / sum(r[f"{key}_n"] for r in subset)
            mean = sum(r[f"{key}_d"] / r[f"{key}_n"] for r in subset) / len(subset)
            exp_p = lookup(group, "current", "cer", "none", prof, "codepoint", "pooled")
            exp_m = lookup(group, "current", "cer", "none", prof, "codepoint", "mean")
            assert abs(pooled - exp_p) < 1e-12, f"{group} {prof} pooled {pooled} != {exp_p}"
            assert abs(mean - exp_m) < 1e-12, f"{group} {prof} mean {mean} != {exp_m}"
        for prof, key in [("strict", "wer_strict"), ("base_letters", "wer_base")]:
            pooled = sum(r[f"{key}_d"] for r in subset) / sum(r[f"{key}_n"] for r in subset)
            exp_p = lookup(group, "current", "wer", "none", prof, "word", "pooled")
            assert abs(pooled - exp_p) < 1e-12, f"{group} wer {prof} pooled {pooled} != {exp_p}"

    leg_mean = sum(r["cer_legacy_d"] / r["cer_legacy_n"] for r in rows) / len(rows)
    exp_leg = lookup("overall", "historical", "cer", "none", "legacy", "codepoint", "mean")
    assert abs(leg_mean - exp_leg) < 1e-12, f"legacy mean {leg_mean} != {exp_leg}"


def main(repo_dir):
    root = pathlib.Path(repo_dir)
    meta_path = root / "data" / "metadata.csv"
    groups_path = root / "data" / "analysis_groups.csv"
    inv_path = root / "research" / "utterance_inventory.csv"
    canon_path = root / "results" / "canonical_results.csv"

    group_map = {}
    if groups_path.exists():
        with open(groups_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                group_map[row["file_name"]] = (row["clip_id"], row["analysis_group"])

    inv_map = {}
    if inv_path.exists():
        with open(inv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("metadata_file_name"):
                    inv_map[row["metadata_file_name"]] = row

    rows = []
    with open(meta_path, encoding="utf-8", newline="") as f:
        for idx, r in enumerate(csv.DictReader(f)):
            fn = r["file_name"]
            m = re.search(r"(\d+)", pathlib.Path(fn).name)
            clip_id = group_map[fn][0] if fn in group_map else (m.group(1) if m else f"{idx + 1:02d}")
            group = group_map[fn][1] if fn in group_map else r["category"]
            gt, out = r["ground_truth"], r["model_output"]
            e, p = int(r["diacritics_expected"]), int(r["diacritics_produced"])
            assert count_family(gt, COUNTED) == e, f"clip {clip_id}: counted E != {e}"
            assert count_family(out, COUNTED) == p, f"clip {clip_id}: counted P != {p}"

            tone_e, tone_p = count_family(gt, TONE), count_family(out, TONE)
            dot_e, dot_p = count_family(gt, DOT), count_family(out, DOT)
            n_opt = alignment_ambiguity(gt, out)

            cs_d, cs_n = cer_edits(gt, out, "strict")
            ct_d, ct_n = cer_edits(gt, out, "tone_stripped")
            cb_d, cb_n = cer_edits(gt, out, "base_letters")
            cl_d, cl_n = cer_edits(gt, out, "legacy")
            ws_d, ws_n = wer_edits(gt, out, "strict")
            wb_d, wb_n = wer_edits(gt, out, "base_letters")

            inv = inv_map.get(fn, {})
            stem = pathlib.Path(fn).stem
            sr_hz = int(float(inv["sample_rate_hz"])) if inv.get("sample_rate_hz") else (16000 if int(clip_id) <= 5 else 48000)
            dur_s = round(float(inv["duration_s"]), 2) if inv.get("duration_s") else None
            kbps = round(float(inv["bitrate_kbps"]), 1) if inv.get("bitrate_kbps") else None
            chain = "16k_gated" if sr_hz == 16000 else "48k_raw"

            rows.append(
                {
                    "id": int(clip_id),
                    "clip_id": clip_id,
                    "row_idx": idx,
                    "file": pathlib.Path(fn).name,
                    "audio": f"audio/{stem}.m4a",
                    "dur_s": dur_s,
                    "sr_hz": sr_hz,
                    "kbps": kbps,
                    "chain": chain,
                    "acoustic_note": ACOUSTIC_NOTES.get(clip_id, ""),
                    "cat": r["category"],
                    "sub": r["subcategory"],
                    "lang": r["language"],
                    "group": group,
                    "gt": gt,
                    "out": out,
                    "e": e,
                    "p": p,
                    "tone_e": tone_e,
                    "tone_p": tone_p,
                    "dot_e": dot_e,
                    "dot_p": dot_p,
                    "n_opt": n_opt,
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
        "source": "chizkidd/igbo-asr-tonal-evaluation-research (metadata.csv + analysis_groups.csv + utterance_inventory.csv, verified against canonical_results.csv)",
        "rows": rows,
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 extract_data.py /path/to/igbo-asr-tonal-evaluation-research > data.json")
    main(sys.argv[1])
