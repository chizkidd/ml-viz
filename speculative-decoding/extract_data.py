"""Pull real measured Tesla T4 numbers from chizkidd/mini-vllm results/stage6..stage11.json.

Usage:
  python3 extract_data.py /home/antigravity/chizkidd/mini-vllm/results > data.json

Everything emitted is measured on a single Tesla T4 (Qwen2.5 0.5B / 1.5B / 3B).
Nothing is fabricated.
"""
import json
import os
import sys

DEFAULT_RESULTS = "/home/antigravity/chizkidd/mini-vllm/results"


def load(d, name):
    with open(os.path.join(d, name), encoding="utf-8") as f:
        return json.load(f)


def pick(rows, keys, nd=4):
    out = []
    for r in rows:
        o = {}
        for k in keys:
            v = r.get(k)
            o[k] = round(v, nd) if isinstance(v, float) else v
        out.append(o)
    return out


def round_nested(obj, nd=4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_nested(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested(v, nd) for v in obj]
    return obj


def main(d):
    s6, s7, s8, s9, s10, s11 = (
        load(d, f"stage{n}.json") for n in (6, 7, 8, 9, 10, 11)
    )

    # Best end-to-end stack per (target, workload) from Stage 11
    best_stack = []
    for target in ("3B", "1.5B"):
        for workload in ("copy", "code", "open"):
            rows = [
                r
                for r in s11["speculation"]
                if r["target"] == target
                and r["workload"] == workload
                and "Stage 10 kernel" not in r["setup"]
            ]
            best = max(rows, key=lambda r: r["tok_s"])
            fp16_eager = next(
                r for r in rows if r["setup"] == "fp16 eager plain"
            )
            fp16_graph = next(
                r for r in rows if r["setup"] == "fp16 graph plain"
            )
            int8_plain = next(
                r for r in rows if r["setup"] == "int8 graph plain"
            )
            int8_spec = max(
                (r for r in rows if r["weights"] == "int8" and r["k"] > 0),
                key=lambda r: r["tok_s"],
            )
            fp16_spec = max(
                (
                    r
                    for r in rows
                    if r["weights"] == "fp16"
                    and r["engine"] == "graph"
                    and r["k"] > 0
                ),
                key=lambda r: r["tok_s"],
            )
            best_stack.append(
                {
                    "target": target,
                    "workload": workload,
                    "best_setup": best["setup"],
                    "best_k": best["k"],
                    "best_tok_s": round(best["tok_s"], 2),
                    "best_vs_eager": round(best["vs_fp16_eager"], 2),
                    "best_vs_graph": round(best["vs_fp16_graph"], 2),
                    "fp16_eager_tok_s": round(fp16_eager["tok_s"], 2),
                    "fp16_graph_tok_s": round(fp16_graph["tok_s"], 2),
                    "int8_plain_tok_s": round(int8_plain["tok_s"], 2),
                    "int8_plain_vs_eager": round(int8_plain["vs_fp16_eager"], 2),
                    "fp16_spec_tok_s": round(fp16_spec["tok_s"], 2),
                    "fp16_spec_k": fp16_spec["k"],
                    "fp16_spec_vs_eager": round(fp16_spec["vs_fp16_eager"], 2),
                    "int8_spec_tok_s": round(int8_spec["tok_s"], 2),
                    "int8_spec_k": int8_spec["k"],
                    "int8_spec_vs_eager": round(int8_spec["vs_fp16_eager"], 2),
                }
            )

    # Filter Stage 11 block_cost to clean variants
    s11_block = [
        r
        for r in s11["block_cost"]
        if r["variant"] in ("fp16", "int8 Stage 10 kernel", "int8 this stage")
    ]

    data = {
        "source": "chizkidd/mini-vllm results/stage6..stage11.json, Tesla T4",
        "gpu": s11["env"]["gpu"],
        "stage6": {
            "model": s6["env"]["model"],
            "verify_cost": pick(
                s6["verify_cost"],
                ("batch", "tokens_per_seq", "ms", "rel_to_one_token", "ms_per_token"),
            ),
            "speculative": pick(
                s6["speculative"],
                (
                    "workload",
                    "k",
                    "tok_s",
                    "speedup_x",
                    "tokens_per_step",
                    "acceptance",
                    "draft_found_frac",
                    "step_ms",
                    "exact",
                    "n_exact",
                    "n_prompts",
                    "predicted_x",
                ),
            ),
            "oracle": pick(s6["oracle"], ("k", "speedup_x")),
            "projection": pick(
                s6["projection"],
                ("workload", "k", "batch", "tokens_per_step", "rel_cost", "projected_x"),
            ),
        },
        "stage7": {
            "cost": pick(
                s7["cost"],
                ("target", "mode", "draft_ms", "target_ms", "c", "weight_ratio", "layer_ratio"),
            ),
            "verify_cost": pick(
                [r for r in s7["verify_cost"] if r["model"] == "target"],
                ("target", "batch", "tokens_per_seq", "ms", "rel_to_one_token"),
            ),
            "sweep": pick(
                s7["sweep"],
                (
                    "target",
                    "workload",
                    "proposer",
                    "k",
                    "tok_s",
                    "speedup_x",
                    "tokens_per_step",
                    "alpha",
                    "expected_tokens",
                    "acceptance",
                    "target_ms",
                    "guess_ms",
                    "exact",
                    "c_micro",
                    "v",
                    "predicted_x",
                ),
            ),
        },
        "stage8": {
            "memory": pick(
                s8["memory"],
                ("model", "variant", "stored_gib", "allocated_gib"),
            ),
            "perplexity": pick(
                s8["accuracy"]["perplexity"],
                ("model", "variant", "ppl", "tokens"),
            ),
            "speed": pick(
                [r for r in s8["speed"] if r["batch"] == 1],
                ("model", "variant", "mode", "median_ms", "floor_ms", "share"),
            ),
        },
        "stage9": {
            "speed": pick(
                [r for r in s9["speed"] if r["batch"] == 1],
                ("model", "variant", "mode", "median_ms", "floor_ms", "share"),
            ),
            "check_cost": pick(
                s9["check_cost"],
                ("model", "variant", "tokens_per_seq", "ms", "rel_to_one_token"),
            ),
            "acceptance": pick(
                s9["acceptance"],
                (
                    "target",
                    "workload",
                    "proposer",
                    "k",
                    "alpha",
                    "tokens_per_step",
                    "acceptance",
                    "exact",
                ),
            ),
            "cost_ratio": pick(
                s9["cost_ratio"],
                ("target", "draft", "mode", "draft_ms", "target_ms", "c"),
            ),
            "speculation": pick(
                [r for r in s9["speculation"] if r["k"] in (0, 4)],
                (
                    "target",
                    "weights",
                    "workload",
                    "proposer",
                    "k",
                    "tok_s",
                    "speedup_x",
                    "vs_fp16_plain",
                    "alpha",
                    "tokens_per_step",
                    "setup",
                    "exact",
                ),
            ),
        },
        "stage10": {
            "plain": pick(
                s10["plain"]["rows"],
                ("model", "variant", "mode", "decode_tok_s", "step_ms", "same_as_eager"),
            ),
            "block_cost": pick(
                [r for r in s10["block_cost"] if r["mode"] == "graph"],
                ("model", "variant", "tokens", "median_ms", "rel_to_one_token"),
            ),
            "multiply": pick(
                s10["multiply"],
                (
                    "target",
                    "workload",
                    "k",
                    "int8_alone",
                    "ngram_alone_fp16",
                    "ngram_alone_int8",
                    "product",
                    "measured_both",
                    "both_over_product",
                    "v_int8",
                    "v_fp16",
                    "vs_fp16_eager_plain",
                ),
            ),
        },
        "stage11": {
            "plain": pick(
                [r for r in s11["plain"]["rows"] if "head not padded" not in r["setup"]],
                ("model", "setup", "decode_tok_s", "step_ms", "same_as_eager"),
            ),
            "block_cost": pick(
                s11_block,
                ("model", "variant", "tokens", "median_ms", "rel_to_one_token", "floor_ms", "share"),
            ),
            "multiply": pick(
                s11["multiply"],
                (
                    "target",
                    "workload",
                    "k",
                    "int8_alone",
                    "ngram_alone_fp16",
                    "ngram_alone_int8",
                    "product",
                    "measured_both",
                    "measured_both_stage10_kernel",
                    "both_over_product",
                    "both_over_product_stage10_kernel",
                    "new_kernel_gain",
                    "v_int8",
                    "v_int8_stage10_kernel",
                    "v_fp16",
                    "vs_fp16_eager_plain",
                ),
            ),
            "speculation": pick(
                s11["speculation"],
                (
                    "target",
                    "weights",
                    "engine",
                    "workload",
                    "proposer",
                    "k",
                    "tok_s",
                    "speedup_x",
                    "tokens_per_step",
                    "alpha",
                    "setup",
                    "vs_fp16_eager",
                    "vs_fp16_graph",
                    "vs_int8_graph",
                    "exact",
                ),
            ),
            "verdict": round_nested(s11["verdict"], 4),
            "best_stack": best_stack,
        },
    }

    json.dump(data, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS
    main(results_dir)
