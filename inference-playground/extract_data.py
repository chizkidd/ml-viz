"""Pull the numbers the playground shows out of chizkidd/minivllm's results/*.json.

Usage: python3 extract_data.py /path/to/minivllm/results > data.json
Everything in the output is a measured value (Tesla T4). Nothing is invented here.
"""
import json, sys, os

def load(d, name):
    with open(os.path.join(d, name)) as f:
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

def main(d):
    s3, s4, s5, vc = (load(d, n) for n in ("stage3.json", "stage4.json", "stage5.json", "vllm_compare.json"))
    rf = s3["roofline"]
    data = {
        "source": "chizkidd/minivllm results/*.json, Tesla T4",
        "roofline": {k: rf[k] for k in ("context_len", "spec_bw_gbps", "measured_bw_gbps", "weight_bytes",
                                       "kv_bytes_per_token", "compute_peak_tflops")},
        "batch_scaling": pick(s3["batch_scaling"], ("B", "ttft_ms", "itl_p50_ms", "agg_tok_s", "per_req_tok_s", "kv_mib")),
        "ceiling": pick(rf["points"], ("B", "ceiling_agg_tok_s", "pct_of_ceiling")),
        "closed": pick(s4["closed"], ("B", "policy", "goodput_tok_s", "lat_mean_s", "lat_p99_s", "ttft_mean_s")),
        "load": pick(s4["load"], ("rate", "policy", "goodput_tok_s", "lat_p50_s", "lat_p99_s", "ttft_p50_s", "utilization")),
        "mem": pick(s5["memory"], ("budget_slots", "policy", "pool_mib", "goodput_tok_s", "lat_p50_s", "mean_running", "kv_fill")),
        "kv_breakdown": pick(s5["kv_breakdown"], ("budget_slots", "policy", "stored", "reserved_empty", "rounding", "not_handed_out")),
        "block_size": pick([r for r in s5["block_size"] if r.get("block_size")], ("block_size", "rounding_pct", "kv_fill", "goodput_tok_s")),
        "vllm_load": pick(vc["load"], ("system", "size", "rate", "tpot_p50_ms", "ttft_p50_s", "goodput_tok_s", "req_s", "slo_attainment")),
        "latency": pick(vc["latency"], ("size", "workload", "system", "tpot_ms", "ttft_ms", "speedup_vs_hf")),
        "capacity": vc["capacity"],
        "slo": vc["config"]["slo"],
    }
    json.dump(data, sys.stdout, separators=(",", ":"))

if __name__ == "__main__":
    main(sys.argv[1])
