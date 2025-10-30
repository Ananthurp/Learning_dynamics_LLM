#!/usr/bin/env python3
import os, json, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to the baseline SFT exp_results folder")
    ap.add_argument("--eval_every", type=int, default=None, help="Override eval_every if not parseable from folder name")
    args = ap.parse_args()

    RUN_DIR = args.run_dir.rstrip("/")
    metrics_path = os.path.join(RUN_DIR, "prob_train_metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Could not find {metrics_path}. Check --run_dir.")

    # --- load JSONL (one dict per line)
    rows = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("No rows in prob_train_metrics.json (file exists but is empty).")

    # --- keys to plot: 'logps_eval_prob_train/<name>'
    prefix = "logps_eval_prob_train/"
    series_keys = [k for k in rows[0].keys() if k.startswith(prefix)]
    if not series_keys:
        raise RuntimeError(f"No keys starting with '{prefix}' found in metrics rows.")
    series_keys.sort()
    names = [k.split("/", 1)[1] for k in series_keys]  # legend names

    # --- build dataframe
    data = {n: [] for n in names}
    for r in rows:
        for k, n in zip(series_keys, names):
            data[n].append(r.get(k, float("nan")))
    df = pd.DataFrame(data)

    # --- x-axis in seen examples
    if args.eval_every is not None:
        EVAL_EVERY = args.eval_every
    else:
        m = re.search(r"eval(\d+)", os.path.basename(RUN_DIR))
        EVAL_EVERY = int(m.group(1)) if m else 100  # fallback if not in folder name
    x = np.arange(len(df)) * EVAL_EVERY

    # --- choose which curves to show (only if present)
    preferred = [
        "chosen",
        "rejected",
        "chosen_initial",
        "chosen_gptsemantic",
        "chosen_gptformat",
        "irr_train",
        "irr_test",
        "irr_hum",
        "random_permute",
        "random_nonhum",
    ]
    to_plot = [n for n in preferred if n in df.columns]

    plt.figure(figsize=(9, 6))
    for n in to_plot:
        plt.plot(x, df[n], label=n, linewidth=2)
    plt.xlabel("Seen training examples")
    plt.ylabel("Avg log-prob on prob_train")
    plt.title(f"SFT learning dynamics — {os.path.basename(RUN_DIR)}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(RUN_DIR, "fig3_sft_learning_dynamics.png")
    plt.savefig(out_path, dpi=160)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()