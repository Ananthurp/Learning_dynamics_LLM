# #!/usr/bin/env python3
# import argparse, json, os, numpy as np
# import matplotlib.pyplot as plt

# def load_series(metrics_path, key_prefix):
#     """Return a list of scalars, one per eval window."""
#     vals = []
#     with open(metrics_path, "r") as f:
#         for line in f:
#             row = json.loads(line)
#             # autodetect prefix if needed
#             if key_prefix is None:
#                 if any(k.startswith("logps_eval_prob_train/") for k in row.keys()):
#                     key_prefix = "logps_eval_prob_train/"
#                 elif any(k.startswith("logps_train_prob_train/") for k in row.keys()):
#                     key_prefix = "logps_train_prob_train/"
#                 else:
#                     raise RuntimeError("Could not find eval/train prob_train keys in metrics row.")
#             v = row.get(key_prefix + "chosen", None)
#             if v is None:
#                 # some dumps store a scalar; most store list-of-scalars; accept both
#                 # try eval prefix for backward compat
#                 raise KeyError(f"Missing key: {key_prefix+'chosen'} in metrics row.")
#             # ensure scalar
#             if isinstance(v, list):
#                 vals.append(float(np.mean(v)))
#             else:
#                 vals.append(float(v))
#     return np.array(vals)

# def steps_to_epochs(n_points, eval_every, n_examples_per_epoch):
#     steps = np.arange(n_points) * eval_every
#     return steps / float(n_examples_per_epoch)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--baseline", required=True, help="run dir for baseline SFT")
#     ap.add_argument("--extend",   required=True, help="run dir for extend SFT")
#     ap.add_argument("--spdpo",    required=True, help="run dir for sparsemax-DPO")
#     ap.add_argument("--out", default="fig5_panel1_chosen.png")
#     # training schedule info (epochs = steps/eval_every / (n_examples/epoch))
#     ap.add_argument("--sft-nexamples", type=int, default=5000,
#                     help="examples per epoch for SFT runs (baseline & extend)")
#     ap.add_argument("--sft-eval-every", type=int, default=100,
#                     help="eval_every used in SFT runs")
#     ap.add_argument("--dpo-nexamples", type=int, default=5000,
#                     help="examples per epoch assumed for spDPO epoch axis")
#     ap.add_argument("--dpo-eval-every", type=int, default=200,
#                     help="eval_every used in spDPO run")
#     args = ap.parse_args()

#     # files
#     mbase = os.path.join(args.baseline, "prob_train_metrics.json")
#     mext  = os.path.join(args.extend,   "prob_train_metrics.json")
#     mspd  = os.path.join(args.spdpo,    "prob_train_metrics.json")
#     for p in [mbase, mext, mspd]:
#         if not os.path.exists(p):
#             raise FileNotFoundError(f"Missing: {p}")

#     # load curves
#     b_vals = load_series(mbase, key_prefix=None)
#     e_vals = load_series(mext,  key_prefix=None)
#     s_vals = load_series(mspd,  key_prefix=None)  # spDPO uses standard log-probs at eval time

#     # x-axes: epochs
#     b_x = steps_to_epochs(len(b_vals), args.sft_eval_every, args.sft_nexamples)
#     e_x = steps_to_epochs(len(e_vals), args.sft_eval_every, args.sft_nexamples)
#     s_x = steps_to_epochs(len(s_vals), args.dpo_eval_every, args.dpo_nexamples)

#     # plot
#     plt.figure(figsize=(6,3.2), dpi=160)
#     plt.plot(b_x, b_vals, label="baseline", linewidth=2)
#     plt.plot(e_x, e_vals, label="extend", linewidth=2)
#     plt.plot(s_x, s_vals, label="spDPO", linewidth=2, linestyle="--")

#     plt.title("Chosen $y_u^+$")
#     plt.xlabel("Number of epochs")
#     plt.ylabel("Log probability")
#     plt.grid(True, alpha=0.25)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(args.out)
#     print(f"Saved {args.out}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse, json, os, math
import numpy as np
import matplotlib.pyplot as plt

def load_series(run_dir: str, key: str) -> np.ndarray:
    """
    Reads exp_results/<run>/prob_train_metrics.json (JSONL),
    extracts a scalar per line under `key`, returns float array.
    """
    path = os.path.join(run_dir, "prob_train_metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    vals = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if key not in row:
                # quietly skip rows that don't have the key (robust to logging changes)
                continue
            v = row[key]
            # guard NaNs/inf
            try:
                v = float(v)
                if not (math.isfinite(v)):
                    continue
                vals.append(v)
            except Exception:
                continue
    if len(vals) == 0:
        raise RuntimeError(f"No values for key '{key}' in {path}")
    return np.asarray(vals, dtype=float)

def make_epoch_axis(num_points: int, total_epochs: int) -> np.ndarray:
    """
    Map the logged evaluation points to epoch positions.
    We spread the N points evenly over [1, total_epochs].
    """
    if num_points == 1:
        return np.array([total_epochs], dtype=float)
    return np.linspace(1.0, float(total_epochs), num_points)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="run dir for baseline SFT")
    ap.add_argument("--extend",   required=True, help="run dir for extend SFT")
    ap.add_argument("--spdpo",    required=True, help="run dir for spDPO")
    ap.add_argument("--key", default="logps_eval_prob_train/chosen",
                    help="metrics key to read (default: logps_eval_prob_train/chosen)")
    ap.add_argument("--sft-epochs", type=int, default=6, help="epochs for SFT runs")
    ap.add_argument("--spdpo-epochs", type=int, default=3, help="epochs for spDPO run")
    ap.add_argument("--out", required=True, help="where to save the PNG")
    args = ap.parse_args()

    # Load series
    y_base = load_series(args.baseline, args.key)
    y_ext  = load_series(args.extend,   args.key)
    y_sp   = load_series(args.spdpo,    args.key)

    # Epoch axes
    x_base = make_epoch_axis(len(y_base), args.sft_epochs)
    x_ext  = make_epoch_axis(len(y_ext),  args.sft_epochs)
    x_sp   = make_epoch_axis(len(y_sp),   args.spdpo_epochs)

    # Plot
    plt.figure(figsize=(6,4), dpi=160)
    plt.plot(x_base, y_base, marker="o", label="Baseline (SFT)")
    plt.plot(x_ext,  y_ext,  marker="o", label="Extend (SFT-extend)")
    plt.plot(x_sp,   y_sp,   marker="o", label="spDPO (sparsemax-DPO)")

    plt.xlabel("Epochs")
    plt.ylabel("Log probability (eval on prob_train)")
    plt.title("Fig. 5 – Panel 1: y⁺ (chosen)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()