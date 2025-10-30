# #!/usr/bin/env python3
# import os, json, argparse, math
# import numpy as np # type: ignore
# import matplotlib.pyplot as plt # type: ignore

# # ---------- Run names (your 30k / 6-epoch runs) ----------
# RUN_NAMES = {
#     "sft_base":   "pythia410m_hh_sft_baseline_e6_tot30k_bs4_lr2e-5_eval100",
#     "sft_ext":    "pythia410m_hh_sft_extend_e6_tot30k_bs4_lr2e-5_eval100",
#     "dpo_base":   "pythia410m_hh_dpo_from_sftbase_e6_tot30k_bs4_lr1e-6_beta0.1_eval100",
#     "dpo_ext":    "pythia410m_hh_dpo_from_sftextend_e6_tot30k_bs4_lr1e-6_beta0.1_eval100",
#     "spdpo_base": "pythia410m_hh_spdpo_from_sftbase_e6_tot30k_bs4_lr1e-6_beta0.1_eval100",
# }

# # eval cadence + epoch scaling
# EVAL_EVERY_SFT = 100
# EVAL_EVERY_DPO = 100
# N_EXAMPLES_PER_EPOCH_SFT = 30000
# N_EXAMPLES_PER_EPOCH_DPO = 30000

# # keys inside prob_train_metrics.json lines
# KEYS = {
#     "chosen": "logps_eval_prob_train/chosen",
#     "rejected": "logps_eval_prob_train/rejected",
#     "chosen_gptsemantic": "logps_eval_prob_train/chosen_gptsemantic",
#     "chosen_gptformat": "logps_eval_prob_train/chosen_gptformat",
#     "reject_gptsemantic": "logps_eval_prob_train/reject_gptsemantic",
#     "reject_gptformat": "logps_eval_prob_train/reject_gptformat",
#     "random_permute": "logps_eval_prob_train/random_permute",
#     "random_nonhum": "logps_eval_prob_train/random_nonhum",
#     "argmax": "argmax_prob_logits",
# }

# def read_series(run_dir: str, key: str):
#     path = os.path.join(run_dir, "prob_train_metrics.json")
#     if not os.path.exists(path):
#         return []
#     out = []
#     with open(path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 row = json.loads(line)
#             except Exception:
#                 continue
#             v = row.get(key, None)
#             if v is None:
#                 continue
#             if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
#                 continue
#             out.append(float(v))
#     return out

# def ema(arr, alpha=0.5):
#     if len(arr) == 0:
#         return np.array([])
#     out = [arr[0]]
#     for v in arr[1:]:
#         out.append(alpha * v + (1 - alpha) * out[-1])
#     return np.array(out)

# def epochs(n_points, eval_every, n_examples_per_epoch):
#     if n_points == 0:
#         return np.array([])
#     delta = eval_every / float(n_examples_per_epoch)
#     return np.arange(1, n_points + 1) * delta

# def concat_xy(xl, yl, xr, yr):
#     if len(yl) == 0:
#         return xr, yr
#     if len(yr) == 0:
#         return xl, yl
#     return np.concatenate([xl, xr + xl[-1]]), np.concatenate([yl, yr])

# def assemble_concat(root, sft_name, post_name, key):
#     sft = read_series(os.path.join(root, sft_name), KEYS[key])
#     post = read_series(os.path.join(root, post_name), KEYS[key])
#     xs = epochs(len(sft), EVAL_EVERY_SFT, N_EXAMPLES_PER_EPOCH_SFT)
#     xp = epochs(len(post), EVAL_EVERY_DPO, N_EXAMPLES_PER_EPOCH_DPO)
#     x, y = concat_xy(xs, np.array(sft, float), xp, np.array(post, float))
#     return x, y

# def first_turning_point(x, y, alpha=0.5, lookahead=8, drop_eps=2.0):
#     """First epoch where smoothed series peaks then drops by >= drop_eps across lookahead points."""
#     if len(y) < lookahead + 2:
#         return None
#     ys = ema(y, alpha=alpha)
#     k = int(np.argmax(ys))
#     if k >= len(ys) - lookahead:
#         return None
#     after = ys[k+1:k+1+lookahead]
#     if ys[k] - after.min() >= drop_eps:
#         return x[k]
#     return None

# def pad_ylim(minv, maxv, frac=0.08):
#     rng = max(maxv - minv, 1.0)
#     pad = rng * frac
#     return minv - pad, maxv + pad

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", required=True)
#     ap.add_argument("--out", required=True)
#     ap.add_argument("--smooth", type=float, default=0.5)
#     args = ap.parse_args()

#     # Colors/styles
#     C_BASE = "#1f77b4"       # blue
#     C_EXT  = "#ff7f0e"       # orange
#     C_SPD  = "#9467bd"       # purple (distinct)
#     LS_DPO = "solid"
#     LS_SPD = "dashed"

#     fig, ax = plt.subplots(1, 4, figsize=(20, 4))

#     # ===== Panel 1: chosen y+ =====
#     xb, yb = assemble_concat(args.root, RUN_NAMES["sft_base"], RUN_NAMES["dpo_base"], "chosen")
#     xe, ye = assemble_concat(args.root, RUN_NAMES["sft_ext"],  RUN_NAMES["dpo_ext"],  "chosen")
#     xs, ys = assemble_concat(args.root, RUN_NAMES["sft_base"], RUN_NAMES["spdpo_base"], "chosen")

#     yb_s, ye_s, ys_s = ema(yb, args.smooth), ema(ye, args.smooth), ema(ys, args.smooth)
#     ax[0].plot(xb, yb_s, color=C_BASE, linestyle=LS_DPO, label="baseline (DPO)")
#     ax[0].plot(xe, ye_s, color=C_EXT,  linestyle=LS_DPO, label="extend (DPO)")
#     ax[0].plot(xs, ys_s, color=C_SPD,  linestyle=LS_SPD, label="baseline (SP-DPO)")
#     ax[0].set_title(r'Chosen $y_u^+$'); ax[0].set_xlabel('Number of epochs'); ax[0].set_ylabel('Log probability'); ax[0].grid(True)
#     if len(yb_s)+len(ye_s)+len(ys_s):
#         ymin = min([v for v in [*yb_s, *ye_s, *ys_s]])
#         ymax = max([v for v in [*yb_s, *ye_s, *ys_s]])
#         ax[0].set_ylim(*pad_ylim(ymin, ymax))
#     tp = first_turning_point(xb, yb, alpha=args.smooth, lookahead=8, drop_eps=2.0)
#     if tp is not None:
#         ax[0].axvline(tp, color="k", linestyle="dashdot")

#     # ===== Panel 2: rejected y- =====
#     xb2, yb2 = assemble_concat(args.root, RUN_NAMES["sft_base"], RUN_NAMES["dpo_base"], "rejected")
#     xe2, ye2 = assemble_concat(args.root, RUN_NAMES["sft_ext"],  RUN_NAMES["dpo_ext"],  "rejected")
#     xs2, ys2 = assemble_concat(args.root, RUN_NAMES["sft_base"], RUN_NAMES["spdpo_base"], "rejected")

#     yb2_s, ye2_s, ys2_s = ema(yb2, args.smooth), ema(ye2, args.smooth), ema(ys2, args.smooth)
#     ax[1].plot(xb2, yb2_s, color=C_BASE, linestyle=LS_DPO, label="baseline (DPO)")
#     ax[1].plot(xe2, ye2_s, color=C_EXT,  linestyle=LS_DPO, label="extend (DPO)")
#     ax[1].plot(xs2, ys2_s, color=C_SPD,  linestyle=LS_SPD, label="baseline (SP-DPO)")
#     ax[1].set_title(r'Rejected $y_u^-$'); ax[1].set_xlabel('Number of epochs'); ax[1].grid(True)
#     if len(yb2_s)+len(ye2_s)+len(ys2_s):
#         ymin = min([*yb2_s, *ye2_s, *ys2_s]); ymax = max([*yb2_s, *ye2_s, *ys2_s])
#         ax[1].set_ylim(*pad_ylim(ymin, ymax))
#         # annotate peak on baseline rejected
#         k = int(np.argmax(yb2_s))
#         ax[1].annotate(r'$y_u^-$ is the'+'\npeak now', xy=(xb2[k], yb2_s[k]),
#                        xytext=(xb2[max(k-10,0)]+0.2, ymin+0.2*(ymax-ymin)),
#                        arrowprops=dict(arrowstyle="simple", color='darkred'),
#                        color='darkred', fontsize=12, ha='left')

#     # ===== Panel 3: other responses =====
#     def stack(sft_name, post_name):
#         xs_list, ys_list, styles = [], [], []
#         lines = [
#             ("chosen_gptsemantic", "solid"),
#             ("chosen_gptformat",   "dotted"),
#             ("reject_gptsemantic", "dashed"),
#             ("reject_gptformat",   "dashdot"),
#             ("random_permute",     "solid"),
#             ("random_nonhum",      "dotted"),
#         ]
#         for key, ls in lines:
#             x, y = assemble_concat(args.root, sft_name, post_name, key)
#             xs_list.append(x); ys_list.append(ema(y, args.smooth)); styles.append(ls)
#         return xs_list, ys_list, styles

#     xsb, ysb, lsb = stack(RUN_NAMES["sft_base"], RUN_NAMES["dpo_base"])
#     xse, yse, lse = stack(RUN_NAMES["sft_ext"],  RUN_NAMES["dpo_ext"])

#     for x, y, ls in zip(xsb, ysb, lsb):
#         ax[2].plot(x, y, color=C_BASE, linestyle=ls, alpha=0.9)
#     for x, y, ls in zip(xse, yse, lse):
#         ax[2].plot(x, y, color=C_EXT,  linestyle=ls, alpha=0.6)

#     ax[2].set_title('Other responses'); ax[2].set_xlabel('Number of epochs'); ax[2].grid(True)
#     all_y = []
#     for ys_ in ysb + yse:
#         if len(ys_): all_y.extend(list(ys_))
#     if all_y:
#         all_y = np.array(all_y)
#         ax[2].set_ylim(*pad_ylim(all_y.min(), all_y.max()))
#         # annotations
#         xmid = np.median(xb) if len(xb) else 2.0
#         ax[2].annotate('All rephrases', xy=(xmid, np.percentile(all_y, 65)),
#                        xytext=(xmid-1.5, np.percentile(all_y, 50)),
#                        arrowprops=dict(arrowstyle="simple", color='darkred'),
#                        color='darkred', fontsize=12, ha='center')
#         ax[2].annotate('Non-human\nsequence', xy=(xmid, np.percentile(all_y, 15)),
#                        xytext=(xmid-1.5, np.percentile(all_y, 8)),
#                        arrowprops=dict(arrowstyle="simple", color='darkred'),
#                        color='darkred', fontsize=12, ha='center')

#     # ===== Panel 4: Argmax response =====
#     # Concatenated x/y for baseline DPO / extend DPO / SP-DPO:
#     xb3, yb3 = assemble_concat(args.root, RUN_NAMES["sft_base"], RUN_NAMES["dpo_base"], "argmax")
#     xe3, ye3 = assemble_concat(args.root, RUN_NAMES["sft_ext"],  RUN_NAMES["dpo_ext"],  "argmax")
#     xs3, ys3 = assemble_concat(args.root, RUN_NAMES["sft_base"], RUN_NAMES["spdpo_base"], "argmax")
#     yb3_s, ye3_s, ys3_s = ema(yb3, args.smooth), ema(ye3, args.smooth), ema(ys3, args.smooth)

#     if len(yb3_s): ax[3].plot(xb3, yb3_s, color=C_BASE, linestyle=LS_DPO)
#     if len(ye3_s): ax[3].plot(xe3, ye3_s, color=C_EXT,  linestyle=LS_DPO)
#     if len(ys3_s): ax[3].plot(xs3, ys3_s, color=C_SPD,  linestyle=LS_SPD)

#     ax[3].set_title('Argmax response'); ax[3].set_xlabel('Number of epochs'); ax[3].grid(True)

#     # auto y-limits so it always shows
#     all_arg = np.array([v for v in [*yb3_s, *ye3_s, *ys3_s]]) if (len(yb3_s)+len(ye3_s)+len(ys3_s)) else np.array([-110,-100])
#     ax[3].set_ylim(*pad_ylim(all_arg.min(), all_arg.max()))

#     # Put the "start of DPO" arrow at the boundary between SFT and post-SFT for baseline
#     # Compute boundary x from SFT portion length:
#     sft_arg_series = read_series(os.path.join(args.root, RUN_NAMES["sft_base"]), KEYS["argmax"])
#     xs_boundary = epochs(len(sft_arg_series), EVAL_EVERY_SFT, N_EXAMPLES_PER_EPOCH_SFT)
#     if len(xs_boundary) and len(xb3):
#         x_boundary = xs_boundary[-1]
#         y_tip = np.percentile(all_arg, 35)
#         ax[3].annotate('Drop at the\nstart of DPO',
#                        xy=(x_boundary, y_tip),
#                        xytext=(min(x_boundary + 0.6, (xb3[-1] if len(xb3) else x_boundary + 1.0)),
#                                y_tip - 0.08*(all_arg.max()-all_arg.min())),
#                        arrowprops=dict(arrowstyle="simple", color='darkred'),
#                        color='darkred', fontsize=12, ha='center')

#     # legend (only once)
#     ax[0].legend(loc="lower left")
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(args.out), exist_ok=True)
#     plt.savefig(args.out, dpi=180)
#     print(f"Saved {args.out}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5-style plots with correct phase stitching:
- SFT for 6 epochs (baseline or extend), then DPO/SP-DPO/ENDPO for the next 6.
- Vertical line marks the start of the second phase (default 6.0 epochs).
- Reads each phase's prob_train_metrics.json from its own run directory.
"""

import os, json, argparse
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# -----------------------------------------------------------------------------
# Your experiment names (adjust only if the folder names change)
# -----------------------------------------------------------------------------
RUNS = {
    "sft_base":    "pythia410m_hh_sft_baseline_e6_tot30k_bs4_lr2e-5_eval100",
    "sft_ext":     "pythia410m_hh_sft_extend_e6_tot30k_bs4_lr2e-5_eval100",
    "dpo_base":    "pythia410m_hh_dpo_from_sftbase_e6_tot30k_bs4_lr1e-6_beta0.1_eval100",
    "dpo_ext":     "pythia410m_hh_dpo_from_sftextend_e6_tot30k_bs4_lr1e-6_beta0.1_eval100",
    "spdpo_base":  "pythia410m_hh_spdpo_from_sftbase_e6_tot30k_bs4_lr1e-6_beta0.1_eval100",
    # NEW: ENDPO on top of baseline SFT
    "endpo_base":  "pythia410m_hh_endpo_from_sftbase_e6_tot30k_bs4_lr1e-6_beta0.1_alpha1.5_gamma0.5_tau1.0_eval100",
}

# Keys written by your evaluation logger (trainers.BasicTrainer.evaluation on prob_train)
KEYS = {
    "chosen":   "logps_eval_prob_train/chosen",
    "rejected": "logps_eval_prob_train/rejected",
    "cgpt":     "logps_eval_prob_train/chosen_gptsemantic",
    "cfmt":     "logps_eval_prob_train/chosen_gptformat",
    "rgpt":     "logps_eval_prob_train/reject_gptsemantic",
    "rfmt":     "logps_eval_prob_train/reject_gptformat",
    "rperm":    "logps_eval_prob_train/random_permute",
    "rnon":     "logps_eval_prob_train/random_nonhum",
    "argmax":   "argmax_prob_logits",
}

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def read_series(jsonl_path: str, key: str) -> List[float]:
    vals: List[float] = []
    if not os.path.isfile(jsonl_path):
        return vals
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if key in row and row[key] is not None:
                try:
                    v = float(row[key])
                except Exception:
                    continue
                if np.isfinite(v):
                    vals.append(v)
    return vals

def idx_to_epochs(n_pts: int, eval_every: int, examples_per_epoch: int, offset_epochs: float = 0.0) -> np.ndarray:
    if n_pts <= 0:
        return np.array([], dtype=float)
    idx = np.arange(n_pts, dtype=float)
    return offset_epochs + idx * (float(eval_every) / float(examples_per_epoch))

def smooth(y: List[float], frac: float) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    k = max(1, int(len(y) * frac))
    if k % 2 == 0:
        k += 1
    return uniform_filter1d(np.asarray(y, dtype=float), size=k, mode="nearest")

def stitch_sft_then_after(root: str,
                          sft_run: str,
                          after_run: str,
                          key: str,
                          sft_eval_every: int,
                          after_eval_every: int,
                          examples_per_epoch: int) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """
    Returns (x, y, boundary_epoch, n_sft_pts, n_after_pts),
    where boundary_epoch is the start of the 'after' phase on the epoch axis.
    """
    p_sft   = os.path.join(root, sft_run,   "prob_train_metrics.json")
    p_after = os.path.join(root, after_run, "prob_train_metrics.json")

    ys = read_series(p_sft,   key)
    ya = read_series(p_after, key)

    xs = idx_to_epochs(len(ys), sft_eval_every,  examples_per_epoch, 0.0)
    boundary = xs[-1] if len(xs) > 0 else 0.0
    xa = idx_to_epochs(len(ya), after_eval_every, examples_per_epoch, boundary)

    if len(xs) and len(xa):
        x = np.concatenate([xs, xa])
        y = np.concatenate([ys, ya])
    elif len(xs):
        x, y = xs, np.asarray(ys, dtype=float)
    else:
        x, y = xa, np.asarray(ya, dtype=float)

    return x, y, float(boundary), len(ys), len(ya)

def only_after(root: str,
               run: str,
               key: str,
               after_eval_every: int,
               examples_per_epoch: int,
               offset_epoch: float) -> Tuple[np.ndarray, np.ndarray, int]:
    p = os.path.join(root, run, "prob_train_metrics.json")
    y = read_series(p, key)
    x = idx_to_epochs(len(y), after_eval_every, examples_per_epoch, offset_epoch)
    return x, np.asarray(y, dtype=float), len(y)

def set_auto_ylim(ax, series_list: List[np.ndarray], pad_frac: float = 0.06):
    data = [np.asarray(s, float) for s in series_list if s is not None and len(s) > 0]
    if not data:
        return
    vals = np.concatenate(data)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    rng = max(1e-6, vmax - vmin)
    ax.set_ylim(vmin - pad_frac * rng, vmax + pad_frac * rng)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder containing exp_results/<exp_name>")
    ap.add_argument("--out",  required=True, help="Output figure path")
    ap.add_argument("--examples-per-epoch", type=int, default=5000)
    ap.add_argument("--sft-eval-every",     type=int, default=100)
    ap.add_argument("--dpo-eval-every",     type=int, default=100)
    ap.add_argument("--smooth-frac",        type=float, default=0.06)
    ap.add_argument("--force-dpo-start",    type=float, default=None,
                    help="Force the vertical line at this epoch (default: compute from SFT length)")
    args = ap.parse_args()

    EPE, SEV, DEV, SMF = int(args.examples_per_epoch), int(args.sft_eval_every), int(args.dpo_eval_every), float(args.smooth_frac)

    # Colors / styles
    C_BASE = "#1f77b4"   # baseline (DPO)
    C_EXT  = "#ff7f0e"   # extend   (DPO)
    C_SPD  = "#7e2f8e"   # SP-DPO   (baseline)
    C_END  = "#2ca02c"   # ENDPO    (baseline) - green
    LS_SPD = (0, (5, 3))
    LS_END = (0, (2, 2))

    fig, axs = plt.subplots(1, 4, figsize=(22, 4.8), constrained_layout=True)

    # --- Panel 1: Chosen y+ ---
    xb, yb, boundary_b, nsft_b, ndpo_b = stitch_sft_then_after(args.root, RUNS["sft_base"], RUNS["dpo_base"], KEYS["chosen"], SEV, DEV, EPE)
    xe, ye, boundary_e, nsft_e, ndpo_e = stitch_sft_then_after(args.root, RUNS["sft_ext"],  RUNS["dpo_ext"],  KEYS["chosen"], SEV, DEV, EPE)

    start_epoch = boundary_b if boundary_b > 0 else boundary_e
    xs_sp, ys_sp, n_sp = only_after(args.root, RUNS["spdpo_base"], KEYS["chosen"], DEV, EPE, start_epoch)
    xs_en, ys_en, n_en = only_after(args.root, RUNS["endpo_base"], KEYS["chosen"], DEV, EPE, start_epoch)

    dpo_start_epoch = float(args.force_dpo_start) if args.force_dpo_start is not None else float(start_epoch)

    ax = axs[0]
    ax.set_title(r"Chosen $y_u^+$"); ax.set_xlabel("Number of epochs"); ax.set_ylabel("Log probability"); ax.grid(True, alpha=0.3)
    if len(xb): ax.plot(xb,  smooth(yb, SMF),  color=C_BASE, label="baseline (DPO)")
    if len(xe): ax.plot(xe,  smooth(ye, SMF),  color=C_EXT,  label="extend (DPO)")
    if len(xs_sp): ax.plot(xs_sp, smooth(ys_sp, SMF), color=C_SPD, linestyle=LS_SPD, label="baseline (SP-DPO)")
    if len(xs_en): ax.plot(xs_en, smooth(ys_en, SMF), color=C_END, linestyle=LS_END, label="baseline (ENDPO)")
    if np.isfinite(dpo_start_epoch): ax.axvline(dpo_start_epoch, color="k", linestyle="dashdot", alpha=0.9)
    ax.legend(loc="best")
    set_auto_ylim(ax, [smooth(yb,SMF), smooth(ye,SMF), smooth(ys_sp,SMF), smooth(ys_en,SMF)])

    # --- Panel 2: Rejected y- ---
    ax = axs[1]
    ax.set_title(r"Rejected $y_u^-$"); ax.set_xlabel("Number of epochs"); ax.grid(True, alpha=0.3)
    xb2, yb2, _, _, _ = stitch_sft_then_after(args.root, RUNS["sft_base"], RUNS["dpo_base"], KEYS["rejected"], SEV, DEV, EPE)
    xe2, ye2, _, _, _ = stitch_sft_then_after(args.root, RUNS["sft_ext"],  RUNS["dpo_ext"],  KEYS["rejected"], SEV, DEV, EPE)
    xs2_sp, ys2_sp, _ = only_after(args.root, RUNS["spdpo_base"], KEYS["rejected"], DEV, EPE, start_epoch)
    xs2_en, ys2_en, _ = only_after(args.root, RUNS["endpo_base"], KEYS["rejected"], DEV, EPE, start_epoch)

    if len(xb2):   ax.plot(xb2,   smooth(yb2, SMF), color=C_BASE)
    if len(xe2):   ax.plot(xe2,   smooth(ye2, SMF), color=C_EXT)
    if len(xs2_sp): ax.plot(xs2_sp, smooth(ys2_sp, SMF), color=C_SPD, linestyle=LS_SPD)
    if len(xs2_en): ax.plot(xs2_en, smooth(ys2_en, SMF), color=C_END, linestyle=LS_END)
    if np.isfinite(dpo_start_epoch): ax.axvline(dpo_start_epoch, color="k", linestyle="dashdot", alpha=0.9)
    set_auto_ylim(ax, [smooth(yb2,SMF), smooth(ye2,SMF), smooth(ys2_sp,SMF), smooth(ys2_en,SMF)])

    # --- Panel 3: Other responses ---
    ax = axs[2]
    ax.set_title("Other responses"); ax.set_xlabel("Number of epochs"); ax.grid(True, alpha=0.3)
    cats   = ["cgpt", "cfmt", "rgpt", "rfmt", "rperm", "rnon"]
    styles = ["solid", "dotted", "dashed", "dashdot", (0,(1,2)), (0,(3,2))]
    plotted = []
    for cat, ls in zip(cats, styles):
        xb3, yb3, _, _, _ = stitch_sft_then_after(args.root, RUNS["sft_base"], RUNS["dpo_base"], KEYS[cat], SEV, DEV, EPE)
        xe3, ye3, _, _, _ = stitch_sft_then_after(args.root, RUNS["sft_ext"],  RUNS["dpo_ext"],  KEYS[cat], SEV, DEV, EPE)
        xs3_sp, ys3_sp, _ = only_after(args.root, RUNS["spdpo_base"], KEYS[cat], DEV, EPE, start_epoch)
        xs3_en, ys3_en, _ = only_after(args.root, RUNS["endpo_base"], KEYS[cat], DEV, EPE, start_epoch)

        if len(xb3):   s = smooth(yb3, SMF); ax.plot(xb3, s, color=C_BASE, linestyle=ls, alpha=.95); plotted.append(s)
        if len(xe3):   s = smooth(ye3, SMF); ax.plot(xe3, s, color=C_EXT,  linestyle=ls, alpha=.95); plotted.append(s)
        if len(xs3_sp): s = smooth(ys3_sp, SMF); ax.plot(xs3_sp, s, color=C_SPD, linestyle=ls, alpha=.85); plotted.append(s)
        if len(xs3_en): s = smooth(ys3_en, SMF); ax.plot(xs3_en, s, color=C_END, linestyle=ls, alpha=.85); plotted.append(s)

    if np.isfinite(dpo_start_epoch): ax.axvline(dpo_start_epoch, color="k", linestyle="dashdot", alpha=0.9)
    set_auto_ylim(ax, plotted)

    # --- Panel 4: Argmax response ---
    ax = axs[3]
    ax.set_title("Argmax response"); ax.set_xlabel("Number of epochs"); ax.grid(True, alpha=0.3)
    xb4, yb4, _, _, _ = stitch_sft_then_after(args.root, RUNS["sft_base"], RUNS["dpo_base"], KEYS["argmax"], SEV, DEV, EPE)
    xe4, ye4, _, _, _ = stitch_sft_then_after(args.root, RUNS["sft_ext"],  RUNS["dpo_ext"],  KEYS["argmax"], SEV, DEV, EPE)
    xs4_sp, ys4_sp, _ = only_after(args.root, RUNS["spdpo_base"], KEYS["argmax"], DEV, EPE, start_epoch)
    xs4_en, ys4_en, _ = only_after(args.root, RUNS["endpo_base"], KEYS["argmax"], DEV, EPE, start_epoch)

    s_list = []
    if len(xb4):   s = smooth(yb4, SMF);   ax.plot(xb4,   s, color=C_BASE); s_list.append(s)
    if len(xe4):   s = smooth(ye4, SMF);   ax.plot(xe4,   s, color=C_EXT ); s_list.append(s)
    if len(xs4_sp): s = smooth(ys4_sp, SMF); ax.plot(xs4_sp, s, color=C_SPD, linestyle=LS_SPD); s_list.append(s)
    if len(xs4_en): s = smooth(ys4_en, SMF); ax.plot(xs4_en, s, color=C_END, linestyle=LS_END); s_list.append(s)
    if np.isfinite(dpo_start_epoch): ax.axvline(dpo_start_epoch, color="k", linestyle="dashdot", alpha=0.9)
    set_auto_ylim(ax, s_list)

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=240)
    print(f"[OK] Saved: {args.out}")

if __name__ == "__main__":
    main()