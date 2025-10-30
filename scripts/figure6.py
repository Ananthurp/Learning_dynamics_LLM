#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot (baseline SFT → SP-DPO and ENDPO): log-probability of chosen y_u^+ vs argmax.

Phases:
  - Epochs 0..6 : baseline SFT
  - Epochs 6..12: SP-DPO / ENDPO trained on top of the baseline SFT

Each phase is read from its own run folder (prob_train_metrics.json).
"""

import os, json, argparse
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

RUN_SFT_BASE    = "pythia410m_hh_sft_baseline_e6_tot30k_bs4_lr2e-5_eval100"
RUN_SPDPO_BASE  = "pythia410m_hh_spdpo_from_sftbase_e6_tot30k_bs4_lr1e-6_beta0.1_eval100"
RUN_ENDPO_BASE  = "pythia410m_hh_endpo_from_sftbase_e6_tot30k_bs4_lr1e-6_beta0.1_alpha1.5_gamma0.5_tau1.0_eval100"

KEY_CHOSEN = "logps_eval_prob_train/chosen"
KEY_ARGMAX = "argmax_prob_logits"

# ----------------------------- helpers -----------------------------
def read_series(path: str, key: str) -> List[float]:
    vals: List[float] = []
    if not os.path.isfile(path):
        return vals
    with open(path, "r") as f:
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

def to_epochs(n_pts: int, eval_every: int, examples_per_epoch: int, offset_epoch: float = 0.0) -> np.ndarray:
    if n_pts <= 0:
        return np.array([], dtype=float)
    idx = np.arange(n_pts, dtype=float)
    return offset_epoch + idx * (float(eval_every) / float(examples_per_epoch))

def smooth(y: List[float], frac: float) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    k = max(1, int(len(y) * frac))
    if k % 2 == 0:
        k += 1
    return uniform_filter1d(np.asarray(y, dtype=float), size=k, mode="nearest")

def concat_two_stages(root: str,
                      run_stage1: str,
                      run_stage2: str,
                      key: str,
                      eval_every_stage1: int,
                      eval_every_stage2: int,
                      examples_per_epoch: int) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """
    Return x (epochs), y (values), boundary_epoch, n_pts_stage1, n_pts_stage2
    for a metric that comes from SFT (stage1) then a method phase (stage2).
    """
    p1 = os.path.join(root, run_stage1, "prob_train_metrics.json")
    p2 = os.path.join(root, run_stage2, "prob_train_metrics.json")
    y1 = read_series(p1, key)
    y2 = read_series(p2, key)

    x1 = to_epochs(len(y1), eval_every_stage1, examples_per_epoch, 0.0)
    boundary = x1[-1] if len(x1) > 0 else 0.0
    x2 = to_epochs(len(y2), eval_every_stage2, examples_per_epoch, boundary)

    if len(x1) and len(x2):
        x = np.concatenate([x1, x2]); y = np.concatenate([y1, y2])
    elif len(x1):
        x, y = x1, np.asarray(y1, dtype=float)
    else:
        x, y = x2, np.asarray(y2, dtype=float)

    return x, y, float(boundary), len(y1), len(y2)

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root containing run folders")
    ap.add_argument("--out",  required=True, help="Output image path")
    ap.add_argument("--sft-run",    default=RUN_SFT_BASE)
    ap.add_argument("--spdpo-run",  default=RUN_SPDPO_BASE)
    ap.add_argument("--endpo-run",  default=RUN_ENDPO_BASE)
    ap.add_argument("--examples-per-epoch", type=int, default=5000)
    ap.add_argument("--sft-eval-every",     type=int, default=100)
    ap.add_argument("--method-eval-every",  type=int, default=100, help="Eval interval for SP-DPO/ENDPO")
    ap.add_argument("--smooth-frac",        type=float, default=0.06)
    ap.add_argument("--force-method-start", type=float, default=None,
                    help="Force the vertical line at this epoch (default: compute from SFT length)")
    args = ap.parse_args()

    EPE, SEV, MEV, SMF = int(args.examples_per_epoch), int(args.sft_eval_every), int(args.method_eval_every), float(args.smooth_frac)

    # Build concatenated curves across SFT → SP-DPO and SFT → ENDPO
    # Chosen
    x_c_sp, y_c_sp, boundary_sp, n_sft_c_sp, n_meth_c_sp = concat_two_stages(
        args.root, args.sft_run, args.spdpo_run, KEY_CHOSEN, SEV, MEV, EPE
    )
    x_c_en, y_c_en, boundary_en, n_sft_c_en, n_meth_c_en = concat_two_stages(
        args.root, args.sft_run, args.endpo_run, KEY_CHOSEN, SEV, MEV, EPE
    )
    # Argmax
    x_a_sp, y_a_sp, _, n_sft_a_sp, n_meth_a_sp = concat_two_stages(
        args.root, args.sft_run, args.spdpo_run, KEY_ARGMAX, SEV, MEV, EPE
    )
    x_a_en, y_a_en, _, n_sft_a_en, n_meth_a_en = concat_two_stages(
        args.root, args.sft_run, args.endpo_run, KEY_ARGMAX, SEV, MEV, EPE
    )

    method_start_epoch = float(args.force_method_start) if args.force_method_start is not None else float(boundary_sp or boundary_en)

    # Smooth
    ys_c_sp = smooth(list(y_c_sp), SMF); ys_a_sp = smooth(list(y_a_sp), SMF)
    ys_c_en = smooth(list(y_c_en), SMF); ys_a_en = smooth(list(y_a_en), SMF)

    # Plot
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    # Colors/linestyles: SP-DPO red; ENDPO green
    C_SPD, C_END = "#d62728", "#2ca02c"
    LS_CH, LS_AM = "solid", (0, (5, 3))

    # SP-DPO curves
    ax.plot(x_c_sp, ys_c_sp, linewidth=2.2, color=C_SPD, linestyle=LS_CH, label=r"SP-DPO: Chosen $y_u^+$")
    ax.plot(x_a_sp, ys_a_sp, linewidth=2.2, color=C_SPD, linestyle=LS_AM, label="SP-DPO: Argmax")

    # ENDPO curves
    ax.plot(x_c_en, ys_c_en, linewidth=2.2, color=C_END, linestyle=LS_CH, label=r"ENDPO: Chosen $y_u^+$")
    ax.plot(x_a_en, ys_a_en, linewidth=2.2, color=C_END, linestyle=LS_AM, label="ENDPO: Argmax")

    ax.set_title("Baseline SFT → SP-DPO vs ENDPO: Chosen vs Argmax")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Log probability")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    # Vertical line at method start (default 6.0)
    if np.isfinite(method_start_epoch):
        ax.axvline(method_start_epoch, color="k", linestyle="dashdot", alpha=0.9)

    # Auto y-limits
    stacks = [ys_c_sp, ys_a_sp, ys_c_en, ys_a_en]
    vals = np.concatenate([v for v in stacks if len(v)])
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    rng = vmax - vmin if vmax > vmin else max(1.0, abs(vmin) * 0.1)
    ax.set_ylim(vmin - 0.07 * rng, vmax + 0.07 * rng)

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=240, bbox_inches="tight")
    print(f"[OK] Saved {args.out}")

    # Sanity report
    per_ep = EPE // SEV  # e.g., 5000/100 = 50 evals per epoch
    print("\n[Sanity]")
    print(f"Expected points per epoch: {per_ep}")
    print(f"SFT pts expected (6 epochs): {per_ep * 6}")
    print(f"SP-DPO chosen  SFT_pts={n_sft_c_sp:4d}  AFTER_pts={n_meth_c_sp:4d}  TOTAL={len(x_c_sp):4d}")
    print(f"SP-DPO argmax  SFT_pts={n_sft_a_sp:4d}  AFTER_pts={n_meth_a_sp:4d}  TOTAL={len(x_a_sp):4d}")
    print(f"ENDPO  chosen  SFT_pts={n_sft_c_en:4d}  AFTER_pts={n_meth_c_en:4d}  TOTAL={len(x_c_en):4d}")
    print(f"ENDPO  argmax  SFT_pts={n_sft_a_en:4d}  AFTER_pts={n_meth_a_en:4d}  TOTAL={len(x_a_en):4d}")
    print(f"Vertical line epoch (method start): {method_start_epoch:.3f}")

if __name__ == "__main__":
    main()