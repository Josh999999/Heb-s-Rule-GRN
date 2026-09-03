#!/usr/bin/env python3
"""
Run the developmental-memory experiments and collect their figures.

Examples
--------
  python3 run_experiments.py                  # all experiments, paper scale
  python3 run_experiments.py 1 2              # only experiments 1 and 2
  python3 run_experiments.py --quick          # fast smoke run (few generations)
  python3 run_experiments.py 3 --gens 5000    # override generation count
  python3 run_experiments.py --outdir results # write figures under results/

Notes
-----
* Experiment 2 is Experiment 1's model run against BOTH targets (S1 and S2),
  i.e. the varying-environment / "developmental memory" condition.
* Experiment 3 uses N = SIDE_3^2 (large). At paper scale (400,000 generations)
  it needs far more time/memory than a laptop run; use --quick or --gens.
"""

import argparse
import glob
import os
import shutil
import sys
import time
import traceback

import numpy as np

import Global
from GRN import diag_mask, sparse_topology, tanh_sigma


# --------------------------------------------------------------------------
# Per-experiment configuration: paper scale and a fast "quick" scale.
# --------------------------------------------------------------------------
PAPER = {
    1: dict(n_generations=200_000),
    2: dict(n_generations=800_000),
    3: dict(n_generations=400_000, V=20),
    4: dict(n_generations=500_000, R=100),
}

QUICK = {
    1: dict(n_generations=5_000),
    2: dict(n_generations=8_000),
    3: dict(n_generations=150, V=6),
    4: dict(n_generations=5_000, R=100),
}


def _banner(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def _collect_strays(folder):
    """Experiment 3 writes some image strips to bare filenames; move them in."""
    moved = 0
    strays = sorted(set(glob.glob("Figure*_imgstrip.png")) | set(glob.glob("Figure[0-9]*.png")))
    for stray in strays:
        if not os.path.isfile(stray):
            continue
        if os.path.dirname(os.path.abspath(stray)) == os.path.abspath(folder):
            continue
        shutil.move(stray, os.path.join(folder, os.path.basename(stray)))
        moved += 1
    return moved


# --------------------------------------------------------------------------
# Experiment runners
# --------------------------------------------------------------------------
def run_experiment1(folder, cfg):
    """Single target (S1), N = 8."""
    Global.reseed()
    Global.N = 8
    Global.FIGURES_OUTPUT = folder

    import Experiment1 as E1
    E1.N = Global.N
    E1.TARGET = Global.S1                      # single-target condition

    B = np.zeros((Global.N, Global.N)) * diag_mask()
    E1.experiment1(alpha=B.copy(), targets=None, R=30, K=4, **cfg)


def run_experiment2(folder, cfg):
    """Two targets (S1 and S2) with switching -- the varying environment."""
    Global.reseed()
    Global.N = 8
    Global.FIGURES_OUTPUT = folder

    import Experiment1 as E1
    E1.N = Global.N
    E1.TARGET = Global.S1

    B = np.zeros((Global.N, Global.N)) * diag_mask()
    E1.experiment1(alpha=B.copy(), targets=Global.TARGETS,
                   switch_every=2000, R=30, K=4, **cfg)


def run_experiment3(folder, cfg):
    """Image recall, N = SIDE_3^2 (large)."""
    Global.reseed()
    Global.N = Global.SIDE_3 * Global.SIDE_3
    Global.FIGURES_OUTPUT = folder

    import Experiment3 as E3
    # Experiment3 binds N / TARGETS / FIGURES_OUTPUT at import time, so set them.
    E3.N = Global.N
    E3.TARGETS = [Global.IMAGE_DARWIN_FLAT, Global.IMAGE_HEB_FLAT]
    E3.FIGURES_OUTPUT = folder

    mask = sparse_topology(k=10, self_interaction=False)
    B = np.zeros((Global.N, Global.N)) * mask
    E3.experiment3(alpha=B.copy(), mask=mask, sigmoid=tanh_sigma,
                   switch_every=2000, speedup=40, dev_time_steps=10,
                   t1_coef=1.0, t2_coef=0.2, prob_mutB=1.0, **cfg)


def run_experiment4(folder, cfg):
    """Eight target patterns, N = SIDE_4^2."""
    Global.reseed()
    Global.N = Global.SIDE_4 * Global.SIDE_4
    Global.FIGURES_OUTPUT = folder

    import Experiment4 as E4
    E4.N = Global.N
    E4.FIGURES_OUTPUT = folder                 # bound at import time

    mask = sparse_topology(k=10, self_interaction=False)
    B = np.zeros((Global.N, Global.N)) * mask
    E4.experiment4(alpha=B.copy(), mask=mask, sigmoid=tanh_sigma,
                   switch_every=2000, speedup=20, dev_time_steps=10,
                   t1_coef=1.0, t2_coef=0.2, prob_mutB=1.0, **cfg)


RUNNERS = {1: run_experiment1, 2: run_experiment2,
           3: run_experiment3, 4: run_experiment4}

DESCRIPTIONS = {
    1: "single target S1 (N=8)",
    2: "two targets S1+S2, varying environment (N=8)",
    3: "image recall, Darwin/Hebb (N=SIDE_3^2)",
    4: "eight patterns, modular environment (N=SIDE_4^2)",
}


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", type=int, choices=[1, 2, 3, 4],
                    help="which experiments to run (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run with few generations")
    ap.add_argument("--gens", type=int, default=None,
                    help="override the generation count for every experiment")
    ap.add_argument("--outdir", default=".",
                    help="parent directory for the figure folders (default: .)")
    args = ap.parse_args(argv)

    todo = args.experiments or [1, 2, 3, 4]
    scales = QUICK if args.quick else PAPER

    os.makedirs(args.outdir, exist_ok=True)

    results, t_all = [], time.perf_counter()

    for n in todo:
        folder = os.path.join(args.outdir, f"Experiment{n}")
        os.makedirs(folder, exist_ok=True)

        cfg = dict(scales[n])
        if args.gens is not None:
            cfg["n_generations"] = args.gens

        _banner(f"Experiment {n} - {DESCRIPTIONS[n]}\n"
                f"generations = {cfg['n_generations']:,} -> {folder}")

        t0 = time.perf_counter()
        try:
            RUNNERS[n](folder, cfg)
            _collect_strays(folder)
            figs = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
            dt = time.perf_counter() - t0
            print(f"  done in {dt:.1f}s - {len(figs)} figures: {', '.join(figs)}")
            results.append((n, "ok", len(figs), dt))
        except Exception:
            dt = time.perf_counter() - t0
            traceback.print_exc()
            print(f"  FAILED after {dt:.1f}s")
            results.append((n, "FAILED", 0, dt))

    _banner("Summary")
    total_figs = 0
    for n, status, nfig, dt in results:
        total_figs += nfig
        print(f"  Experiment {n}: {status:6s}  {nfig:2d} figures  {dt:7.1f}s")
    print(f"\n  {total_figs} figures in {time.perf_counter() - t_all:.1f}s "
          f"(under {os.path.abspath(args.outdir)})")

    return 1 if any(s == "FAILED" for _, s, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
