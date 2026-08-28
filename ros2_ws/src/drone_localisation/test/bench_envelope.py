"""
bench_envelope.py -- not a test. Prints the achievable-performance table:
profile x scale policy, in the RPE terms filter_design.md 7 specifies.

Run from the package root (or anywhere the package imports):
    python3 -m drone_localisation.ekf.bench_envelope     # if installed
    python3 test/bench_envelope.py                       # from source
"""
import contextlib
import dataclasses
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drone_localisation.ekf import sim, so3
from drone_localisation.ekf.filter import EkfCore, Kind
from drone_localisation.ekf.frontend import Scheduler
from drone_localisation.ekf.params import EkfParams

S_TRUE = 1.37
DT = 0.10


def rpe(log, win_s):
    k = int(round(win_s / DT))
    P = np.array([l["p"] for l in log])
    T = np.array([l["p_true"] for l in log])
    e = np.linalg.norm((P[k:] - P[:-k]) - (T[k:] - T[:-k]), axis=1)
    L = np.linalg.norm(T[k:] - T[:-k], axis=1)
    m = L > 0.05
    rel = 100 * np.median(e[m] / L[m]) if m.any() else float("nan")
    return float(np.sqrt((e ** 2).mean())), float(rel)


def run(profile, estimate, s_guess, duration=80.0):
    p = dataclasses.replace(EkfParams(), estimate_scale=1.0 if estimate else 0.0)
    steps, truth = sim.simulate(profile, duration=duration)
    core = EkfCore(p)
    sim.init_core(core, steps[0], s_guess=s_guess, p0=truth["p"][1])
    with contextlib.redirect_stdout(io.StringIO()):
        log = sim.run_frontend(Scheduler(p, core), steps)
    ate = float(np.sqrt(np.mean(
        [np.sum((l["p"] - l["p_true"]) ** 2) for l in log])))
    r1, rel1 = rpe(log, 1.0)
    _, rel2 = rpe(log, 2.0)
    return dict(s=core.x.s, sig=core.x.sigma_s, ate=ate, r1=r1,
                rel1=rel1, rel2=rel2,
                altrej=core.n_rejected[Kind.ALTITUDE])


def main():
    hdr = (f"{'profile':<12}{'policy':<22}{'s':>8}{'sigma_s':>9}"
           f"{'ATErms':>9}{'RPE1s':>9}{'RPE1s%':>9}{'RPE2s%':>9}{'altrej':>8}")
    print(hdr)
    print("-" * len(hdr))
    for prof in ["box", "mixed", "inspection", "vertical", "hover"]:
        for lbl, est, sg in [("estimate", True, 1.0),
                             ("hold, seed exact", False, S_TRUE),
                             ("hold, seed +3%", False, S_TRUE * 1.03)]:
            r = run(prof, est, sg)
            print(f"{prof:<12}{lbl:<22}{r['s']:8.3f}{r['sig']:9.4f}"
                  f"{r['ate']:9.3f}{r['r1']:9.4f}{r['rel1']:9.2f}"
                  f"{r['rel2']:9.2f}{r['altrej']:8d}")
        print()


if __name__ == "__main__":
    main()
    