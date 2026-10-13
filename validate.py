"""
Independent sanity checks for the Criteo A/B analysis (no statsmodels / xgboost needed).
1) Recompute lifts by plain aggregation on the FULL file (all ~14M rows).
2) Recompute lifts on the 15%-per-chunk random sample using several different seeds.
Seed 0 should reproduce the main analysis (rel. conversion lift ~60.1%, n=2,096,939).
Usage: python validate.py criteo-uplift-v2.1.csv | tee validate_output.txt
"""
import sys
import numpy as np, pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "criteo-uplift-v2.1.csv"
SEEDS = [0, 1, 2, 3, 4]
FRAC = 0.15

def agg(d):
    """rows: control, treated; cols: n, conversions, visits"""
    out = np.zeros((2, 3))
    for tv in (0, 1):
        x = d[d["treatment"] == tv]
        out[tv] = [len(x), x["conversion"].sum(), x["visit"].sum()]
    return out

def report(name, a):
    n_c, cv_c, vs_c = a[0]
    n_t, cv_t, vs_t = a[1]
    pc, pt = cv_c / n_c, cv_t / n_t
    vc, vt = vs_c / n_c, vs_t / n_t
    se = np.sqrt(pt * (1 - pt) / n_t + pc * (1 - pc) / n_c)
    print(f"{name:<14} n={int(n_c + n_t):>10,}  treated share={n_t/(n_c+n_t):.4f}  "
          f"conv {pc:.4%} -> {pt:.4%}  rel lift={(pt - pc) / pc:6.1%}  "
          f"(abs lift 95% CI {pt-pc-1.96*se:.4%} to {pt-pc+1.96*se:.4%})  "
          f"visit rel lift={(vt - vc) / vc:5.1%}")

full = np.zeros((2, 3))
acc = {s: np.zeros((2, 3)) for s in SEEDS}
for chunk in pd.read_csv(PATH, chunksize=1_000_000, usecols=["treatment", "conversion", "visit"]):
    full += agg(chunk)
    for s in SEEDS:
        acc[s] += agg(chunk.sample(frac=FRAC, random_state=s))

print("\n=== Independent recomputation (plain aggregation) ===")
report("FULL DATA", full)
for s in SEEDS:
    report(f"sample seed {s}", acc[s])
print("\nCheck: seed 0 should match the main analysis (n=2,096,939; conv 0.1946% -> 0.3116%; ~60.1% lift).")
