"""
Criteo Uplift A/B test + uplift modeling  (rough, end-to-end, ~1 run)
Usage:  python criteo_ab_project.py path/to/criteo-uplift-v2.1.csv.gz
Deps :  pip install pandas numpy scipy statsmodels scikit-learn xgboost matplotlib
Outputs: printed numbers (copy into resume/write-up) + figures in ./figs
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

PATH = sys.argv[1] if len(sys.argv) > 1 else "criteo-uplift-v2.1.csv.gz"
RUN_PEEK = False       # step 6 skipped in the quick version
N_ADJ = 2_000_000      # rows used for regression adjustment
N_MODEL = 2_000_000    # rows used for uplift modeling
VALUE_PER_CONV = 50.0  # ASSUMPTION: revenue per conversion ($)
COST_PER_TREAT = 0.02  # ASSUMPTION: cost per ad exposure ($)
RNG = np.random.default_rng(42)
os.makedirs("figs", exist_ok=True)
FEATS = [f"f{i}" for i in range(12)]

# Read in chunks and randomly sample across the WHOLE file (file may be ordered, so head() is not random)
FRAC = 0.15  # ~2M rows out of ~14M
df = pd.concat(
    (c.sample(frac=FRAC, random_state=0) for c in pd.read_csv(PATH, chunksize=1_000_000)),
    ignore_index=True,
)
print("rows:", len(df)); print(df.head(3))
t, conv, visit = df["treatment"].values, df["conversion"].values, df["visit"].values

# ---------------- 1. Health checks ----------------
print("\n=== 1. Health checks ===")
n_t, n_c = (t == 1).sum(), (t == 0).sum()
print(f"treated={n_t:,} control={n_c:,} ratio={n_t/(n_t+n_c):.4f}")
# SRM: design ratio is ~85/15 for Criteo; test against that
exp_ratio = 0.85
chi = stats.chisquare([n_t, n_c], [exp_ratio*(n_t+n_c), (1-exp_ratio)*(n_t+n_c)])
print(f"SRM chi2 p-value vs 85/15: {chi.pvalue:.4f}  (small p => investigate; huge N makes tiny deviations 'significant')")

# Covariate balance: standardized mean difference
smd = {}
for f in FEATS:
    a, b = df.loc[t == 1, f], df.loc[t == 0, f]
    smd[f] = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
smd = pd.Series(smd)
print(f"max |SMD|: {float(smd.abs().max()):.4f} (<0.1 is balanced)")
print("most imbalanced features:", smd.abs().sort_values(ascending=False).head(3).round(4).to_dict())

# Power / MDE
base = conv[t == 0].mean()
print(f"baseline conversion (control): {base:.5%}")
for rel in [0.05, 0.10, 0.20]:
    es = proportion_effectsize(base * (1 + rel), base)
    n = NormalIndPower().solve_power(es, alpha=0.05, power=0.8, ratio=n_t / n_c)
    print(f"  detect {rel:.0%} relative lift -> need ~{n:,.0f} control users (ratio {n_t/n_c:.1f}:1)")

# ---------------- 2. Effect estimation ----------------
print("\n=== 2. Effect estimation ===")
res = []
for name, y in [("conversion", conv), ("visit", visit)]:
    yt, yc = y[t == 1], y[t == 0]
    pt, pc = yt.mean(), yc.mean()
    z, p = proportions_ztest([yt.sum(), yc.sum()], [len(yt), len(yc)])
    se = np.sqrt(pt*(1-pt)/len(yt) + pc*(1-pc)/len(yc))
    ci = (pt - pc - 1.96*se, pt - pc + 1.96*se)
    # bootstrap on aggregated counts (fast binomial resampling)
    bt = RNG.binomial(len(yt), pt, 5000) / len(yt)
    bc = RNG.binomial(len(yc), pc, 5000) / len(yc)
    boot_lift = (bt - bc) / bc
    print(f"{name}: treat={pt:.4%} ctrl={pc:.4%} abs lift={pt-pc:.4%} "
          f"rel lift={(pt-pc)/pc:.1%} 95%CI abs=({ci[0]:.4%},{ci[1]:.4%}) p={p:.2e}")
    print(f"   bootstrap rel-lift 95% CI: ({np.percentile(boot_lift,2.5):.1%}, {np.percentile(boot_lift,97.5):.1%})")
    res.append((name, p))
rej, p_adj, _, _ = multipletests([r[1] for r in res], method="holm")
print("Holm-adjusted p:", dict(zip([r[0] for r in res], p_adj)))

# ---------------- 3. Variance reduction (regression adjustment) ----------------
print("\n=== 3. Regression adjustment ===")
sub = df.sample(min(N_ADJ, len(df)), random_state=1)
X0 = sm.add_constant(sub[["treatment"]])
m0 = sm.OLS(sub["conversion"], X0).fit(cov_type="HC1")
Xc = sub[FEATS].copy(); Xc = (Xc - Xc.mean()) / Xc.std()
X1 = sm.add_constant(pd.concat([sub[["treatment"]], Xc], axis=1))
m1 = sm.OLS(sub["conversion"], X1).fit(cov_type="HC1")
w0 = np.diff(m0.conf_int().loc["treatment"])[0]; w1 = np.diff(m1.conf_int().loc["treatment"])[0]
print(f"ATE unadjusted={m0.params['treatment']:.5f}  CI width={w0:.5f}")
print(f"ATE adjusted  ={m1.params['treatment']:.5f}  CI width={w1:.5f}  -> {1-w1/w0:.1%} narrower")
print(f"adjusted relative lift vs control baseline: {m1.params['treatment']/conv[t == 0].mean():.1%}")
print("(Note: this is covariate/regression adjustment, not CUPED -- no pre-period outcome.)")

# ---------------- 4. Uplift modeling (T-learner) ----------------
print("\n=== 4. Uplift modeling (T-learner, XGBoost) ===")
m = df.sample(min(N_MODEL, len(df)), random_state=2).reset_index(drop=True)
cut = int(len(m) * 0.6)
tr, te = m.iloc[:cut], m.iloc[cut:]
params = dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
              tree_method="hist", n_jobs=-1, eval_metric="logloss")
mt = XGBClassifier(**params).fit(tr.loc[tr.treatment == 1, FEATS], tr.loc[tr.treatment == 1, "conversion"])
mc = XGBClassifier(**params).fit(tr.loc[tr.treatment == 0, FEATS], tr.loc[tr.treatment == 0, "conversion"])
score = mt.predict_proba(te[FEATS])[:, 1] - mc.predict_proba(te[FEATS])[:, 1]

yv, tv = te["conversion"].values, te["treatment"].values
order = np.argsort(-score)
y_s, t_s = yv[order], tv[order]
cyt, cyc = np.cumsum(y_s * t_s), np.cumsum(y_s * (1 - t_s))
cnt, cnc = np.cumsum(t_s), np.cumsum(1 - t_s)
k = np.arange(1, len(y_s) + 1)
uplift_rate = cyt / np.maximum(cnt, 1) - cyc / np.maximum(cnc, 1)
inc = uplift_rate * k                       # cumulative incremental conversions if we treat top-k
qini = cyt - cyc * cnt / np.maximum(cnc, 1) # standard Qini (treated-scale)
idx = np.linspace(0, len(k) - 1, 200).astype(int)
x = k[idx] / len(k)
rand_line = qini[-1] * x
_trap = getattr(np, "trapezoid", None) or np.trapz
auuc_gain = _trap(qini[idx] - rand_line, x)
print(f"Qini area vs random (higher=better): {auuc_gain:.1f}")
plt.figure(figsize=(6, 4)); plt.plot(x, qini[idx], label="T-learner"); plt.plot(x, rand_line, "--", label="Random")
plt.xlabel("Fraction of users targeted"); plt.ylabel("Qini (incremental conversions)")
plt.legend(); plt.tight_layout(); plt.savefig("figs/qini.png", dpi=150); plt.close()

# ---------------- 5. Business simulation ----------------
print("\n=== 5. Targeting policy simulation ===")
total_inc = inc[-1]
profit = inc * VALUE_PER_CONV - COST_PER_TREAT * k
profit_all = profit[-1]
best = profit.argmax()
print(f"Treat everyone: incremental conv={total_inc:.0f}, profit=${profit_all:,.0f}")
print(f"Best policy: treat top {k[best]/len(k):.0%} -> captures {inc[best]/total_inc:.0%} of incremental conv, "
      f"profit=${profit[best]:,.0f} (vs ${profit_all:,.0f} blanket)")
for f in [0.1, 0.2, 0.3, 0.5]:
    i = int(len(k) * f) - 1
    print(f"  top {f:.0%}: captures {inc[i]/total_inc:.0%} of incremental conversions")
print("NOTE: top-k estimates are noisy at small k; value/cost are assumptions.")

if not RUN_PEEK:
    print("\nDone (step 6 skipped). Figures in ./figs."); sys.exit()

# ---------------- 6. Peeking simulation (A/A) ----------------
print("\n=== 6. Peeking simulation (A/A test, no true effect) ===")
n_sim, n_per, looks, p0 = 2000, 20000, 10, 0.003
fp_final = fp_peek = 0
for _ in range(n_sim):
    a = RNG.random(n_per) < p0; b = RNG.random(n_per) < p0
    hit = False
    for L in range(1, looks + 1):
        n = n_per * L // looks
        sa, sb = a[:n].sum(), b[:n].sum()
        if sa + sb == 0: continue
        _, p = proportions_ztest([sa, sb], [n, n])
        if p < 0.05: hit = True
        if L == looks: fp_final += p < 0.05
    fp_peek += hit
print(f"False-positive rate: single look at end={fp_final/n_sim:.1%}, peeking {looks}x={fp_peek/n_sim:.1%} (nominal 5%)")
print("Fix: pre-register sample size, or use sequential testing (e.g. alpha spending / mSPRT).")
print("\nDone. Figures in ./figs. Copy the numbers above into your resume bullets.")
