"""Effect of a late delivery on the review score.

Late orders differ from on-time ones before anything goes wrong: they ship
farther, to some states more than others, carry more items and cluster around
peak dates. The naive gap in scores mixes the effect of lateness with those
differences, so four estimators are compared, from naive to doubly robust:

1. naive difference in means;
2. OLS with all purchase-time covariates;
3. inverse propensity weighting (IPW);
4. augmented IPW (AIPW) with 5-fold cross-fitting, LightGBM for both the
   propensity and outcome models. Consistent if either model is right.

The estimand is the effect on late orders (ATT): how much higher would the
score of the orders that arrived late have been had they arrived on time.
All reviews are used. Olist sends the survey on the promised delivery date, so
for late orders the review is usually written before the parcel arrives - by a
customer who already knows it is late. Keeping only post-delivery reviews would
drop three quarters of the late orders, and the latest ones. A Cinelli-Hazlett
sensitivity analysis asks how strong an unobserved confounder would need to
be to explain the effect away.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold

NUM = ["distance_km", "same_state", "promised_days", "price", "freight", "freight_share",
       "weight_g", "n_items", "n_sellers", "purchase_dow", "seller_prior_orders"]
CAT = ["customer_state", "seller_state", "category", "purchase_month"]
CLIP = 0.01


def sample(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.review_score.notna()].copy()
    d = d.dropna(subset=["distance_km", "weight_g"])
    for c in CAT:
        d[c] = d[c].fillna("missing").astype("category")
    return d


def _lgb(objective: str, seed: int):
    common = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    return lgb.LGBMClassifier(**common) if objective == "binary" else lgb.LGBMRegressor(**common)


def cross_fit(d: pd.DataFrame, t: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0):
    """Out-of-fold propensity e(x) and outcome models mu0(x), mu1(x)."""
    X = d[NUM + CAT]
    e, mu0, mu1 = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, t)):
        ps = _lgb("binary", seed + k).fit(X.iloc[tr], t[tr])
        e[te] = ps.predict_proba(X.iloc[te])[:, 1]
        for arm, out in ((0, mu0), (1, mu1)):
            idx = tr[t[tr] == arm]
            out[te] = _lgb("reg", seed + k).fit(X.iloc[idx], y[idx]).predict(X.iloc[te])
    return np.clip(e, CLIP, 1 - CLIP), mu0, mu1


def att_aipw(t, y, e, mu0):
    """Doubly robust ATT with an influence-function standard error."""
    p = t.mean()
    psi = (t * (y - mu0) - (1 - t) * e / (1 - e) * (y - mu0)) / p
    att = psi.mean()
    # Subtract the estimand's own contribution for the treated in the influence function.
    infl = psi - t * att / p
    return float(att), float(infl.std(ddof=1) / np.sqrt(len(y)))


def att_ipw(t, y, e):
    w = e / (1 - e)
    return float(y[t == 1].mean() - np.average(y[t == 0], weights=w[t == 0]))


def ols(d: pd.DataFrame, treat: str = "late"):
    f = f"review_score ~ {treat} + " + " + ".join(NUM) + " + " + " + ".join(f"C({c})" for c in CAT)
    return smf.ols(f, data=d).fit(cov_type="HC1")


def robustness_value(t_stat: float, dof: int, q: float = 1.0) -> float:
    """Cinelli & Hazlett (2020): partial R2 an unobserved confounder needs with both
    treatment and outcome to reduce the estimate by 100*q%."""
    f = q * abs(t_stat) / np.sqrt(dof)
    return float(0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2))


def partial_r2(t_stat: float, dof: int) -> float:
    return float(t_stat ** 2 / (t_stat ** 2 + dof))


def benchmark(d: pd.DataFrame) -> dict:
    """Partial R2 of the strongest observed covariate groups, for comparison with the RV."""
    out = {}
    for name in ("distance_km", "promised_days", "n_items", "freight"):
        m_y = ols(d)
        t_y = m_y.tvalues.get(name, 0.0)
        r2_y = partial_r2(t_y, int(m_y.df_resid))
        f = "late ~ " + " + ".join(c for c in NUM) + " + " + " + ".join(f"C({c})" for c in CAT)
        m_t = smf.ols(f, data=d.assign(late=d.late.astype(float))).fit()
        r2_t = partial_r2(m_t.tvalues.get(name, 0.0), int(m_t.df_resid))
        out[name] = {"partial_r2_outcome": r2_y, "partial_r2_treatment": r2_t}
    return out


def dose_response(d: pd.DataFrame, seed: int = 0) -> list[dict]:
    """AIPW ATT for each lateness band against on-time orders."""
    bands = [(1, 3), (4, 7), (8, 14), (15, 1000)]
    rows = []
    on_time = d[d.late == 0]
    for lo, hi in bands:
        sub = pd.concat([on_time, d[(d.late_days >= lo) & (d.late_days <= hi)]])
        t = sub.late.to_numpy()
        y = sub.review_score.to_numpy(dtype=float)
        e, mu0, _ = cross_fit(sub, t, y, seed=seed)
        att, se = att_aipw(t, y, e, mu0)
        rows.append({"band": f"{lo}-{hi if hi < 1000 else '+'} days", "n_late": int(t.sum()),
                     "naive": float(y[t == 1].mean() - y[t == 0].mean()), "att": att, "se": se})
    return rows


def main() -> None:
    df = pd.read_parquet("data/orders.parquet")
    d = sample(df)
    t = d.late.to_numpy()
    y = d.review_score.to_numpy(dtype=float)
    e, mu0, mu1 = cross_fit(d, t, y)
    m = ols(d)
    att, se = att_aipw(t, y, e, mu0)
    rv = robustness_value(m.tvalues["late"], int(m.df_resid))
    rv_ci = robustness_value(m.tvalues["late"] - 1.96 * np.sign(m.tvalues["late"]), int(m.df_resid))
    res = {
        "n": int(len(d)), "n_late": int(t.sum()),
        "late_reviews_written_before_delivery": float(d.loc[d.late == 1, "review_after_delivery"].eq(0).mean()),
        "naive": float(y[t == 1].mean() - y[t == 0].mean()),
        "ols": {"est": float(m.params["late"]), "se": float(m.bse["late"])},
        "ipw": att_ipw(t, y, e),
        "aipw_att": {"est": att, "se": se, "ci95": [att - 1.96 * se, att + 1.96 * se]},
        "overlap": {"propensity_late_p01_p99": np.percentile(e[t == 1], [1, 99]).round(4).tolist(),
                    "propensity_ontime_p01_p99": np.percentile(e[t == 0], [1, 99]).round(4).tolist(),
                    "share_clipped": float(((e <= CLIP) | (e >= 1 - CLIP)).mean())},
        "sensitivity": {"robustness_value": rv, "robustness_value_ci": rv_ci,
                        "partial_r2_late_outcome": partial_r2(m.tvalues["late"], int(m.df_resid)),
                        "benchmarks": benchmark(d)},
        "dose_response": dose_response(d),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/causal.json").write_text(json.dumps(res, indent=2))
    np.save("reports/propensity.npy", np.c_[t, e])
    print(json.dumps({k: res[k] for k in ("n", "n_late", "naive", "ols", "ipw", "aipw_att")}, indent=1))
    print("RV", round(rv, 3), "RV_ci", round(rv_ci, 3))
    for r in res["dose_response"]:
        print(r)
    print(json.dumps(res["sensitivity"]["benchmarks"], indent=1))


if __name__ == "__main__":
    main()
