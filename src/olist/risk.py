"""Which orders will arrive late, predicted at the moment of purchase.

Only information available at checkout is used. The seller's track record is
the share of the seller's earlier orders that had *already been delivered*
late by the time of this purchase, so no outcome from the future leaks in.

Validation is temporal: fit on orders bought up to 2018-03-31, score orders
bought from 2018-04-01 on. Three models are compared:
- the rate of late deliveries to the customer's state in the training period;
- logistic regression on the numeric features;
- LightGBM on everything.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .causal import CAT, NUM

CUTOFF = pd.Timestamp("2018-04-01")


def seller_track_record(df: pd.DataFrame) -> pd.Series:
    """Late share of the seller's orders delivered before this order was placed."""
    done = df[["seller_id", "order_delivered_customer_date", "late"]].dropna().sort_values(
        "order_delivered_customer_date")
    done["cum_late"] = done.groupby("seller_id").late.cumsum()
    done["cum_n"] = done.groupby("seller_id").cumcount() + 1
    q = df[["seller_id", "order_purchase_timestamp"]].reset_index().sort_values("order_purchase_timestamp")
    m = pd.merge_asof(q, done[["seller_id", "order_delivered_customer_date", "cum_late", "cum_n"]],
                      left_on="order_purchase_timestamp", right_on="order_delivered_customer_date",
                      by="seller_id", direction="backward", allow_exact_matches=False)
    # Shrink toward the overall rate so a seller with two orders is not 0% or 100%.
    prior, k = df.late.mean(), 20
    rate = (m.cum_late.fillna(0) + k * prior) / (m.cum_n.fillna(0) + k)
    return pd.Series(rate.to_numpy(), index=m["index"]).sort_index()


def capture(y, p, frac):
    order = np.argsort(-p)
    k = max(1, int(frac * len(y)))
    return float(y[order[:k]].sum() / y.sum())


def main() -> None:
    df = pd.read_parquet("data/orders.parquet")
    df["seller_late_rate"] = seller_track_record(df)
    df["seller_has_history"] = (df.seller_prior_orders > 0).astype(int)
    for c in CAT:
        df[c] = df[c].fillna("missing").astype("category")
    feats = NUM + ["seller_late_rate", "seller_has_history"] + [c for c in CAT if c != "purchase_month"]
    tr = df[df.order_purchase_timestamp < CUTOFF]
    te = df[df.order_purchase_timestamp >= CUTOFF]
    ytr, yte = tr.late.to_numpy(), te.late.to_numpy()

    state_rate = tr.groupby("customer_state", observed=True).late.mean()
    p_state = te.customer_state.map(state_rate).astype(float).fillna(ytr.mean()).to_numpy()

    num = NUM + ["seller_late_rate", "seller_has_history"]
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logit.fit(tr[num].fillna(0), ytr)
    p_logit = logit.predict_proba(te[num].fillna(0))[:, 1]

    gbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    gbm.fit(tr[feats], ytr)
    p_gbm = gbm.predict_proba(te[feats])[:, 1]

    res = {"train_orders": int(len(tr)), "test_orders": int(len(te)),
           "late_rate_train": float(ytr.mean()), "late_rate_test": float(yte.mean()), "models": {}}
    for name, p in (("state_rate", p_state), ("logistic", p_logit), ("lightgbm", p_gbm)):
        res["models"][name] = {"auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                               "capture_top10": capture(yte, p, 0.10), "capture_top20": capture(yte, p, 0.20)}
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    res["top_features"] = (imp / imp.sum()).round(4).head(10).to_dict()
    Path("reports/risk.json").write_text(json.dumps(res, indent=2))
    np.save("reports/risk_test.npy", np.c_[yte, p_gbm, te.late_days.to_numpy()])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
