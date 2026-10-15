"""Does a late first delivery stop the customer from coming back?

Only first orders are used (a customer's later orders are partly caused by
how the first one went). Two views:

- Cox proportional hazards on time to the second order, censored at the end
  of the data, adjusted for the purchase-time covariates;
- AIPW on "bought again within 180 days", restricted to first orders placed at
  least 180 days before the data ends, so every customer had the full window.

Repeat buying is rare on Olist (about 3% of customers), so the absolute
numbers are small and the relative ones carry wide intervals.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .causal import CAT, NUM, att_aipw, cross_fit

WINDOW = 180


def main() -> None:
    df = pd.read_parquet("data/orders.parquet")
    first = df[df.order_seq == 0].copy()
    first["event"] = first.days_to_next_order.notna().astype(int)
    first["duration"] = first.days_to_next_order.fillna(first.days_observed).clip(lower=0.5)

    cox_cols = ["late"] + [c for c in NUM if c != "seller_prior_orders"]
    cdf = first[cox_cols + ["duration", "event"]].dropna()
    cph = CoxPHFitter(penalizer=0.01).fit(cdf, "duration", "event")
    hr = cph.summary.loc["late"]

    win = first[first.days_observed >= WINDOW].dropna(subset=["distance_km", "weight_g"]).copy()
    win["back"] = (win.days_to_next_order <= WINDOW).astype(int)
    for c in CAT:
        win[c] = win[c].fillna("missing").astype("category")
    t = win.late.to_numpy()
    y = win.back.to_numpy(dtype=float)
    e, mu0, _ = cross_fit(win, t, y)
    att, se = att_aipw(t, y, e, mu0)

    res = {
        "first_orders": int(len(first)), "repeat_share": float(first.event.mean()),
        "cox": {"hazard_ratio": float(hr["exp(coef)"]), "ci95": [float(hr["exp(coef) lower 95%"]),
                                                                 float(hr["exp(coef) upper 95%"])],
                "p": float(hr["p"])},
        "window_days": WINDOW, "window_customers": int(len(win)),
        "back_rate_on_time": float(y[t == 0].mean()), "back_rate_late": float(y[t == 1].mean()),
        "aipw_att": {"est": att, "se": se, "ci95": [att - 1.96 * se, att + 1.96 * se],
                     "relative_to_on_time": att / float(y[t == 0].mean())},
    }
    Path("reports/retention.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
