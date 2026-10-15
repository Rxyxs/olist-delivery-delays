import numpy as np
import pandas as pd
import pytest

from olist.causal import att_aipw, att_ipw, robustness_value
from olist.data import haversine_km
from olist.risk import seller_track_record


def _confounded(n=20000, effect=-2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = 1 / (1 + np.exp(-(x - 2)))           # treatment more likely when x is high
    t = (rng.random(n) < e).astype(int)
    y = 4 + 0.8 * x + effect * t + rng.normal(scale=0.5, size=n)
    mu0 = 4 + 0.8 * x
    return t, y, e, mu0, x


def test_aipw_recovers_effect_under_confounding():
    t, y, e, mu0, _ = _confounded()
    naive = y[t == 1].mean() - y[t == 0].mean()
    att, se = att_aipw(t, y, e, mu0)
    assert abs(naive - (-2.0)) > 0.5          # confounding biases the naive gap
    assert att == pytest.approx(-2.0, abs=4 * se + 0.02)


def test_aipw_is_doubly_robust_to_a_wrong_outcome_model():
    t, y, e, _, _ = _confounded(seed=1)
    att, se = att_aipw(t, y, e, np.full_like(y, y.mean()))  # useless outcome model
    assert att == pytest.approx(-2.0, abs=4 * se + 0.05)


def test_ipw_recovers_effect():
    t, y, e, _, _ = _confounded(seed=2)
    assert att_ipw(t, y, e) == pytest.approx(-2.0, abs=0.1)


def test_robustness_value_grows_with_t_stat():
    assert 0 < robustness_value(5, 10000) < robustness_value(50, 10000) < 1


def test_haversine_santiago_sao_paulo():
    assert haversine_km(-33.45, -70.67, -23.55, -46.63) == pytest.approx(2590, rel=0.02)


def test_seller_track_record_uses_only_past_deliveries():
    t = pd.Timestamp
    df = pd.DataFrame({
        "seller_id": ["a", "a", "a"],
        "order_purchase_timestamp": [t("2018-01-01"), t("2018-01-02"), t("2018-01-20")],
        "order_delivered_customer_date": [t("2018-01-15"), t("2018-01-10"), t("2018-02-01")],
        "late": [1, 0, 1],
    })
    r = seller_track_record(df)
    prior = df.late.mean()
    # order 2 is placed before any delivery; order 3 after both earlier deliveries (one late)
    assert r.iloc[1] == pytest.approx(prior)
    assert r.iloc[2] == pytest.approx((1 + 20 * prior) / (2 + 20))
