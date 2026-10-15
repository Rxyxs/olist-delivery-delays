"""README figures: dose-response, propensity overlap, and risk-model capture."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("docs/img")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = json.loads(Path("reports/causal.json").read_text())

    # Dose-response
    rows = c["dose_response"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["naive"] for r in rows], 0.4, color="#BAB0AC", label="naive difference")
    ax.bar(x + 0.2, [r["att"] for r in rows], 0.4, color="#E15759", label="doubly robust effect (AIPW)",
           yerr=[1.96 * r["se"] for r in rows], capsize=3)
    for xi, r in zip(x, rows):
        ax.text(xi + 0.2, r["att"] - 0.12, f"{r['att']:.2f}", ha="center", va="top", fontsize=9)
    ax.set_xticks(x, [f"{r['band']}\n(n={r['n_late']:,})" for r in rows])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("change in review score (1-5)")
    ax.set_ylim(-3.0, 0.1)
    ax.set_title("Effect of a late delivery on the review, by days late")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "dose_response.png", dpi=130)
    plt.close(fig)

    # Overlap
    te = np.load("reports/propensity.npy")
    t, e = te[:, 0], te[:, 1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.linspace(0, 0.8, 81)
    ax.hist(e[t == 0], bins, density=True, alpha=0.6, label="on time", color="#4E79A7")
    ax.hist(e[t == 1], bins, density=True, alpha=0.6, label="late", color="#E15759")
    ax.set_xlabel("estimated probability of arriving late (cross-fitted)")
    ax.set_ylabel("density")
    ax.set_title("Propensity overlap: late orders have on-time look-alikes")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "overlap.png", dpi=130)
    plt.close(fig)

    # Risk capture on the temporal test period
    r = np.load("reports/risk_test.npy")
    y, p = r[:, 0], r[:, 1]
    order = np.argsort(-p)
    frac = np.arange(1, len(y) + 1) / len(y)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(frac * 100, np.cumsum(y[order]) / y.sum() * 100, color="#59A14F", lw=2, label="LightGBM at checkout")
    ax.plot([0, 100], [0, 100], "k:", lw=1, label="random")
    ax.set_xlabel("% of orders flagged (riskiest first)")
    ax.set_ylabel("% of late deliveries flagged")
    ax.set_title("Late deliveries caught, orders placed Apr-Aug 2018")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "risk_capture.png", dpi=130)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
