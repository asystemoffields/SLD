"""Plot the SLD frontier: sequential core rounds and wall-clock speedup vs depth k.

Usage: PYTHONPATH=../SMOKE:.. python bench/plot.py [tag]  ->  results/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    d = json.loads((RES / f"frontier_{tag}.json").read_text())
    ds = d["depth_sweep"]
    ks = [r["k"] for r in ds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(ks, [r["full_rounds"] for r in ds], "o-", label="full loop", color="#444")
    ax1.plot(ks, [r["ee_rounds"] for r in ds], "s--", label="early-exit", color="#c0392b")
    ax1.plot(ks, [r["oracle_log2"] for r in ds], "^:", label="parallel-scan oracle (log2 k)", color="#e67e22")
    ax1.plot(ks, [r["sld_rounds"] for r in ds], "D-", label="SLD (lossless)", color="#2980b9", lw=2.5)
    ax1.set_xlabel("recurrence depth k"); ax1.set_ylabel("sequential core rounds")
    ax1.set_title("SLD collapses depth-k looping to O(1) rounds (lossless)")
    ax1.legend(); ax1.grid(alpha=0.3)

    wc = d["wallclock_b1"]
    ax2.axhline(1.0, color="#999", ls=":", lw=1)
    ax2.plot([r["k"] for r in wc], [r["speedup"] for r in wc], "D-", color="#27ae60", lw=2.5)
    ax2.set_xlabel("recurrence depth k"); ax2.set_ylabel("wall-clock speedup vs full loop")
    ax2.set_title("Batch-1 latency speedup grows with depth")
    ax2.grid(alpha=0.3)
    for r in wc:
        if r["k"] in (6, 16):
            ax2.annotate(f"{r['speedup']:.2f}x", (r["k"], r["speedup"]),
                         textcoords="offset points", xytext=(0, 8), ha="center")

    fig.tight_layout()
    out = RES / f"frontier_{tag}.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
