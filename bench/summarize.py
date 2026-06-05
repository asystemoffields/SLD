"""Turn results/frontier_<tag>.json into a markdown results block + ASCII plot.

Usage: python bench/summarize.py [tag]   ->   prints markdown to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def bar(v, vmax, width=24, ch="#"):
    n = int(round(width * v / vmax)) if vmax else 0
    return ch * n + "." * (width - n)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    d = json.loads((RES / f"frontier_{tag}.json").read_text())
    ds = d["depth_sweep"]
    out = []
    out.append("### Depth sweep — sequential core rounds (lossless asserted)\n")
    out.append("| k | full-loop | early-exit | parallel-scan oracle | **SLD** | mean accept | lossless |")
    out.append("|--:|--:|--:|--:|--:|--:|:--:|")
    for r in ds:
        ll = "✓" if abs(r["sld_lossless"] - 1) < 1e-9 else f"{r['sld_lossless']:.3f}"
        out.append(f"| {r['k']} | {r['full_rounds']:.0f} | {r['ee_rounds']:.2f} | "
                   f"{r['oracle_log2']} | **{r['sld_rounds']:.2f}** | {r['sld_mean_accept']:.2f} | {ll} |")
    out.append("")
    # ASCII plot of rounds vs k
    vmax = max(r["full_rounds"] for r in ds)
    out.append("```")
    out.append("sequential core rounds vs depth k   (full=#  SLD=o)")
    for r in ds:
        f = bar(r["full_rounds"], vmax, 28, "#")
        s = int(round(28 * r["sld_rounds"] / vmax))
        line = list(f)
        for i in range(min(s, 28)):
            line[i] = "o" if line[i] == "." else "O"
        out.append(f"k={r['k']:>2} |" + "".join(line) + f"  full={r['full_rounds']:.0f} sld={r['sld_rounds']:.1f}")
    out.append("```")
    out.append("")
    # controls
    out.append("### Controls (all lossless; isolate where the win comes from)\n")
    out.append("| k | no-draft rounds | blind-draft rounds | Anderson (training-free) rounds | draft-only acc (lossy, no verify) |")
    out.append("|--:|--:|--:|--:|--:|")
    for r in ds:
        out.append(f"| {r['k']} | {r['nodraft_rounds']:.2f} | {r['blind_rounds']:.2f} | "
                   f"{r['anderson_rounds']:.2f} | {r['draftonly_acc']:.3f} |")
    out.append("")
    # length generalization
    if "lengthgen" in d:
        out.append("### Length generalization — why lossless matters\n")
        out.append("Draft horizon H; for k>H the draft is out-of-distribution. The lossy one-shot "
                   "jump collapses; SLD stays exactly lossless by spending one more verified round.\n")
        out.append("| k | OOD? | draft-only acc (lossy) | SLD acc | SLD lossless | SLD rounds |")
        out.append("|--:|:--:|--:|--:|:--:|--:|")
        for r in d["lengthgen"]:
            ll = "✓" if abs(r["sld_lossless"] - 1) < 1e-9 else f"{r['sld_lossless']:.3f}"
            ood = "" if r["in_train"] else "**OOD**"
            out.append(f"| {r['k']} | {ood} | {r['draftonly_acc']:.3f} | {r['sld_acc']:.3f} | {ll} | {r['sld_rounds']:.2f} |")
        out.append("")
    # horizon
    out.append("### Horizon sweep (k = max)\n")
    out.append("| horizon H | rounds | core rows/example | mean accept | lossless |")
    out.append("|--:|--:|--:|--:|:--:|")
    for r in d["horizon_sweep"]:
        ll = "✓" if abs(r["lossless"] - 1) < 1e-9 else f"{r['lossless']:.3f}"
        out.append(f"| {r['horizon']} | {r['rounds']:.2f} | {r['rows']:.1f} | {r['mean_accept']:.2f} | {ll} |")
    out.append("")
    # wallclock
    for key, label in [("wallclock_b1", "batch-1 latency"), ("wallclock_b64", "batch-64 throughput")]:
        out.append(f"### Wall-clock, {label} (6 threads)\n")
        out.append("| k | full-loop ms | SLD ms | speedup |")
        out.append("|--:|--:|--:|--:|")
        for r in d[key]:
            out.append(f"| {r['k']} | {r['full_ms']:.3f} | {r['sld_ms']:.3f} | {r['speedup']:.2f}× |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
