"""Real-model validation on CPU: does parcae-140m's loop converge, and by how
much? This quantifies the headroom SLD / early-exit have on a real stable looped
LM, using parcae's own `num_steps_pair` API to set the recurrence count (no
fragile hooks). Slow on CPU but real.

Run: /data/llm/.venv/bin/python bench/parcae_cpu.py
"""
from __future__ import annotations

import json, time
from pathlib import Path
import torch

RES = Path(__file__).resolve().parents[1] / "results"


def get_prompts():
    texts = [
        "The capital of France is",
        "Water is made of hydrogen and",
        "In 1969, humans first landed on the",
        "The opposite of hot is",
        "Two plus two equals",
        "The quick brown fox jumps over the lazy",
        "Photosynthesis converts sunlight into",
        "She opened the door and saw a",
    ]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("SandyResearch/parcae-140m")
        enc = [tok(t, return_tensors="pt").input_ids for t in texts]
        return enc, tok
    except Exception as e:
        print("tokenizer load failed (", repr(e)[:80], "); using random ids", flush=True)
        g = torch.Generator().manual_seed(0)
        return [torch.randint(0, 30000, (1, 8), generator=g) for _ in range(8)], None


@torch.no_grad()
def main():
    import parcae_lm
    torch.set_num_threads(6)
    print("loading parcae-140m ...", flush=True)
    m = parcae_lm.from_pretrained("SandyResearch/parcae-140m").eval()
    Tmax = int(getattr(m.config, "mean_recurrence", 8))
    print(f"loaded. mean_recurrence T={Tmax}", flush=True)
    prompts, tok = get_prompts()

    # next-token logits as a function of the number of recurrent loops
    def logits_at(ids, T):
        out = m(ids, num_steps_pair=torch.tensor([T, 0]), return_logits=True)
        return out["logits"][:, -1, :]

    print("\n=== parcae-140m: how the next token settles with recurrence depth ===", flush=True)
    settle_steps = []
    rows = []
    t0 = time.time()
    for pi, ids in enumerate(prompts):
        toks = [logits_at(ids, T).argmax(-1).item() for T in range(1, Tmax + 1)]
        final = toks[-1]
        # first loop count whose next token equals the full-T answer and stays
        settle = Tmax
        for T in range(Tmax, 0, -1):
            if toks[T - 1] == final:
                settle = T
            else:
                break
        settle_steps.append(settle)
        dec = (lambda i: tok.decode([i]) if tok else str(i))
        rows.append({"prompt_idx": pi, "settle_loop": settle, "final_token": final,
                     "token_by_loop": toks})
        print(f"  prompt {pi}: settles by loop {settle}/{Tmax}  "
              f"final={dec(final)!r}  ({time.time()-t0:.0f}s)", flush=True)

    ss = torch.tensor(settle_steps).float()
    print(f"\nmean loops actually needed: {ss.mean():.2f} / {Tmax}", flush=True)
    for T in range(1, Tmax + 1):
        frac = (ss <= T).float().mean().item()
        print(f"  by loop {T}: {frac*100:5.1f}% of prompts already at the final token", flush=True)
    headroom = (Tmax - ss.mean().item())
    print(f"\n=> early-exit / SLD headroom: ~{headroom:.1f} of {Tmax} loops are redundant on average.", flush=True)
    print("   (SLD would verify-and-skip these losslessly; the win scales with depth and, on GPU,", flush=True)
    print("    with batch -- the regime CPU cannot show.)", flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "parcae_cpu.json").write_text(json.dumps(
        {"T": Tmax, "mean_loops_needed": ss.mean().item(),
         "frac_settled_by_loop": {T: (ss <= T).float().mean().item() for T in range(1, Tmax + 1)},
         "per_prompt": rows}, indent=2))
    print("[saved] results/parcae_cpu.json", flush=True)


if __name__ == "__main__":
    main()
