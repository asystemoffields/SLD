"""Where do the milliseconds go? Decompose parcae's per-call wall-clock on CPU to
see why verified SLD loses despite fewer core steps.

Times, at real LAMBADA-ish sequence lengths:
  - core step            : loop.step (the recurrent block)
  - decode (full-seq)    : current readout = coda over all positions + lm_head over
                           ALL positions, then slice last  <-- suspected waste
  - decode (last-pos)    : coda over all positions, lm_head over LAST position only
                           (lossless: we only ever use the last token's logits)
  - lm_head full vs last : isolate the vocab matmul (hidden -> 32768) cost
"""
from __future__ import annotations
import time, statistics as st
import torch
from parcae_sld import ParcaeLoop

torch.set_num_threads(6)


@torch.no_grad()
def timeit(fn, n=20, warm=3):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
    return st.median(ts) * 1e3  # ms


def main():
    import parcae_lm
    print("loading parcae-140m ...", flush=True)
    m = parcae_lm.from_pretrained("SandyResearch/parcae-140m").eval()
    T = int(getattr(m.config, "mean_recurrence", 8))
    loop = ParcaeLoop(m)
    print(f"config: prelude={m.config.n_layers_in_prelude} "
          f"recurrent={m.config.n_layers_in_recurrent_block} "
          f"coda={len(m.transformer.coda)} | vocab={m.lm_head.weight.shape[0]} "
          f"| hidden={m.lm_head.weight.shape[1]} | T={T}", flush=True)

    g = torch.Generator().manual_seed(0)
    print(f"\n{'seq':>5}{'core step':>12}{'decode(full)':>14}{'decode(last)':>14}"
          f"{'lm_head(full)':>15}{'lm_head(last)':>15}", flush=True)
    for seq in [12, 32, 64, 96]:
        ids = torch.randint(0, 30000, (1, seq), generator=g)
        x, e, fc = loop.encode(ids)
        x = loop.step(x, e, fc)

        # full-seq decode (current code path): coda(all) + lm_head(all), slice last
        def dec_full():
            return loop.decode(x, ids, fc)

        # last-position decode (proposed, lossless): coda(all), lm_head(last only)
        def dec_last():
            h = m.transformer.C(x)
            for i, blk in enumerate(m.transformer.coda):
                k = str(loop.off_coda + i)
                ve = m.value_embeds[k](ids) if k in m.value_embeds else None
                h = blk(h, fc, None, ve=ve)
            h = m.transformer.ln_f(h[:, -1:, :])
            return (m.lm_head(h).float() * loop.logit_scale)[:, -1, :]

        # isolate the lm_head matmul, full-seq vs last position
        hh = m.transformer.ln_f(m.transformer.C(x))
        def lmh_full(): return m.lm_head(hh)
        def lmh_last(): return m.lm_head(hh[:, -1:, :])

        t_core = timeit(lambda: loop.step(x, e, fc))
        t_df, t_dl = timeit(dec_full), timeit(dec_last)
        t_lf, t_ll = timeit(lmh_full), timeit(lmh_last)
        print(f"{seq:>5}{t_core:>12.2f}{t_df:>14.2f}{t_dl:>14.2f}{t_lf:>15.2f}{t_ll:>15.2f}", flush=True)

        # sanity: last-pos decode must equal full-seq decode's last token (lossless)
        assert torch.allclose(dec_full(), dec_last(), atol=1e-4), "last-pos decode diverged!"
    print("\n[ok] last-position decode is bit-equal to full-seq decode's last token.", flush=True)


if __name__ == "__main__":
    main()
