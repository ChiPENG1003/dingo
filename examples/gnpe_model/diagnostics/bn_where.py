"""Locate the degenerate BN units (running_var < eps) by module group.

If they cluster in the embedding net's early blocks, the cause is the input /
RB-coefficient distribution. If they cluster in the flow's coupling nets, the
cause is the sharpness of the posterior being modelled.
"""
import sys, re, gc, torch, numpy as np
EPS = 1e-3

def group(k):
    if k.startswith("embedding_net"):
        m = re.search(r"blocks\.(\d+)\.", k)
        return f"embedding block {int(m.group(1)):02d}" if m else "embedding other"
    if k.startswith("flow"):
        m = re.search(r"_transforms\.(\d+)\.", k)
        return f"flow step {int(m.group(1)):02d}" if m else "flow other"
    return "other"

for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    agg = {}
    for k, t in sd.items():
        if not k.endswith("running_var"):
            continue
        v = t.float().numpy()
        g = group(k)
        n, bad = agg.get(g, (0, 0))
        agg[g] = (n + v.size, bad + int((v < EPS).sum()))
    tot_n = sum(n for n, _ in agg.values())
    tot_b = sum(b for _, b in agg.values())
    print(f"\n=== {path}  epoch {d['epoch']}   {tot_b}/{tot_n} units < {EPS:g} "
          f"({100*tot_b/tot_n:.2f}%)")
    emb = sum(b for g, (n, b) in agg.items() if g.startswith("embedding"))
    flw = sum(b for g, (n, b) in agg.items() if g.startswith("flow"))
    print(f"    embedding net: {emb:6d}      flow: {flw:6d}")
    for g in sorted(agg, key=lambda g: -agg[g][1])[:8]:
        n, b = agg[g]
        if b:
            print(f"      {g:22s} {b:6d}/{n:6d}  ({100*b/n:5.1f}%)")
    del d, sd; gc.collect()
