"""Within the flow's transform_net (a small ResidualNet), which depth do the
degenerate BN units sit at? blocks.0 is nearest the input, blocks.4 nearest the
output that emits the spline parameters."""
import sys, re, torch, numpy as np
EPS = 1e-3
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
agg = {}
for k, t in d["model_state_dict"].items():
    if not k.endswith("running_var") or "transform_net" not in k:
        continue
    m = re.search(r"blocks\.(\d+)\.batch_norm_layers\.(\d+)", k)
    if not m:
        continue
    key = (int(m.group(1)), int(m.group(2)))
    v = t.float().numpy()
    n, b = agg.get(key, (0, 0))
    agg[key] = (n + v.size, b + int((v < EPS).sum()))
print(f"{sys.argv[1]}  epoch {d['epoch']}")
print("  block.bn    degenerate / total     %")
for key in sorted(agg):
    n, b = agg[key]
    print(f"   blocks.{key[0]}.bn{key[1]}   {b:6d} / {n:6d}   {100*b/n:6.2f}%")
