"""Compare the BatchNorm running_var distribution across checkpoints.

eval() divides by sqrt(running_var + eps) with eps=1e-3 (dingo/glasflow use
BatchNorm1d(features, eps=1e-3)). Units whose running_var sits far below eps are
ones where the stored population statistic carries almost no signal, so eval()
and train() see very different normalisations there.
"""
import sys, gc, torch, numpy as np

for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    v = torch.cat([t.float().flatten() for k, t in sd.items()
                   if k.endswith("running_var")]).numpy()
    m = torch.cat([t.float().flatten() for k, t in sd.items()
                   if k.endswith("running_mean")]).numpy()
    q = np.percentile(v, [0, 1, 5, 50, 95, 100])
    print(f"\n{path}   epoch {d['epoch']}   {v.size} BN units")
    print(f"   running_var  min {q[0]:.3e}  p1 {q[1]:.3e}  p5 {q[2]:.3e}  "
          f"median {q[3]:.3e}  p95 {q[4]:.3e}  max {q[5]:.3e}")
    for thr in (1e-3, 1e-2, 1e-1):
        print(f"     units with running_var < {thr:.0e}: {(v < thr).sum():7d}"
              f"  ({100*(v < thr).mean():5.2f}%)")
    print(f"   |running_mean|  median {np.median(abs(m)):.3e}  max {abs(m).max():.3e}")
    del d, sd, v, m
    gc.collect()
