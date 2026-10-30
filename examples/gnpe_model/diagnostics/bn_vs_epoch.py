"""Degenerate-BN-unit count vs training epoch, for one train_dir.

Same model, same data, same ASD -- the only thing changing is how sharp the
posterior has become. If the count tracks the loss, sharpness is the cause.
"""
import sys, gc, torch, numpy as np
EPS = 1e-3
for path in sys.argv[1:]:
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{path}: {e}"); continue
    v = np.concatenate([t.float().numpy().ravel()
                        for k, t in d["model_state_dict"].items()
                        if k.endswith("running_var")])
    print(f"epoch {d['epoch']:4d}  units<eps {int((v<EPS).sum()):6d} "
          f"({100*(v<EPS).mean():5.2f}%)  min {v.min():.2e}  median {np.median(v):.3f}",
          flush=True)
    del d, v; gc.collect()
