#!/usr/bin/env python
"""Cheap BN-health check for a training run. Reads only the checkpoint's tensors
-- no dataset, no forward pass -- so it takes seconds, not minutes.

The RB weight is split by input channel (columns are channel-major:
[ch0 Re | ch1 Im | ch2 1/ASD]):

  signal drift  |W_sig - W_sig_ref| / |W_sig_ref| over ch0+ch1. This is the SVD
                basis. It must stay 0 under freeze_rb_layer: true or signal.
                It was 1.12 on the ET run that collapsed (16 epochs unfrozen).
  signal orth   max|W_sig W_sig^T - I|: 8e-07 for an intact basis, 1.24 collapsed.
  ASD columns   norm of ch2. Starts at exactly 0 and stays 0 while frozen; is
                EXPECTED to grow under freeze_rb_layer: signal with many ASDs
                (LVK learned 0 -> 16.8), so it is reported, not flagged.

Measured eval()-vs-train() gaps for calibration:
  ET  model_stage_0.pt (RB frozen,   signal drift 0)     gap  0.241
  ET  model_latest.pt  (RB unfrozen, signal drift 1.12)  gap 40.87  <- collapsed
  LVK model_stage_1.pt (RB unfrozen)                     gap  0.059  (few degenerate units)

Usage:
  python bn_health.py <ckpt.pt> [reference.pt]
Exit code 1 if the signal basis moved or lost orthonormality.
"""
import gc, sys, torch, numpy as np

EPS = 1e-3
DRIFT_WARN, ORTH_WARN = 0.05, 0.05
N_CHANNELS = 3  # RepackageStrainsAndASDS: Re, Im, 1/(ASD*1e23)

def rb_weights(sd):
    return {k: v.float() for k, v in sd.items()
            if "layers_rb" in k and k.endswith("weight")}

def split(w):
    k = w.shape[1] // N_CHANNELS
    return w[:, : 2 * k], w[:, 2 * k :]

# One checkpoint at a time, keeping only what is needed: a .pt with optimizer
# state is ~4.4 GB, and holding two at once gets killed on a login node.
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch, sd = ckpt["epoch"], ckpt["model_state_dict"]
v = np.concatenate([t.float().numpy().ravel()
                    for k, t in sd.items() if k.endswith("running_var")])
W_now = rb_weights(sd)
del ckpt, sd
gc.collect()
print(f"{sys.argv[1]}  epoch {epoch}")
print(f"  degenerate BN units (running_var < {EPS:g}): {int((v < EPS).sum())} "
      f"/ {v.size} ({100*(v < EPS).mean():.2f}%)")
print("     -- amplifiers, not damage; harmless while activation means are stable")

ref = None
if len(sys.argv) > 2:
    ref = rb_weights(torch.load(sys.argv[2], map_location="cpu",
                                weights_only=False)["model_state_dict"])
    gc.collect()

bad = False
for k, w in sorted(W_now.items()):
    sig, asd = split(w)
    orth = (sig @ sig.T - torch.eye(sig.shape[0])).abs().max().item()
    line = f"  {k.split('.')[-2]}: signal max|WW^T-I| {orth:.3g}"
    if ref is not None:
        rsig, rasd = split(ref[k])
        drift = ((sig - rsig).norm() / rsig.norm()).item()
        line += f"   signal drift {drift:.4f}"
        if drift > DRIFT_WARN:
            line += " <-- WARNING"; bad = True
        line += f"   ASD columns |W| {rasd.norm():.3f} -> {asd.norm():.3f}"
    else:
        line += f"   ASD columns |W| {asd.norm():.3f}"
    if orth > ORTH_WARN:
        line += "  <-- SIGNAL BASIS LOST ORTHONORMALITY"; bad = True
    print(line)

print("\nVERDICT:", "RB signal basis is drifting -- run bn_probe.py before trusting eval()/inference"
      if bad else "RB signal basis intact; eval() should track train()")
sys.exit(1 if bad else 0)
