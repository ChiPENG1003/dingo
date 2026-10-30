"""
Split the RB layer weight by input channel and compare two checkpoints.

Layout (LinearProjectionRB): the layer sees x[:, block] of shape
(num_channels, num_bins) flattened channel-major, so columns are
[ch0 = Re | ch1 = Im | ch2.. = auxiliary/PSD]. init_layers writes the SVD basis
V into columns [:2k] only -- the PSD columns start at EXACTLY zero and are never
initialised with anything.
"""
import sys, torch

def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    W = {k: v.float() for k, v in d["model_state_dict"].items()
         if "layers_rb" in k and k.endswith("weight")}
    return d["epoch"], W

e0, A = load(sys.argv[1])
e1, B = load(sys.argv[2])
print(f"A = epoch {e0}   B = epoch {e1}\n")

for k in sorted(A):
    a, b = A[k], B[k]
    ncols = a.shape[1]
    nch = 3                      # Re, Im, PSD
    kbin = ncols // nch
    print(f"{k}  {tuple(a.shape)}   num_bins={kbin}")
    for ci, name in enumerate(["ch0 Re ", "ch1 Im ", "ch2 PSD"]):
        sa = a[:, ci*kbin:(ci+1)*kbin]
        sb = b[:, ci*kbin:(ci+1)*kbin]
        # per-row squared norm: how much this channel contributes to diag(WW^T)
        print(f"   {name}  |A| {sa.norm():9.4f}  |B| {sb.norm():9.4f}   "
              f"|B-A| {(sb-sa).norm():9.4f}   "
              f"mean row-norm^2  A {sa.pow(2).sum(1).mean():8.4f}  B {sb.pow(2).sum(1).mean():8.4f}")
    print()
