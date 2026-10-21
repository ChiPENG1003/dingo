"""
Refresh the BatchNorm running statistics of a dingo checkpoint.

Why this is needed: stage_1 unfreezes the RB layer (freeze_rb_layer: False).
The RB layer sits directly in front of BatchNorm, which normalises away its
output scale, so the loss puts no constraint on its norm -- it drifts. By
epoch 214 of main_train_dir_ET its weight norm had grown 20.0 -> 28.9 and the
rows had lost orthonormality (mean diag(WW^T) 1.0 -> 2.08). The 348 BN layers'
running_mean/running_var no longer describe the real activations, so eval()
-- which test_epoch and ALL inference use -- collapses, while train() (batch
stats) is fine.

Measured on model_latest.pt @ epoch 214:
    eval(), stored running stats   +30.27   (training log said +30.10)
    train(), batch stats           -10.60
    eval(), after 50 recal batches -10.18

Usage:
    python recalibrate_bn.py <in.pt> <out.pt> [device] [n_batches]
"""
import copy, sys, torch, numpy as np
sys.path.insert(0, "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo")
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.training.train_builders import set_train_transforms, build_dataset
from dingo.core.utils.torchutils import build_train_and_test_loaders

IN   = sys.argv[1]
OUT  = sys.argv[2]
DEV  = sys.argv[3] if len(sys.argv) > 3 else "cpu"
NMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 200

pm = build_model_from_kwargs(filename=IN, device=DEV)
ts, net = pm.metadata["train_settings"], pm.network
print(f"{IN}  epoch {pm.epoch}", flush=True)

stage = ts["training"]["stage_1"]
wfd = build_dataset(data_settings=copy.deepcopy(ts["data"]), leave_waveforms_on_disk=True)
set_train_transforms(wfd, ts["data"], stage["asd_dataset_path"])
train_loader, test_loader = build_train_and_test_loaders(
    wfd, ts["data"]["train_fraction"], stage["batch_size"], 4)

val = [[d.to(DEV) for d in b] for b, _ in zip(test_loader, range(4))]

def val_loss():
    net.eval()
    with torch.no_grad():
        return float(torch.cat([-net(b[0], *b[1:]) for b in val]).mean())

print(f"before recalibration: val loss {val_loss():8.3f}", flush=True)

# Walk training batches in train() mode under no_grad: weights are untouched,
# only the BN running_mean/running_var buffers are updated.
checkpoints = sorted({25, 50, 100, NMAX})
with torch.no_grad():
    n = 0
    for b in train_loader:
        net.train()
        net(*[d.to(DEV) for d in b])
        n += 1
        if n in checkpoints:
            print(f"  after {n:4d} batches: val loss {val_loss():8.3f}", flush=True)
        if n >= NMAX:
            break

pm.save_model(OUT, save_training_info=True)
print(f"\nwrote {OUT}", flush=True)
