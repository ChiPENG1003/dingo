"""Decompose the flow's loss into its two terms, in eval() vs train() mode.

For a normalising flow,
    -log p(theta|x) = |z|^2/2 + (d/2) log(2*pi) - log|det J|
with z = T(theta; context) the point mapped into the standard-normal base
distribution. z is not a prediction error: it is the truth expressed in units of
the predicted posterior width. If BN mis-normalisation shifts the predicted
posterior centre by Delta while the posterior width is sigma, then |z| ~ Delta/sigma
and the loss penalty grows as (Delta/sigma)^2 -- so the SAME shift costs
100x more on a posterior that is 10x narrower.
"""
import copy, sys, math, torch, numpy as np
sys.path.insert(0, "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo")
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.training.train_builders import set_train_transforms, build_dataset
from dingo.core.utils.torchutils import build_train_and_test_loaders

CKPT = sys.argv[1]
STAGE = sys.argv[2] if len(sys.argv) > 2 else "stage_1"
pm = build_model_from_kwargs(filename=CKPT, device="cpu")
ts, net = pm.metadata["train_settings"], pm.network
print(f"{CKPT}  epoch {pm.epoch}", flush=True)

stage = ts["training"][STAGE]
wfd = build_dataset(data_settings=copy.deepcopy(ts["data"]), leave_waveforms_on_disk=True)
set_train_transforms(wfd, ts["data"], stage["asd_dataset_path"])
_, test_loader = build_train_and_test_loaders(
    wfd, ts["data"]["train_fraction"], stage["batch_size"], 4)
val = [[d.to("cpu") for d in b] for b, _ in zip(test_loader, range(2))]
bn_backup = {k: b.detach().clone() for k, b in net.named_buffers()}

def parts(batch):
    theta, ctx = batch[0], batch[1:]
    emb = net.embedding_net(*ctx)
    z, logabsdet = net.flow._transform(theta, context=emb)
    return z, logabsdet

d = val[0][0].shape[1]
const = 0.5 * d * math.log(2 * math.pi)
print(f"dim(theta) = {d},  (d/2)log(2pi) = {const:.3f}\n", flush=True)

with torch.no_grad():
    for tag, mode in [("eval()  [test_epoch]", False), ("train() [train_epoch]", True)]:
        net.train() if mode else net.eval()
        Z, L = zip(*[parts(b) for b in val])
        z = torch.cat(Z); lad = torch.cat(L)
        z2 = z.pow(2).sum(1)
        loss = 0.5 * z2 + const - lad
        print(f"{tag}")
        print(f"   |z|^2        mean {z2.mean():10.2f}   median {z2.median():10.2f}"
              f"   (chi^2_{d} would be ~{d})")
        print(f"   per-dim |z|  rms  {z.pow(2).mean().sqrt():10.3f}   "
              f"<- truth in units of predicted posterior sigma")
        print(f"   log|det J|   mean {lad.mean():10.2f}")
        print(f"   -> loss      mean {loss.mean():10.3f}\n", flush=True)
        if mode:
            for k, v in bn_backup.items():
                dict(net.named_buffers())[k].copy_(v)
