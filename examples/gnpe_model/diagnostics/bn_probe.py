import copy, sys, torch, numpy as np
sys.path.insert(0, "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo")
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.training.train_builders import set_train_transforms, build_dataset
from dingo.core.utils.torchutils import build_train_and_test_loaders

CKPT = sys.argv[1]
DEV  = sys.argv[2] if len(sys.argv) > 2 else "cpu"
print(f"loading {CKPT} on {DEV}", flush=True)
pm = build_model_from_kwargs(filename=CKPT, device=DEV)
ts = pm.metadata["train_settings"]
print("epoch", pm.epoch, flush=True)
net = pm.network

stage = ts["training"]["stage_1"]
wfd = build_dataset(data_settings=copy.deepcopy(ts["data"]), leave_waveforms_on_disk=True)
set_train_transforms(wfd, ts["data"], stage["asd_dataset_path"])
train_loader, test_loader = build_train_and_test_loaders(
    wfd, ts["data"]["train_fraction"], stage["batch_size"], 4)

# back up ONLY the BN buffers (cheap) so train-mode forwards can be undone
bn_keys = [k for k, _ in net.named_buffers()]
bn_backup = {k: b.detach().clone() for k, b in net.named_buffers()}
def restore():
    d = dict(net.named_buffers())
    for k, v in bn_backup.items():
        d[k].copy_(v)

val_batches = [ [d.to(DEV) for d in b] for b, _ in zip(test_loader, range(4)) ]
print(f"got {len(val_batches)} val batches of {val_batches[0][0].shape[0]}", flush=True)

def report(tag, losses):
    l = torch.cat(losses).cpu().numpy()
    q = np.percentile(l, [50, 90, 99, 100])
    print(f"\n{tag}")
    print(f"   mean {l.mean():10.3f}   median {q[0]:8.3f}  p90 {q[1]:8.3f}  p99 {q[2]:8.3f}  max {q[3]:12.1f}")
    print(f"   samples with loss > 0 : {(l>0).sum()} / {l.size}", flush=True)

with torch.no_grad():
    net.eval()
    report("A) eval()  -- exactly what test_epoch does", [-net(b[0], *b[1:]) for b in val_batches])

    net.train()   # dropout is 0.0, so BN batch-stats is the ONLY difference
    ls = [-net(b[0], *b[1:]) for b in val_batches]
    restore()
    report("B) train() -- same val data, BN uses batch stats", ls)

    # C) recalibrate BN running stats on TRAINING data, then eval again
    net.train()
    for i, b in enumerate(train_loader):
        net(*[d.to(DEV) for d in b])
        if i >= 49: break
    net.eval()
    report("C) eval() after 50 train batches of BN recalibration",
           [-net(b[0], *b[1:]) for b in val_batches])
    restore()
