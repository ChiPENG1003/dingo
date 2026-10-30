"""Rewrite stage_1's batch_size (and lr) inside a checkpoint.

Training settings are read from the .pt on resume, not from the yaml, so this
has to be done here. lr is scaled linearly with the batch size by default
(the standard rule); the result is still well below stage_0's validated 1e-4.
"""
import sys, torch

src, dst, new_bs = sys.argv[1], sys.argv[2], int(sys.argv[3])
lr_mode = sys.argv[4] if len(sys.argv) > 4 else "auto"

d = torch.load(src, map_location="cpu", weights_only=False)
st = d["metadata"]["train_settings"]["training"]["stage_1"]
old_bs, old_lr = st["batch_size"], st["optimizer"]["lr"]
new_lr = old_lr * new_bs / old_bs if lr_mode == "auto" else (
    old_lr if lr_mode == "keep" else float(lr_mode))
st["batch_size"], st["optimizer"]["lr"] = new_bs, new_lr

print(f"epoch {d['epoch']}   stage_progress {d['metadata'].get('stage_progress')}")
print(f"  batch_size {old_bs} -> {new_bs}")
print(f"  lr         {old_lr:g} -> {new_lr:g}")
print(f"  freeze_rb_layer  {st.get('freeze_rb_layer')}   epochs {st['epochs']}")
assert st.get("freeze_rb_layer") is True, "RB layer must stay frozen"
torch.save(d, dst)
print(f"wrote {dst}")
