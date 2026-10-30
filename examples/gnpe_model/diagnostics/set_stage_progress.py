"""
Record the real stage boundaries on a checkpoint and restore stage_0's
configured length.

patch_checkpoint_stage1.py predates metadata["stage_progress"], so it encodes
"stage_0 finished at epoch 200" by truncating stage_0.epochs to 200. train_stages
now records boundaries explicitly, which says the same thing without falsifying
the configured stage length (and lets the preflight report it).

Usage: python set_stage_progress.py <ckpt.pt> <end_epochs csv> <stage_0 epochs>
"""
import sys, torch

path, ends, stage0_epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
end_epochs = [int(x) for x in ends.split(",") if x]

d = torch.load(path, map_location="cpu", weights_only=False)
tr = d["metadata"]["train_settings"]["training"]
print(f"epoch {d['epoch']}")
print(f"stage_0.epochs: {tr['stage_0']['epochs']} -> {stage0_epochs}")
tr["stage_0"]["epochs"] = stage0_epochs
d["metadata"].setdefault("stage_progress", {})["end_epochs"] = end_epochs
print(f"stage_progress.end_epochs -> {end_epochs}")

# Mirror train_stages' loop entry to confirm where the resume lands.
stages, n = [], 0
while f"stage_{n}" in tr:
    stages.append(tr[f"stage_{n}"]); n += 1
start = len(end_epochs)
s0 = end_epochs[-1] if end_epochs else 0
st = stages[start]
print(f"-> stage_{start}, epochs {d['epoch']+1}..{s0 + st['epochs']}, "
      f"fresh optimizer: {d['epoch'] == s0}, freeze_rb={st.get('freeze_rb_layer')}")
assert d["epoch"] == s0, "checkpoint epoch is not on the recorded boundary"

torch.save(d, path)
print(f"wrote {path}")
