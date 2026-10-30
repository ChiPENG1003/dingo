#!/usr/bin/env python
"""
Patch a dingo checkpoint so that training resumes at the *start of stage_1*
with a batch size that fits on a 10 GB GPU.

Edits to pm.metadata["train_settings"]["training"]:

0. stage_1 is inserted from a yaml file if the checkpoint has none.  A run
   started while stage_1 was commented out in train_settings.yaml has a
   stage_0-only settings dict baked into the .pt, and train_stages() reads the
   stages from there.

1. stage_0.epochs -> the epoch the checkpoint actually stopped at.
   train_stages() picks the starting stage with
   np.searchsorted(cumsum([stage["epochs"] ...]), pm.epoch + 1).
   When early stopping or a wall-clock timeout ends stage_0 before its
   configured number of epochs, a plain resume lands back in stage_0.
   Truncating stage_0 puts the checkpoint exactly on the stage_0/stage_1
   boundary, which also makes initialize_stage() take the resume=False branch
   and build a fresh optimizer/scheduler from the stage_1 settings.
   Pass --finish-stage-0 to instead run stage_0 out to its configured length
   and roll into stage_1 afterwards.

2. stage_1.batch_size -> smaller, with the learning rate linearly rescaled
   (batch 2048 @ lr 1e-5 does not fit on an RTX 3080).

Training settings are read from the checkpoint on resume, not from
train_settings.yaml, which is why this has to be done here.
"""

import argparse

import torch
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Input .pt (e.g. model_latest.pt)")
    p.add_argument("--out", required=True, help="Output .pt (input is left untouched)")
    p.add_argument(
        "--stage1-settings",
        default=None,
        help="YAML file holding the stage_1 dict. Required when the checkpoint has "
        "no stage_1; overwrites the existing one when it does.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the checks and print the resulting stage plan, write nothing.",
    )
    p.add_argument(
        "--finish-stage-0",
        action="store_true",
        help="Leave stage_0.epochs alone, so the resume finishes the remaining "
        "stage_0 epochs first and enters stage_1 only afterwards. By default "
        "stage_0 is truncated at the checkpoint epoch and stage_1 starts now.",
    )
    p.add_argument(
        "--batch-size",
        default="keep",
        help="New stage_1 batch size, or 'keep' (default) to leave it alone. "
        "Only needs shrinking if the GPU cannot hold the configured batch.",
    )
    p.add_argument(
        "--lr",
        default="auto",
        help="New stage_1 lr. 'auto' (default) scales it linearly with the batch "
        "size, 'keep' leaves it unchanged, or give a number.",
    )
    args = p.parse_args()

    print(f"loading {args.checkpoint} ...")
    d = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    epoch = d["epoch"]
    training = d["metadata"]["train_settings"]["training"]

    print(f"\ncheckpoint epoch: {epoch}")
    print("stages before:")
    for k in sorted(training):
        s = training[k]
        print(
            f"  {k}: epochs={s['epochs']} batch_size={s['batch_size']} "
            f"lr={s['optimizer']['lr']} freeze_rb={s.get('freeze_rb_layer')}"
        )

    # 0. Insert stage_1 if the checkpoint predates it.
    if args.stage1_settings is not None:
        with open(args.stage1_settings) as f:
            stage_1 = yaml.safe_load(f)
        verb = "replacing" if "stage_1" in training else "inserting"
        print(f"\n{verb} stage_1 from {args.stage1_settings}:")
        print(yaml.dump(stage_1, default_flow_style=False, sort_keys=False))
        training["stage_1"] = stage_1
    elif "stage_1" not in training:
        raise SystemExit(
            "ERROR: checkpoint has no stage_1 and no --stage1-settings was given"
        )

    # 1. Truncate stage_0 at the epoch training actually stopped at.
    old_epochs = training["stage_0"]["epochs"]
    if epoch >= old_epochs:
        print(f"stage_0.epochs unchanged ({old_epochs}); stage_0 ran to completion")
    elif args.finish_stage_0:
        print(
            f"stage_0.epochs unchanged ({old_epochs}); --finish-stage-0 given, so "
            f"epochs {epoch + 1}..{old_epochs} of stage_0 run first"
        )
    else:
        training["stage_0"]["epochs"] = epoch
        print(f"stage_0.epochs: {old_epochs} -> {epoch}")

    # 2. Optionally shrink the stage_1 batch, rescaling the lr to match.
    old_bs = training["stage_1"]["batch_size"]
    new_bs = old_bs if args.batch_size == "keep" else int(args.batch_size)
    training["stage_1"]["batch_size"] = new_bs
    print(f"stage_1.batch_size: {old_bs} -> {new_bs}")

    old_lr = training["stage_1"]["optimizer"]["lr"]
    if args.lr == "auto":
        new_lr = old_lr * new_bs / old_bs
    elif args.lr == "keep":
        new_lr = old_lr
    else:
        new_lr = float(args.lr)
    training["stage_1"]["optimizer"]["lr"] = new_lr
    print(f"stage_1.optimizer.lr: {old_lr} -> {new_lr}")

    # Sanity check: reproduce the stage selection that train_stages() will do.
    end_epochs = []
    total = 0
    n = 0
    while f"stage_{n}" in training:
        total += training[f"stage_{n}"]["epochs"]
        end_epochs.append(total)
        n += 1
    import numpy as np

    start = int(np.searchsorted(end_epochs, epoch + 1))
    fresh = epoch == end_epochs[start] - training[f"stage_{start}"]["epochs"]
    print(f"\nend_epochs={end_epochs}")
    print(f"-> resumes in stage_{start}, epochs {epoch + 1}..{end_epochs[start]}")
    print(f"-> new optimizer/scheduler: {fresh}")
    if args.finish_stage_0:
        # Mid-stage_0 resume: keep the checkpoint's optimizer/scheduler, then
        # train_stages() enters stage_1 fresh once stage_0's epochs run out.
        if start != 0:
            raise SystemExit("ERROR: --finish-stage-0 but stage_0 is already over")
        print(f"-> then stage_1, epochs {end_epochs[0] + 1}..{end_epochs[1]}")
    elif start != 1 or not fresh:
        raise SystemExit("ERROR: would not start cleanly at stage_1, refusing to write")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    print(f"\nwriting {args.out} ...")
    torch.save(d, args.out)
    print("done")


if __name__ == "__main__":
    main()
