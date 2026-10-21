from typing import Optional, Tuple
import os

import h5py
import numpy as np
import yaml
import argparse
import shutil
import textwrap
import time
from copy import deepcopy

from threadpoolctl import threadpool_limits

from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.gw.training.train_builders import (
    build_dataset,
    set_train_transforms,
    build_svd_for_embedding_network,
)
from dingo.core.utils.trainutils import RuntimeLimits
from dingo.core.utils import (
    set_requires_grad_flag,
    get_number_of_model_parameters,
    build_train_and_test_loaders,
)
from dingo.core.utils.trainutils import EarlyStopping
from dingo.gw.dataset import WaveformDataset
from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.nn.enets import LinearProjectionRB


def copy_files_to_local(
    file_path: str, local_dir: Optional[str], leave_keys_on_disk: bool, is_condor: bool = False,
) -> str:
    """
    Copy files to local node if local_dir is provided to minimize network traffic during training.

    Parameters
    ----------
    file_path: str
        Path to file that should be copied.
    local_dir: Optional[str]
        Directory where file should be copied. If None, file will not be copied.
    leave_keys_on_disk: bool
        Whether to leave keys on disk and load them during training. If dataset is not copied and
        leave_keys_on_disk is True, a warning will be raised.
    is_condor: bool
        Whether this is a condor job.

    Returns
    -------
    local_file_path: str
        Modified file path if file was copied to local node, else the original file path.
    """
    local_file_path = file_path
    if local_dir is not None:
        file_name = file_path.split("/")[-1]
        local_file_path = os.path.join(local_dir, file_name)
        print(f"Copying file to {local_file_path}")
        # Copy file
        start_time = time.time()
        shutil.copy(file_path, local_file_path)
        elapsed_time = time.time() - start_time
        print("Done. This took {:2.0f}:{:2.0f} min.".format(*divmod(elapsed_time, 60)))
    elif leave_keys_on_disk and is_condor:
        print(
            f"Warning: leave_waveforms_on_disk defaults to True, but local_cache_path is not specified. "
            f"This means that the waveforms will be loaded during training from {local_file_path} ."
            f"This can lead to unexpected long times for data loading during training due to network traffic. "
            f"To prevent this, specify 'local_cache_path = tmp' in the local settings or set "
            f"leave_waveforms_on_disk = False. However, the latter is not recommended for large datasets since "
            f"it can lead to memory issues when loading the entire dataset into RAM. "
        )

    return local_file_path


def prepare_training_new(
    train_settings: dict, train_dir: str, local_settings: dict
) -> Tuple[BasePosteriorModel, WaveformDataset]:
    """
    Based on a settings dictionary, initialize a WaveformDataset and PosteriorModel.

    For model type 'nsf+embedding' (the only acceptable type at this point) this also
    initializes the embedding network projection stage with SVD V matrices based on
    clean detector waveforms.

    Parameters
    ----------
    train_settings : dict
        Settings which ultimately come from train_settings.yaml file.
    train_dir : str
        This is only used to save diagnostics from the SVD.
    local_settings : dict
        Local settings (e.g., num_workers, device)

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """
    data_settings = deepcopy(train_settings["data"])
    # Optionally copy files to local and update path
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )
    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
    )  # No transforms yet
    initial_weights = {}

    # The embedding network is assumed to have an SVD projection layer. If other types
    # of embedding networks are added in the future, update this code.

    if train_settings["model"].get("embedding_kwargs", None):
        # First, build the SVD for seeding the embedding network.
        print("\nBuilding SVD for initialization of embedding network.")
        initial_weights["V_rb_list"] = build_svd_for_embedding_network(
            wfd,
            train_settings["data"],
            train_settings["training"]["stage_0"]["asd_dataset_path"],
            num_workers=local_settings["num_workers"],
            batch_size=train_settings["training"]["stage_0"]["batch_size"],
            out_dir=train_dir,
            **train_settings["model"]["embedding_kwargs"]["svd"],
        )

    # Now set the transforms for training. We need to do this here so that we can (a)
    # get the data dimensions to configure the network, and (b) save the
    # parameter standardization dict in the PosteriorModel. In principle, (a) could
    # be done without generating data (by careful calculation) and (b) could also
    # be done outside the transform setup. But for now, this is convenient. The
    # transforms will be reset later by initialize_stage().

    set_train_transforms(
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset_path"],
    )

    # This modifies the model settings in-place.
    autocomplete_model_kwargs(train_settings["model"], wfd[0])
    full_settings = {
        "dataset_settings": wfd.settings,
        "train_settings": train_settings,
    }

    print("\nInitializing new posterior model.")
    print("Complete settings:")
    print(yaml.dump(full_settings, default_flow_style=False, sort_keys=False))

    pm = build_model_from_kwargs(
        settings=full_settings,
        initial_weights=initial_weights,
        device=local_settings["device"],
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                config=full_settings,
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    return pm, wfd


def prepare_training_resume(
    checkpoint_name: str, local_settings: dict, train_dir: str
) -> Tuple[BasePosteriorModel, WaveformDataset]:
    """
    Loads a PosteriorModel from a checkpoint, as well as the corresponding
    WaveformDataset, in order to continue training. It initializes the saved optimizer
    and scheduler from the checkpoint.

    Parameters
    ----------
    checkpoint_name : str
        File name containing the checkpoint (.pt format).
    local_settings : dict
        Local settings (e.g., num_workers, device)
    train_dir: str
        Path to training directory where the wandb info is saved.

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """

    pm = build_model_from_kwargs(
        filename=checkpoint_name, device=local_settings["device"]
    )
    data_settings = deepcopy(pm.metadata["train_settings"]["data"])
    # Optionally copy files to local and update path
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )
    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                resume="must",
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    return pm, wfd


def _set_rb_layer_trainability(network, mode) -> None:
    """
    Apply a stage's freeze_rb_layer setting to the reduced-basis (RB) projection.

    true / false freeze or train the whole RB layer, exactly as before.

    "signal" freezes only the part that holds the SVD basis -- the columns acting
    on the strain's real and imaginary channels, plus the bias -- and trains the
    columns acting on the third input channel, 1/ASD. Those columns are initialised
    to zero and stay zero while the layer is frozen, so with a frozen RB layer the
    network never sees the ASD. Unfreezing the whole layer instead lets the signal
    columns drift: nothing in the loss constrains their scale, and on the ET run
    they grew 14.1 -> 21.0 in 16 epochs while eval() collapsed (val -10.6 -> +30).
    "signal" keeps the basis fixed and still lets a multi-ASD stage use the ASD.

    The freeze is a gradient mask. It holds exactly under Adam (zero gradient ->
    zero moments -> zero update) but not under decoupled weight decay (AdamW).
    """
    rb_modules = [m for m in network.modules() if isinstance(m, LinearProjectionRB)]
    for m in rb_modules:
        for handle in getattr(m, "_signal_freeze_hooks", []):
            handle.remove()
        m._signal_freeze_hooks = []

    if isinstance(mode, bool):
        set_requires_grad_flag(network, name_contains="layers_rb", requires_grad=not mode)
        return
    if mode != "signal":
        raise ValueError(f"freeze_rb_layer must be true, false or 'signal', got {mode!r}")
    if not rb_modules:
        raise ValueError("freeze_rb_layer: signal, but the network has no RB layer")

    for m in rb_modules:
        n_signal = 2 * m.num_bins  # columns for channel 0 (Re) and channel 1 (Im)
        for layer in m.layers_rb:
            layer.bias.requires_grad = False
            layer.weight.requires_grad = True

            def mask_signal_columns(grad, n=n_signal):
                grad = grad.clone()
                grad[:, :n] = 0
                return grad

            m._signal_freeze_hooks.append(layer.weight.register_hook(mask_signal_columns))


def _check_stage_asd_datasets(stages: list, first_index: int = 0) -> None:
    """
    Check every remaining stage's ASD dataset when training starts.

    A later stage's asd_dataset_path is otherwise first opened when that stage
    begins, which for stage_1 is hundreds of epochs in. Also reports how many ASDs
    each stage trains on, so a dataset regenerated in place is visible in the log,
    and warns about freeze_rb_layer: signal on a single-ASD dataset, where the ASD
    channel is constant and gives the loss nothing to learn from.
    """
    for offset, stage in enumerate(stages):
        n = first_index + offset
        path = stage["asd_dataset_path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"stage_{n}: asd_dataset_path {path} does not exist")
        with h5py.File(path, "r") as f:
            counts = {det: int(f["asds"][det].shape[0]) for det in f["asds"]}
        print(f"stage_{n}: {path} -- ASDs per detector: {counts}")
        if stage.get("freeze_rb_layer") == "signal" and max(counts.values()) <= 1:
            print(
                f"WARNING: stage_{n} trains the RB layer's ASD columns "
                f"(freeze_rb_layer: signal) on a single ASD per detector. That input "
                f"channel is then constant, so there is nothing to learn; use "
                f"freeze_rb_layer: true."
            )


def initialize_stage(
    pm: BasePosteriorModel,
    wfd: WaveformDataset,
    stage: dict,
    num_workers: int,
    resume: bool = False,
):
    """
    Initializes training based on PosteriorModel metadata and current stage:
        * Builds transforms (based on noise settings for current stage);
        * Builds DataLoaders;
        * At the beginning of a stage (i.e., if not resuming mid-stage), initializes
        a new optimizer and scheduler;
        * Freezes / unfreezes SVD layer of embedding network

    Parameters
    ----------
    pm : BasePosteriorModel
    wfd : WaveformDataset
    stage : dict
        Settings specific to current stage of training
    num_workers : int
    resume : bool
        Whether training is resuming mid-stage. This controls whether the optimizer and
        scheduler should be re-initialized based on contents of stage dict.

    Returns
    -------
    (train_loader, test_loader)
    """

    train_settings = pm.metadata["train_settings"]

    # Rebuild transforms based on possibly different noise.
    set_train_transforms(wfd, train_settings["data"], stage["asd_dataset_path"])

    # Allows for changes in batch size between stages.
    train_loader, test_loader = build_train_and_test_loaders(
        wfd,
        train_settings["data"]["train_fraction"],
        stage["batch_size"],
        num_workers,
    )

    pm.lr_warmup = None
    if not resume:
        # New optimizer and scheduler. If we are resuming, these should have been
        # loaded from the checkpoint.
        print("Initializing new optimizer and scheduler.")
        pm.optimizer_kwargs = stage["optimizer"]
        pm.scheduler_kwargs = stage["scheduler"]
        pm.initialize_optimizer_and_scheduler()

        # Optional linear lr warmup over the first steps of the stage; see
        # _apply_lr_warmup in base_model.py for why a fresh Adam needs it. It must
        # finish inside the first epoch: CosineAnnealingLR computes each epoch's lr
        # from the current group lr, so a ramp still running at the epoch-end
        # scheduler step would scale the whole remaining schedule down.
        warmup_steps = int(stage.get("lr_warmup_steps", 0))
        if warmup_steps > 0:
            if warmup_steps > len(train_loader):
                print(
                    f"lr_warmup_steps {warmup_steps} exceeds the {len(train_loader)} "
                    f"steps in one epoch; clamping to {len(train_loader)}."
                )
                warmup_steps = len(train_loader)
            pm.lr_warmup = {
                "steps": warmup_steps,
                "step": 0,
                "target_lrs": [g["lr"] for g in pm.optimizer.param_groups],
            }
            print(f"Linear lr warmup over the first {warmup_steps} steps.")

    # Freeze/unfreeze RB layer if necessary
    if "freeze_rb_layer" in stage:
        _set_rb_layer_trainability(pm.network, stage["freeze_rb_layer"])
    n_grad = get_number_of_model_parameters(pm.network, (True,))
    n_nograd = get_number_of_model_parameters(pm.network, (False,))
    print(f"Fixed parameters: {n_nograd}\nLearnable parameters: {n_grad}\n")
    if stage.get("freeze_rb_layer") == "signal":
        n_masked = sum(
            2 * m.num_bins * layer.weight.shape[0]
            for m in pm.network.modules()
            if isinstance(m, LinearProjectionRB)
            for layer in m.layers_rb
        )
        print(
            f"  of the learnable parameters, {n_masked} are RB signal weights whose "
            f"gradients are masked, so they stay fixed\n"
        )

    return train_loader, test_loader


def _infer_completed_stages(pm: BasePosteriorModel, stages: list) -> list:
    """Reconstruct the stage boundaries of a checkpoint written before they were
    recorded, using the old assumption that every stage ran its full length.

    That assumption is exactly what early stopping breaks, so warn when the
    checkpoint's epoch does not sit on a configured boundary: the run is then
    either mid-stage (fine) or past a stage that ended early (not fine, and the
    stage index inferred here will be too low).
    """
    end_epochs = [int(e) for e in np.cumsum([stage["epochs"] for stage in stages])]
    n = int(np.searchsorted(end_epochs, pm.epoch + 1))
    completed = [int(e) for e in end_epochs[:n]]
    if pm.epoch not in ([0] + completed):
        print(
            f"WARNING: this checkpoint (epoch {pm.epoch}) has no recorded stage "
            f"boundaries, so they were inferred as {completed} from the configured "
            f"stage lengths {end_epochs}. If a stage ended early, that inference is "
            f"wrong and training will re-enter a stage it already finished. Set "
            f"metadata['stage_progress']['end_epochs'] on the checkpoint to the "
            f"epochs at which stages actually ended."
        )
    return completed


def train_stages(
    pm: BasePosteriorModel, wfd: WaveformDataset, train_dir: str, local_settings: dict
) -> bool:
    """
    Train the network, iterating through the sequence of stages. Stages can change
    certain settings such as the noise characteristics, optimizer, and scheduler settings.

    Parameters
    ----------
    pm : BasePosteriorModel
    wfd : WaveformDataset
    train_dir : str
        Directory for saving checkpoints and train history.
    local_settings : dict

    Returns
    -------
    bool
        True if all stages are complete
        False otherwise
    """

    train_settings = pm.metadata["train_settings"]
    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch, **local_settings["runtime_limits"]
    )

    # Extract list of stages from settings dict
    stages = []
    num_stages = 0
    while True:
        try:
            stages.append(train_settings["training"][f"stage_{num_stages}"])
            num_stages += 1
        except KeyError:
            break
    # Epoch at which each COMPLETED stage actually ended. This cannot be derived
    # from the configured stage lengths: early stopping ends a stage before its
    # configured epoch count, and every cumulative boundary computed from those
    # counts is then wrong for the rest of the run -- which stage we are in,
    # whether a new optimizer is built, when the stage ends, and whether the run
    # is finished. Record what happened instead of assuming.
    progress = pm.metadata.setdefault("stage_progress", {})
    if progress.get("end_epochs") is None:
        progress["end_epochs"] = _infer_completed_stages(pm, stages)
    completed = progress["end_epochs"]
    _check_stage_asd_datasets(stages[len(completed):], first_index=len(completed))

    for n in range(len(completed), num_stages):
        stage = stages[n]
        stage_start_epoch = completed[-1] if completed else 0
        stage_end_epoch = stage_start_epoch + stage["epochs"]

        # Beginning the stage iff no epoch of it has run yet. Anything else is a
        # mid-stage resume, where the optimizer and scheduler must be kept as
        # they were rather than rebuilt from the stage settings.
        if pm.epoch == stage_start_epoch:
            print(f"\nBeginning training stage {n}. Settings:")
            best = os.path.join(train_dir, "best_model.pt")
            if n > 0 and os.path.isfile(best):
                # EarlyStopping is rebuilt for every stage and its first epoch always
                # counts as best, so stage n's first epoch overwrites best_model.pt
                # whatever its loss. Keep the one the previous stage produced.
                kept = os.path.join(train_dir, f"best_model_before_stage_{n}.pt")
                shutil.copy2(best, kept)
                print(f"Copied {best} to {kept} before stage {n} overwrites it.")
            print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(
                pm, wfd, stage, local_settings["num_workers"], resume=False
            )
        else:
            print(f"\nResuming training in stage {n}. Settings:")
            print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(
                pm, wfd, stage, local_settings["num_workers"], resume=True
            )
        early_stopping = None
        if stage.get("early_stopping"):
            try:
                early_stopping = EarlyStopping(**stage["early_stopping"])
            except Exception:
                print(
                    "Early stopping settings invalid. Please pass 'patience', 'delta', 'metric'"
                )
                raise
        stage_start = time.time()
        runtime_limits.max_epochs_total = stage_end_epoch
        pm.train(
            train_loader,
            test_loader,
            train_dir=train_dir,
            runtime_limits=runtime_limits,
            checkpoint_epochs=local_settings["checkpoint_epochs"],
            use_wandb=local_settings.get("wandb", False),
            test_only=local_settings.get("test_only", False),
            early_stopping=early_stopping,
        )
        stage_elapsed = time.time() - stage_start
        elapsed_min, elapsed_sec = divmod(stage_elapsed, 60)
        print(f"Stage {n} training complete. This took {elapsed_min:.0f}:{elapsed_sec:02.0f} min.")
        # if test_only, model should not be saved, and run is complete
        if local_settings.get("test_only", False):
            return True

        # The stage is over either because it ran its configured length or
        # because it stopped early. Both end it; only a runtime limit leaves it
        # unfinished, to be resumed mid-stage by the next job.
        stopped_early = early_stopping is not None and early_stopping.early_stop
        if pm.epoch == stage_end_epoch or stopped_early:
            if stopped_early:
                print(
                    f"Stage {n} stopped early at epoch {pm.epoch} of a configured "
                    f"{stage_end_epoch}. Stage {n + 1} will start from here."
                )
            # Record the boundary BEFORE saving, so the checkpoint carries it.
            completed.append(pm.epoch)
            save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
            print(f"Training stage complete. Saving to {save_file}.")
            pm.save_model(save_file, save_training_info=True)
        if runtime_limits.local_limits_exceeded(pm.epoch):
            print("Local runtime limits reached. Ending program.")
            break

    return len(completed) == num_stages


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Train a neural network for gravitational-wave single-event inference.
        
        This program can be called in one of two ways:
            a) with a settings file. This will create a new network based on the 
            contents of the settings file.
            b) with a checkpoint file. This will resume training from the checkpoint.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        help="YAML file containing training settings.",
    )
    parser.add_argument(
        "--train_dir", required=True, help="Directory for Dingo training output."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file from which to resume training.",
    )
    parser.add_argument(
        "--exit_command",
        type=str,
        default="",
        help="Optional command to execute after completion of training.",
    )
    args = parser.parse_args()

    # The settings file and checkpoint are mutually exclusive.
    if args.checkpoint is None and args.settings_file is None:
        parser.error("Must specify either a checkpoint file or a settings file.")
    if args.checkpoint is not None and args.settings_file is not None:
        parser.error("Cannot specify both a checkpoint file and a settings file.")

    return args


def train_local():
    args = parse_args()

    os.makedirs(args.train_dir, exist_ok=True)

    if args.settings_file is not None:
        print("Beginning new training run.")
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

        # Extract the local settings from train settings file, save it separately. This
        # file can later be modified, and the settings take effect immediately upon
        # resuming.

        local_settings = train_settings.pop("local")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "w") as f:
            if (
                local_settings.get("wandb", False)
                and "id" not in local_settings["wandb"].keys()
            ):
                try:
                    import wandb

                    local_settings["wandb"]["id"] = wandb.util.generate_id()
                except ImportError:
                    print("wandb not installed, cannot generate run id.")
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

        pm, wfd = prepare_training_new(train_settings, args.train_dir, local_settings)

    else:
        print("Resuming training run.")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)
        pm, wfd = prepare_training_resume(
            args.checkpoint, local_settings, args.train_dir
        )

    with threadpool_limits(limits=1, user_api="blas"):
        complete = train_stages(pm, wfd, args.train_dir, local_settings)

    if complete:
        if args.exit_command:
            print(
                f"All training stages complete. Executing exit command: {args.exit_command}."
            )
            os.system(args.exit_command)
        else:
            print("All training stages complete.")
    else:
        print("Program terminated due to runtime limit.")
