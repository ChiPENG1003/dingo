# Overlapping Signal Training Pipeline — Pending Modifications

These modifications are needed to support training on overlapping gravitational wave signals
(two superimposed BBH events). The waveform dataset is assumed to be pre-generated externally,
with detector-frame strains already projected and superimposed.

**Assumed dataset format:**
- `parameters`: pandas DataFrame with columns `chirp_mass_1`, `mass_ratio_1`, ..., `ra_1`, `dec_1`,
  `geocent_time_1`, `psi_1`, `luminosity_distance_1`, and the same set with `_2` suffix for the second event.
- `polarizations`: dict `{"H1": array, "L1": array}` — pre-projected, pre-superimposed, noise-free
  complex frequency-domain strains.

---

## Modification 2 — `dingo/gw/training/train_builders.py`

**Function:** `set_train_transforms` (line 66)

**Prompt for implementation:**

```
In dingo/gw/training/train_builders.py, modify the function set_train_transforms to support
overlapping signals where all parameters (including extrinsic parameters for both events) are
already stored in the waveform dataset, and the waveform dataset stores pre-projected
detector-frame strains (keyed by ifo name, e.g. "H1", "L1") instead of geocentric polarizations
(h_plus, h_cross).

Specifically:

1. Remove the following transforms from the transforms list:
   - SampleExtrinsicParameters  (line ~111)
   - GetDetectorTimes            (line ~112)
   - GNPECoalescenceTimes        (the entire gnpe_time_shifts block, lines ~116-126)
   - ProjectOntoDetectors        (line ~156)

2. Remove the call to get_extrinsic_prior_dict (line ~101) and the variable
   extrinsic_prior_dict, since there is no extrinsic prior to sample at training time.

3. Replace the call to get_standardization_dict (lines ~148-154) with a call to the new
   function get_standardization_dict_overlapping(wfd, data_settings["inference_parameters"]).
   This new function is defined in dingo/gw/gwutils.py (see Modification 3 below).
   Remove the context_parameters logic since GNPE proxies are no longer generated.

4. Remove the context_parameters handling (lines ~115, ~131-135, ~179-182) since there are
   no GNPE proxy variables. The UnpackDict at the end should always use
   selected_keys = ["inference_parameters", "waveform"].

5. Keep the following transforms unchanged:
   - SampleNoiseASD
   - WhitenAndScaleStrain
   - AddWhiteNoiseComplex
   - SelectStandardizeRepackageParameters
   - RepackageStrainsAndASDS
   - CropMaskStrainRandom (if present)
   - UnpackDict

After the change, the transforms list should be:
    [SampleNoiseASD, WhitenAndScaleStrain, AddWhiteNoiseComplex,
     SelectStandardizeRepackageParameters, RepackageStrainsAndASDS, UnpackDict]
```

---

## Modification 3 — `dingo/gw/gwutils.py`

**Function:** new function `get_standardization_dict_overlapping`, to be added after the
existing `get_standardization_dict` function (after line 146).

**Prompt for implementation:**

```
In dingo/gw/gwutils.py, add a new function get_standardization_dict_overlapping after the
existing get_standardization_dict function (after line 146).

The new function replaces the three-source standardization logic (dataset + prior + Monte Carlo)
used in the single-event case with a simpler version that computes mean and std directly from
the waveform dataset. This is correct for overlapping signals because all parameters, including
extrinsic parameters for both events, are pre-stored in the dataset.

The function signature and body:

    def get_standardization_dict_overlapping(wfd, selected_parameters):
        """
        Compute standardization mean and std for overlapping signal training,
        where all inference parameters are stored directly in the waveform dataset.

        Parameters
        ----------
        wfd : WaveformDataset
        selected_parameters : list[str]
            List of parameter names to standardize. All must be columns in wfd.parameters.

        Returns
        -------
        dict with keys "mean" and "std", each a dict mapping parameter name to float.
        """
        mean_all, std_all = wfd.parameter_mean_std()

        missing = [p for p in selected_parameters if p not in mean_all]
        if missing:
            raise KeyError(
                f"Parameters not found in waveform dataset: {missing}. "
                f"All inference_parameters must be stored as columns in the dataset."
            )

        return {
            "mean": {k: mean_all[k] for k in selected_parameters},
            "std":  {k: std_all[k]  for k in selected_parameters},
        }

Also update the import in train_builders.py to import get_standardization_dict_overlapping
from dingo.gw.gwutils alongside the existing get_standardization_dict (or replace it,
depending on whether single-event training still needs to work from the same file).
```

---

## train_settings.yaml changes (already decided)

When applying the above modifications, the `train_settings.yaml` must also be updated:

- Remove the `extrinsic_prior` block entirely.
- Remove the `gnpe_time_shifts` block entirely.
- Update `inference_parameters` to list all 28 parameters (14 per event, with `_1`/`_2` suffixes).

Example `inference_parameters` section:
```yaml
inference_parameters:
  - chirp_mass_1
  - mass_ratio_1
  - a_1_1
  - a_2_1
  - tilt_1_1
  - tilt_2_1
  - phi_12_1
  - phi_jl_1
  - theta_jn_1
  - luminosity_distance_1
  - geocent_time_1
  - ra_1
  - dec_1
  - psi_1
  - chirp_mass_2
  - mass_ratio_2
  - a_1_2
  - a_2_2
  - tilt_1_2
  - tilt_2_2
  - phi_12_2
  - phi_jl_2
  - theta_jn_2
  - luminosity_distance_2
  - geocent_time_2
  - ra_2
  - dec_2
  - psi_2
```
