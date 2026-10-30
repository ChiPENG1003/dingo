"""The x axes of the heat maps, and the rule that turns a cell into a signal 2.

Every sweep shares the same y axis (10 log-spaced time separations, defined in
combo_common) and differs only in what the x axis controls on the distraction
signal. Signal 1 -- the target -- is never touched by anything in here.

Within a cell the axis parameter is pinned to the bin centre, everything else is
redrawn from the model's own prior, and the distance is solved so the distraction
signal lands at a prescribed SNR. Nuisance draws are keyed to the realisation
index alone, never to the cell, so realisation k is the *same* set of random
spins / inclination / sky across every bin of every sweep (common random
numbers): differences between cells then come from the axis, not from noise.
"""

import numpy as np

import combo_common as cc
import inference as inf


class Context:
    """Everything a cell needs that is fixed for a whole sweep."""

    def __init__(self, injection_generator, theta1, snr1, snr1_detectors,
                 nuisance_base_seed=200000, noise_base_seed=300000):
        self.ig = injection_generator
        self.theta1 = theta1
        self.snr1 = snr1
        self.snr1_detectors = snr1_detectors
        self.nuisance_base_seed = nuisance_base_seed
        self.noise_base_seed = noise_base_seed

    @property
    def q1(self):
        return float(self.theta1["mass_ratio"])

    @property
    def mc1(self):
        return float(self.theta1["chirp_mass"])

    @property
    def d1(self):
        return float(self.theta1["luminosity_distance"])

    def nuisance_seed(self, realization):
        return self.nuisance_base_seed + realization

    def noise_seed(self, realization):
        return self.noise_base_seed + realization


class Axis:
    """One x axis: 10 bin centres plus the rule that applies a centre to theta2.

    apply() returns nothing and mutates theta2; it may set chirp_mass,
    mass_ratio and/or luminosity_distance. If target_snr2 is not None the
    distance it leaves behind is overwritten by the SNR solve afterwards.
    """

    def __init__(self, name, label, centers, apply_fn, target_snr2=None, notes=(),
                 independent_replicas=False):
        self.name = name
        self.label = label
        self.centers = np.asarray(centers, dtype=float)
        self._apply = apply_fn
        self.target_snr2 = target_snr2
        self.notes = list(notes)
        # A controlled axis reuses realisation k's nuisance draw in every column
        # on purpose (common random numbers). The population axis has no columns
        # to hold fixed, so each one must be an INDEPENDENT sample instead --
        # otherwise the ten columns would be ten identical copies.
        self.independent_replicas = independent_replicas

    def replica_offset(self, x_index):
        return int(x_index) * 10000 if self.independent_replicas else 0

    def apply(self, theta2, x_value, ctx, rng):
        self._apply(theta2, float(x_value), ctx, rng)

    @property
    def edges(self):
        """Cell boundaries for pcolormesh: midpoints, extended by half a step.

        Geometric midpoints when the centres are geometrically spaced, so a log
        axis gets cells of equal width on screen.
        """
        c = self.centers
        if len(c) == 1:
            return np.array([c[0] * 0.9, c[0] * 1.1])
        log_like = np.all(c > 0) and np.allclose(
            np.diff(np.log(c)), np.log(c[1]) - np.log(c[0]), rtol=1e-3
        )
        if log_like:
            mid = np.sqrt(c[:-1] * c[1:])
            return np.concatenate([[c[0] ** 2 / mid[0]], mid, [c[-1] ** 2 / mid[-1]]])
        mid = 0.5 * (c[:-1] + c[1:])
        return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


# --------------------------------------------------------------------------
# The individual axes
# --------------------------------------------------------------------------

def _apply_nothing(theta2, x_value, ctx, rng):
    """snr_ratio: masses stay as drawn from the prior; only the distance moves,
    and that is done by the SNR solve in build_theta2()."""
    return


def _apply_distance(theta2, x_value, ctx, rng):
    theta2["luminosity_distance"] = x_value


def _apply_separation(theta2, x_value, ctx, rng):
    """Transverse separation axis, D = d_L * gamma.

    Signal 2 is put at signal 1's distance -- that is what makes a single d_L
    meaningful in the product -- and then placed at the sky angle gamma = D/d_L
    from it, at a uniformly random azimuth around signal 1.

    Consequence to keep in mind: SNR_2 cannot be pinned here, because the
    distance is spent enforcing d_2 = d_1. Worse, gamma itself moves SNR_2 --
    changing the sky position changes the antenna response - so this axis mixes
    geometric separability with how loud the intruder happens to be at that
    patch of sky. SNR_2 is recorded per run so the mixture can be checked.
    """
    theta2["luminosity_distance"] = ctx.d1
    gamma = x_value / ctx.d1  # radians
    if not 0.0 <= gamma <= np.pi:
        raise ValueError(
            f"D = {x_value:g} Mpc needs gamma = {gamma:.3f} rad at "
            f"d_L = {ctx.d1:.1f} Mpc; only 0 to pi ({np.pi * ctx.d1:.0f} Mpc) exists."
        )
    theta2["ra"], theta2["dec"] = cc.place_at_separation(
        float(ctx.theta1["ra"]), float(ctx.theta1["dec"]), gamma, rng
    )


def _apply_separation_matched(theta2, x_value, ctx, rng):
    """Separation axis with the masses matched to signal 1 as well.

    The plain `separation` axis left the masses on the prior, and the answer was
    a completely flat grid (range 0.023, below the per-cell noise): at
    d_2 = d_1 a prior-drawn intruder comes out 2-8x weaker than the target, so
    nothing was ever damaged and the geometry never got a chance to matter.

    Matching the masses makes signal 2 a twin of signal 1 -- same chirp mass,
    same mass ratio, same distance -- so SNR_2 lands near SNR_1 and differs only
    through inclination, spins, phase and antenna response. Sky position is then
    the only systematic difference between the two sources, which is what the
    axis is meant to isolate.
    """
    theta2["chirp_mass"] = ctx.mc1
    theta2["mass_ratio"] = ctx.q1
    _apply_separation(theta2, x_value, ctx, rng)


def _apply_separation_clone(theta2, x_value, ctx, rng):
    """Signal 2 is an exact clone of signal 1 except for its sky position.

    Every intrinsic and orientation parameter is copied -- masses, spins,
    inclination, phase, polarisation, distance -- so the ONLY difference
    between the two sources is (ra, dec), plus the time shift on the y axis.
    SNR_2 then differs from SNR_1 solely through the antenna response at the new
    sky location, which is the cleanest possible isolation of geometry.

    Consequence: within a cell the 100 realisations differ only in the azimuth
    of signal 2 around signal 1 and in the noise, since the nuisance draw is
    entirely overwritten.
    """
    for name, value in ctx.theta1.items():
        theta2[name] = float(value)
    _apply_separation(theta2, x_value, ctx, rng)


def _apply_chirp_mass(theta2, x_value, ctx, rng):
    """Chirp mass axis. The mass ratio is pinned to signal 1's, not redrawn:
    the admissible q window shrinks hard at both ends of the chirp mass range
    (at Mc = 10 only q >= 0.755 survives, at Mc = 65 only q >= 0.872), so a
    redrawn q would sweep along with the axis and confound it."""
    theta2["chirp_mass"] = x_value
    theta2["mass_ratio"] = ctx.q1


def _apply_mass_ratio(theta2, x_value, ctx, rng):
    """Mass ratio axis. The chirp mass has to be redrawn conditionally: at
    q = 0.2 only Mc in [18.4, 29.4] keeps both components inside [10, 80] Msun,
    so an unconditional prior draw would be rejected almost always."""
    theta2["mass_ratio"] = x_value
    mc_range = cc.chirp_mass_range_for_q(x_value)
    if mc_range is None:
        raise ValueError(f"No chirp mass admits q = {x_value:g} with masses in "
                         f"[{cc.MASS_MIN}, {cc.MASS_MAX}] Msun.")
    theta2["chirp_mass"] = cc.sample_truncated(
        ctx.ig.prior["chirp_mass"], mc_range[0], mc_range[1], rng
    )


def _apply_free(theta2, x_value, ctx, rng):
    """population: nothing is controlled but the time separation. The distraction
    signal is exactly a prior draw, so SNR_1/SNR_2 comes out as a DERIVED
    quantity with its own natural distribution, to be binned afterwards."""
    return


def build_axis(name, ctx, num_bins=10, **overrides):
    if name == "population":
        return Axis(
            "population", "replica", np.arange(num_bins, dtype=float), _apply_free,
            target_snr2=None, independent_replicas=True,
            notes=["signal 2 is a pure prior draw; only |dt| is controlled",
                   "the 10 columns are independent replicas, NOT a parameter axis "
                   "-- bin the result with summarize_grid.py --bin-by snr_ratio",
                   "measured natural distribution at SNR_1 = 25: median ratio "
                   "3.68, only 10.5% of draws have ratio < 1"],
        )

    """Resolve an axis against a concrete signal 1.

    Two of the four axes cannot be written down before signal 1 is chosen: the
    chirp mass range depends on signal 1's mass ratio, and every SNR-pinned axis
    depends on signal 1's SNR.
    """
    if name == "snr_ratio":
        lo = overrides.get("ratio_min", 0.2)
        hi = overrides.get("ratio_max", 5.0)
        centers = np.geomspace(lo, hi, num_bins)
        return Axis(
            "snr_ratio", r"$\mathrm{SNR}_1 / \mathrm{SNR}_2$", centers, _apply_nothing,
            target_snr2=None,  # per-bin; filled in by build_theta2 via the centre
            notes=["distance solved per realisation so the realised ratio is exact"],
        )

    if name in ("separation", "separation_matched", "separation_clone"):
        matched = name == "separation_matched"
        clone = name == "separation_clone"
        # D = d_L * gamma with d_2 = d_1, so D is exactly proportional to the
        # sky angle: the axis is gamma in disguise, expressed as a length.
        d_max = float(np.pi * ctx.d1)
        lo = overrides.get("separation_min", 0.0)
        hi = min(overrides.get("separation_max", d_max), d_max)
        centers = np.linspace(lo, hi, num_bins)
        return Axis(
            name, r"$D = d_L\,\gamma$ [Mpc]", centers,
            _apply_separation_clone if clone else
            (_apply_separation_matched if matched else _apply_separation),
            target_snr2=None,
            notes=[f"d_2 pinned to d_1 = {ctx.d1:.1f} Mpc, so D = d_1 * gamma "
                   f"and D_max = pi * d_1 = {d_max:.0f} Mpc",
                   "transverse ARC at a common distance, not the 3-D chord",
                   "SNR_2 is NOT controlled: the distance is spent on d_2 = d_1, "
                   "and gamma itself changes the antenna response"]
                  + ([f"masses matched to signal 1: Mc_2 = {ctx.mc1:.2f}, "
                      f"q_2 = {ctx.q1:.4f}, so SNR_2 lands near SNR_1"] if matched
                     else [])
                  + (["signal 2 is an exact clone of signal 1 except (ra, dec): "
                      "SNR_2 differs only through the antenna response"] if clone
                     else []),
        )

    if name == "luminosity_distance":
        lo = overrides.get("distance_min", 100.0)
        hi = overrides.get("distance_max", 2000.0)
        centers = np.geomspace(lo, hi, num_bins)
        return Axis(
            "luminosity_distance", r"$d_{L,2}$ [Mpc]", centers, _apply_distance,
            target_snr2=None,
            notes=["SNR_2 is NOT controlled here: at fixed distance it still "
                   "scatters over an order of magnitude with inclination, sky "
                   "position and masses. That scatter is the point of this axis."],
        )

    if name == "chirp_mass":
        lo = overrides.get("chirp_mass_min", 10.0)
        hi = overrides.get("chirp_mass_max", 65.0)
        allowed = cc.chirp_mass_range_for_q(ctx.q1)
        if allowed is None:
            raise ValueError(f"Signal 1 has q = {ctx.q1:.4f}, for which no chirp "
                             f"mass keeps both components in [10, 80] Msun.")
        lo_eff, hi_eff = max(lo, allowed[0]), min(hi, allowed[1])
        notes = []
        if (lo_eff, hi_eff) != (lo, hi):
            notes.append(
                f"axis clipped from [{lo:g}, {hi:g}] to [{lo_eff:.2f}, {hi_eff:.2f}] "
                f"by the component-mass constraint at q = q1 = {ctx.q1:.4f}"
            )
        centers = np.linspace(lo_eff, hi_eff, num_bins)
        return Axis("chirp_mass", r"$\mathcal{M}_2$ [$M_\odot$]", centers,
                    _apply_chirp_mass,
                    target_snr2=overrides.get("pinned_snr2", ctx.snr1),
                    notes=notes + [f"q_2 pinned to q_1 = {ctx.q1:.4f}"])

    if name == "mass_ratio":
        lo = overrides.get("mass_ratio_min", 0.2)
        hi = overrides.get("mass_ratio_max", 1.0)
        centers = np.linspace(lo, hi, num_bins)
        return Axis("mass_ratio", r"$q_2$", centers, _apply_mass_ratio,
                    target_snr2=overrides.get("pinned_snr2", ctx.snr1),
                    notes=["chirp mass redrawn from the prior truncated to the "
                           "window admitted by each q",
                           "q < 0.2 is excluded: q = 0.125 is reachable only at "
                           "the single point (m1, m2) = (80, 10) Msun"])

    raise ValueError(f"Unknown axis {name!r}. Known: {sorted(AXIS_NAMES)}")


AXIS_NAMES = ("population", "snr_ratio", "luminosity_distance", "chirp_mass",
              "mass_ratio", "separation", "separation_matched",
              "separation_clone")


# --------------------------------------------------------------------------
# Turning a cell into a distraction signal
# --------------------------------------------------------------------------

def build_theta2(ctx, axis, x_index, shift_seconds, realization):
    """Parameters of the distraction signal for one cell and one realisation.

    Returns (theta2, info) where info records what the SNR solve actually
    achieved -- the realised SNR_2 and the distance it needed, both of which go
    into the per-run summary so the sweep can be re-binned later by either.
    """
    x_value = float(axis.centers[x_index])
    seed = ctx.nuisance_seed(realization + axis.replica_offset(x_index))

    # Full prior draw: spins, inclination, phase, sky, polarisation. This also
    # reseeds numpy/torch/bilby, which is what makes realisation k identical
    # across cells.
    theta2 = inf._draw_reproducible_truth(ctx.ig, seed)

    # Tie the geocentric time to signal 1. The prior has geocent_time ~ U(-0.1,
    # 0.1) s; leaving it free would smear the separation by +-0.2 s and merge
    # the 0.05 s and 0.1 s rows of the grid into each other. With this tie the
    # only offset is the applied shift, up to the ~10 ms detector-arrival
    # difference that signal 2's own sky position produces.
    theta2["geocent_time"] = float(ctx.theta1["geocent_time"])

    rng = np.random.default_rng(seed + 7919)
    axis.apply(theta2, x_value, ctx, rng)

    info = {"x_value": x_value, "snr2_target": None}

    target_snr2 = axis.target_snr2
    if axis.name == "snr_ratio":
        # The axis IS the ratio, so each bin has its own SNR_2 target.
        target_snr2 = ctx.snr1 / x_value

    if target_snr2 is not None:
        distance, _ = cc.distance_for_snr(
            ctx.ig, theta2, target_snr2, shift_seconds=shift_seconds
        )
        theta2["luminosity_distance"] = float(distance)
        info["snr2_target"] = float(target_snr2)

    return theta2, info
