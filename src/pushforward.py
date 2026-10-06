"""The prior pushforward: the marginal priors, and the datasets they generate.

Every study is a statement about what an approximator was trained on, so the training prior and
the data it produces are part of the method, not a preliminary. This module draws that as one
figure per experiment/model, in three blocks:

* **the marginal priors** -- one panel per parameter of `mcmc_param_names`, i.e. the
  subject-level parameters the NPE infers plus, for a hierarchical model, the hyperparameter it
  is amortized over. Where a study's test cases *shift* a prior hyperparameter (studies 2 and
  3), each case's prior is outlined over the training prior, so the figure shows how far the
  sweep moves it -- and shows, on the panels that do not move, that the priors factorize and
  only the swept factor differs;
* **the data the prior generates** -- mean accuracy and the fastest and slowest response time
  per simulated dataset. These are exactly the statistics studies 5 and 6 select on, so the band
  a model was trained on is shaded here and its test cases' bands are marked;
* **example datasets** -- a handful of individual pushforward draws as response-time
  histograms, correct responses up and errors down, split by instruction for the
  speed/accuracy models.

Nothing here reads a checkpoint: the pushforward is a property of the simulator alone, so this
runs before `train_npe` as a check that a configuration generates plausible data, and after it
as the figure that documents what was trained on. `scripts/prior_stats.py` is the numeric
counterpart -- it dumps the same draws as a CSV for the R figures; this one is self-contained
because a configuration that generates implausible data should be visible without a second
tool.
"""

import logging
from dataclasses import dataclass, field
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from pipeline import accuracy, rt_range
from simulation import ACCURACY_CHANNEL, CONDITION_CHANNEL, RT_CHANNEL, bands_from_bounds

logger = logging.getLogger(__name__)

# The keys of `Case.sim_kwargs` that are *not* prior hyperparameters: `num_obs` is the design,
# and the four band bounds restrict the generated data rather than the prior it is drawn from
# (studies 5 and 6). Whatever else a case carries shifts the prior itself, which is what studies
# 2 and 3 do and what `case_overlays` below redraws.
BAND_KEYS = ("acc_lower", "acc_upper", "rt_lower", "rt_upper")
DESIGN_KEYS = ("num_obs",)

# Correct responses up, errors down -- the response-time histogram this literature reads.
CORRECT_COLOR = "#2f6f9f"
ERROR_COLOR = "#b3523a"
# Speed against accuracy instruction, for the models that have both.
CONDITION_COLORS = ("#2f6f9f", "#d9963c")
CONDITION_LABELS = ("speed", "accuracy")

# How many rejection rounds the figure's draw may take. `DataBandSimulator` defaults to 12,
# which is right for training -- there the cap exists to fail fast on a band the prior cannot
# reach, rather than stall a job for hours -- but study 5's chance-accuracy band accepts about
# 2% of draws, so a couple of thousand accepted datasets need a hundred thousand simulated ones
# and more rounds than that to collect them. A figure is drawn once, so it can afford to wait;
# a band the prior genuinely cannot reach still raises, just later.
MAX_ROUNDS = 48

PARAM_COLUMNS = 4
EXAMPLE_COLUMNS = 3
NUM_BINS = 40


def pushforward_num_obs(cfg):
    """The trial count to push the prior through.

    `pushforward_num_obs` when the config sets it, otherwise the count this study tests at.
    Study 1 has no single one -- the trial count is exactly what it sweeps -- so it falls back to
    `eval_num_obs`, and the figure's example panels say how many trials they hold either way.
    """
    configured = cfg.get("pushforward_num_obs")

    if configured is not None:
        return configured

    return cfg.get("test_num_obs") or cfg["eval_num_obs"]


def prior_hyperparameters(sim_kwargs):
    """The entries of a case's `sim_kwargs` that move the prior rather than the data.

    Studies 2 and 3 shift a prior hyperparameter; studies 5 and 6 leave the prior alone and
    reject on a statistic of `x`; study 1 varies only the trial count. Splitting them by key is
    what lets one figure serve all six without knowing which study it is drawing.
    """
    return {
        name: value
        for name, value in sim_kwargs.items()
        if name not in BAND_KEYS and name not in DESIGN_KEYS
    }


def case_bands(sim_kwargs):
    """The band a case imposes on the generated data, as `{name: (lower, upper)}`, or `{}`."""
    return bands_from_bounds(**{name: sim_kwargs[name] for name in BAND_KEYS if name in sim_kwargs})


@dataclass(frozen=True)
class CaseOverlay:
    """How one test case's prior differs from the training prior, as the figure draws it.

    A shifted hyperparameter reaches the parameter panels in one of two ways, and which one
    depends on the model rather than on the study:

    * as a `draw` -- the case's value is pushed through the prior, so the parameters it governs
      are outlined over the training prior. This is every parameter the prior samples;
    * as a `pin` -- a hierarchical model *samples* the hyperparameter the case fixes
      (`drift_slope_loc` in study 2, `threshold_diff_loc` in study 3 are parameters of those
      models, drawn from the meta simulator's Uniform), so the case's prior for it is a point
      mass and a line is the honest way to draw it.
    """

    label: str
    draw: dict = field(default_factory=dict)
    pins: dict = field(default_factory=dict)


def case_overlays(prior_simulator, cases, num_draws, param_names=()):
    """Build one `CaseOverlay` per test case that shifts a prior hyperparameter.

    Parameters only -- no dataset is simulated here, so this costs a fraction of a second even
    where the pushforward itself is expensive, and it stays affordable for the banded studies
    whose rejection rates make simulating a case's data anything but.

    A hyperparameter passed at call time overrides the one bound into the `_partial_` in
    `conf/simulator/prior_simulator/*.yaml`, which is the same mechanism `CustomSimulator.sample`
    relies on to make studies 2 and 3 measure anything -- so the outline is the prior those cases
    are actually drawn from, not a reconstruction of it.

    Returns:
        A list of `CaseOverlay`, empty when no case shifts the prior (studies 1, 4, 5 and 6).
    """
    overlays = []

    for case in cases:
        shifted = prior_hyperparameters(case.sim_kwargs)

        if not shifted:
            continue

        draw = prior_simulator.sample(num_draws, **shifted)

        overlays.append(
            CaseOverlay(
                label=", ".join(f"{name} = {np.asarray(value).item():g}" for name, value in shifted.items()),
                draw=draw,
                pins={
                    name: float(np.asarray(value))
                    for name, value in shifted.items()
                    if name in param_names and name not in draw
                },
            ),
        )

    return overlays


def example_indices(data_x, num_examples):
    """Pick example datasets spanning the pushforward rather than an arbitrary handful.

    Evenly spaced quantiles of mean accuracy, so the panels run from the least to the most
    accurate dataset the prior produced and a reader sees the range rather than the centre.
    Deterministic given the draw, which matters because the figure is regenerated whenever the
    prior changes and two versions have to be comparable.
    """
    rates = accuracy(data_x)
    num_examples = min(num_examples, rates.size)
    positions = np.linspace(0, rates.size - 1, num_examples).round().astype(int)

    return np.argsort(rates)[positions]


def _panels(subfigure, num_panels, num_columns, **kwargs):
    """Lay out `num_panels` axes on a grid, hiding the unused remainder of the last row."""
    num_rows = max(1, ceil(num_panels / num_columns))
    axes = subfigure.subplots(num_rows, num_columns, squeeze=False, **kwargs).ravel()

    for axis in axes[num_panels:]:
        axis.set_visible(False)

    return axes[:num_panels]


def _marginal_panel(axis, values, name, overlays=(), colors=()):
    """One parameter's training prior, with any shifted case priors drawn over it.

    A case whose shift reaches this parameter through the prior is outlined; one that fixes this
    parameter outright -- a hierarchical model's hyperparameter -- is a vertical line, in the
    same colour as its outline elsewhere in the row.
    """
    values = np.reshape(np.asarray(values), -1)
    shifted = [np.reshape(np.asarray(overlay.draw[name]), -1) for overlay in overlays if name in overlay.draw]
    pinned = [overlay.pins[name] for overlay in overlays if name in overlay.pins]

    low = min([values.min(), *(other.min() for other in shifted), *pinned])
    high = max([values.max(), *(other.max() for other in shifted), *pinned])
    edges = np.linspace(low, high, NUM_BINS + 1)

    axis.hist(values, bins=edges, density=True, color="0.75", edgecolor="none")

    for overlay, color in zip(overlays, colors, strict=True):
        if name in overlay.draw:
            axis.hist(
                np.reshape(np.asarray(overlay.draw[name]), -1),
                bins=edges,
                density=True,
                histtype="step",
                color=color,
                linewidth=1.0,
                label=overlay.label,
            )
        elif name in overlay.pins:
            axis.axvline(overlay.pins[name], color=color, linewidth=1.0, label=overlay.label)

    axis.set_xlabel(name)
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)


def _statistic_panel(axis, values, name, trained=None, marks=()):
    """One pushforward statistic, with the trained-on band shaded and the test bands marked."""
    values = np.asarray(values)

    # Long right tails: a few prior draws produce a slowest response time an order of magnitude
    # past the bulk, which squashes every bin into the left edge. The axis stops at the 99.5th
    # percentile -- or at the top of the band this model was trained under, which has to stay
    # visible -- and the draws past it pile into the last bin, clipped rather than dropped.
    #
    # Deliberately *not* stretched to reach every test band: study 6's widest window ends at 20
    # seconds against response times of under one, so honouring it would leave the figure a spike
    # at the origin. Bounds past the axis are counted instead.
    trained_bound = [bound for bound in (trained or ()) if np.isfinite(bound)]
    limit = max([float(np.quantile(values, 0.995)), *trained_bound])

    # ...and only when there is a tail worth cutting. Mean accuracy reaches 1 and stops, so
    # clipping its top half-percent would cost a visible bin and buy no room at all; the
    # slowest response time runs to 70 seconds against a bulk under 5, and there it buys the
    # whole panel.
    if float(values.max()) <= 1.5 * limit:
        limit = float(values.max())

    num_beyond = int((values > limit).sum())

    axis.hist(np.clip(values, None, limit), bins=NUM_BINS, density=True, color="0.75", edgecolor="none")

    notes = [f"{num_beyond} draws beyond axis"] if num_beyond else []

    if trained is not None:
        axis.axvspan(*trained, color=CORRECT_COLOR, alpha=0.12, zorder=0, label="trained on")

    bounds = sorted({float(bound) for lower, upper in marks for bound in (lower, upper)})

    for bound in bounds:
        if bound <= limit:
            axis.axvline(bound, color="0.35", linestyle=":", linewidth=0.8)

    num_bounds_beyond = sum(bound > limit for bound in bounds)

    if num_bounds_beyond:
        notes.append(f"{num_bounds_beyond} test bounds beyond axis")

    for offset, note in enumerate(notes):
        axis.text(
            0.98,
            0.93 - 0.07 * offset,
            note,
            transform=axis.transAxes,
            ha="right",
            fontsize=6,
            color="0.35",
        )

    axis.set_xlabel(name)
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)


def _example_panel(axis, data_x, param_names, params, rt_limit):
    """One dataset: response times, correct up and errors down, split by instruction if present.

    Counts rather than densities, and one shared limit across the panels, so the panels are
    comparable to each other -- the datasets share a trial count, so the vertical scale means
    the same thing in each.
    """
    data_x = np.asarray(data_x)
    # Clipped to the shared limit rather than cut off by it: the panels share an axis so they can
    # be read against each other, and a dataset with a slower tail than that should show the tail
    # piling into the last bin instead of quietly losing those trials.
    response_time = np.clip(data_x[:, RT_CHANNEL], None, rt_limit)
    is_correct = data_x[:, ACCURACY_CHANNEL] == 1
    edges = np.linspace(0, rt_limit, NUM_BINS + 1)

    if data_x.shape[-1] > CONDITION_CHANNEL:
        groups = [
            (data_x[:, CONDITION_CHANNEL] == value, color, label)
            for value, (color, label) in enumerate(zip(CONDITION_COLORS, CONDITION_LABELS, strict=True))
        ]
    else:
        groups = [(np.ones_like(is_correct), CORRECT_COLOR, None)]

    for in_group, color, label in groups:
        for correct, sign in ((True, 1), (False, -1)):
            selected = response_time[in_group & (is_correct == correct)]
            counts, _ = np.histogram(selected, bins=edges)
            axis.stairs(
                sign * counts,
                edges,
                fill=True,
                alpha=0.55,
                color=color if label is not None else (CORRECT_COLOR if correct else ERROR_COLOR),
                label=label if correct and label is not None else None,
            )

    axis.axhline(0, color="0.3", linewidth=0.8)
    axis.set_xlim(0, rt_limit)
    axis.set_xlabel("response time (s)")
    axis.set_ylabel("trials", fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(f"accuracy = {np.mean(is_correct):.2f}", fontsize=8, pad=14)

    # The parameters behind the panel, so a reader can tie an odd-looking dataset back to the
    # draw that produced it rather than only to its position in the accuracy ordering. Above the
    # axis rather than inside it: the panel's own histogram fills both halves of the frame, and
    # which half is free depends on the draw.
    axis.text(
        0.5,
        1.01,
        "  ".join(f"{name} {value:.2f}" for name, value in zip(param_names, params, strict=True)),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="0.35",
    )


def prior_pushforward_figure(  # noqa: PLR0913
    draws,
    param_names,
    *,
    num_examples=6,
    overlays=(),
    trained_bands=None,
    test_bands=(),
    title=None,
):
    """Build the prior pushforward figure.

    Args:
        draws: one simulator draw, as `simulator.sample` returns it -- a parameter array per
            name plus `x` of shape `(dataset, trial, channel)`.
        param_names: the parameters to show marginals for, in the order they should appear.
        num_examples: how many individual datasets to draw as histograms.
        overlays: `CaseOverlay`s from `case_overlays`, drawn over the marginals.
        trained_bands: the band this model was trained under, `{name: (lower, upper)}`, shaded
            on the statistic it restricts.
        test_bands: the bands this study's test cases impose, marked on the same panels.
        title: figure title; the experiment and model the run composed.

    Returns:
        The `matplotlib` figure, for the caller to save.
    """
    data_x = np.asarray(draws["x"])
    trained_bands = trained_bands or {}

    examples = example_indices(data_x, num_examples)
    # One limit for every example panel, taken over the whole draw rather than the examples, so
    # that redrawing with a different `num_examples` does not silently rescale the axis.
    rt_limit = float(np.nanquantile(data_x[..., RT_CHANNEL], 0.995))

    param_rows = ceil(len(param_names) / PARAM_COLUMNS)
    example_rows = ceil(len(examples) / EXAMPLE_COLUMNS)

    figure = plt.figure(
        figsize=(3.2 * PARAM_COLUMNS, 2.5 * (param_rows + 1 + example_rows)),
        layout="constrained",
    )
    top, middle, bottom = figure.subfigures(3, 1, height_ratios=[param_rows, 1, example_rows])

    top.suptitle("Marginal prior distributions", fontsize=11)

    # One colour per test case, held across the panels, so a line in the hyperparameter panel and
    # the outline it produces in the parameter panels are recognisably the same case.
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(overlays))) if overlays else []

    for axis, name in zip(_panels(top, len(param_names), PARAM_COLUMNS), param_names, strict=True):
        _marginal_panel(axis, draws[name], name, overlays=overlays, colors=colors)

    if overlays:
        # Collected over every panel and de-duplicated: a case appears as an outline in one panel
        # and as a line in another, and only the panels it actually reaches carry a handle.
        entries = {}

        for axis in top.axes:
            entries.update(dict(zip(*reversed(axis.get_legend_handles_labels()), strict=True)))

        ordered = [overlay.label for overlay in overlays if overlay.label in entries]
        top.legend(
            [entries[label] for label in ordered],
            ordered,
            loc="outside right upper",
            fontsize=7,
            title="test cases",
            title_fontsize=7,
        )

    middle.suptitle("Data the prior generates", fontsize=11)

    lowest_rt, highest_rt = rt_range(data_x)
    statistics = (
        ("mean accuracy", accuracy(data_x), "acc"),
        ("fastest response time (s)", lowest_rt, "rt"),
        ("slowest response time (s)", highest_rt, "rt"),
    )

    for axis, (name, values, band) in zip(_panels(middle, len(statistics), 3), statistics, strict=True):
        _statistic_panel(
            axis,
            values,
            name,
            trained=trained_bands.get(band),
            marks=[bounds[band] for bounds in test_bands if band in bounds],
        )

    # Whichever statistic the band restricts carries the handle, so it is collected across the
    # row rather than taken from the first panel -- study 6's window shades the response-time
    # panels and leaves the accuracy one bare.
    entries = {}

    for axis in middle.axes:
        entries.update(dict(zip(*reversed(axis.get_legend_handles_labels()), strict=True)))

    if entries:
        middle.legend(list(entries.values()), list(entries), loc="outside right upper", fontsize=7)

    bottom.suptitle(
        f"Example datasets ({data_x.shape[1]} trials each; correct up, error down)", fontsize=11,
    )

    parameters = np.column_stack([np.reshape(np.asarray(draws[name]), -1) for name in param_names])

    for axis, index in zip(_panels(bottom, len(examples), EXAMPLE_COLUMNS), examples, strict=True):
        _example_panel(axis, data_x[index], param_names, parameters[index], rt_limit)

    if data_x.shape[-1] > CONDITION_CHANNEL:
        handles, labels = bottom.axes[0].get_legend_handles_labels()
        bottom.legend(handles, labels, loc="outside right upper", fontsize=7, title="instruction", title_fontsize=7)

    if title is not None:
        figure.suptitle(title, fontsize=13)

    return figure
