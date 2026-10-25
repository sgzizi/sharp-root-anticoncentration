# Reproduction protocol

This document records the conventions behind the reference outputs. The
default commands are intentionally lightweight: they run on a CPU, use fixed
seeds, and complete in about one minute on a recent laptop.

## Environment and entry points

Python 3.10 or newer is required. Install the package and test dependencies
from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

The committed reference files were generated with CPython 3.12.8, NumPy
2.5.1, Matplotlib 3.11.1, and pytest 9.1.1; two consecutive generations on
this stack are byte-identical. Other compatible dependency versions satisfy
the same tests and reproduce the same numerical conclusions, but last-bit
floating-point values and figure typography may differ.

The complete reproduction command is:

```bash
bash reproduce_all.sh
```

It executes the test suite first and stops immediately if a consistency check
fails. Each experiment also has a standalone command and exposes its numerical
settings through `--help`.

## Dimension-free cube incidence

Command:

```bash
python experiments/dimension_free.py
```

The default feature curve has two active coordinates, regardless of the
ambient dimension. Consequently, the root event depends only on the first two
uniform cube coordinates; the other coordinates are integrated out exactly.
The experiment samples only the active pair and compares the resulting
interval-hitting ratio with the exact value. Its pointwise 95% intervals are
Wilson score intervals for the Bernoulli root event.

Outputs:

- `figures/dimension_free.png` and `figures/dimension_free.pdf`
- `results/dimension_free.csv`

## Cost-sensitive Gaussian-RBF graph SSL

Command:

```bash
python experiments/rbf_graph_ssl.py
```

### Online protocol

A single seeded sequence of base graph instances is generated before online
learning begins. For a horizon `T`, the learner uses the first `T` instances
and a parameter mesh with spacing proportional to `T^{-1/2}`. The cost pairs
and Hedge actions are resampled independently across repetitions.

The oblivious environment precomputes utility matrices for efficiency. These
matrices are never supplied to `act()`: the learner selects an action first and
receives the current utility row only through `observe()`. The accompanying
tests enforce this interface.

### Regret quantities

The experiment reports two distinct finite-comparator quantities:

- `grid_regret` is exact regret relative to the algorithm's parameter grid for
  that horizon;
- `dense_proxy_regret` uses the union of a 401-point evaluation grid and the
  algorithm grid. It is a finite diagnostic, not an exact continuous-parameter
  comparator.

The plotted intervals are pointwise normal-approximation 95% intervals for the
mean over 30 cost/action repetitions, conditional on the fixed base-instance
sequence.

### Illustrative transition

The transition panel is chosen deterministically from the maximum-horizon,
repetition-zero run. The selection rule maximizes the number of sign changes
on the dense parameter grid and breaks ties by proximity to the center of the
parameter interval. The utility panel contains the selected transition round
and the rounds with the largest dense-grid utility-change counts. The complete
selection rule and selected indices are stored in the result metadata.

Grounded harmonic systems are solved directly. The implementation never forms
a matrix inverse or expands a determinant numerically.

Outputs:

- `figures/rbf_graph_ssl/regret_scaling.{png,pdf}`
- `figures/rbf_graph_ssl/transition_curve.{png,pdf}`
- `figures/rbf_graph_ssl/utility_curves.{png,pdf}`
- `results/rbf_graph_ssl/configuration.csv`
- `results/rbf_graph_ssl/regret_runs.csv`
- `results/rbf_graph_ssl/regret_summary.csv`
- `results/rbf_graph_ssl/online_trace.csv`
- `results/rbf_graph_ssl/transition_curve.csv`
- `results/rbf_graph_ssl/transition_diagnostics.csv`
- `results/rbf_graph_ssl/utility_curves.csv`
- `results/rbf_graph_ssl/utility_selection.csv`

## Common-offset polynomial law

Command:

```bash
python experiments/common_offset_demo.py
```

The fixture is a fixed monic cubic with known coefficients and roots. Before
plotting, the implementation checks the polynomial shift identity directly.
The left panel is a two-coordinate projection of the full one-parameter
coefficient curve. The right panel compares a seeded histogram of one
translated ordered root with its exact uniform density. The roots are not
estimated numerically.

Outputs:

- `figures/common_offset_demo.png` and `figures/common_offset_demo.pdf`
- `results/common_offset/coefficient_curve.csv`
- `results/common_offset/configuration.csv`

## Test coverage

The test suite checks:

- the exact dimension-free root probability, inert-coordinate reduction, and
  seeded Monte Carlo estimates;
- the two-dimensional cube-section incidence identity;
- analytic harmonic-score derivatives against finite differences;
- the support-body incidence speed used by the RBF calculation;
- the separation between algorithm-grid regret and the finite proxy;
- the non-anticipating `act()`/`observe()` protocol;
- polynomial coefficient translation and rigid translation of ordered roots.

Continuous integration runs the tests and reduced-size smoke versions of all
three experiments under Python 3.10 and Python 3.12.
<!-- Maintenance annotation; functional content is unchanged. -->
