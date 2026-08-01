# Sharp Root Anti-Concentration: Reproducibility Artifact

[![tests](https://github.com/sgzizi/sharp-root-anticoncentration/actions/workflows/tests.yml/badge.svg)](https://github.com/sgzizi/sharp-root-anticoncentration/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-2E8B57.svg)](LICENSE)

This repository accompanies *Sharp Root Anti-Concentration via Projective
Incidence and Ordered Root Laws* by Zijun Wang and Yuchen Miao.  It provides
small, deterministic checks of the analytic interfaces used in the paper and
reproducible illustrations of the two graph-learning mechanisms.

The proofs are entirely analytic.  The computations here are designed to
expose the geometry, test the implementations against exact identities, and
make the transition-to-regret pipeline inspectable; they are not empirical
evidence in place of the theorems.

## What is reproduced

| Command | Paper component | Output |
|---|---|---|
| `python experiments/dimension_free.py` | Equation (19): the dimension-free cube example for $F_{\delta,N}(x)=(1,x/\delta,0,\ldots,0)$ | Monte Carlo interval-hitting ratios with exact value and Wilson intervals |
| `python experiments/rbf_graph_ssl.py` | Equation (56) and Theorem 5.1: cost-sensitive Gaussian-RBF harmonic classification | Score crossings, piecewise-constant utilities, and full-information regret |
| `python experiments/common_offset_demo.py` | The algebraic common-offset mechanism used in Theorem 5.2 | A one-parameter coefficient curve and rigid translation of ordered roots |
| `python -m pytest` | Equation-level consistency checks | Exact dimension-free ratio, cube-section incidence identity, harmonic derivative, online protocol, and polynomial root translation |

The RBF experiment reports regret to the finite algorithm grid exactly and a
separate finite proxy formed by the union of a dense evaluation grid and the
algorithm grid.  The common-offset
plot visualizes the exact one-scalar parameterization of the induced
coefficient and root laws.

## Quick start

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
bash reproduce_all.sh
```

The script first runs the complete test suite and then regenerates every
artifact.  Figures are written to `figures/` and tabular summaries to
`results/`.  All pseudorandom generators use fixed, explicitly recorded seeds.
The reference run takes about one minute on a recent laptop and uses well under
one gigabyte of memory.

To run only the fast correctness checks:

```bash
python -m pytest
```

## Experimental scope

- The dimension-free calculation takes
  $\alpha\sim\operatorname{Unif}([-1,1]^N)$, hence $A=1$, and uses
  $I=[-\varepsilon,\varepsilon]$ with $0<\varepsilon\le\delta$.  Its exact
  reference value is
  $\Pr\{x_\alpha\in I\}/|I|=1/(4\delta)$; the general Equation (19) constant
  is $A/(4\delta)$.
  Only the two active coordinates are sampled; the remaining cube coordinates
  are analytically inert and are therefore omitted without changing the law of
  the root event.
- For efficiency, the oblivious RBF environment precomputes its utility
  matrices.  They remain outside the learner: `act()` accepts no utility input,
  and the current row is passed to `observe()` only after the action is fixed.
  Grounded systems are solved directly; no matrix inverse or determinant
  expansion is used numerically.
- The common-offset experiment verifies the polynomial shift identity before
  plotting the coefficient curve and the translated root law.

## Reference outputs

The horizontal line in the first figure is the exact $A=1$ hitting ratio for
the displayed fixed interval.  Error bars are pointwise 95% Wilson intervals.

<p align="center">
  <img src="figures/dimension_free.png" width="650" alt="Dimension-free interval-hitting ratio">
</p>

The RBF panels expose illustrative harmonic-score crossings and
piecewise-constant utilities from the fixed maximum-horizon run.  The script
deterministically selects a curve with the largest dense-grid sign-change
count, breaking ties by proximity to the center, and then isolates the shown
crossings by bisection.  The utility panel includes that round and the rounds
with the largest dense-grid change counts; its step locations remain a
dense-grid rendering.  The selection metadata are recorded in `results/`.

Regret to the algorithm grid and regret to the dense-plus-algorithm-grid finite
proxy are reported separately.  The pointwise normal-approximation 95%
intervals are over 30 independent cost/action repetitions, conditional on the
fixed base-instance sequence.  This is a finite-horizon diagnostic, not an
estimate of the theorem's asymptotic exponent or continuous-comparator bound.

<p align="center">
  <img src="figures/rbf_graph_ssl/regret_scaling.png" width="820" alt="RBF finite-horizon regret diagnostics">
</p>

<p align="center">
  <img src="figures/rbf_graph_ssl/transition_curve.png" width="48%" alt="RBF transition curve">
  <img src="figures/rbf_graph_ssl/utility_curves.png" width="48%" alt="RBF piecewise-constant utilities">
</p>

The common-offset figure separates coefficient-space geometry from root-space
regularity: shifted coefficient vectors trace a one-dimensional polynomial
curve, while each ordered root is an exact translate of the scalar offset.

<p align="center">
  <img src="figures/common_offset_demo.png" width="760" alt="Common-offset coefficient curve and root law">
</p>

## Repository layout

```text
experiments/          reproducible figure and table generators
src/sharp_root_ac/    incidence, harmonic-extension, online-learning,
                      and polynomial-shift routines
tests/                deterministic equation- and protocol-level checks
figures/              generated PDF/PNG figures
results/              generated CSV summaries and selection metadata
```

## License

The code is released under the MIT License.
