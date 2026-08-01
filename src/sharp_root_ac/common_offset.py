"""Algebra for a polynomial observed through a common additive offset.

Coefficients use NumPy's ascending-power convention: ``a[k]`` multiplies
``x**k``.  If

    p(x) = sum_k a[k] x**k,

then :func:`shift_coefficients` returns the coefficients of ``p(x + offset)``.
The implementation is the finite binomial identity used by the common-offset
application; it does not estimate coefficients by interpolation or root
finding.
"""

from __future__ import annotations

from math import comb
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


NumericArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]


def _coefficient_array(coefficients: ArrayLike) -> NumericArray:
    """Return a finite, nonempty one-dimensional coefficient array."""

    raw = np.asarray(coefficients)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    array = np.asarray(raw, dtype=dtype)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("coefficients must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("coefficients must be finite")
    return array


def polynomial_degree(coefficients: ArrayLike, *, atol: float = 0.0) -> int:
    """Return the degree after removing numerically zero trailing terms.

    The default ``atol=0`` performs exact zero trimming.  A positive tolerance
    is useful only when the input coefficients themselves arose from floating
    point arithmetic.
    """

    if atol < 0:
        raise ValueError("atol must be nonnegative")
    array = _coefficient_array(coefficients)
    nonzero = np.flatnonzero(np.abs(array) > atol)
    if nonzero.size == 0:
        return -1
    return int(nonzero[-1])


def shift_coefficients(coefficients: ArrayLike, offset: float) -> NumericArray:
    r"""Return coefficients of :math:`p(x+v)` for ``v=offset``.

    For ascending coefficients :math:`a_0,\ldots,a_m`, the returned vector is

    .. math::

       b_j(v)=\sum_{k=j}^m {k\choose j}a_kv^{k-j}.

    Translation leaves the leading coefficient unchanged.  In particular,
    when the degree is at least two, the normalized pair
    :math:`(b_{m-1}/b_m,b_{m-2}/b_m)` traces a one-scalar polynomial curve.
    """

    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    source = _coefficient_array(coefficients)
    shifted = np.zeros_like(source)
    degree = source.size - 1
    for j in range(degree + 1):
        shifted[j] = sum(
            comb(k, j) * source[k] * offset ** (k - j)
            for k in range(j, degree + 1)
        )
    return shifted


def evaluate_polynomial(coefficients: ArrayLike, points: ArrayLike) -> np.ndarray:
    """Evaluate an ascending-coefficient polynomial by Horner's rule."""

    array = _coefficient_array(coefficients)
    return np.polynomial.polynomial.polyval(points, array)


def translate_roots(roots: ArrayLike, offset: ArrayLike) -> np.ndarray:
    r"""Translate roots of :math:`p` to roots of :math:`x\mapsto p(x+v)`.

    The function deliberately performs no numerical root finding.  It records
    the exact rigid translation :math:`z\mapsto z-v`.  ``offset`` may be a
    scalar or an array broadcast-compatible with ``roots``; the latter form is
    convenient for drawing independent samples of a single translated root.
    """

    root_array = np.asarray(roots)
    if root_array.ndim != 1:
        raise ValueError("roots must be a one-dimensional array")
    if not np.all(np.isfinite(root_array)):
        raise ValueError("roots must be finite")
    offset_array = np.asarray(offset)
    if not np.all(np.isfinite(offset_array)):
        raise ValueError("offset must be finite")
    try:
        return root_array - offset_array
    except ValueError as error:
        raise ValueError("roots and offsets must be broadcast-compatible") from error


def normalized_subleading_curve(
    coefficients: ArrayLike,
    offsets: ArrayLike,
    *,
    atol: float = 0.0,
) -> NDArray[np.float64]:
    r"""Evaluate the normalized top-two nonleading coefficients after shifts.

    The output has shape ``(len(offsets), 2)``.  Its columns are
    :math:`b_{m-1}(v)/b_m` and :math:`b_{m-2}(v)/b_m`.  A degree of at least
    two is required.  Real coefficients and offsets are used because this
    helper is intended for the geometric common-offset illustration.
    """

    source = _coefficient_array(coefficients)
    if np.iscomplexobj(source):
        raise ValueError("the coefficient-curve illustration requires real coefficients")
    degree = polynomial_degree(source, atol=atol)
    if degree < 2:
        raise ValueError("at least a quadratic polynomial is required")
    source = np.asarray(source[: degree + 1], dtype=np.float64)
    leading = source[degree]
    offset_array = np.asarray(offsets, dtype=np.float64)
    if offset_array.ndim != 1:
        raise ValueError("offsets must be a one-dimensional array")
    if not np.all(np.isfinite(offset_array)):
        raise ValueError("offsets must be finite")

    curve = np.empty((offset_array.size, 2), dtype=np.float64)
    for index, offset in enumerate(offset_array):
        shifted = shift_coefficients(source, float(offset))
        curve[index, 0] = float(shifted[degree - 1] / leading)
        curve[index, 1] = float(shifted[degree - 2] / leading)
    return curve
