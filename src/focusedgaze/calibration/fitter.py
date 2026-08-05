"""Polynomial calibration fit, with robust outlier rejection. FIT TIME ONLY.

This is the only module in the package that needs scikit-learn, and it is not
imported by anything on the runtime path. Applying a calibration is pure NumPy
(:meth:`focusedgaze.calibration.profile.CalibrationProfile.apply`); fitting one
needs a least-squares solver and a polynomial feature expansion, and there is no
reason to reimplement either.

The import is done inside the functions, not at module scope, so that
``import focusedgaze.calibration.fitter`` succeeds without the ``calibration``
extra and a missing dependency surfaces as a :class:`CalibrationError` naming the
remedy rather than a bare ``ImportError``.

THE DEGREE DISCREPANCY - PRESERVED DELIBERATELY
-----------------------------------------------
The legacy ``fit_calibration`` defaults to ``degree=2``. The legacy
``robust_fit_samples`` passes ``degree=3``. Nothing in the live system reaches
the fitter except through the robust path, so effective behaviour has always
been degree 3, while the plain entry point's own default was quadratic. The
README's "degree-3" claim is true only via the robust path (audit F3).

**Both defaults are reproduced exactly.** They are not tidied into agreement:
that would be a behaviour change disguised as a cleanup, and any caller of the
plain function would silently get a different model. The two constants below name
the discrepancy instead of hiding it.

WHAT IS NEW HERE (and how the old numbers are still guaranteed)
---------------------------------------------------------------
The legacy fitter recorded only a **training-set** error, which its own comment
admitted was "not a true held-out validation". The profile format requires a real
one. So the fitters also measure a held-out error - but they measure it with a
**throwaway parallel fit** on a subset, and the coefficients that get shipped are
still fitted on every sample exactly as before. The shipped model is therefore
bit-for-bit what the legacy code produced; ``validation_error`` is an extra
observation about it, not an influence on it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ..exceptions import CalibrationError
from .profile import (
    DEFAULT_CAMERA_SIZE,
    CalibrationProfile,
    _check_powers_match_sklearn,
    polynomial_powers,
)

__all__ = [
    "DEFAULT_DEGREE",
    "DEFAULT_MAD_FACTOR",
    "DEFAULT_MIN_KEEP",
    "DEFAULT_VALIDATION_FRACTION",
    "DEFAULT_VALIDATION_SEED",
    "MAD_TO_SIGMA",
    "ROBUST_DEFAULT_DEGREE",
    "FitResult",
    "fit_calibration",
    "robust_fit_samples",
]

_log = logging.getLogger("focusedgaze")

#: ``fit_calibration``'s legacy default. Quadratic. See the module docstring.
DEFAULT_DEGREE: Final = 2

#: ``robust_fit_samples``'s legacy default, and what the live system runs.
ROBUST_DEFAULT_DEGREE: Final = 3

#: Residual outlier threshold, in robust standard deviations.
DEFAULT_MAD_FACTOR: Final = 2.5

#: Refuse to drop outliers if it would leave fewer than this many samples.
DEFAULT_MIN_KEEP: Final = 60

#: Scale factor turning a median absolute deviation into a standard-deviation
#: estimate for normally distributed residuals. Carried over verbatim.
MAD_TO_SIGMA: Final = 1.4826

#: Guards against a zero MAD when more than half the residuals are identical.
#: Carried over verbatim: it shifts the threshold, so it is behaviour.
_MAD_EPS: Final = 1e-9

#: Fraction of samples held out to measure ``validation_error``.
DEFAULT_VALIDATION_FRACTION: Final = 0.2

#: The hold-out split is a seeded permutation, so a given sample set always
#: produces the same validation number. An unseeded split would make the
#: recorded error irreproducible, and an unreproducible number in a profile is
#: worse than no number.
DEFAULT_VALIDATION_SEED: Final = 0


@dataclass(frozen=True)
class FitResult:
    """A fitted profile and what the robust pass did to get there.

    Args:
        profile: The fitted calibration.
        n_dropped: Outliers the residual pass **identified**. Legacy semantics,
            preserved: when the refit is declined because too few samples would
            remain, this still reports what was found, and ``n_samples`` is what
            reveals that nothing was actually removed.
        n_samples: Samples the final fit was given.
    """

    profile: CalibrationProfile
    n_dropped: int
    n_samples: int


def _require_sklearn() -> tuple[Any, Any]:
    """Import scikit-learn, or explain how to get it.

    D1's rule applied to the calibration extra: a missing optional dependency is
    an expected state with a known remedy, not a stack trace from an import.
    """
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
        from sklearn.preprocessing import PolynomialFeatures  # type: ignore[import-untyped]
    except ImportError as exc:
        raise CalibrationError(
            "fitting a calibration needs scikit-learn, which the base install does "
            "not require because applying a calibration does not. Install it with:\n"
            "    pip install 'focusedgaze[calibration]'"
        ) from exc
    return PolynomialFeatures, LinearRegression


def _as_sample_array(samples: Iterable[Sequence[float]]) -> NDArray[np.float64]:
    """Validate and normalise ``(pitch, yaw, target_x, target_y)`` rows.

    Raises:
        CalibrationError: If the samples are empty, wrongly shaped, or contain a
            value that is not a finite number.
    """
    try:
        rows = samples if isinstance(samples, np.ndarray) else list(samples)
        data = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"calibration samples are not numeric: {exc}") from exc
    if data.size == 0:
        raise CalibrationError("no calibration samples: nothing to fit")
    if data.ndim != 2 or data.shape[1] != 4:
        raise CalibrationError(
            "calibration samples must be (pitch, yaw, target_x, target_y) rows, got "
            f"shape {data.shape}"
        )
    if not bool(np.isfinite(data).all()):
        raise CalibrationError("calibration samples contain a non-finite value")
    return data


def _core_fit(data: NDArray[np.float64], degree: int) -> CalibrationProfile:
    """The legacy ``fit_calibration`` arithmetic, expressed as a profile.

    Every step is the original one, in the original order:
    ``PolynomialFeatures(degree=degree)`` with its default ``include_bias=True``,
    two ``LinearRegression`` fits against the same expanded matrix, and the
    training error as the mean Euclidean residual over the fitted points. The
    only difference is what comes out: coefficients instead of live estimators.
    """
    polynomial_features, linear_regression = _require_sklearn()

    gaze = data[:, 0:2]
    targets_x = data[:, 2]
    targets_y = data[:, 3]

    n_terms = int(polynomial_powers(degree).shape[0])
    if data.shape[0] < n_terms:
        raise CalibrationError(
            f"a degree-{degree} fit has {n_terms} terms but only {data.shape[0]} samples "
            "were given; the system would be underdetermined and the resulting profile "
            "meaningless. Collect more samples or lower the degree."
        )

    poly = polynomial_features(degree=degree)
    gaze_poly = poly.fit_transform(gaze)

    reg_x = linear_regression().fit(gaze_poly, targets_x)
    reg_y = linear_regression().fit(gaze_poly, targets_y)

    # Read the term order out of the fitted transformer, then check it against
    # the order this package generates. See profile._check_powers_match_sklearn.
    powers = np.asarray(poly.powers_, dtype=np.int64)
    _check_powers_match_sklearn(powers, degree)

    # Training-fit error, verbatim from the legacy fitter including its use of
    # the UNCLAMPED predictions. It is a sanity check, not a validation; that is
    # what validation_error is for.
    pred_x = reg_x.predict(gaze_poly)
    pred_y = reg_y.predict(gaze_poly)
    err_x = np.abs(pred_x - targets_x)
    err_y = np.abs(pred_y - targets_y)
    fit_error = float(np.mean(np.sqrt(err_x**2 + err_y**2)))

    return CalibrationProfile(
        degree=degree,
        powers=powers,
        coef_x=np.asarray(reg_x.coef_, dtype=np.float64),
        coef_y=np.asarray(reg_y.coef_, dtype=np.float64),
        intercept_x=float(reg_x.intercept_),
        intercept_y=float(reg_y.intercept_),
        fit_error=fit_error,
        n_samples=int(data.shape[0]),
        camera_size=None,
    )


def _keep_mask(
    data: NDArray[np.float64], profile: CalibrationProfile, mad_factor: float,
) -> NDArray[np.bool_]:
    """Which samples survive the MAD outlier pass. Legacy arithmetic, verbatim.

    Two details that look incidental and are not:

    * the residuals use the **clamped** prediction, because the legacy code
      called ``apply_calibration``, which clamps. A sample whose prediction lands
      off-screen therefore has its residual capped, which makes it less likely to
      be rejected, not more;
    * the predictions are computed one at a time rather than as a batch. A batch
      matrix product can differ from a dot product in the last bit, and this
      threshold comparison would turn that into a different set of retained
      samples and so a different final model.
    """
    predictions = np.asarray(
        [profile.apply(float(pitch), float(yaw)) for pitch, yaw in data[:, 0:2]],
        dtype=np.float64,
    )
    residuals = np.linalg.norm(predictions - data[:, 2:4], axis=1)
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median)) + _MAD_EPS
    return np.asarray(residuals <= median + mad_factor * MAD_TO_SIGMA * mad)


def _holdout_error(
    data: NDArray[np.float64],
    degree: int,
    *,
    robust: bool,
    mad_factor: float,
    min_keep: int,
    fraction: float,
    seed: int,
) -> float | None:
    """Mean held-out error, as a fraction of the screen, or None if unmeasurable.

    Fits a **throwaway** model on a random training subset and scores it on the
    rest. The profile that gets returned to the caller is fitted separately on
    every sample, so this measurement never moves the shipped coefficients.

    The same pipeline is used for the throwaway fit as for the real one,
    including the outlier pass when there is one: a validation number measured
    from a different procedure than the model it describes is misleading.

    Returns None when the split cannot be made honestly: no hold-out fraction
    requested, fewer validation points than one, or too few training points to
    determine the polynomial.
    """
    n = int(data.shape[0])
    if fraction <= 0.0:
        return None
    n_val = int(round(n * fraction))
    n_train = n - n_val
    n_terms = int(polynomial_powers(degree).shape[0])
    if n_val < 1 or n_train < n_terms:
        _log.info(
            "skipping held-out validation: %d samples cannot be split into a "
            "training set of at least %d and a non-empty hold-out set",
            n, n_terms,
        )
        return None

    order = np.random.default_rng(seed).permutation(n)
    train = data[order[n_val:]]
    validate = data[order[:n_val]]

    profile = _core_fit(train, degree)
    if robust:
        keep = _keep_mask(train, profile, mad_factor)
        n_dropped = int((~keep).sum())
        if n_dropped and int(keep.sum()) >= min_keep:
            profile = _core_fit(train[keep], degree)

    # Scored through the clamped apply(), which is how the legacy calibration
    # routine measured its on-screen validation error too.
    errors: list[float] = []
    for row in validate:
        x, y = profile.apply(float(row[0]), float(row[1]))
        errors.append(float(np.hypot(x - float(row[2]), y - float(row[3]))))
    return float(np.mean(errors))


def fit_calibration(
    samples: Iterable[Sequence[float]],
    *,
    degree: int = DEFAULT_DEGREE,
    name: str = "default",
    screen_size: tuple[int, int] | None = None,
    camera_size: tuple[int, int] | None = DEFAULT_CAMERA_SIZE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    validation_seed: int = DEFAULT_VALIDATION_SEED,
) -> CalibrationProfile:
    """Fit a polynomial calibration with no outlier rejection.

    The direct replacement for the legacy ``fit_calibration``, **including its
    ``degree=2`` default**. Most callers want :func:`robust_fit_samples`, which
    is what the shipping system uses and which defaults to degree 3.

    Args:
        samples: ``(pitch, yaw, target_x, target_y)`` rows. Angles in radians,
            targets as fractions of the screen.
        degree: Polynomial degree. Defaults to the legacy quadratic.
        name: Name for the resulting profile.
        screen_size: Display calibrated against, if known. Recorded, not used in
            the arithmetic: the outputs are screen fractions.
        camera_size: Capture resolution, likewise recorded.
        validation_fraction: Portion held out to measure ``validation_error``.
            Zero disables the measurement.
        validation_seed: Seed for the hold-out split, so the number is
            reproducible.

    Returns:
        The fitted profile.

    Raises:
        CalibrationError: If scikit-learn is missing, or the samples are empty,
            malformed, non-finite, or too few for the requested degree.
    """
    data = _as_sample_array(samples)
    profile = _core_fit(data, degree)
    validation_error = _holdout_error(
        data, degree, robust=False, mad_factor=DEFAULT_MAD_FACTOR,
        min_keep=DEFAULT_MIN_KEEP, fraction=validation_fraction, seed=validation_seed,
    )
    return replace(
        profile,
        name=name,
        screen_size=screen_size,
        camera_size=camera_size,
        validation_error=validation_error,
        source=f"focusedgaze.calibration.fitter.fit_calibration degree={degree}",
    )


def robust_fit_samples(
    samples: Iterable[Sequence[float]],
    *,
    degree: int = ROBUST_DEFAULT_DEGREE,
    mad_factor: float = DEFAULT_MAD_FACTOR,
    min_keep: int = DEFAULT_MIN_KEEP,
    name: str = "default",
    screen_size: tuple[int, int] | None = None,
    camera_size: tuple[int, int] | None = DEFAULT_CAMERA_SIZE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    validation_seed: int = DEFAULT_VALIDATION_SEED,
) -> FitResult:
    """Fit, reject residual outliers by MAD, refit. The shipping path.

    Blinks and glances away from the target produce samples whose label is
    simply wrong, and least squares has no defence against them. So the model is
    fitted once, each sample's residual is measured against it, and anything
    further than ``mad_factor`` robust standard deviations out is dropped before
    a second fit.

    Two conditions gate the refit, both carried over unchanged: the pass must
    have found at least one outlier, and it must leave at least ``min_keep``
    samples. If dropping the outliers would leave too little data, the first fit
    stands and ``n_dropped`` is reported as zero, because nothing was in fact
    dropped.

    Args:
        samples: ``(pitch, yaw, target_x, target_y)`` rows.
        degree: Polynomial degree. Defaults to 3, the shipping value.
        mad_factor: Rejection threshold in robust standard deviations.
        min_keep: Refuse to refit if fewer than this many samples would remain.
        name: Name for the resulting profile.
        screen_size: Display calibrated against, if known.
        camera_size: Capture resolution.
        validation_fraction: Portion held out to measure ``validation_error``.
        validation_seed: Seed for the hold-out split.

    Returns:
        A :class:`FitResult` carrying the profile, the number of samples
        dropped, and the number the final fit used.

    Raises:
        CalibrationError: If scikit-learn is missing, or the samples are empty,
            malformed, non-finite, or too few for the requested degree.
    """
    data = _as_sample_array(samples)

    profile = _core_fit(data, degree)
    keep = _keep_mask(data, profile, mad_factor)
    # n_dropped counts outliers IDENTIFIED, not necessarily removed. That is the
    # legacy meaning and it is preserved rather than tidied: when the refit is
    # declined below, the original code still returned the count it had found,
    # and any caller comparing the two implementations would see a difference.
    n_dropped = int((~keep).sum())
    data_kept = data
    if n_dropped and int(keep.sum()) >= min_keep:
        data_kept = data[keep]
        profile = _core_fit(data_kept, degree)
    elif n_dropped:
        _log.info(
            "outlier pass identified %d outliers, but keeping only %d samples would "
            "fall below min_keep=%d, so the original fit stands and they were retained",
            n_dropped, int(keep.sum()), min_keep,
        )

    validation_error = _holdout_error(
        data, degree, robust=True, mad_factor=mad_factor, min_keep=min_keep,
        fraction=validation_fraction, seed=validation_seed,
    )
    final = replace(
        profile,
        name=name,
        screen_size=screen_size,
        camera_size=camera_size,
        validation_error=validation_error,
        n_dropped=n_dropped,
        source=(
            "focusedgaze.calibration.fitter.robust_fit_samples "
            f"degree={degree} mad_factor={mad_factor} min_keep={min_keep}"
        ),
    )
    return FitResult(profile=final, n_dropped=n_dropped, n_samples=int(data_kept.shape[0]))
