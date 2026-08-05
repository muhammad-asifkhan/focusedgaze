"""Versioned, self-describing calibration profiles.

WHAT CHANGED FROM THE ORIGINAL (and why)
----------------------------------------
The legacy system persisted its calibration with ``pickle``, and the pickled
dict held **live** ``PolynomialFeatures`` and ``LinearRegression`` objects. Three
consequences, all of them defects rather than inconveniences:

* Loading a profile required scikit-learn **at runtime**, so a library whose
  inference path is pure ONNX still dragged in a fitting framework.
* The file was version-fragile. sklearn does not promise that an estimator
  pickled by one release unpickles into an equivalent object in the next, so a
  profile could stop loading, or load subtly differently, after a routine
  dependency upgrade.
* ``pickle.load`` executes arbitrary code, so a profile from an untrusted source
  is a code-execution vector.

A calibration is a polynomial. A polynomial is a table of exponents and two
vectors of coefficients. That is all this format stores, in JSON, alongside the
metadata needed to tell one profile from another:

    schema_version, created_at, screen resolution, camera resolution,
    validation_error (held-out error as a fraction of the screen)

:meth:`CalibrationProfile.apply` is therefore **pure NumPy**. scikit-learn is
needed to *fit* a profile (see :mod:`focusedgaze.calibration.fitter`) and to
*migrate* an old pickle, never to use one. That is the whole point of the phase.

TERM ORDERING - THE TRAP
------------------------
The coefficients come out of sklearn ordered to match
``PolynomialFeatures.powers_``. For two inputs that order is::

    1, a, b, a^2, ab, b^2, a^3, a^2 b, a b^2, b^3

that is, ascending total degree, and within a degree the exponent of the first
input descending. Getting this wrong does **not** raise: it evaluates a smooth,
plausible surface in the wrong place. So the exponent table is written into the
profile explicitly, :func:`polynomial_powers` reproduces sklearn's order for
fitting and migration, and both are pinned by tests that were shown to fail when
the order is perturbed (see ``tests/test_calibration_profile.py`` and the
mutation-check record in MIGRATION_AUDIT.md).

Numbers are stored as JSON floats. ``json`` writes a float through ``repr``,
which is the shortest string that round-trips a ``float64`` exactly, so nothing
is lost. The Phase 1 fixture-rounding trap (audit F2) does not apply here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ..exceptions import CalibrationError, ProfileVersionError

__all__ = [
    "ACTIVE_POINTER_NAME",
    "DEFAULT_CAMERA_SIZE",
    "DEFAULT_CLAMP",
    "FORMAT_ID",
    "PROFILE_DIR_ENV",
    "PROFILE_SUFFIX",
    "SCHEMA_VERSION",
    "CalibrationProfile",
    "active_profile_name",
    "delete_profile",
    "iter_profiles",
    "list_profiles",
    "load_active_profile",
    "migrate_pickle",
    "polynomial_powers",
    "profiles_dir",
    "set_active_profile",
]

_log = logging.getLogger("focusedgaze")

#: Bumped whenever the on-disk layout changes in a way an older reader cannot
#: cope with. A profile written by a newer schema is refused, not guessed at.
SCHEMA_VERSION: Final = 1

#: Written into every file so a stray JSON document is never mistaken for one.
FORMAT_ID: Final = "focusedgaze.calibration-profile"

PROFILE_SUFFIX: Final = ".json"

#: The active-profile pointer. Deliberately not ``.json``, so it can never
#: collide with a profile called "active" and never shows up in a listing.
ACTIVE_POINTER_NAME: Final = "active.pointer"

#: Overrides the platform config directory. Exists so a test, a CI job or a
#: portable install can redirect profiles without a hard-coded path (rule 5).
PROFILE_DIR_ENV: Final = "FOCUSEDGAZE_PROFILE_DIR"

#: The shipping capture resolution (audit section 7). Used as the literal
#: default for ``camera_size`` so no caller has to invent one. There is no
#: equivalent default for the screen: a made-up screen resolution would be a
#: false record, so an unknown screen is stored as null.
DEFAULT_CAMERA_SIZE: Final = (1280, 720)

#: The legacy output clamp. ``apply_calibration`` squeezed its prediction into
#: [0, 1] because a polynomial extrapolates freely outside the calibrated range
#: and a cursor off the edge of the screen is worse than one stuck to it.
DEFAULT_CLAMP: Final = (0.0, 1.0)

_N_INPUTS: Final = 2
_INPUT_NAMES: Final = ("pitch", "yaw")
_OUTPUT_NAMES: Final = ("x", "y")

# A profile name becomes a filename, so it may not contain a separator, may not
# start with a dot, and may not be empty. Checked rather than sanitised: quietly
# rewriting a name containing a parent-directory escape into something else
# would be a surprising success.
_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def polynomial_powers(degree: int, n_inputs: int = _N_INPUTS) -> NDArray[np.int64]:
    """Exponent table in scikit-learn's ``PolynomialFeatures`` order.

    Reproduces ``PolynomialFeatures(degree=degree).powers_`` for
    ``include_bias=True, interaction_only=False``: row ``i`` gives the exponent
    of each input in term ``i``, so term ``i`` of ``(a, b)`` is
    ``a**powers[i, 0] * b**powers[i, 1]``.

    sklearn builds its terms from ``combinations_with_replacement`` over the
    input indices, for total degrees 0, 1, ... degree in that order. This
    function does the same thing, which is why the orders agree; it is not a
    transcription of an observed sequence. ``test_calibration_profile.py``
    asserts the equality against the real sklearn for degrees 1 to 8.

    Args:
        degree: Highest total degree. Must be >= 1.
        n_inputs: Number of input variables. Two for (pitch, yaw).

    Returns:
        An ``(n_terms, n_inputs)`` array of non-negative integer exponents.

    Raises:
        CalibrationError: If ``degree`` or ``n_inputs`` is out of range.
    """
    if degree < 1:
        raise CalibrationError(f"polynomial degree must be >= 1, got {degree}")
    if n_inputs < 1:
        raise CalibrationError(f"n_inputs must be >= 1, got {n_inputs}")

    rows: list[list[int]] = []
    for total in range(degree + 1):
        for combo in combinations_with_replacement(range(n_inputs), total):
            row = [0] * n_inputs
            for index in combo:
                row[index] += 1
            rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def _check_powers_match_sklearn(powers: NDArray[np.int64], degree: int) -> None:
    """Assert a fitted transformer's term order is the one we reproduce.

    Called on every path that reads coefficients out of scikit-learn (fitting
    and pickle migration). If a future sklearn release reorders its features,
    the coefficients would silently attach to the wrong terms; this turns that
    into a loud failure at the only moment it can still be noticed.
    """
    n_inputs = powers.shape[1] if powers.ndim == 2 else _N_INPUTS
    expected = polynomial_powers(degree, n_inputs)
    if powers.shape != expected.shape or not np.array_equal(powers, expected):
        raise CalibrationError(
            "scikit-learn's polynomial term order is not the one this package "
            "reproduces, so the fitted coefficients cannot be attached to terms "
            "safely.\n"
            f"  sklearn : {powers.tolist()}\n"
            f"  expected: {expected.tolist()}\n"
            "This is a hard failure on purpose: the wrong order does not crash, "
            "it evaluates a plausible surface in the wrong place."
        )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _check_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise CalibrationError(
            f"invalid profile name {name!r}. Names become filenames, so they must "
            "start with a letter or digit and contain only letters, digits, dot, "
            "hyphen and underscore (max 64 characters)."
        )
    return name


def _as_size(value: object, field: str) -> tuple[int, int] | None:
    """Coerce a (width, height) pair, or None for 'not recorded'."""
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise CalibrationError(f"{field} must be a (width, height) pair or None, got {value!r}")
    if len(value) != 2:
        raise CalibrationError(f"{field} must have exactly two entries, got {value!r}")
    try:
        width, height = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{field} entries must be integers: {value!r}") from exc
    if width <= 0 or height <= 0:
        raise CalibrationError(f"{field} must be positive, got {(width, height)!r}")
    return (width, height)


def _as_optional_error(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str | bytes) or not isinstance(value, (int, float, np.floating)):
        raise CalibrationError(f"{field} must be a number or None, got {value!r}")
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise CalibrationError(f"{field} must be finite and non-negative, got {out!r}")
    return out


@dataclass(frozen=True, eq=False)
class CalibrationProfile:
    """A fitted (pitch, yaw) -> normalised (x, y) polynomial, and its provenance.

    The mapping is::

        x = intercept_x + sum_i coef_x[i] * pitch**powers[i, 0] * yaw**powers[i, 1]
        y = intercept_y + sum_i coef_y[i] * pitch**powers[i, 0] * yaw**powers[i, 1]

    then clamped to :attr:`clamp`. Outputs are **fractions of the screen**, not
    pixels, which is why :attr:`screen_size` plays no part in the arithmetic: it
    is recorded so a profile replayed on a different display or a different
    camera can be recognised as stale, not to scale anything.

    Equality is disabled (``eq=False``): the coefficient fields are arrays, and a
    generated ``__eq__`` over arrays raises rather than answering. Compare
    :meth:`to_dict` outputs, or the arrays themselves.

    Args:
        degree: Highest total degree of the polynomial.
        powers: ``(n_terms, 2)`` exponent table, term ``i`` in row ``i``.
        coef_x: ``(n_terms,)`` coefficients for the horizontal output.
        coef_y: ``(n_terms,)`` coefficients for the vertical output.
        intercept_x: Constant added to the horizontal output.
        intercept_y: Constant added to the vertical output.
        screen_size: ``(width, height)`` in pixels of the display calibrated
            against, or None when it was not recorded.
        camera_size: ``(width, height)`` in pixels of the capture stream.
        created_at: ISO-8601 UTC timestamp. Filled in when omitted.
        validation_error: Mean held-out error in normalised screen units, the
            same "fraction of screen" the legacy routine printed. None when it
            was not measured.
        fit_error: Mean training-set error, in the same units. This is the
            legacy ``fit_error_normalized``, kept because old profiles carry it
            and because the gap between the two numbers is the overfitting
            signal.
        n_samples: Samples the final fit used.
        n_dropped: Outliers the robust pass identified. Not always the same as
            the number removed: the refit is declined when it would leave too
            few samples, and the legacy count is preserved as-is. Compare
            against ``n_samples`` to tell the two cases apart.
        name: The profile's name, which is also its filename stem.
        source: Free-text provenance, e.g. which fitter produced it.
        clamp: ``(low, high)`` bounds applied to both outputs.
        schema_version: On-disk layout version.

    Raises:
        CalibrationError: If the arguments do not describe a usable polynomial.
    """

    degree: int
    powers: NDArray[np.int64]
    coef_x: NDArray[np.float64]
    coef_y: NDArray[np.float64]
    intercept_x: float
    intercept_y: float
    screen_size: tuple[int, int] | None = None
    camera_size: tuple[int, int] | None = DEFAULT_CAMERA_SIZE
    created_at: str = ""
    validation_error: float | None = None
    fit_error: float | None = None
    n_samples: int | None = None
    n_dropped: int = 0
    name: str = "default"
    source: str = ""
    clamp: tuple[float, float] = DEFAULT_CLAMP
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # frozen dataclass: normalise in place

        degree = int(self.degree)
        if degree < 1:
            raise CalibrationError(f"degree must be >= 1, got {degree}")

        # np.array, not np.asarray: the profile marks these read-only below, and
        # doing that to an array the caller still holds would be a side effect
        # on their data.
        powers = np.array(self.powers, dtype=np.int64)
        if powers.ndim != 2 or powers.shape[1] != _N_INPUTS:
            raise CalibrationError(
                f"powers must be an (n_terms, {_N_INPUTS}) array, got shape {powers.shape}"
            )
        if powers.shape[0] == 0:
            raise CalibrationError("powers must contain at least one term")
        if bool((powers < 0).any()):
            raise CalibrationError("polynomial exponents must be non-negative")
        reached = int(powers.sum(axis=1).max())
        if reached != degree:
            raise CalibrationError(
                f"powers reach total degree {reached} but the profile declares degree {degree}"
            )
        if len({tuple(row) for row in powers.tolist()}) != powers.shape[0]:
            raise CalibrationError("powers contains duplicate terms")

        coef_x = np.array(self.coef_x, dtype=np.float64)
        coef_y = np.array(self.coef_y, dtype=np.float64)
        for label, coef in (("coef_x", coef_x), ("coef_y", coef_y)):
            if coef.shape != (powers.shape[0],):
                raise CalibrationError(
                    f"{label} has shape {coef.shape}, expected ({powers.shape[0]},) "
                    "to match the number of polynomial terms"
                )
            if not bool(np.isfinite(coef).all()):
                raise CalibrationError(f"{label} contains a non-finite value")

        intercept_x, intercept_y = float(self.intercept_x), float(self.intercept_y)
        if not (np.isfinite(intercept_x) and np.isfinite(intercept_y)):
            raise CalibrationError("intercepts must be finite")

        low, high = float(self.clamp[0]), float(self.clamp[1])
        if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
            raise CalibrationError(
                f"clamp must be a finite (low, high) with low < high, got {self.clamp!r}"
            )

        if int(self.n_dropped) < 0:
            raise CalibrationError(f"n_dropped must be >= 0, got {self.n_dropped}")
        if self.n_samples is not None and int(self.n_samples) < 0:
            raise CalibrationError(f"n_samples must be >= 0, got {self.n_samples}")

        # Arrays are made read-only: a frozen dataclass that hands out a mutable
        # array is not frozen in any sense a caller can rely on.
        powers.flags.writeable = False
        coef_x.flags.writeable = False
        coef_y.flags.writeable = False

        set_(self, "degree", degree)
        set_(self, "powers", powers)
        set_(self, "coef_x", coef_x)
        set_(self, "coef_y", coef_y)
        set_(self, "intercept_x", intercept_x)
        set_(self, "intercept_y", intercept_y)
        set_(self, "screen_size", _as_size(self.screen_size, "screen_size"))
        set_(self, "camera_size", _as_size(self.camera_size, "camera_size"))
        set_(self, "created_at", self.created_at or _utc_now_iso())
        set_(self, "validation_error",
             _as_optional_error(self.validation_error, "validation_error"))
        set_(self, "fit_error", _as_optional_error(self.fit_error, "fit_error"))
        set_(self, "n_samples", None if self.n_samples is None else int(self.n_samples))
        set_(self, "n_dropped", int(self.n_dropped))
        set_(self, "name", _check_name(self.name))
        set_(self, "source", str(self.source))
        set_(self, "clamp", (low, high))
        set_(self, "schema_version", int(self.schema_version))

    # -- evaluation ---------------------------------------------------------

    @property
    def n_terms(self) -> int:
        """How many polynomial terms the profile carries."""
        return int(self.powers.shape[0])

    def _terms(self, pitch: NDArray[np.float64], yaw: NDArray[np.float64]) -> NDArray[np.float64]:
        """Expand (pitch, yaw) into the profile's polynomial terms.

        The last axis of the result indexes terms, in the order ``powers``
        declares, so ``terms @ coef`` attaches every coefficient to the term it
        was fitted for. Nothing here assumes any particular order: the table is
        the authority, which is what makes a profile self-describing.
        """
        stacked = np.stack((pitch, yaw), axis=-1)          # (..., 2)
        return np.prod(stacked[..., None, :] ** self.powers, axis=-1)

    def apply(self, pitch: float, yaw: float) -> tuple[float, float]:
        """Map one gaze reading to a normalised screen position.

        Pure NumPy: this is the runtime path and it must not need scikit-learn.

        Args:
            pitch: Vertical gaze angle in radians, as the model reports it.
            yaw: Horizontal gaze angle in radians.

        Returns:
            ``(x, y)`` as fractions of the screen, clamped to :attr:`clamp`.
            The clamp is the legacy behaviour: the polynomial extrapolates
            without limit outside the calibrated range.

        Raises:
            CalibrationError: If either input is not finite.
        """
        if not (np.isfinite(pitch) and np.isfinite(yaw)):
            raise CalibrationError(f"gaze angles must be finite, got ({pitch!r}, {yaw!r})")
        # Scalar evaluation, one reading at a time, exactly as the legacy
        # apply_calibration did. Deliberately NOT routed through the batch path:
        # a matrix product and a dot product may differ in the last bit, and
        # `fitter.robust_fit_samples` compares residuals against a threshold, so
        # a one-ULP difference there could change WHICH samples get rejected and
        # therefore what the final coefficients are.
        terms = self._terms(np.float64(pitch), np.float64(yaw))
        low, high = self.clamp
        x = float(terms @ self.coef_x) + self.intercept_x
        y = float(terms @ self.coef_y) + self.intercept_y
        return (max(low, min(high, x)), max(low, min(high, y)))

    def apply_array(
        self, pitch: object, yaw: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Vectorised :meth:`apply` over arrays of readings.

        Convenience for offline analysis and accuracy measurement. Results may
        differ from :meth:`apply` in the final bit or two, because BLAS picks a
        different kernel for a matrix product than for a dot product, so this is
        never used where a decision turns on an exact comparison.

        Raises:
            CalibrationError: If the shapes differ or an input is not finite.
        """
        p = np.asarray(pitch, dtype=np.float64)
        y_in = np.asarray(yaw, dtype=np.float64)
        if p.shape != y_in.shape:
            raise CalibrationError(f"pitch and yaw shapes differ: {p.shape} vs {y_in.shape}")
        if not (bool(np.isfinite(p).all()) and bool(np.isfinite(y_in).all())):
            raise CalibrationError("gaze angles must be finite")
        terms = self._terms(p, y_in)
        low, high = self.clamp
        x = np.clip(terms @ self.coef_x + self.intercept_x, low, high)
        y = np.clip(terms @ self.coef_y + self.intercept_y, low, high)
        return (x, y)

    __call__ = apply

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The exact JSON document this profile serialises to.

        Self-describing on purpose: a reader that has never heard of sklearn can
        evaluate the polynomial from ``powers``, ``coef_*`` and ``intercept_*``
        alone.
        """
        return {
            "format": FORMAT_ID,
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at": self.created_at,
            "source": self.source,
            "degree": self.degree,
            "input_names": list(_INPUT_NAMES),
            "output_names": list(_OUTPUT_NAMES),
            "powers": self.powers.tolist(),
            "coef_x": self.coef_x.tolist(),
            "coef_y": self.coef_y.tolist(),
            "intercept_x": self.intercept_x,
            "intercept_y": self.intercept_y,
            "clamp": list(self.clamp),
            "screen_size": list(self.screen_size) if self.screen_size else None,
            "camera_size": list(self.camera_size) if self.camera_size else None,
            "validation_error": self.validation_error,
            "fit_error": self.fit_error,
            "n_samples": self.n_samples,
            "n_dropped": self.n_dropped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationProfile:
        """Rebuild a profile from :meth:`to_dict` output.

        Raises:
            ProfileVersionError: If the document announces a schema this build
                does not know how to read. Separate from ``CalibrationError``
                because the remedy differs: a version mismatch is migrated, a
                broken profile is recollected.
            CalibrationError: If the document is not a profile, or describes an
                unusable polynomial.
        """
        if not isinstance(data, dict):
            raise CalibrationError(
                f"a calibration profile must be a JSON object, got {type(data).__name__}"
            )

        fmt = data.get("format")
        if fmt != FORMAT_ID:
            raise CalibrationError(
                f"not a focusedgaze calibration profile (format={fmt!r}, expected {FORMAT_ID!r})"
            )

        raw_version = data.get("schema_version")
        if not isinstance(raw_version, int) or isinstance(raw_version, bool):
            raise CalibrationError(f"schema_version is not an integer: {raw_version!r}")
        if raw_version != SCHEMA_VERSION:
            remedy = (
                "The profile was written by a newer focusedgaze; upgrade the package."
                if raw_version > SCHEMA_VERSION
                else "Re-run calibration to produce a current profile."
            )
            raise ProfileVersionError(
                f"calibration profile schema version {raw_version} is not supported by this "
                f"build (it reads version {SCHEMA_VERSION}). {remedy}"
            )

        try:
            degree = int(data["degree"])
            powers = data["powers"]
            coef_x = data["coef_x"]
            coef_y = data["coef_y"]
            intercept_x = float(data["intercept_x"])
            intercept_y = float(data["intercept_y"])
        except KeyError as exc:
            raise CalibrationError(f"calibration profile is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"calibration profile has a malformed field: {exc}") from exc

        clamp_raw = data.get("clamp") or DEFAULT_CLAMP
        try:
            clamp = (float(clamp_raw[0]), float(clamp_raw[1]))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"clamp is malformed: {clamp_raw!r}") from exc

        try:
            powers_arr = np.asarray(powers, dtype=np.int64)
            coef_x_arr = np.asarray(coef_x, dtype=np.float64)
            coef_y_arr = np.asarray(coef_y, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"calibration profile has malformed arrays: {exc}") from exc

        return cls(
            degree=degree,
            powers=powers_arr,
            coef_x=coef_x_arr,
            coef_y=coef_y_arr,
            intercept_x=intercept_x,
            intercept_y=intercept_y,
            screen_size=_as_size(data.get("screen_size"), "screen_size"),
            camera_size=_as_size(data.get("camera_size"), "camera_size"),
            created_at=str(data.get("created_at") or ""),
            validation_error=_as_optional_error(data.get("validation_error"), "validation_error"),
            fit_error=_as_optional_error(data.get("fit_error"), "fit_error"),
            n_samples=None if data.get("n_samples") is None else int(data["n_samples"]),
            n_dropped=int(data.get("n_dropped") or 0),
            name=str(data.get("name") or "default"),
            source=str(data.get("source") or ""),
            clamp=clamp,
            schema_version=raw_version,
        )

    def to_json(self, *, indent: int | None = 1) -> str:
        """Serialise to a JSON string.

        Floats go through ``repr``, which is the shortest string that round-trips
        a ``float64`` exactly, so this is lossless. ``allow_nan=False`` refuses
        to emit the non-standard NaN and Infinity tokens; the constructor already
        rejects non-finite coefficients, so reaching that error means something
        else is wrong.
        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    # -- files --------------------------------------------------------------

    def save_to(self, path: str | Path) -> Path:
        """Write the profile to an explicit path, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        _log.info("calibration profile written to %s", target)
        return target

    @classmethod
    def load_from(cls, path: str | Path) -> CalibrationProfile:
        """Read a profile from an explicit path.

        Raises:
            CalibrationError: If the file is absent or is not a valid profile.
            ProfileVersionError: If its schema version is unsupported.
        """
        target = Path(path)
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise CalibrationError(f"cannot read calibration profile {target}: {exc}") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise CalibrationError(f"{target} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def save(self, name: str | None = None, *, directory: str | Path | None = None) -> Path:
        """Save under a name in the profile directory.

        Args:
            name: Profile name. Defaults to :attr:`name`. Passing a different
                one renames the profile, and the stored ``name`` field follows,
                so a file and its contents never disagree about what it is.
            directory: Override the profile directory. See :func:`profiles_dir`.

        Returns:
            The path written.
        """
        chosen = _check_name(name) if name is not None else self.name
        profile = self if chosen == self.name else replace(self, name=chosen)
        return profile.save_to(profiles_dir(directory) / f"{chosen}{PROFILE_SUFFIX}")

    @classmethod
    def load(
        cls, name: str = "default", *, directory: str | Path | None = None,
    ) -> CalibrationProfile:
        """Load a named profile from the profile directory.

        Raises:
            CalibrationError: If no profile of that name exists. The message
                lists what is there, because "not found" without the
                alternatives is the least useful error in any CLI.
            ProfileVersionError: If its schema version is unsupported.
        """
        _check_name(name)
        base = profiles_dir(directory)
        path = base / f"{name}{PROFILE_SUFFIX}"
        if not path.is_file():
            known = list_profiles(directory)
            listing = ", ".join(known) if known else "none"
            raise CalibrationError(
                f"no calibration profile named {name!r} in {base} (available: {listing})"
            )
        return cls.load_from(path)


def profiles_dir(directory: str | Path | None = None) -> Path:
    """Where named profiles live.

    Resolution order, first match wins:

    1. the ``directory`` argument;
    2. the ``FOCUSEDGAZE_PROFILE_DIR`` environment variable;
    3. ``platformdirs.user_config_dir("focusedgaze")`` plus ``profiles``.

    The directory is not created here; :meth:`CalibrationProfile.save` creates
    it, so merely asking where profiles would live never writes anything.
    """
    if directory is not None:
        return Path(directory)
    env = os.environ.get(PROFILE_DIR_ENV)
    if env:
        return Path(env)
    # Imported here rather than at module scope so that pointing the profile
    # directory somewhere explicit never depends on platformdirs resolving.
    import platformdirs

    return Path(platformdirs.user_config_dir("focusedgaze", appauthor=False)) / "profiles"


def list_profiles(directory: str | Path | None = None) -> list[str]:
    """Names of every saved profile, sorted. Empty when the directory is absent."""
    base = profiles_dir(directory)
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob(f"*{PROFILE_SUFFIX}") if p.is_file())


def iter_profiles(directory: str | Path | None = None) -> Iterator[CalibrationProfile]:
    """Load every saved profile, skipping any that will not parse.

    One corrupt file must not make the whole list unreadable, so failures are
    logged and passed over rather than raised.
    """
    for name in list_profiles(directory):
        try:
            yield CalibrationProfile.load(name, directory=directory)
        except CalibrationError as exc:   # ProfileVersionError is a subclass
            _log.warning("skipping unreadable calibration profile %r: %s", name, exc)


def delete_profile(name: str, directory: str | Path | None = None) -> bool:
    """Delete a named profile. Returns False if it was not there.

    Clears the active pointer when it named the deleted profile, so "load the
    active profile" can never resolve to something that no longer exists.
    """
    _check_name(name)
    base = profiles_dir(directory)
    path = base / f"{name}{PROFILE_SUFFIX}"
    if not path.is_file():
        return False
    was_active = active_profile_name(directory) == name
    path.unlink()
    if was_active:
        (base / ACTIVE_POINTER_NAME).unlink(missing_ok=True)
    _log.info("deleted calibration profile %r", name)
    return True


def set_active_profile(name: str, directory: str | Path | None = None) -> None:
    """Make ``name`` the profile :func:`load_active_profile` returns.

    Raises:
        CalibrationError: If no profile of that name exists. Pointing at a
            missing profile would only defer the failure to whoever loads it.
    """
    _check_name(name)
    base = profiles_dir(directory)
    if not (base / f"{name}{PROFILE_SUFFIX}").is_file():
        raise CalibrationError(f"cannot activate {name!r}: no such profile in {base}")
    base.mkdir(parents=True, exist_ok=True)
    (base / ACTIVE_POINTER_NAME).write_text(name, encoding="utf-8")
    _log.info("active calibration profile is now %r", name)


def active_profile_name(directory: str | Path | None = None) -> str | None:
    """The active profile's name, or None when none has been chosen."""
    pointer = profiles_dir(directory) / ACTIVE_POINTER_NAME
    if not pointer.is_file():
        return None
    return pointer.read_text(encoding="utf-8").strip() or None


def load_active_profile(directory: str | Path | None = None) -> CalibrationProfile:
    """Load whichever profile is active.

    Falls back to a profile named ``default`` when no pointer has been set, so a
    single-user install never has to think about switching.

    Raises:
        CalibrationError: If neither an active pointer nor a ``default`` profile
            exists.
    """
    name = active_profile_name(directory) or "default"
    return CalibrationProfile.load(name, directory=directory)


def migrate_pickle(
    path: str | Path,
    *,
    name: str = "default",
    screen_size: tuple[int, int] | None = None,
    camera_size: tuple[int, int] | None = DEFAULT_CAMERA_SIZE,
) -> CalibrationProfile:
    """Convert a legacy ``.pkl`` calibration into a :class:`CalibrationProfile`.

    One-shot, and the only place in this package that unpickles anything.

    **This executes the file.** ``pickle.load`` runs arbitrary code by design,
    and it needs scikit-learn present because the legacy dict holds live
    ``PolynomialFeatures`` and ``LinearRegression`` objects. Both facts are the
    reason the format is being replaced. Run it on a profile you produced
    yourself, keep the JSON, and delete the pickle.

    What is carried across, and what cannot be:

    * carried: degree, the exponent table, both coefficient vectors, both
      intercepts, and the legacy ``fit_error_normalized`` as :attr:`fit_error`;
    * ``created_at`` comes from the file's modification time, because the pickle
      records no timestamp of its own and stamping it "now" would make every
      migrated profile look freshly calibrated;
    * ``validation_error`` stays None. The legacy pickle never stored one (the
      routine printed the held-out error and discarded it), so there is nothing
      to migrate and putting a number there would be a fabrication;
    * ``screen_size``/``camera_size`` are whatever the caller passes, for the
      same reason: the pickle does not record them.

    Args:
        path: The ``.pkl`` file to read.
        name: Name for the resulting profile.
        screen_size: Display the pickle was calibrated against, if known.
        camera_size: Capture resolution it was calibrated against.

    Raises:
        CalibrationError: If the file is missing, is not a legacy calibration
            dict, cannot be unpickled, or its term order is not the one this
            package reproduces.
    """
    import pickle  # local import: nothing else in the package unpickles, by design

    source = Path(path)
    if not source.is_file():
        raise CalibrationError(f"no calibration pickle at {source}")

    _log.warning(
        "unpickling %s to migrate it. pickle.load executes arbitrary code: only do "
        "this with a file you created yourself.", source,
    )
    try:
        with source.open("rb") as handle:
            model = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - unpickling can raise essentially anything
        raise CalibrationError(
            f"cannot unpickle {source}: {exc!r}. Legacy profiles embed scikit-learn "
            "estimator objects, so migration needs scikit-learn installed "
            "(pip install 'focusedgaze[calibration]') and a version close enough to "
            "the one that wrote the file. That version fragility is exactly what the "
            "JSON profile format removes."
        ) from exc

    if not isinstance(model, dict):
        raise CalibrationError(
            f"{source} does not contain a legacy calibration dict (found {type(model).__name__})"
        )
    missing = {"poly", "reg_x", "reg_y", "degree"} - set(model)
    if missing:
        raise CalibrationError(f"{source} is missing legacy calibration keys: {sorted(missing)}")

    poly = model["poly"]
    try:
        powers = np.asarray(poly.powers_, dtype=np.int64)
        n_features_in = int(poly.n_features_in_)
        include_bias = bool(poly.include_bias)
        interaction_only = bool(poly.interaction_only)
    except AttributeError as exc:
        raise CalibrationError(f"{source} does not hold a fitted PolynomialFeatures: {exc}") from exc

    if n_features_in != _N_INPUTS:
        raise CalibrationError(
            f"{source} was fitted on {n_features_in} inputs; this package calibrates on "
            f"exactly {_N_INPUTS} (pitch, yaw)"
        )
    if not include_bias or interaction_only:
        raise CalibrationError(
            f"{source} used PolynomialFeatures(include_bias={include_bias}, "
            f"interaction_only={interaction_only}); only the shipping configuration "
            "(include_bias=True, interaction_only=False) can be migrated"
        )

    degree = int(model["degree"])
    # The migration READS the term order out of the file rather than assuming
    # it, then CHECKS it against the order this package generates. Either alone
    # would be weaker: reading without checking would silently accept a file
    # whose order we cannot reproduce when fitting, and checking without reading
    # would trust an assumption over the evidence in front of us.
    _check_powers_match_sklearn(powers, degree)

    try:
        coef_x = np.asarray(model["reg_x"].coef_, dtype=np.float64)
        coef_y = np.asarray(model["reg_y"].coef_, dtype=np.float64)
        intercept_x = float(model["reg_x"].intercept_)
        intercept_y = float(model["reg_y"].intercept_)
    except AttributeError as exc:
        raise CalibrationError(f"{source} does not hold fitted regressors: {exc}") from exc

    created_at = datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat(timespec="seconds")
    profile = CalibrationProfile(
        degree=degree,
        powers=powers,
        coef_x=coef_x,
        coef_y=coef_y,
        intercept_x=intercept_x,
        intercept_y=intercept_y,
        screen_size=screen_size,
        camera_size=camera_size,
        created_at=created_at,
        validation_error=None,
        fit_error=_as_optional_error(model.get("fit_error_normalized"), "fit_error"),
        name=name,
        source=f"migrated from pickle: {source.name}",
    )
    _log.info("migrated %s to a degree-%d profile with %d terms", source, degree, profile.n_terms)
    return profile
