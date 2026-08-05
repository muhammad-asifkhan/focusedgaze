"""Calibration: collect, fit, persist.

A gaze model reports where the eyes point. It does not know where the screen is,
where the user is sitting, or how their eyes differ from the training set's. A
per-person calibration closes that gap with a low-degree polynomial from
``(pitch, yaw)`` in radians to ``(x, y)`` as fractions of the screen.

The split that matters
----------------------
:mod:`~focusedgaze.calibration.profile` is the **runtime** half: load a profile
and evaluate it in pure NumPy. It imports nothing beyond numpy, and platformdirs
only when asked where profiles live.

:mod:`~focusedgaze.calibration.fitter` is the **fit-time** half: it needs
scikit-learn, and it is the only module in the package that does. Importing this
package does not import scikit-learn; calling a fitter does, and says how to
install it if it is absent.

That separation is the point of Phase 5. The legacy system pickled live
scikit-learn estimator objects, which made a fitting framework a runtime
dependency and made every profile fragile across sklearn releases. See
:mod:`~focusedgaze.calibration.profile` for the full account.

Typical use::

    from focusedgaze.calibration import CalibrationCollector, CalibrationProfile

    collector = CalibrationCollector(screen_size=(1920, 1080))
    for pitch, yaw, tx, ty in readings:
        collector.add_sample(pitch, yaw, tx, ty)
    result = collector.fit(name="alice")        # needs scikit-learn
    result.profile.save()                       # JSON, in the platform config dir

    profile = CalibrationProfile.load("alice")  # needs numpy only
    x, y = profile.apply(pitch, yaw)

Migrating a legacy ``.pkl``::

    from focusedgaze.calibration import migrate_pickle
    migrate_pickle("models/calibration_model.pkl", name="alice").save()
"""

from __future__ import annotations

from .collector import CalibrationCollector, CalibrationSample
from .fitter import (
    DEFAULT_DEGREE,
    DEFAULT_MAD_FACTOR,
    DEFAULT_MIN_KEEP,
    ROBUST_DEFAULT_DEGREE,
    FitResult,
    fit_calibration,
    robust_fit_samples,
)
from .profile import (
    PROFILE_DIR_ENV,
    SCHEMA_VERSION,
    CalibrationProfile,
    active_profile_name,
    delete_profile,
    iter_profiles,
    list_profiles,
    load_active_profile,
    migrate_pickle,
    polynomial_powers,
    profiles_dir,
    set_active_profile,
)

__all__ = [
    "DEFAULT_DEGREE",
    "DEFAULT_MAD_FACTOR",
    "DEFAULT_MIN_KEEP",
    "PROFILE_DIR_ENV",
    "ROBUST_DEFAULT_DEGREE",
    "SCHEMA_VERSION",
    "CalibrationCollector",
    "CalibrationProfile",
    "CalibrationSample",
    "FitResult",
    "active_profile_name",
    "delete_profile",
    "fit_calibration",
    "iter_profiles",
    "list_profiles",
    "load_active_profile",
    "migrate_pickle",
    "polynomial_powers",
    "profiles_dir",
    "robust_fit_samples",
    "set_active_profile",
]
