"""Distance estimation and the positioning gate.

Calibration only holds while you sit where you sat when you recorded it, so the
system constrains the user to a zone: 45-65 cm from the screen, face near the
centre of frame. This module decides whether they are in it.

Distance comes from the pinhole camera model applied to the interpupillary
distance. A closer face spans more pixels between the pupils:

    distance_cm = focal_px * REAL_IPD_CM / ipd_px

WHAT CHANGED FROM THE ORIGINAL (and nothing else did)
-----------------------------------------------------
The legacy `PositioningGate` read its focal length from `models/camera_focal.json`
through a **relative** path at construction time. That made its output depend on
the process working directory: the golden harness measured **117.406 cm from one
directory and 121.244 cm from another for identical landmarks**, a 3.8 cm swing,
larger than the system's entire accuracy budget, decided by where Python was
launched.

Focal length is now explicit. You pass a `focal` calibration or you do not, and
what you get is a function of the arguments. Loading one from a file is a
separate, opt-in call with an explicit path. No hidden lookups, no cwd.

The arithmetic is otherwise untouched, and
``tests/test_golden_tier1.py::test_positioning_gate_matches_golden`` pins it to
the pre-refactor recording.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "LEFT_IRIS_CENTER",
    "NOSE_TIP",
    "RIGHT_IRIS_CENTER",
    "FocalCalibration",
    "PositioningConfig",
    "PositioningGate",
    "PositioningStatus",
]

# MediaPipe FaceLandmarker indices. Iris points require the refined-landmarks
# model (478 points); the base 468-point model does not contain them.
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
NOSE_TIP = 1


class _Landmark(Protocol):
    """The only thing this module needs from a landmark: normalised x/y."""

    x: float
    y: float


@dataclass(frozen=True)
class PositioningConfig:
    """The zone the user must sit in, and how it is judged.

    Defaults are the shipping values and must not drift during extraction.
    """

    min_distance_cm: float = 45.0
    max_distance_cm: float = 65.0
    #: Amber band inside each edge of the green zone.
    warn_margin_cm: float = 5.0
    #: How far the nose may sit from frame centre, as a fraction of width/height.
    center_tolerance: float = 0.12
    #: Adult average interpupillary distance.
    real_ipd_cm: float = 6.3
    #: Fallback horizontal field of view when no focal calibration is supplied.
    assumed_hfov_deg: float = 60.0


@dataclass(frozen=True)
class FocalCalibration:
    """A measured focal length, in pixels, and the frame width it was measured at.

    Measure once by sitting at a known distance; see :meth:`PositioningGate.measure_focal`.
    Storing the frame width lets the value be rescaled if capture resolution changes.
    """

    focal_px: float
    frame_width: int

    def for_width(self, frame_width: int) -> float:
        """This focal length rescaled to a different capture width."""
        return self.focal_px * (frame_width / self.frame_width)

    def to_dict(self) -> dict[str, Any]:
        return {"focal_px": self.focal_px, "frame_w": self.frame_width}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FocalCalibration:
        return cls(float(data["focal_px"]), int(data["frame_w"]))

    @classmethod
    def load(cls, path: str | Path) -> FocalCalibration:
        """Load from a JSON file. The path is explicit: there is no default.

        Raises:
            FileNotFoundError: if the file is absent.
            ValueError: if it does not contain a usable calibration.
        """
        raw = Path(path).read_text(encoding="utf-8")
        try:
            return cls.from_dict(json.loads(raw))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} is not a valid focal calibration: {exc}") from exc

    def save(self, path: str | Path) -> None:
        """Write to a JSON file, creating the parent directory if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()), encoding="utf-8")


@dataclass(frozen=True)
class PositioningStatus:
    """Where the user is, and whether that is good enough to calibrate or track."""

    distance_cm: float
    #: Nose offset from frame centre, signed, as a fraction of frame size.
    dx: float
    dy: float
    distance_ok: bool
    centered: bool
    in_zone: bool
    #: "GREEN" comfortably inside, "YELLOW" near an edge, "RED" outside.
    zone: str
    #: Human-readable guidance, e.g. "MOVE BACK".
    messages: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Legacy-shaped dict, for code written against the original API."""
        return {
            "distance_cm": self.distance_cm,
            "dx": self.dx,
            "dy": self.dy,
            "distance_ok": self.distance_ok,
            "centered": self.centered,
            "in_zone": self.in_zone,
            "zone": self.zone,
            "messages": list(self.messages),
        }


class PositioningGate:
    """Distance and centring constraints, evaluated per frame.

    Args:
        config: Zone thresholds. Defaults match the shipping system.
        focal: Measured focal length. When omitted, distance is derived from
            ``config.assumed_hfov_deg``, which is a rough estimate. A real
            measurement is worth taking.

    Stateless and therefore thread-safe: :meth:`evaluate` reads only its
    arguments and the immutable config.
    """

    __slots__ = ("config", "focal")

    def __init__(self, config: PositioningConfig | None = None,
                 focal: FocalCalibration | None = None) -> None:
        self.config = config or PositioningConfig()
        self.focal = focal

    def focal_px(self, frame_width: int) -> float:
        """Focal length in pixels for this capture width."""
        if self.focal is not None:
            return self.focal.for_width(frame_width)
        half = math.radians(self.config.assumed_hfov_deg) / 2.0
        return frame_width / (2.0 * math.tan(half))

    def measure_focal(self, ipd_px: float, frame_width: int,
                      known_distance_cm: float = 50.0) -> FocalCalibration:
        """Derive a focal length from a face at a known distance.

        Sit at ``known_distance_cm``, measure the inter-pupil distance in pixels,
        and call this once. Returns the calibration; it is the caller's job to
        save it and to pass it back in. Nothing is written implicitly.
        """
        focal = ipd_px * known_distance_cm / self.config.real_ipd_cm
        return FocalCalibration(focal, frame_width)

    def evaluate(self, landmarks: Sequence[_Landmark],
                 frame_shape: tuple[int, ...]) -> PositioningStatus | None:
        """Judge one frame's face position.

        Args:
            landmarks: MediaPipe landmarks, including the iris points: the
                refined 478-point model is required.
            frame_shape: ``(height, width, ...)`` as produced by OpenCV.

        Returns:
            The status, or ``None`` when the eyes are too close together in the
            image to measure: under one pixel apart, which means the face is
            far too distant or the landmarks are degenerate.
        """
        cfg = self.config
        h, w = frame_shape[0], frame_shape[1]

        li = landmarks[LEFT_IRIS_CENTER]
        ri = landmarks[RIGHT_IRIS_CENTER]
        ipd_px = math.hypot((ri.x - li.x) * w, (ri.y - li.y) * h)
        if ipd_px < 1:
            return None
        distance_cm = self.focal_px(w) * cfg.real_ipd_cm / ipd_px

        nose = landmarks[NOSE_TIP]
        dx = nose.x - 0.5
        dy = nose.y - 0.5

        distance_ok = cfg.min_distance_cm <= distance_cm <= cfg.max_distance_cm
        centered = abs(dx) <= cfg.center_tolerance and abs(dy) <= cfg.center_tolerance
        in_zone = distance_ok and centered

        messages: list[str] = []
        if distance_cm < cfg.min_distance_cm:
            messages.append("MOVE BACK")
        elif distance_cm > cfg.max_distance_cm:
            messages.append("COME CLOSER")
        if dx > cfg.center_tolerance:
            messages.append("MOVE LEFT")
        elif dx < -cfg.center_tolerance:
            messages.append("MOVE RIGHT")
        if dy > cfg.center_tolerance:
            messages.append("SIT UP / RAISE CAMERA")
        elif dy < -cfg.center_tolerance:
            messages.append("LOWER YOURSELF / CAMERA")

        if not in_zone:
            zone = "RED"
        elif (distance_cm < cfg.min_distance_cm + cfg.warn_margin_cm
              or distance_cm > cfg.max_distance_cm - cfg.warn_margin_cm
              or abs(dx) > cfg.center_tolerance * 0.7
              or abs(dy) > cfg.center_tolerance * 0.7):
            zone = "YELLOW"
        else:
            zone = "GREEN"

        return PositioningStatus(
            distance_cm=distance_cm, dx=dx, dy=dy,
            distance_ok=distance_ok, centered=centered, in_zone=in_zone,
            zone=zone, messages=tuple(messages),
        )
