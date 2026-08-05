"""Record Tier 2 golden fixtures: real frames -> expected (pitch, yaw).

Tier 1 covers everything downstream of the model with pure numbers. This tier
covers the one stage that genuinely needs images: MediaPipe landmarks, the
smoothed square face crop, and L2CS ONNX inference.

PRIVACY: the frames contain a real face. Everything this writes goes to
`tests/fixtures/tier2/`, which is gitignored. It is never committed and never
enters a wheel. The manifest records SHA-256 digests so a fixture can be checked
for corruption without the frames being in version control.

REGENERATING ON ANOTHER MACHINE
    Anyone can produce their own Tier 2 fixture. You do not need mine:

        python tests/golden/record_tier2.py --frames 60

    Sit in the normal position, look around the screen while it records, and it
    will write frames.npz plus manifest.json. The hardware test then replays
    YOUR recording against whichever implementation is selected. Because the
    expected values are recorded from the pipeline at record time, the fixture
    is self-consistent regardless of whose face is in it.

Locating the legacy pipeline (rule 4, no hard-coded paths):
    FOCUSEDGAZE_LEGACY_DIR   points at the original gaze-detection/ folder.
    If unset, falls back to ../gaze-detection relative to this repository.

Usage:
    python tests/golden/record_tier2.py [--frames N] [--camera INDEX]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Same rule as record_tier1.py: environment variable first, relative fallback
# second, absolute drive path never.
LEGACY = pathlib.Path(
    os.environ.get("FOCUSEDGAZE_LEGACY_DIR") or (REPO_ROOT.parent / "gaze-detection")
)
OUT = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "tier2"


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=60, help="frames to keep (default 60)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--settle-timeout", type=float, default=15.0,
                    help="seconds to wait for auto-exposure (default 15)")
    ap.add_argument("--min-brightness", type=float, default=40.0,
                    help="refuse to record below this mean brightness (default 40/255)")
    ap.add_argument("--min-face-rate", type=float, default=0.5,
                    help="refuse to write below this fraction of frames with a face "
                         "(default 0.5)")
    args = ap.parse_args(argv)

    if not LEGACY.is_dir():
        print(f"Legacy pipeline not found at: {LEGACY}\n\n"
              "Set FOCUSEDGAZE_LEGACY_DIR to the original gaze-detection/ folder,\n"
              "or check the two projects out side by side:\n"
              "    <parent>/gaze-detection/\n"
              "    <parent>/focusedgaze-sdk/     <- this repository")
        return 1

    prev = pathlib.Path.cwd()
    os.chdir(LEGACY)
    sys.path.insert(0, str(LEGACY))
    try:
        import cv2
        import gaze_pipeline
        import numpy as np
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_facial_transformation_matrixes=True,
            )
        )

        cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print(f"ERROR: could not open camera {args.camera}. Is it muted or in use?")
            return 1

        # Wait for auto-exposure, do not assume it. This used to be sleep(1.0),
        # which is roughly five times too short on the reference webcam: measured
        # here, mean brightness sat at ~52/255 for the first 3.5 s and only
        # climbed to ~100/255 by 6.5 s. A fixed short sleep therefore recorded
        # the first frames at about half their settled brightness, and a Tier 2
        # attempt that failed with a near-black image was read at the time as
        # "the room was dark" when the camera had simply not opened up yet.
        #
        # Poll until brightness stops climbing rather than sleeping a magic
        # number, so a faster camera costs less and a slower one still settles.
        print("Waiting for auto-exposure to settle...")
        # Compare against a reading from a WINDOW ago, not the previous frame.
        # Comparing consecutive frames is what the first version of this did and
        # it converged after 1.4 s at 50.9/255, while the camera was still on its
        # way to ~100/255: at 33 ms apart a slow ramp looks flat, so "10 steady
        # frames" is satisfied part-way up the curve. Brightness has to be still
        # over a span comparable to the ramp to mean anything.
        window_s = 1.5
        history: list[tuple[float, float]] = []
        prev_b = -1.0
        settle_start = time.time()
        while time.time() - settle_start < args.settle_timeout:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.time()
            prev_b = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            history.append((now, prev_b))
            old = [b for t, b in history if now - t >= window_s]
            if old and now - settle_start >= 2.0:
                then = old[-1]
                if abs(prev_b - then) <= max(1.0, then * 0.03):
                    break
        settled_for = time.time() - settle_start
        print(f"  settled at {prev_b:.1f}/255 after {settled_for:.1f}s")
        if prev_b < args.min_brightness:
            print(f"\nREFUSING TO RECORD: brightness {prev_b:.1f}/255 is below "
                  f"{args.min_brightness}/255.")
            print("The image is too dark for reliable landmark detection. Turn on a")
            print("light and run again. Recording anyway would produce a fixture that")
            print("passes while pinning degraded behaviour, which is worse than none.")
            cap.release()
            return 3

        # The bbox smoothing is stateful across frames, so the fixture must
        # replay from a known-clean start.
        gaze_pipeline.reset_bbox_smoothing()

        frames, expected = [], []
        print(f"Recording {args.frames} frames. Look around the screen...")
        idx = 0
        attempts = 0
        while len(frames) < args.frames and attempts < args.frames * 6:
            attempts += 1
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)          # as gaze_server does
            pitch, yaw, bbox = gaze_pipeline.get_gaze_reading(frame, landmarker, idx)
            idx += 1
            frames.append(frame)
            expected.append({
                "frame_index": len(frames) - 1,
                "pitch": None if pitch is None else float(pitch),
                "yaw": None if yaw is None else float(yaw),
                "bbox": None if bbox is None else [int(v) for v in bbox],
            })
        cap.release()

        n_face = sum(1 for e in expected if e["pitch"] is not None)
        if not frames:
            print("ERROR: captured no frames.")
            return 1
        face_rate = n_face / len(frames)
        if n_face == 0:
            print(f"REFUSING TO WRITE: {len(frames)} frames captured but NO FACE was")
            print("detected in any of them. A fixture with no face exercises only the")
            print("null path and would give false confidence. Check the camera is not")
            print("muted, sit in frame, and run again.")
            return 2
        if face_rate < args.min_face_rate:
            # Zero faces was already refused, but a handful is barely better: the
            # fixture would pin mostly the null path while looking like coverage,
            # and the number that matters is the one nobody reads twice.
            print(f"REFUSING TO WRITE: only {n_face}/{len(frames)} frames "
                  f"({face_rate:.0%}) contain a face, below the required "
                  f"{args.min_face_rate:.0%}.")
            print("Sit square to the camera at a normal working distance, keep your")
            print("whole face in frame, and run again. Lower the bar with")
            print("--min-face-rate only if you know why you are doing it.")
            return 4

        OUT.mkdir(parents=True, exist_ok=True)
        frames_path = OUT / "frames.npz"
        np.savez_compressed(frames_path, frames=np.asarray(frames, dtype=np.uint8))

        manifest = {
            "schema_version": 1,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "legacy gaze-detection pipeline (unmodified)",
            "frame_count": len(frames),
            "frames_with_face": n_face,
            # Recording conditions, kept as data. A fixture that looks wrong later
            # is much easier to explain when the brightness it was captured at is
            # written down rather than reconstructed from memory.
            "settled_brightness": round(prev_b, 1),
            "settle_seconds": round(settled_for, 1),
            "face_rate": round(face_rate, 3),
            "frame_shape": list(frames[0].shape),
            "mirrored": True,
            "frames_file": frames_path.name,
            "frames_sha256": sha256_of(frames_path),
            "expected": expected,
            "note": ("Frames contain a real face and are gitignored. Regenerate your "
                     "own with record_tier2.py --frames N."),
        }
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

        print(f"\nTier 2 fixture written to {OUT}")
        print(f"  frames        : {len(frames)}")
        print(f"  with a face   : {n_face}/{len(frames)}")
        print(f"  frames.npz    : {frames_path.stat().st_size / 1e6:.1f} MB")
        print(f"  sha256        : {manifest['frames_sha256'][:16]}…")
        return 0
    finally:
        os.chdir(prev)


if __name__ == "__main__":
    raise SystemExit(main())
