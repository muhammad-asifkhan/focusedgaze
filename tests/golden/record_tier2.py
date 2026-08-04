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
        time.sleep(1.0)

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
        if n_face == 0:
            print(f"REFUSING TO WRITE: {len(frames)} frames captured but NO FACE was")
            print("detected in any of them. A fixture with no face exercises only the")
            print("null path and would give false confidence. Check the camera is not")
            print("muted, sit in frame, and run again.")
            return 2

        OUT.mkdir(parents=True, exist_ok=True)
        frames_path = OUT / "frames.npz"
        np.savez_compressed(frames_path, frames=np.asarray(frames, dtype=np.uint8))

        manifest = {
            "schema_version": 1,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "legacy gaze-detection pipeline (unmodified)",
            "frame_count": len(frames),
            "frames_with_face": n_face,
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
