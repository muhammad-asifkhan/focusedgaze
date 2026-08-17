# Running focusedgaze on NixOS.
#
# THE PROBLEM THIS SOLVES
# -----------------------
# Every heavy dependency here ships as a manylinux wheel with compiled
# extensions: numpy, opencv-python, mediapipe, openvino, scikit-learn. Those
# wheels are linked against libraries at FHS paths -- /usr/lib/libstdc++.so.6
# and friends -- which NixOS does not have. The failure is a bare
#
#     ImportError: libstdc++.so.6: cannot open shared object file
#
# raised from inside numpy's C extension, which reads like a broken numpy
# install and is nothing of the kind. pip installed everything correctly; the
# dynamic loader simply cannot find a C++ runtime.
#
# `buildFHSEnv` answers it by giving the shell a real /usr/lib assembled from
# the packages below. Inside it, ordinary `pip install` works the way every
# other distribution's does, which is what the project's own instructions
# assume.
#
# WHY NOT THE ALTERNATIVES
# ------------------------
# * Packaging this with `python3.withPackages` would be cleaner in principle
#   and does not work in practice: mediapipe and openvino are the two hardest
#   things in this dependency set to get from nixpkgs, and they are exactly the
#   two that cannot be dropped.
# * `programs.nix-ld.enable` is the other correct answer and is a *system*
#   change requiring a rebuild and root. This file needs neither, and it lives
#   in the repository, so it travels with the project.
#
# USAGE
#     nix-shell                       # you are now in an FHS shell
#     python3.12 -m venv .venv
#     source .venv/bin/activate
#     pip install -e ".[intel,calibration,server]"
#     focusedgaze setup && focusedgaze check
#
# Re-enter with `nix-shell` in future sessions; the venv persists.
#
# NOTE ON THE PYTHON VERSION
# --------------------------
# 3.12, not 3.13. `requires-python` allows 3.13, but 3.12 is the version this
# project's wheel coverage has actually been exercised against -- mediapipe in
# particular has historically lagged new Python releases, and a missing wheel
# there means building from source, which is a much longer detour than this
# file is trying to save you from. Use 3.13 only if you want to test it.

{ pkgs ? import <nixpkgs> { } }:

(pkgs.buildFHSEnv {
  name = "focusedgaze";

  targetPkgs = pkgs: with pkgs; [
    # ---- Python itself -------------------------------------------------
    python312
    python312Packages.pip
    python312Packages.virtualenv

    # ---- The C++ runtime whose absence causes the reported error -------
    stdenv.cc.cc.lib
    zlib

    # ---- OpenCV's shared-library needs ---------------------------------
    # libGL and glib are the two that bite first: `import cv2` fails on
    # libGL.so.1 and then on libgthread-2.0.so.0, one after the other.
    libGL
    libGLU
    glib
    glibc

    # ---- X11, for the full-screen dot ----------------------------------
    # `focusedgaze calibrate` and `accuracy` open a real OpenCV window. Without
    # these the import succeeds and the window creation fails later, which is a
    # worse place to discover it.
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libXi
    xorg.libXrandr
    xorg.libXfixes
    xorg.libXcursor
    xorg.libSM
    xorg.libICE
    xorg.libxcb
    libxkbcommon
    fontconfig
    freetype

    # ---- Wayland sessions ----------------------------------------------
    # Harmless on X11. Present because a NixOS desktop is as likely to be
    # Wayland, and OpenCV's Qt backend looks for these.
    wayland
    wayland-protocols

    # ---- Camera ---------------------------------------------------------
    # v4l-utils is not linked against; it is here so `v4l2-ctl --list-devices`
    # is available when the camera check fails and you need to know why.
    v4l-utils
    libv4l

    # ---- Convenience -----------------------------------------------------
    git
    curl                                # the model downloads use urllib, but
                                        # curl is what you reach for to test a
                                        # proxy or a blocked host
  ];

  # Keeps the venv's own bin ahead of the FHS one once activated.
  profile = ''
    export FOCUSEDGAZE_NIX_SHELL=1
    echo "focusedgaze FHS shell. Python: $(python3.12 --version)"
    echo "If .venv exists:  source .venv/bin/activate"
  '';

  runScript = "bash";
}).env
