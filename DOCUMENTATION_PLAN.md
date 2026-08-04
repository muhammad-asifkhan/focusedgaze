# focusedgaze: user documentation plan and writing brief

What each document is for, what goes in it, and the voice rules that apply to all of them.

This file is itself written under the no-em-dash rule below, so it works as a small example
of the rule as well as a statement of it.

---

## Principle

Write each document when the code it describes lands, not at Phase 8. A page written three
months after the API is a reconstruction. A page written the same week is a record.

## The documents

| File | Purpose | Write when |
|---|---|---|
| `README.md` | The front door. What it does, whether it fits, how to start. | Written. Refine per phase |
| `docs/install.md` | Providers, extras, model weights, the licence constraint | Phase 6 |
| `docs/calibration.md` | Why it is per person, how to run it, when to redo | Phase 5 |
| `docs/usage.md` | The two API layers, with real working examples | Phase 4 |
| `docs/api.md` | Reference: every public class, method, argument, exception | Per phase, as each lands |
| `docs/accuracy.md` | What the numbers actually are and where it is worse | Phase 8, after the baseline |
| `docs/troubleshooting.md` | Symptom, cause, fix | Phase 6 |
| `examples/` | Four runnable scripts | Phase 4 and 6 |

### README.md

Order matters. Someone deciding whether to use this needs the disqualifying facts early,
not buried under a feature list.

1. One sentence: webcam eye-gaze tracking as a Python library.
2. The pipeline, as the six-line ASCII flow. It tells a technical reader more in six lines
   than three paragraphs would.
3. **What it needs before it works.** A webcam, a per-person calibration, and model weights
   the user fetches themselves. Say this before the quickstart, not after.
4. Quickstart: install, download models, calibrate, ten lines of code.
5. The two layers, with the honest reason the pure one exists: video files, tests, a shared
   camera.
6. Accuracy, with the real figures, including where it is worst.
7. Platform support table. Windows tested, Linux and macOS structurally supported, untested.
8. Licence: MIT code, restricted weights, link to NOTICE.

### docs/install.md

The provider choice is the thing people will get wrong, so lead with it. The base install is
provider-agnostic on purpose, and the user picks `[directml]`, `[cuda]` or `[cpu]`. Give the
error they will see if they pick none. Then extras, then model weights, including why
`download-models` fetches one asset automatically and deliberately refuses the other.

### docs/calibration.md

The concept people miss is that calibration is per person, per machine, per seating
position. Explain the smooth-pursuit routine in the terms the existing cheat sheet uses
("follow the dot with your eyes"), what a good validation error looks like, and when a redo
is actually warranted rather than a recenter.

### docs/usage.md

Real code that runs, not fragments. Four progressions: the simplest webcam loop, supplying
your own frames, the callback API, and headless calibration. Each one complete enough to
paste and execute.

### docs/api.md

Reference, not narrative. Every public name, its arguments with types and defaults, what it
returns, what it raises, and thread-safety where it matters. Generate it from docstrings if
that is cleaner, but the docstrings must be good enough to generate from.

### docs/accuracy.md

This is the page that builds trust, because it is the one that admits things. Same-session
and held-out figures, the degradation at the bottom edge, what changes it (lighting,
seating, glasses, distance), and how someone measures it for themselves.

### docs/troubleshooting.md

Symptom first, in a table. The existing cheat sheet's table is already the right shape, with
entries like "cursor is in the wrong place" and "it clicked the wrong button". Adapt it
rather than replacing it with a list of exception names.

### examples/

`minimal.py`, `video_file.py`, `callback.py`, `headless_calibration.py`. Every one must run
as-is. An example that does not run is worse than no example.

---

## Voice

**There is already a model for the voice in this project.** Part 2 of `GAZE_SYSTEM_DOCS.md`
in the originating game repository, the plain-English cheat sheet. It says things like "this
is the #1 thing people forget", and it organises troubleshooting by what the user feels
rather than by what the code does. That is the register.

The failure mode to avoid is documentation that reads as though nobody had used the thing.

**Never write these.** Marketing adjectives: powerful, seamless, robust, elegant,
lightweight, blazing, effortless, cutting-edge. Openers like "In today's world" or "Whether
you're a researcher or a hobbyist". The word "simply" or "just" in front of an instruction,
because if it were simple the instruction would not be needed. "It's worth noting that".
Tricolons where three items are listed for rhythm rather than because there are three. Emoji
in headings.

**Specifics beat adjectives, always.** Not "fast inference" but "about 15 ms on an RTX 4060
via DirectML, about 104 ms on CPU". Not "highly accurate" but "held-out error around 8.9% of
screen size, 3 to 8% across the top and centre, 13 to 14% along the bottom edge". Where a
number is not measured, say so rather than reaching for an adjective.

> **The example above was corrected, and the correction is the point.** An earlier version of
> this brief used "2.0 to 2.4 cm within a session, around 3.0 cm on a held-out session" as its
> illustration of specificity. Neither figure has a source in this repository. The first
> traces to an accuracy script in the originating project whose output has never been
> captured here, and the second appears nowhere at all. The percentage figures that replace
> them come from that project's own documentation and are real.
>
> A brief that demands sourced numbers cannot itself contain an unsourced one, because the
> next writer will copy it straight into a public page. Standing rule 8 applies to
> instructions exactly as it applies to code: if it is not in the repo, ask, do not assert.
> Re-measuring accuracy in centimetres is scheduled before the milestone scripts are deleted.

**Say what does not work, early and plainly.** It needs calibrating per person. It is worse
at the bottom of the screen. It is only tested on Windows. It will not fetch the model
weights for you. A reader who finds a limitation in paragraph two trusts the rest of the
page. One who finds it in the FAQ stops trusting all of it.

**Let sections be different lengths.** Uniform section length is one of the strongest tells
of generated text. The install page needs more room than the licence note. Some sections are
two sentences, and that is fine.

**Address the reader as "you". Refer to the library by name**, not as "we" or "the package".
Contractions are fine. A dry aside is fine where it is earned.

**No filler transitions.** Cut "Additionally", "Furthermore", "That said", "It's important to
note". Start the sentence.

## No em-dashes

This applies to every document, every commit message, every docstring and every code comment.

**Do not find-and-replace.** Mechanically swapping every em-dash for a hyphen or a comma
leaves sentences limping where the dash was carrying structure. Rewrite each sentence
according to what the dash was doing:

| The dash was doing this | Rewrite as |
|---|---|
| A parenthetical aside | Commas, or parentheses |
| Introducing an explanation | A colon |
| A hard break or reversal | Two sentences |
| Summarising a list | A colon |
| A trailing afterthought | Usually cut it. It was rarely needed |

**Two exceptions, both important.**

1. Leave en-dashes in numeric ranges alone. "2.0–2.4 cm" and "3.12–3.14" are correct
   typography, not a stylistic tic.
2. Leave every dash inside a code block, command, file path or filter-repo rule exactly as
   it is. This project has already been broken twice by dash characters in shell commands.
   Do not touch anything executable.

**Any sweep asserts its rules fire before it writes.** A replacement that silently matches
nothing is the most repeated failure in this project. The sweep that produced the current
state caught an inert rule on its first run.

**Current state.** `README`, `NOTICE`, `CONTEXT_HANDOFF`, `CHANGELOG`, `requirements-dev.txt`
and everything under `src/` and `tests/` are clean, except for six occurrences inside code
fences in `CONTEXT_HANDOFF` that the rule deliberately leaves. `MIGRATION_AUDIT.md` and
`STANDING_BRIEF.md` are a deliberate backlog: see audit section 36.

## Content rules

- Every code example must actually run. Execute each one before including it.
- Every command must be copy-pasteable, with the correct paths for this project. The legacy
  pipeline is on the `D:` drive, not where older documents say it is.
- Do not document APIs that do not exist yet. `README.md` may show the intended shape, but
  it must be marked clearly as in development.
- Do not claim the `pip install focusedgaze[...]` commands have been verified. The package
  publishes 0.0.0 only, which is a placeholder with no working code in it.
- Anywhere behaviour is surprising, say why. The fullscreen requirement, the per-person
  calibration, the refusal to download the gaze weights: each has a real reason, and the
  reason is what stops people fighting it.
- The Gaze360 restriction goes in the README, not only in `NOTICE`. Someone evaluating the
  library commercially needs it before they invest time.

## What to flag rather than invent

If a number, behaviour or limitation is not recorded anywhere in the repo, say so and leave a
marked gap. Do not estimate a benchmark, invent a supported platform, or soften a limitation
to make a sentence read better. Standing rule 8 applies to prose exactly as it applies to
code, and as the corrected example above shows, it applies to this brief too.
