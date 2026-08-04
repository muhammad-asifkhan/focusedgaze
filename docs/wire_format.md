# The WebSocket wire format

Phase 7 reconnaissance. This document establishes ground truth for the contract
that `focusedgaze serve` must honour, read from the source rather than from any
summary of it.

**Authority:** `<LEGACY>\gaze_server.py` and `<GAME>\input-manager.js`. Where a
statement is inferred rather than read, it says so in bold.

**Status:** analysis only. No server code has been written. Nothing in the legacy
pipeline or the game was modified to produce this.

`<LEGACY>` and `<GAME>` are the paths recorded in `CONTEXT_HANDOFF.md` section 2.

---

## 1. Transport

| Property | Value | Source |
|---|---|---|
| Scheme | `ws://` (no TLS) | `input-manager.js:41`, `gaze_test.html:70` |
| Host default | `"localhost"` | `gaze_server.py:69` |
| Port default | `8765` | `gaze_server.py:70` |
| Path | `/` (no path is used or checked) | client URL is bare host:port |
| Subprotocol | none requested, none negotiated | `serve(handler, HOST, PORT)`, `gaze_server.py:496` |
| Origin check | none | `serve()` called without `origins`; the library performs no Origin validation when it is unset |
| Auth | none | - |
| Frame type | Text | `broadcast()` sends `str` via `send_text` |
| Encoding | UTF-8, `ensure_ascii=True` | `json.dumps` defaults |
| Keepalive | `ping_interval=20`, `ping_timeout=20` seconds | library defaults, not overridden; `websockets/asyncio/server.py:72-73` |
| Backpressure | **none** | `broadcast()` "pushes the message synchronously to all connections even if their write buffers are overflowing" |

`HOST = "localhost"` binds only what `localhost` resolves to. The stream is
webcam-derived and unauthenticated; the default must not become `"0.0.0.0"`.

### Byte-level note

The brief renders the format as `{"type":"gaze","ok":true,...}`. The actual
bytes carry `json.dumps` default separators, so there is a space after every
colon and comma:

```
{"type": "gaze", "ok": true, "x": 0.4242, "y": 0.7071, "t": 1690000000.123}
```

No JSON parser cares. It matters only if anyone tries to assert byte-identity
against the brief's rendering rather than against the server's output.

---

## 2. Messages the server emits

Three types. Python dicts preserve insertion order and `json.dumps` writes in
that order, so the field order below is the wire order.

### 2.1 `type: "gaze"` - the raw gaze feed

Constructed in four places, all with identical key order:
`gaze_server.py:108` (module initial value), `:345` (camera handover reset),
`:369` (no face), `:377` (good reading).

| Field | Type | Notes |
|---|---|---|
| `type` | string | always the literal `"gaze"` |
| `ok` | boolean | `true` only on the good-reading path |
| `x` | number or `null` | `round(sx, 4)`; `null` when `ok` is `false` |
| `y` | number or `null` | `round(sy, 4)`; `null` when `ok` is `false` |
| `t` | number | `time.time()`, Unix epoch seconds, **not rounded** |

Good reading:

```
{"type": "gaze", "ok": true, "x": 0.4242, "y": 0.7071, "t": 1690000000.1234567}
```

No face:

```
{"type": "gaze", "ok": false, "x": null, "y": null, "t": 1690000000.1234567}
```

The very first broadcast after startup, before the gaze loop has produced
anything, carries the module initial value, whose `t` is the literal `0.0`:

```
{"type": "gaze", "ok": false, "x": null, "y": null, "t": 0.0}
```

**When `ok` is `false`.** Exactly three causes, and no others:

1. `get_gaze_reading` returned `pitch is None` (`gaze_server.py:368`), which
   happens when MediaPipe found no face or the crop came back empty
   (`gaze_pipeline.py:146-153`).
2. The gaze pipeline does not own the camera, i.e. gesture mode has it
   (`gaze_server.py:343-346`). This fires once on the transition, not per tick.
3. The module initial value, before the first reading.

There is **no positioning gate in the server.** `gaze_server.py` never imports
or calls `PositioningGate`. The docstring at `:24` and the HUD text at
`gaze_test.html:107` both say "no face / out of zone"; the "out of zone" half is
not implemented. Wiring the positioning gate into the `ok` decision would look
like a natural improvement and would be a behaviour change: the cursor would
vanish whenever the player leans outside 45-65 cm.

**Coordinates.** `x` and `y` are normalised to `[0, 1]`, origin top-left, **y
increases downward**. The clamp is applied by `apply_calibration`
(`calibration_utils.py:62-63`) before smoothing. The 1 euro filter output is a
convex combination of clamped inputs, so it cannot leave `[0, 1]`
(**inferred from the filter algebra at `gaze_server.py:291`, not measured**).

The frame is mirrored with `cv2.flip(frame, 1)` at `gaze_server.py:362` before
inference, because calibration was recorded through a flipping reader. This is
not cosmetic: dropping it does not mirror the output, it makes the calibration
polynomial wrong.

**Float formatting.** `round(v, 4)` returns a Python float, and `json.dumps`
serialises floats with `repr`, i.e. the shortest string that round-trips. So
`0.5` is emitted as `0.5`, never `0.5000`, and a value that rounds to a whole
number is emitted as `1.0`, never `1`. Trailing zeros are never padded.

`t` is deliberately not rounded, and it is load-bearing: see section 4.

### 2.2 `type: "mode"` - input-mode status

`gaze_server.py:226-236`, `mode_status()`.

| Field | Type | Notes |
|---|---|---|
| `type` | string | literal `"mode"` |
| `mode` | string | `"gaze"` or `"gesture"` |
| `gesture_available` | boolean | whether `gesture_bridge` imported at startup |
| `error` | string | `""` on success; on failure either the caller's message or `GESTURE_IMPORT_ERROR`, which is `f"{type(exc).__name__}: {exc}"` |
| `t` | number | `time.time()` |

Emitted on two occasions:

- **Per client, on connect**, via `websocket.send` (`:440`). This is the first
  application message any client receives.
- **To all clients**, via `broadcast`, after a mode switch completes or fails
  (`:452`).

### 2.3 `type: "input"` - the resolved pointer

`gaze_server.py:388-427`, `build_input_message()`. **This is the only message
the game reads.**

| Field | Type | Source |
|---|---|---|
| `type` | string | literal `"input"` |
| `mode` | string | `_mode` |
| `source` | string | `"gesture"` when `mode == "gesture"`, else `"gaze"` |
| `ok` | boolean | `bool(ok)` from the active pipeline |
| `x` | number or `null` | from the active pipeline, already rounded by its producer |
| `y` | number or `null` | as above |
| `hand_ok` | boolean | gesture snapshot, `false` when no gesture engine |
| `gesture` | string | gesture label, `""` when none |
| `click_seq` | integer | monotonic counter, `0` when no gesture engine |
| `clear_seq` | integer | monotonic counter, `0` when no gesture engine |
| `swipe_seq` | integer | monotonic counter, `0` when no gesture engine |
| `swipe` | string | last swipe direction, `""` when none |
| `t` | number | `time.time()`, taken at message build time |

The resolver branch is `:403`:

```python
    if mode == MODE_GESTURE and gesture is not None:
        ok, x, y = gesture["ok"], gesture["x"], gesture["y"]
    else:
        ok, x, y = gaze["ok"], gaze["x"], gaze["y"]
```

`hand_ok`, `gesture`, `click_seq`, `clear_seq`, `swipe_seq` and `swipe` are
populated in **every** mode, so the HUD can show hand state while the eyes
steer. When no gesture engine exists they take the constants above rather than
being omitted.

The three `*_seq` fields are counters, not booleans, so a dropped or coalesced
frame can neither swallow an event nor replay one. The gesture side rounds its
own `x`/`y` to 4 places (`gesture_bridge.py:232-233`), which is why the units
match across sources.

### 2.4 Version and mode dependence

There is **no protocol version field** anywhere in the wire format. Nothing
negotiates. A client cannot tell which server it is talking to.

Mode dependence:

- The `gaze` message shape never varies. Its **emission** does: in gesture mode
  `LATEST["t"]` stops changing after the one-shot reset, so the gaze feed goes
  silent entirely (section 4).
- The `input` message shape never varies. Its `mode`, `source`, `ok`, `x` and
  `y` values do.
- The `mode` message's `gesture_available` and `error` are fixed at import time
  by whether `gesture_bridge` loaded.

---

## 3. Messages the server accepts

Exactly one command. `gaze_server.py:441-452`.

```json
{"cmd": "mode", "mode": "gaze"}
```

| Field | Type | Handling |
|---|---|---|
| `cmd` | string | only `"mode"` is recognised; anything else is silently ignored |
| `mode` | string | coerced with `str(msg.get("mode", ""))`; validated against `("gaze", "gesture")` |

Sent by the client at `input-manager.js:78`. Handling detail:

- Non-JSON payloads are caught by `except (ValueError, TypeError)` and ignored.
- The switch runs on `asyncio.to_thread`, because it hands the camera between
  pipelines and builds a MediaPipe landmarker, which takes seconds. Running it
  inline would freeze the broadcaster and every other client.
- The resulting `mode` status is broadcast to **all** clients, not just the
  requester.

**Edge case, see risk R-11:** valid JSON that is not an object (`"hi"`, `3`,
`null`) makes `msg.get` raise `AttributeError`, which the inner handler does not
catch. The outer handler prints a traceback and the connection closes.

---

## 4. Pacing, connection lifecycle, multiple clients

`broadcaster()`, `gaze_server.py:464-484`.

```python
    period = 1.0 / SEND_HZ
```

with `SEND_HZ = 60` (`:93`), so the tick is nominally 60 Hz. `MIGRATION_AUDIT.md`
section 7 records a **measured** 35-44 Hz, attributed to Windows timer
granularity. That measurement was not repeated for this document.

**Two different rates on one tick.** This is the part a rewrite is most likely
to get wrong:

| Message | Cadence | Rule |
|---|---|---|
| `gaze` | only when `LATEST["t"] != last_gaze_t`, roughly 15-19 Hz per the docstring at `:470` | one send per new gaze reading |
| `input` | every tick | gesture state can change between gaze readings |

Nothing is sent at all when `CLIENTS` is empty; the loop still ticks.

**Ordering within a tick:** `gaze` is broadcast before `input`. Across the
connection lifetime: the per-client `mode` message is sent before the client
enters the broadcast set's traffic.

**Multiple clients** share one `CLIENTS` set and receive byte-identical frames.
`broadcast()` skips connections whose protocol state is not OPEN and ignores
per-connection write failures, continuing to the rest.

**A late joiner does not get a gaze replay.** `last_gaze_t` is broadcaster-global,
not per-client, so a client that connects mid-stream waits for the next new
reading. At 15-19 Hz that is tens of milliseconds. In gesture mode it is
forever, because `LATEST["t"]` has stopped changing.

**Disconnect:** the `async for` loop ends, `finally` discards the socket from
`CLIENTS`. No cleanup beyond that; no state is per-client.

**Reconnect:** `input-manager.js` reconnects every `RECONNECT_MS = 1200`
indefinitely, and on close resets `lastClick = lastSwipe = lastClear = null` so
the next first message re-syncs the counters without replaying events.
`gaze_test.html` reconnects at 1000 ms. The server must therefore tolerate rapid
repeated connects and must never replay a backlog on connect.

---

## 5. What the client actually consumes

This is the section that decides what is safe to change.

There is exactly **one** WebSocket client in the browser game:
`<GAME>\input-manager.js`, line 117. Every other game file reads
`window.GameInput` instead. Searched the whole game tree for `WebSocket`,
`ws://`, `8765`, `m.type`, `click_seq`, `hand_ok` and `gesture_available` to
confirm.

The only other client anywhere is `<LEGACY>\gaze_test.html:86`, which is a
diagnostic page, not part of the game.

### 5.1 The finding that matters most

`input-manager.js:133-135`

```js
      // "gaze" messages are the raw device feed, kept for gaze_test.html. The
      // game consumes the resolved "input" message instead.
      if (m.type !== "input") return;
```

**The game discards every `type: "gaze"` message.**

A server that emits only the `gaze` message, however byte-identical, leaves the
game's cursor at its initial `(0.5, 0.5)` with `ok` false, forever, with no
console error. The Phase 7 exit criterion as written cannot be met that way.
See section 8.

### 5.2 Field-by-field

Read = the client branches on it or stores it. Ignored = it arrives and is
dropped.

**`type: "gaze"`**

| Field | Game (`input-manager.js`) | `gaze_test.html` |
|---|---|---|
| `type` | read, then the message is dropped | read (`:94`) |
| `ok` | ignored | **read** (`:96`) |
| `x` | ignored | **read** (`:98`, `:102`) |
| `y` | ignored | **read** (`:99`, `:103`) |
| `t` | ignored | ignored |

**`type: "mode"`**

| Field | Game | Where |
|---|---|---|
| `type` | **read** | `:124` |
| `mode` | **read**, assigned to `api.mode` | `:125` |
| `gesture_available` | **read**, coerced with `!!` | `:126` |
| `error` | **read**, logged as a console warning if truthy | `:128` |
| `t` | ignored | - |

**`type: "input"`**

| Field | Game | Where |
|---|---|---|
| `type` | **read** | `:135` |
| `ok` | **read**, coerced with `!!`; gates the x/y assignment | `:137-138` |
| `x` | **read** only when `ok` | `:138` |
| `y` | **read** only when `ok` | `:138` |
| `hand_ok` | **read**, coerced with `!!` | `:139` |
| `gesture` | **read**, compared against the literal `"pinch"` | `:140`, `:145` |
| `mode` | **read**, only when truthy and different | `:150` |
| `source` | **read**, falls back to `api.mode` when falsy | `:151` |
| `click_seq` | **read**, compared with `>` against the last value | `:157`, `:160` |
| `clear_seq` | **read**, same | `:157`, `:161` |
| `swipe_seq` | **read**, same | `:157`, `:162` |
| `swipe` | **read**, `\|\| ""` | `:164` |
| `t` | ignored | - |

### 5.3 Consequences

**`t` is read by nobody.** Not by the game, not by `gaze_test.html`, on any
message type. Its precision and its presence are free to change **on the wire**.

But `t` is load-bearing **inside** the server: `broadcaster()` dedups the gaze
feed on `LATEST["t"] != last_gaze_t`. Rounding it to milliseconds would collapse
two readings that land in the same millisecond into one broadcast. Keep the
full-precision float internally whatever is done on the wire.

**Constraints that cannot be relaxed**, because the client branches on them:

- the literal string values `"gaze"`, `"gesture"`, `"input"`, `"mode"`, `"pinch"`
- `x` and `y` normalised `[0, 1]`, y down (`marshes.js:285` converts to NDC as
  `x*2-1`, `-(y*2-1)`, so a y-axis flip breaks 3D picking silently)
- `ok` gating x/y, so that a false reading never moves the cursor
- the three `*_seq` fields being monotonically non-decreasing integers compared
  with `>`
- `x`/`y` being `null` rather than a stale point when `ok` is false, which is
  what lets the client keep its last good position while hiding the cursor

**Constraints that can be relaxed:**

- `t` on the wire, entirely
- exact float formatting of `x` and `y` (both consumers do arithmetic on them)
- the whole `type: "gaze"` message, as far as the *game* is concerned; it is
  required only by `gaze_test.html`
- key order and whitespace

### 5.4 The downstream API surface

`window.GameInput` is what the rest of the game reads. Recorded so a wrapper
author can see the full blast radius:

`ok`, `x`, `y`, `mode`, `source`, `handOk`, `gesture`, `connected`,
`gestureAvailable`, `pinching`, `switching`, `modeLabel()`, `setMode()`,
`cycleMode()`, `onClick()`, `onSwipe()`, `onClear()`, `onStatus()`,
`onPinchStart()`, `onPinchEnd()`.

Consumers: `gaze-client.js` (2D quest), `forest.js` (3D), `marshes.js`
(stage 4). None of them touches the socket.

---

## 6. The gaze / gesture seam

### 6.1 Classification of every piece of module state

| Name | Line | Class |
|---|---|---|
| `GesturePointer`, `GESTURE_AVAILABLE`, `GESTURE_IMPORT_ERROR` | 59-66 | gesture |
| `HOST`, `PORT` | 69-70 | shared (transport) |
| `MODE_GAZE`, `MODE_GESTURE`, `VALID_MODES` | 81-82 | gesture |
| `GESTURE_DETECT_EVERY` | 89 | gesture, and dead (see 6.3) |
| `FACE_MODEL_PATH` | 90 | gaze |
| `CAM_INDEX`, `CAM_W`, `CAM_H` | 91-92 | shared (gaze opens with them, `set_mode` passes them to the gesture engine) |
| `SEND_HZ` | 93 | shared (paces both feeds) |
| `ONE_EURO_MIN_CUTOFF`, `_BETA`, `_DCUTOFF` | 100-102 | gaze |
| `LATEST` | 108 | gaze |
| `_running` | 109 | shared |
| `_frame_lock`, `_latest_frame`, `_frame_seq` | 114-116 | gaze (exposed to gesture only through dead code) |
| `_gaze_camera_on`, `_gaze_camera_free` | 122-123 | **shared: the camera lease** |
| `_mode`, `_gesture`, `_mode_lock` | 129-131 | gesture |
| `CLIENTS` | 385 | shared |

### 6.2 Classification of every function and branch

| Unit | Line | Class |
|---|---|---|
| `os.chdir(...)` | 40 | gaze prerequisite (see R-2) |
| `open_gaze_camera` | 134 | gaze; called by `set_mode` for rollback and hand-back |
| `release_gaze_camera` | 148 | lease; exists only because gesture exists |
| `read_shared_frame` | 163 | gesture by intent, dead in practice |
| `set_mode` | 177 | gesture |
| `mode_status` | 226 | gesture |
| `_capture_thread` | 239 | gaze, except its exit condition and `_gaze_camera_free.set()` |
| `OneEuro` | 259 | gaze (already extracted to `core/filters.py`) |
| `build_landmarker` | 295 | gaze |
| `gaze_loop` | 310 | gaze, except the lease branch at 343-350 |
| `build_input_message` | 388 | **shared: the entanglement point** |
| `handler` | 430 | shared, with a gesture-only first send (440) and a gesture-only command branch (446-452) |
| `broadcaster` | 464 | shared, with a gesture-only line (483) |
| `main` | 487 | shared, with gesture-only prints (498-502) |
| `__main__` teardown | 507-517 | shared, with a gesture-only stop (515-516) |
| branch `if not _gaze_camera_on.is_set()` | 343 | lease |
| branch `if LATEST["ok"]` (reset on handover) | 344 | gaze reset policy |
| branch `if pitch is None` | 368 | gaze, the `ok:false` path |
| branch `if mode == MODE_GESTURE and gesture is not None` | 403 | the entanglement |
| branch `if msg.get("cmd") == "mode"` | 446 | gesture |
| branch `if LATEST["t"] != last_gaze_t` | 480 | gaze pacing |
| branch `if CLIENTS` | 477 | shared |

### 6.3 One coupling is already gone

`read_shared_frame()` at `:163` is defined and **never called anywhere in
`gaze_server.py`**. `set_mode` builds `GesturePointer(own_camera=True, ...)`
(`:204-206`), and `gesture_bridge.py:183-184` then sets `self._tracker = None`
so the engine constructs its own `MediaPipeHandTracker`. `SharedFrameTracker` is
never instantiated in the shipped configuration.

So the frame-sharing data path between the two pipelines is already severed. The
only surviving coupling is the two-event camera lease. `read_shared_frame` and
`GESTURE_DETECT_EVERY` should not be carried into the SDK; they are game-repo
concerns and are currently unused even there.

### 6.4 Proposed cut

**`focusedgaze.server.websocket` ships:**

- the transport: host and port defaults, `serve()`, the client set, connect and
  disconnect bookkeeping
- the gaze producer and its `LATEST` equivalent, including the 1 euro reset
  policy
- the `type: "gaze"` message, shape preserved exactly
- the broadcaster and **both** of its pacing rules
- a camera lease: `pause(timeout=4.0)` and `resume()`, carrying the `4.0` wait
  and both `0.3` driver sleeps from `release_gaze_camera` and `set_mode:216`.
  These are Windows webcam driver facts, and they belong where the camera is
  opened.
- three extension points, which are the minimum for the game to keep working:

| Hook | Signature | SDK default | Wrapper uses it for |
|---|---|---|---|
| `on_connect` | `() -> dict \| None` | `None` | returning `mode_status()` |
| `on_command` | `(dict) -> dict \| None` | `None` | handling `cmd == "mode"`, returning the status to broadcast to all clients |
| `resolve_input` | `(gaze: dict) -> dict \| None` | the gaze-only `input` message | returning the full legacy `input` message |

`resolve_input` must be able to **replace** the SDK's `input` message, not merely
append to it. A hook that only appends would put two `input` messages on the wire
per tick with different `x`/`y`, and the client would take whichever arrived
last. That is why it is not a generic `extra_messages` list.

**The game repo's thin wrapper keeps:** the `gesture_bridge` import and
`GESTURE_AVAILABLE`, `_mode` / `_gesture` / `_mode_lock`, `set_mode`,
`mode_status`, the full twelve-field `input` message, the `cmd == "mode"`
command, and the calls to `pause()` / `resume()` around the hand-over.

No gesture vocabulary enters the SDK: no mode concept, no `hand_ok`, no
`click_seq`, no pinch.

### 6.5 What cannot be cleanly separated

1. **`build_input_message` is the resolver, and the game only reads its output.**
   This is the whole difficulty. Section 8 sets out the options.
2. **`SEND_HZ` paces both feeds.** The SDK owns the tick, so the wrapper's
   gesture state is sampled at the SDK's rate. That is what happens today, so it
   is not a change, but it means the wrapper cannot choose its own cadence
   without running a second loop.
3. **The camera lease is genuinely shared.** It is not a gaze concept and not a
   gesture concept; it is an operating-system fact about exclusive webcam
   ownership. It has to live on the SDK side because the SDK opens the camera,
   which means the SDK ships an API whose only current caller is a gesture
   feature it knows nothing about.
4. **`os.chdir()` cannot be delegated to the wrapper.** It has to be deleted and
   replaced by explicit paths. See R-2.

---

## 7. Risks

Ordered by how quietly they break the game.

**R-1. The game ignores `type: "gaze"`.** A gaze-only server emitting only the
gaze message produces a game that connects, reports "socket: connected", shows
face "-", and never moves the cursor. No console error. This is the single
largest Phase 7 trap and it is invisible to any test that only checks the gaze
message against a fixture. Source: `input-manager.js:135`.

**R-2. The `os.chdir()` at startup, and the four lookups riding on it.**
`gaze_server.py:33-40` chdirs to the script's own directory **before** importing
the gaze modules, because `gaze_pipeline` builds its ONNX session from
`pathlib.Path.cwd()` at import time. Four cwd-relative lookups depend on it:

| Path | Owner |
|---|---|
| the ONNX gaze model | `gaze_pipeline.py:15-33`, at import time |
| `face_landmarker.task` | `gaze_server.py:90` |
| `models/calibration_model.pkl` | `load_calibration()` |
| `models/camera_focal.json` | `positioning_gate.py:43` |

A library must not chdir; it is process-global and it would break any host
application. But removing it without replacing **all four** lookups changes
behaviour silently. `MIGRATION_AUDIT.md` section 1.4 finding F1 already measured
this: identical landmarks gave 117.406 cm versus 121.244 cm depending only on
the launch directory, a 3.8 cm difference larger than the system's entire claimed
accuracy.

**R-3. Coordinate convention.** `[0, 1]`, origin top-left, y down. The mirror at
`gaze_server.py:362` is required by the calibration, not by aesthetics.
`marshes.js:285` converts to NDC with `-(y*2-1)`, so a y-axis change breaks 3D
picking without breaking 2D picking, which would look like a bug in the forest
scene rather than in the server.

**R-4. Two different cadences on one tick.** `input` every tick, `gaze` only on a
new reading. Sending the gaze message every tick would roughly triple its volume
and change what `gaze_test.html`'s msgs/sec readout shows. It would not break
anything, which is exactly why it needs to be a stated decision rather than an
accident.

**R-5. `t` looks droppable and is not, internally.** No client reads it. The
broadcaster's dedup key is `LATEST["t"]`. Reduce its precision and simultaneous
readings collapse; remove it from the internal snapshot and the gaze feed either
stops or floods.

**R-6. The `ok: false` payload must keep `null` for `x` and `y`.** The client's
`if (m.ok)` guard is what preserves the last good position. Emitting `0` instead
of `null` is harmless against today's client and lethal against any future one
that drops the guard.

**R-7. The 1 euro reset policy is server-level, not filter-level.**
`fx.reset(); fy.reset()` runs on the transition to no-face (`:371`) and on the
camera handover (`:347`). `core/filters.py` already exists; the *policy* does
not live there. Losing it makes the cursor glide from a stale point when the
face returns.

**R-8. Reconnect storms.** Both clients reconnect forever, at 1200 ms and
1000 ms. Any per-connection setup cost is paid repeatedly while the server is
down or restarting. A connection limit, or a handshake that requires the client
to send something first, breaks both clients.

**R-9. No backpressure, by design.** `broadcast()` ignores full write buffers and
ignores per-connection write failures. A slow client accumulates until
`ping_timeout` reaps it. Setting `ping_interval=None`, a common reflex when
connections drop, removes the reaper and turns a slow client into a memory leak.

**R-10. Startup ordering versus the launcher.** `main()` starts the gaze thread,
sleeps 0.5 s, then checks `_running` so a missing calibration model or camera
fails before the port opens. `run-game.ps1:270` treats "port is listening" as
"ready" and waits up to 90 s for it. If the SDK opens the port before the tracker
is up, the launcher reports success for a server with no gaze source.

**R-11. `handler` closes the connection on non-object JSON.** `json.loads('"hi"')`
returns a string; `msg.get` raises `AttributeError`, which
`except (ValueError, TypeError)` at `:443` does not catch. The outer handler
prints a traceback and the socket closes; the browser reconnects 1.2 s later.
Fixing this is correct and is a behaviour change under standing rule 4.

**R-12. `cap.release()` in `gaze_loop`'s `finally` is a `NameError`.**
`gaze_server.py:380`. `cap` is never bound in `gaze_loop`; it is a local of
`open_gaze_camera` and a parameter of `_capture_thread`. Every exit from the main
loop raises. It has gone unnoticed because it only fires at shutdown, on a daemon
thread, as the process is dying. Do not port it. Phase 4's guaranteed camera
release is this bug's fix.

**R-13. `build_input_message` reads `_mode` and `_gesture` without the lock**
(`:400-401`) while `set_mode` mutates both under it. A tick landing mid-switch
can pair the new mode with a stale or absent gesture engine, producing one frame
of `ok: false`. Harmless today; it means the `mode`/`source`/`x`/`y` tuple is not
guaranteed self-consistent.

**R-14. `HOST` must stay `"localhost"`.** Changing the default to `"0.0.0.0"`
publishes an unauthenticated webcam-derived stream to the LAN.

**R-15. There is no version field.** Nothing on the wire identifies the protocol.
Adding one is the only forward-compatible move and both current clients would
ignore it (**inferred**: neither client enumerates fields, both read by name).

---

## 8. Open questions for a human

**Q7-1. The Phase 7 exit criterion is not satisfiable as written. Which
resolution?**

The criterion is "the existing browser game connects to `focusedgaze serve` and
plays end to end, unmodified". The game reads only `type: "input"`. Four options:

- **(a) Restate the criterion.** The game connects to the *game repo's wrapper*,
  which embeds the SDK server and adds the `input` and `mode` messages. Bare
  `focusedgaze serve` is verified against `gaze_test.html`, which does read the
  gaze message. Cleanest SDK, but the stated criterion changes.
- **(b) Hooks only** (section 6.4). Same as (a) in effect: bare
  `focusedgaze serve` still does not drive the game.
- **(c) Move the full `input` message into the SDK.** Bare `focusedgaze serve`
  drives the game. Puts six gesture-shaped fields (`hand_ok`, `gesture`,
  `click_seq`, `clear_seq`, `swipe_seq`, `swipe`) into the SDK's wire format,
  which is what Q6 exists to prevent.
- **(d) Recommended: a minimal gaze-only `input` message, plus the hooks.** The
  SDK emits `type: "input"` with `type`, `mode`, `source`, `ok`, `x`, `y`, `t`,
  where `mode` and `source` are the constant `"gaze"`. No gesture fields at all.
  The wrapper's `resolve_input` hook replaces it with the full legacy message.

  Reading `input-manager.js` line by line against a message with the gesture
  fields absent: `handOk` becomes `false`, `gesture` becomes `""`, `pinching`
  stays `false`, `gestureAvailable` stays `false` so `[m]` correctly does
  nothing, `api.mode` stays `"gaze"` so dwell-to-click stays enabled, and the
  counter sync at `:156-159` stores `undefined` and then compares
  `undefined > undefined`, which is `false`, so no click ever fires spuriously.
  Dwell handles clicking in gaze mode, so the game is fully playable.

  **This is inferred from reading the client and the JavaScript comparison rules.
  It has not been executed.** It should be confirmed by running the game against
  a stub server before it is relied on, and that needs a human.

**Q7-2. Should the SDK send explicit zeros for the sequence counters?**
Option (d) relies on `undefined > undefined` being false. Sending
`click_seq: 0, clear_seq: 0, swipe_seq: 0` is more defensive and more honest
about the contract, at the cost of putting three gesture-shaped field names into
a gaze-only SDK. Both work against today's client.

**Q7-3. Q6 has no recorded answer.** `MIGRATION_AUDIT.md` section 10 item 6 poses
it and recommends option (a); grepping the whole audit finds only the places that
pose it. `CONTEXT_HANDOFF.md` section 5 does not list it as decided. Q7-1
supersedes it and needs answering first.

**Q7-4. Fix R-11 and R-12 during extraction, or after?** Standing rule 4 says
move first, improve second, in separate commits. Both are latent bugs that a
faithful port would reproduce. R-12 in particular is guaranteed to be caught by
Phase 4's "guaranteed camera release" requirement, so it will be fixed whether or
not it is planned.

**Q7-5. Does `gaze_test.html` stay a supported consumer?** It is the only reader
of `type: "gaze"`. If it does, the gaze message is a public contract and the SDK
must keep emitting it. If it does not, the gaze message could be dropped
entirely and only `input` shipped, which is a much smaller surface. It lives in
the legacy directory, not the game, so this is a scope question.

**Q7-6. Where does the wrapper live, and what is it called?** Section 6.4 assumes
a file in the game repo that imports `focusedgaze.server.websocket`. Phase 11
plans to delete the duplicated modules from the game repo; this wrapper is the
one file that must survive that deletion.
