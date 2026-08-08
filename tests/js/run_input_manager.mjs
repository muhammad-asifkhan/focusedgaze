/* Drive the REAL browser client against a running focusedgaze server.
 *
 * Audit section 39 recorded, in bold, that the game playing against the minimal
 * gaze-only `input` message was INFERRED from reading `input-manager.js` and had
 * never been executed. That is the same defect class as a docstring describing a
 * positioning check that never ran. This harness executes it.
 *
 * It loads the unmodified `input-manager.js`, shims the browser globals it
 * touches, points it at a server we control, and prints the resulting public API
 * state plus every event the client fired. The Python side asserts on that.
 *
 * Usage:  node run_input_manager.mjs <ws-url> <run-ms> <game-dir>
 * Output: one JSON object on stdout.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

const [, , wsUrl, runMsRaw, gameDir] = process.argv;
const runMs = Math.max(800, Number(runMsRaw) || 1500);

const sourcePath = join(gameDir, "input-manager.js");
const original = readFileSync(sourcePath, "utf8");

// Point the client at our ephemeral port. This is the ONLY edit, and it is a
// single string literal, not logic.
//
// The replacement is ASSERTED to have fired. Standing brief A2: a text
// replacement that silently no-ops is worse than one that errors, because the
// run then reports success while testing something else -- here, a client still
// dialling 8765, which would look like "the server sent nothing" rather than
// "the harness is broken".
const NEEDLE = "ws://localhost:8765";
if (!original.includes(NEEDLE)) {
  console.error(`FATAL: ${NEEDLE} not found in ${sourcePath}; the URL literal moved`);
  process.exit(2);
}
const patched = original.split(NEEDLE).join(wsUrl);
if (patched === original) {
  console.error("FATAL: URL replacement produced no change");
  process.exit(2);
}

// --- browser globals the client touches -----------------------------------
globalThis.window = globalThis;
globalThis.window.addEventListener = () => {};      // it binds the [m] key
globalThis.document = { addEventListener: () => {} };

// Node 22+ ships a global WebSocket, so the transport is real, not a mock.
const RealWebSocket = globalThis.WebSocket;
if (typeof RealWebSocket !== "function") {
  console.error("FATAL: this Node has no global WebSocket");
  process.exit(2);
}

// Count what actually arrives, so "we read N messages" is measured rather than
// inferred from a sleep. Intercepting construction is the only place to do this:
// the client owns its socket and never exposes it.
let seen = 0;
const byType = { gaze: 0, input: 0, mode: 0, other: 0 };
globalThis.WebSocket = function (...args) {
  const sock = new RealWebSocket(...args);
  sock.addEventListener("message", (ev) => {
    seen += 1;
    let t = "other";
    try {
      t = JSON.parse(ev.data).type ?? "other";
    } catch {
      t = "other";
    }
    byType[t] = (byType[t] ?? 0) + 1;
  });
  return sock;
};

// Run the unmodified client.
const events = [];
new Function(patched)();

const api = globalThis.window.GameInput;
if (!api) {
  console.error("FATAL: input-manager.js did not publish window.GameInput");
  process.exit(2);
}

// Record every event it fires. click/swipe/clear are the ones that must NOT
// fire: a spurious click activates whatever the cursor is resting on.
api.onClick(() => events.push({ event: "click" }));
api.onSwipe((d) => events.push({ event: "swipe", detail: d }));
api.onClear(() => events.push({ event: "clear" }));
api.onStatus(() => events.push({ event: "status" }));
api.onPinchStart(() => events.push({ event: "pinchStart" }));
api.onPinchEnd(() => events.push({ event: "pinchEnd" }));

setTimeout(() => {
  console.log(
    JSON.stringify({
      messages_seen: seen,
      by_type: byType,
      ok: api.ok,
      x: api.x,
      y: api.y,
      mode: api.mode,
      source: api.source,
      handOk: api.handOk,
      gesture: api.gesture,
      connected: api.connected,
      gestureAvailable: api.gestureAvailable,
      pinching: api.pinching,
      events,
    })
  );
  process.exit(0);
}, runMs);
