/**
 * Pulse Studio Infinity in the browser: auto-continue, and a film's controls.
 *
 * One queue run renders one beat. To render an open-ended film without queueing
 * by hand, auto-continue queues the graph again when a beat has been committed --
 * but only when every one of these holds:
 *
 *   - the Observer said the film continues (`pulse_infinity.beat_done`), for a
 *     session whose Loop node in THIS tab's graph has auto-continue on. The
 *     event goes to the tab that queued the prompt, never to every tab, so one
 *     tab's film cannot requeue another's;
 *   - the prompt then finished cleanly (`execution_success`) -- never after an
 *     error or an interrupt, which is how a stuck director stops the loop;
 *   - ComfyUI's queue is empty, so a batch the person queued themselves is not
 *     doubled.
 *
 * The toggle lives in `node.properties`, not in a widget: a widget is a saved
 * slot under the slot contract, and the pack allows one DOM widget in total
 * (tests/test_js_guard.py). The controls -- pause, resume, reroll, replan,
 * stop -- are context-menu entries that ask ComfyUI's own server, through a
 * relative route registered in __init__.py. Nothing here reaches any origin
 * but the one the page came from.
 *
 * Everything is wrapped: a throw in a LiteGraph hook can abort a workflow load.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOOP = "PulseInfinityLoopController";
const AUTO = "pulse_infinity_auto_continue";

/**
 * Should a finished prompt be followed by another? Pure, for the tests.
 *
 * `pending` is the last beat_done event's detail (or null), `autoSessions` the
 * sessions whose Loop node in this graph has auto-continue on, `queueEmpty`
 * whether ComfyUI has nothing else running or waiting.
 */
export function shouldRequeue(pending, autoSessions, queueEmpty) {
  if (!pending || !pending.continue || pending.complete) return false;
  if (!autoSessions.has(pending.session)) return false;
  return Boolean(queueEmpty);
}

function loopNodes() {
  return (app.graph?._nodes ?? []).filter((n) => (n.comfyClass ?? n.type) === LOOP);
}

function sessionOf(node) {
  return String(node.widgets?.find((w) => w.name === "session")?.value ?? "");
}

function autoSessions() {
  return new Set(loopNodes().filter((n) => n.properties?.[AUTO]).map(sessionOf));
}

async function queueIsEmpty() {
  try {
    const queue = await api.getQueue();
    return !(queue?.Running?.length) && !(queue?.Pending?.length);
  } catch (err) {
    console.warn("[PulseStudio] could not read the queue; not continuing:", err);
    return false;
  }
}

function tell(severity, summary, detail) {
  try {
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
      toast.add({ severity, summary, detail, life: 6000 });
      return;
    }
  } catch (err) {
    // Fall through to the console.
  }
  console.info(`[PulseStudio] ${summary}: ${detail}`);
}

async function control(session, action, note = "") {
  try {
    const res = await api.fetchApi("/pulse_infinity/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session, action, note }),
    });
    const data = await res.json();
    if (data.status === "ok") tell("info", `Pulse Infinity: ${action}`, data.message);
    else tell("warn", `Pulse Infinity: ${action} refused`, data.message);
  } catch (err) {
    tell("error", "Pulse Infinity", `could not ${action}: ${err}`);
  }
}

let pending = null;

try {
  api.addEventListener("pulse_infinity.beat_done", (event) => {
    pending = event?.detail ?? null;
  });
  api.addEventListener("execution_success", async () => {
    const finished = pending;
    pending = null;
    try {
      if (!finished) return;
      const sessions = autoSessions();
      if (!sessions.has(finished.session)) return;
      if (shouldRequeue(finished, sessions, await queueIsEmpty())) {
        await app.queuePrompt(0, 1);
      }
    } catch (err) {
      console.warn("[PulseStudio] auto-continue failed:", err);
    }
  });
  for (const name of ["execution_error", "execution_interrupted"]) {
    api.addEventListener(name, () => {
      pending = null;
    });
  }
  api.addEventListener("pulse_infinity.complete", (event) => {
    tell("success", "Pulse Infinity", `${event?.detail?.session}: the film is complete`);
  });
  api.addEventListener("pulse_infinity.paused", (event) => {
    tell("info", "Pulse Infinity", `${event?.detail?.session} is paused; resume it from the Loop node`);
  });
  api.addEventListener("pulse_infinity.error", (event) => {
    tell("warn", "Pulse Infinity", event?.detail?.message ?? "the director's turn failed");
  });
  api.addEventListener("pulse_infinity.status", (event) => {
    console.info(`[PulseStudio] ${event?.detail?.text ?? ""}`);
  });
} catch (err) {
  console.warn("[PulseStudio] Pulse Infinity events not attached:", err);
}

app.registerExtension({
  name: "comfyui_pulse_studio.infinity",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== LOOP) return;
    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
      const result = getExtraMenuOptions?.apply(this, arguments);
      try {
        const node = this;
        const session = sessionOf(node);
        const on = Boolean(node.properties?.[AUTO]);
        options.push(
          null,
          {
            content: on ? "Pulse Infinity: auto-continue is ON (turn off)"
                        : "Pulse Infinity: auto-continue is off (turn on)",
            callback: () => {
              node.properties = node.properties ?? {};
              node.properties[AUTO] = !on;
            },
          },
          { content: "Pulse Infinity: pause after this beat",
            callback: () => control(session, "pause") },
          { content: "Pulse Infinity: resume", callback: () => control(session, "resume") },
          { content: "Pulse Infinity: reroll the last beat",
            callback: () => control(session, "reroll") },
          {
            content: "Pulse Infinity: replan the last beat...",
            callback: () => {
              const note = window.prompt("What should the director change about the last beat?");
              if (note !== null) control(session, "replan", note);
            },
          },
          { content: "Pulse Infinity: stop the film", callback: () => control(session, "stop") },
        );
      } catch (err) {
        console.warn("[PulseStudio] Pulse Infinity menu failed:", err);
      }
      return result;
    };
  },
});
