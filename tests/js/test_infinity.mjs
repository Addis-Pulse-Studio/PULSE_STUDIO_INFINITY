/**
 * Pulse Infinity's auto-continue decision.
 *
 * `js/ps_infinity.js` imports ComfyUI's app and api modules, which do not exist
 * outside a browser, so -- as test_sockets.mjs does -- the pure function is lifted
 * out of the real file and driven here. The rule it encodes is the one that
 * keeps an open-ended film from running away: continue only on a clean finish,
 * for a session this tab has auto-continue on for, with nothing else queued.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE = readFileSync(join(HERE, "..", "..", "js", "ps_infinity.js"), "utf8");

let failures = 0;
let passed = 0;
function test(name, fn) {
  try {
    fn();
    passed++;
  } catch (error) {
    failures++;
    console.error(`  ✗ ${name}\n    ${error.message}`);
  }
}

function lift(name, params) {
  const start = SOURCE.indexOf(`export function ${name}`);
  assert.ok(start > -1, `${name} not found in ps_infinity.js`);
  let depth = 0;
  let i = SOURCE.indexOf("{", start);
  const open = i;
  for (; i < SOURCE.length; i++) {
    if (SOURCE[i] === "{") depth++;
    else if (SOURCE[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  return new Function(...params, SOURCE.slice(open + 1, i));
}

const shouldRequeue = lift("shouldRequeue", ["pending", "autoSessions", "queueEmpty"]);
const ON = new Set(["clock_garden"]);
const BEAT = { session: "clock_garden", beat: 3, continue: true, complete: false };

test("a committed beat of an auto-continued film queues the next", () => {
  assert.equal(shouldRequeue(BEAT, ON, true), true);
});

test("nothing pending, nothing queued", () => {
  assert.equal(shouldRequeue(null, ON, true), false);
});

test("a complete film does not continue", () => {
  assert.equal(shouldRequeue({ ...BEAT, complete: true }, ON, true), false);
  assert.equal(shouldRequeue({ ...BEAT, continue: false }, ON, true), false);
});

test("a session without auto-continue in this graph does not continue", () => {
  assert.equal(shouldRequeue(BEAT, new Set(["another_film"]), true), false);
  assert.equal(shouldRequeue(BEAT, new Set(), true), false);
});

test("a queue with work in it is not doubled", () => {
  assert.equal(shouldRequeue(BEAT, ON, false), false);
});

test("the controls go to ComfyUI's own server, never another origin", () => {
  assert.ok(SOURCE.includes('api.fetchApi("/pulse_infinity/control"'));
  assert.ok(!/https?:\/\//.test(SOURCE.replace(/^\s*\*.*$/gm, "")));
});

test("the toggle is a node property, not a widget", () => {
  assert.ok(SOURCE.includes("node.properties[AUTO] = !on;"));
  assert.ok(!SOURCE.includes("addWidget("));
  assert.ok(!SOURCE.includes("addDOMWidget("));
});

if (failures) {
  console.error(`\n${failures} Pulse Infinity JS test(s) failed`);
  process.exit(1);
}
console.log(`ok - ${passed} Pulse Infinity JS tests passed`);
