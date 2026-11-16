// The page's one job: feed the ported live chain with real audio and show
// what it heard. The microphone and a dropped file go through the SAME
// `LivePitchStream`, so a file is analysed exactly like a take — the web
// page has no offline pipeline to fall back on (librosa/pYIN stay desktop).

import { LivePitchStream, midiToName } from "./segmenter.js";
import { buildBundle } from "./bundle.js";
import { writeMidi } from "./midi.js";
import { BALANCE_DEFAULT, balanceGains, normalizeVoice, peakOf } from "./mix.js";
import {
  BALANCE_KEY,
  LANG_KEY,
  MODE_KEY,
  THEME_KEY,
  readStoredBalance,
  readStoredChoice,
  resumeMode,
  themeFor,
} from "./prefs.js";
import { renderNotes } from "./piano.js";
import { centsVerdict, formatCents } from "./cents.js";
// audio_io.GATE_RMS, through the generated constants: below it a block is
// silence for the level meter AND for the pitch.
import { GATE_RMS } from "./constants.js";
import { DEMO_SR, demoLoopsAfterSeek, demoRepeats, demoTake } from "./demo.js";
import { summarizeIntonation } from "./intonation.js";
import { formatClock, noteIndexAt, playheadX, timeAtX } from "./playhead.js";
import { spaceTogglesPlayback } from "./shortcuts.js";

const BLOCK = 1024; // capture tap block (~21 ms at 48 kHz)
const FILE_BLOCK = 2048;
const VIEW_SECONDS = 10; // the live roll shows the last ten seconds

const $ = (id) => document.getElementById(id);

const THEME_LABELS = { dark: "theme_dark", light: "theme_light" };
const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)") ?? null;

/**
 * The stored theme, or null when storage is unavailable.
 *
 * A blocked storage must not stop the page from opening: the same rule then
 * answers with the system's preference, which is what a first visit gets.
 */
function readStoredTheme() {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch {
    return null;
  }
}

// The tap runs in an AudioWorklet, not a ScriptProcessor: a deprecated node
// on the main thread would glitch under the roll's repaints and rewrite the
// timing the segmenter reads.
const TAP_SOURCE = `
class BlockTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(${BLOCK});
    this.filled = 0;
  }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.filled++] = channel[i];
      if (this.filled === this.buffer.length) {
        this.port.postMessage(this.buffer.slice(0));
        this.filled = 0;
      }
    }
    return true;
  }
}
registerProcessor("block-tap", BlockTap);
`;

let strings = { fr: {}, en: {} };
let lang = "fr";
let stream = null;
let processed = 0;
let lastT = 0;
let context = null;
let node = null;
let media = null;
let drawing = false;
let rowsSignature = "";
let listening = null; // null | "piano" | "mix"
let player = null; // the longer source: it decides when the take ends
let voicePlayer = null;
let playbackContext = null;
let voiceGainNode = null;
let pianoGainNode = null;
let limiter = null;
let peaks = { voice: 0, piano: 0 };
// The take's OWN signal, kept for the voice + piano comparison: the notes
// say what should sound, only the recording says what the singer produced.
let takeChunks = [];
let takeLength = 0;
let takeVoiceCache = null;
// Where the take came from (microphone, dropped file, demo): the bundle records
// it, because a comparison across takes starts by knowing what was compared.
let takeSource = "microphone";
// The playhead runs on the AUDIO clock, not the wall clock: the sound starts
// when the context says it does, and a rAF-driven bar must sit on the note
// being heard.
let playhead = null;
let playheadStartedAt = 0;
let playheadDuration = 0;
let playingIndex = -1; // the ONE answer both views read while it sounds
let lastMode = "piano"; // the roll resumes the mode the singer last chose
let balance = BALANCE_DEFAULT; // 0 = piano alone, 1 = voice alone
let looping = false; // the demo repeats until its button (or any stop) ends it
let theme = "light"; // "light" | "dark": the sheet the page is painted on
let rollPalette = null; // the canvas colours, read from the sheet once per theme

// The head script painted the first frame; this re-derives the SAME decision
// through the tested rule, so the module and that paint cannot disagree.
theme = themeFor({ stored: readStoredTheme(), prefersDark: prefersDark?.matches ?? false });
document.documentElement.dataset.theme = theme;

const t = (key) => strings[lang][key] ?? key;

function setStatus(key) {
  $("status").textContent = t(key);
}

function applyLanguage() {
  for (const el of document.querySelectorAll("[data-i18n]")) {
    el.textContent = t(el.dataset.i18n);
  }
  // An aria-label is a label, not content: it must follow the language too,
  // and textContent would have wiped the input it belongs to.
  for (const el of document.querySelectorAll("[data-i18n-aria]")) {
    el.setAttribute("aria-label", t(el.dataset.i18nAria));
  }
  // Its label follows the LOOP as well as the language, so the generic scan
  // above is corrected rather than trusted.
  updateDemoLabel();
  $("lang").textContent = lang === "fr" ? "EN" : "FR";
  // The theme pill names the CURRENT appearance, as the desktop's selector
  // does: it is a state, not an instruction, so a dark page says "Sombre".
  $("theme").textContent = t(THEME_LABELS[theme]);
  $("theme").title = t("theme_aria");
  document.documentElement.lang = lang;
  rowsSignature = "";
  updateListenLabels();
  draw();
}

function setTheme(next) {
  theme = next;
  document.documentElement.dataset.theme = theme;
  rollPalette = null; // the canvas reads the sheet, so drop its cached colours
  $("theme").textContent = t(THEME_LABELS[theme]);
  draw();
}

function toggleTheme() {
  const next = theme === "dark" ? "light" : "dark";
  setTheme(next);
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    // no storage: the choice still holds for this visit
  }
}

function resetTake(rate, source = "microphone") {
  stopListening();
  takeSource = source;
  stream = new LivePitchStream(rate, { gateRms: GATE_RMS });
  processed = 0;
  lastT = 0;
  takeChunks = [];
  takeLength = 0;
  takeVoiceCache = null;
  rowsSignature = "";
  draw();
}

function rememberVoice(block) {
  takeChunks.push(block);
  takeLength += block.length;
  takeVoiceCache = null;
}

function takeVoice() {
  if (!takeChunks.length) return null;
  if (!takeVoiceCache) {
    takeVoiceCache = new Float32Array(takeLength);
    let offset = 0;
    for (const chunk of takeChunks) {
      takeVoiceCache.set(chunk, offset);
      offset += chunk.length;
    }
  }
  return takeVoiceCache;
}

function onBlock(block) {
  if (!stream) return;
  lastT = (processed + block.length) / stream.sr;
  const { level } = stream.feed(block, lastT);
  processed += block.length;
  rememberVoice(block);
  $("level").style.width = `${Math.min(100, (level / 0.08) * 100)}%`;
  const reading = stream.seg.confirmed() ?? stream.seg.current();
  // An empty slot, not a dash: the readout is where a NOTE goes, and a 52 px
  // em dash reads as a stray rule on the page.
  $("note").textContent = reading ? midiToName(reading.midi) : "";
  $("hz").textContent = reading && Number.isFinite(reading.hz) ? `${reading.hz.toFixed(1)} Hz` : "";
  scheduleDraw();
}

function scheduleDraw() {
  if (drawing) return;
  drawing = true;
  requestAnimationFrame(() => {
    drawing = false;
    draw();
  });
}

function draw() {
  drawRoll();
  drawRows();
}

function viewRange() {
  const notes = stream ? stream.seg.notes : [];
  const live = node !== null;
  if (live) {
    const now = Math.max(VIEW_SECONDS, lastT);
    return [now - VIEW_SECONDS, now];
  }
  const end = notes.length ? notes[notes.length - 1].end : lastT;
  // the tail past the last note's end must stay visible while it sounds
  const reach = playhead !== null ? Math.max(end, playhead) : end;
  return [0, Math.max(4, reach + 0.5)];
}

/**
 * The roll's colours, read from the stylesheet.
 *
 * The canvas paints itself, so a dark page would otherwise keep a light roll:
 * reading the same tokens the page uses means the two palettes cannot drift
 * apart. Read ONCE per theme — a per-frame `getComputedStyle` would re-run
 * style resolution sixty times a second for a value that never changes.
 */
function rollColors() {
  if (rollPalette !== null) return rollPalette;
  const style = getComputedStyle(document.documentElement);
  const read = (name) => style.getPropertyValue(name).trim();
  rollPalette = {
    lane: read("--roll-lane"),
    grid: read("--roll-grid"),
    rule: read("--roll-rule"),
    label: read("--roll-label"),
    time: read("--roll-time"),
    bar: read("--roll-bar"),
    preview: read("--roll-bar-preview"),
    now: read("--roll-bar-now"),
    glint: read("--roll-glint"),
    outline: read("--roll-outline"),
    playhead: read("--roll-playhead"),
    glow: read("--roll-playhead-glow"),
  };
  return rollPalette;
}

/**
 * A bar with soft corners: the roll's notes are struck keys, not blocks of
 * paint, and a rounded end reads as one at 4 px tall.
 */
function roundedBar(painter, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  painter.beginPath();
  painter.moveTo(x + r, y);
  painter.lineTo(x + width - r, y);
  painter.quadraticCurveTo(x + width, y, x + width, y + r);
  painter.lineTo(x + width, y + height - r);
  painter.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  painter.lineTo(x + r, y + height);
  painter.quadraticCurveTo(x, y + height, x, y + height - r);
  painter.lineTo(x, y + r);
  painter.quadraticCurveTo(x, y, x + r, y);
  painter.closePath();
}

function drawRoll() {
  const canvas = $("roll");
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const width = canvas.clientWidth || 800;
  const height = 220;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const painter = canvas.getContext("2d");
  painter.setTransform(dpr, 0, 0, dpr, 0, 0);
  painter.clearRect(0, 0, width, height);
  // The roll is an instrument, not a chart: black keys shade their lane the
  // way every piano roll does, octave rules are copper, bars are lit copper
  // with a glint, and the playhead is a needle with a head. Pitch and time
  // read before any number is counted.
  const colors = rollColors();
  painter.lineWidth = 1;

  const notes = stream ? stream.seg.notes : [];
  const active = stream ? stream.seg.activeNote(lastT) : null;
  const [t0, t1] = viewRange();
  const span = Math.max(0.5, t1 - t0);

  let low = 60;
  let high = 72;
  for (const n of notes) {
    low = Math.min(low, n.midi);
    high = Math.max(high, n.midi);
  }
  if (active) {
    low = Math.min(low, active.midi);
    high = Math.max(high, active.midi);
  }
  low = Math.max(21, low - 2);
  high = Math.min(108, high + 2);
  const lanes = high - low + 1;
  const lane = height / lanes;
  const x = (seconds) => ((seconds - t0) / span) * width;

  // Black keys first, under everything: a shaded lane is the cue that reads
  // contour at a glance instead of counting rows.
  const blackKey = (midi) => [1, 3, 6, 8, 10].includes(((midi % 12) + 12) % 12);
  for (let midi = low; midi <= high; midi++) {
    if (!blackKey(midi)) continue;
    const top = height - (midi - low + 1) * lane;
    painter.fillStyle = colors.lane;
    painter.fillRect(0, top, width, lane);
  }
  painter.font = "10px 'Segoe UI', Arial, sans-serif";
  painter.fillStyle = colors.time;
  painter.strokeStyle = colors.grid;
  for (let second = Math.ceil(t0); second <= t1; second++) {
    const secondX = Math.round(x(second)) + 0.5;
    painter.beginPath();
    painter.moveTo(secondX, 0);
    painter.lineTo(secondX, height - 14);
    painter.stroke();
    painter.fillText(`${second.toFixed(0)} s`, secondX + 4, height - 3);
  }
  painter.strokeStyle = colors.rule;
  painter.font = "11px Georgia, 'Times New Roman', serif";
  painter.fillStyle = colors.label;
  for (let midi = low; midi <= high; midi++) {
    if (midi % 12 !== 0) continue;
    const y = Math.round(height - (midi - low + 1) * lane) + 0.5;
    painter.beginPath();
    painter.moveTo(0, y);
    painter.lineTo(width, y);
    painter.stroke();
    painter.fillText(midiToName(midi), 5, y - 4);
  }

  const bar = (note, preview, now) => {
    const left = x(note.start);
    const right = x(note.end);
    const top = height - (note.midi - low + 1) * lane;
    const w = Math.max(2, right - left);
    const h = Math.max(3, lane - 3);
    painter.fillStyle = preview ? colors.preview : now ? colors.now : colors.bar;
    roundedBar(painter, left, top + 1.5, w, h, 3);
    painter.fill();
    // A one-pixel glint along the top edge: the bar reads as a struck key
    // rather than a block of paint. The preview (still sounding) gets none,
    // so a sung note is never mistaken for a decided one.
    if (!preview) {
      painter.fillStyle = colors.glint;
      painter.fillRect(left + 2, top + 2.5, Math.max(1, w - 4), 1);
    }
    if (now) {
      painter.strokeStyle = colors.outline;
      painter.lineWidth = 1.5;
      roundedBar(painter, left + 0.75, top + 2.25, Math.max(1, w - 1.5), h - 1.5, 2.5);
      painter.stroke();
      painter.lineWidth = 1;
    }
    if (note.voiceDoubt) {
      // The published doubt is a SHAPE, so it survives a monochrome print
      // and a colour-blind reader: dashed, never a different hue alone.
      painter.setLineDash([4, 3]);
      painter.strokeStyle = colors.playhead;
      painter.lineWidth = 1.2;
      roundedBar(painter, left + 0.75, top + 2.25, Math.max(1, w - 1.5), h - 1.5, 2.5);
      painter.stroke();
      painter.setLineDash([]);
      painter.lineWidth = 1;
    }
  };
  notes.forEach((n, index) => {
    if (n.end >= t0 && n.start <= t1) bar(n, false, index === playingIndex);
  });
  if (active) bar(active, true, false);
  if (playhead !== null && playhead >= t0 && playhead <= t1) {
    const px = playheadX(playhead, t0, t1, width);
    painter.save();
    painter.shadowColor = colors.glow;
    painter.shadowBlur = 8;
    painter.strokeStyle = colors.playhead;
    painter.lineWidth = 2;
    painter.beginPath();
    painter.moveTo(px, 0);
    painter.lineTo(px, height);
    painter.stroke();
    painter.restore();
    // The needle's head: where the eye lands when the take starts sounding.
    painter.fillStyle = colors.playhead;
    painter.beginPath();
    painter.moveTo(px - 5, 0);
    painter.lineTo(px + 5, 0);
    painter.lineTo(px, 7);
    painter.closePath();
    painter.fill();
  }
}

function tickPlayhead() {
  if (listening === null || playbackContext === null) return;
  playhead = Math.min(playheadDuration, playbackContext.currentTime - playheadStartedAt);
  const notes = stream ? stream.seg.notes : [];
  const index = noteIndexAt(notes, playhead);
  $("note").textContent = index >= 0 ? midiToName(notes[index].midi) : "";
  $("hz").textContent = "";
  updateClock(playhead);
  setPlayingIndex(index);
  drawRoll();
  if (playhead < playheadDuration) requestAnimationFrame(tickPlayhead);
}

/**
 * The row of the note being heard. Marked without rebuilding the table (a
 * rebuild per frame would drop the selection and cost a hundred nodes a
 * frame) and scrolled only when the row CHANGES, with `nearest` so a row
 * already on screen never makes the list jump.
 */
function setPlayingIndex(index) {
  if (index === playingIndex) return;
  playingIndex = index;
  paintPlayingRow();
  if (index < 0) return;
  const row = $("rows").querySelector(`tr[data-index="${index}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
}

function paintPlayingRow() {
  for (const row of $("rows").children) {
    row.classList.toggle("playing", Number(row.dataset.index) === playingIndex);
  }
}

const VERDICT_KEYS = { ok: "verdict_ok", near: "verdict_near", off: "verdict_off" };

/** `{key}` substitution for the few sentences that carry numbers. */
function fill(template, values) {
  return template.replace(/\{(\w+)\}/g, (_, key) => values[key] ?? "");
}

function tendencyText(tendency) {
  if (tendency === "sharp") return t("intonation_sharp");
  if (tendency === "flat") return t("intonation_flat");
  if (tendency === "even") return t("intonation_even");
  return "";
}

/**
 * The take's intonation in one line: how many notes sat on their key, and the
 * lean only when the desktop's own evidence rule backs it. Silence, not a
 * row of zeros, when there is no note to summarise.
 */
function renderIntonation(notes) {
  const line = $("intonation");
  const summary = summarizeIntonation(notes);
  if (!summary.n) {
    line.hidden = true;
    line.textContent = "";
    $("registers").hidden = true;
    $("registers").textContent = "";
    $("position").hidden = true;
    $("position").textContent = "";
    return;
  }
  const sentence = fill(t("intonation_line"), {
    in_tune: String(summary.inTune),
    n: String(summary.n),
    share: String(Math.round(summary.share * 100)),
  });
  const tendency = tendencyText(summary.tendency);
  line.textContent = tendency ? `${sentence} — ${tendency}` : sentence;
  line.hidden = false;
  renderLeanLine(
    "registers",
    "registers_label",
    [
      ["register_low", summary.byRegister?.low],
      ["register_mid", summary.byRegister?.mid],
      ["register_high", summary.byRegister?.high],
    ],
  );
  renderLeanLine(
    "position",
    "position_label",
    [
      ["position_start", summary.byPosition?.start],
      ["position_end", summary.byPosition?.end],
    ],
  );
}

/**
 * One line per kind of habit (register, position), naming only the groups the
 * evidence backs: a group of three notes leans however those three notes
 * leaned, so it stays unsaid — the desktop's rule, and the reason the line can
 * vanish instead of printing a dash.
 */
function renderLeanLine(id, labelKey, groups) {
  const line = $(id);
  const items = [];
  for (const [nameKey, lean] of groups) {
    if (!lean) continue;
    items.push(
      fill(t("lean_line"), {
        name: t(nameKey),
        tendency: t(lean.mean > 0 ? "intonation_sharp" : "intonation_flat"),
      }),
    );
  }
  line.textContent = items.length ? `${t(labelKey)} — ${items.join(" · ")}` : "";
  line.hidden = !items.length;
}

function drawRows() {
  const notes = stream ? stream.seg.notes : [];
  const signature = `${notes.length}:${notes.length ? notes[notes.length - 1].end.toFixed(3) : ""}:${lang}`;
  if (signature === rowsSignature) return;
  rowsSignature = signature;

  const body = $("rows");
  body.replaceChildren();
  // The roll's history is the take; a very long file would otherwise build
  // thousands of rows nobody reads.
  for (let i = Math.max(0, notes.length - 200); i < notes.length; i++) {
    const note = notes[i];
    const row = document.createElement("tr");
    row.dataset.index = String(i);
    const cells = [
      String(i + 1),
      `${note.start.toFixed(2)} s`,
      `${note.duration.toFixed(2)} s`,
      midiToName(note.midi),
      // the note's own distance from the key it was rounded to, so the lit row
      // answers "just or off, and by how much" while it sounds
      formatCents(note.dev) ?? t("none"),
      note.voiceDoubt ? t("doubt") : t("none"),
    ];
    const verdict = centsVerdict(note.dev);
    cells.forEach((text, column) => {
      const cell = document.createElement("td");
      cell.textContent = text;
      if (column === 5 && note.voiceDoubt) cell.className = "doubt";
      if (column === 4 && verdict) {
        // The colour is the verdict; the word in the tooltip is the redundant
        // cue, so the scale is never carried by colour alone.
        cell.className = `cents-${verdict}`;
        cell.title = t(VERDICT_KEYS[verdict]);
      }
      row.append(cell);
    });
    body.append(row);
  }
  // a rebuild drops the class, so the sounding row is repainted on top of
  // the new nodes instead of waiting for the next playhead tick
  paintPlayingRow();
  renderIntonation(notes);
  $("roll").classList.toggle("seekable", notes.length > 0);
  $("empty").hidden = notes.length > 0;
  // An empty table with its header row is a table OF nothing: no check can
  // pass it (`th-has-data-cells` only ever reports "incomplete"), and a
  // keyboard visitor lands in a scrollable region with nothing to scroll. The
  // empty sentence speaks for it until the first note.
  $("notes-table").hidden = notes.length === 0;
  const capturing = node !== null;
  $("export").disabled = notes.length === 0;
  $("bundle").disabled = notes.length === 0 || takeChunks.length === 0;
  $("listen").disabled = notes.length === 0 || capturing;
  $("voice").disabled = notes.length === 0 || capturing || takeChunks.length === 0;
  $("balance").disabled = takeChunks.length === 0;
  // A reset with no take would move a slider that does nothing: the two
  // controls share the "there is a voice to balance" condition.
  $("balance-reset").disabled = takeChunks.length === 0;
  $("demo").disabled = capturing;
  $("guide-demo").disabled = capturing;
}

async function startCapture() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("error_mic");
    return;
  }
  stopListening();
  $("record").disabled = true;
  setStatus("listening");
  try {
    context = new AudioContext({ latencyHint: "interactive" });
    if (context.state === "suspended") await context.resume();
    const url = URL.createObjectURL(new Blob([TAP_SOURCE], { type: "text/javascript" }));
    await context.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    // Raw input on purpose: the browser's own filters rewrite the signal
    // BEFORE the analyser sees it, exactly like the Windows endpoint chain.
    media = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
    resetTake(context.sampleRate);
    const source = context.createMediaStreamSource(media);
    node = new AudioWorkletNode(context, "block-tap", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    node.port.onmessage = (event) => onBlock(event.data);
    source.connect(node);
    node.connect(context.destination); // keeps the graph pulling; the tap is silent
    $("stop").disabled = false;
  } catch {
    await stopCapture(true);
    setStatus("error_mic");
  }
}

async function stopCapture(failed = false) {
  if (node) {
    node.port.onmessage = null;
    node.disconnect();
    node = null;
  }
  if (media) {
    for (const track of media.getTracks()) track.stop();
    media = null;
  }
  if (context) {
    await context.close();
    context = null;
  }
  if (stream) stream.seg.flush(lastT);
  $("record").disabled = false;
  $("stop").disabled = true;
  if (!failed) setStatus("stopped");
  $("level").style.width = "0%";
  draw();
}

/**
 * The take's samples through the live chain, in the same blocks a capture
 * arrives in — one path for a dropped file and for the demo signal, so a file
 * is read exactly like a take and the demo like a file.
 */
function feedTake(samples, rate) {
  for (let offset = 0; offset < samples.length; offset += FILE_BLOCK) {
    const block = samples.subarray(offset, Math.min(samples.length, offset + FILE_BLOCK));
    lastT = (offset + block.length) / rate;
    stream.feed(block, lastT);
  }
  stream.seg.flush(lastT);
}

/**
 * The sample take, so the listening path can be tried with no microphone at
 * all. The samples go through `feedTake` like any dropped file: the notes and
 * their cents are the real chain's answer, never a written-out table.
 */
async function loadDemo() {
  stopListening();
  setStatus("analyzing");
  $("record").disabled = true;
  $("stop").disabled = true;
  await new Promise((resolve) => setTimeout(resolve, 0)); // let the status paint
  const samples = demoTake();
  resetTake(DEMO_SR, "demo");
  rememberVoice(samples);
  feedTake(samples, DEMO_SR);
  setStatus("demo_ready");
  $("record").disabled = false;
  draw();
}

/**
 * The demo button: ONE gesture that loads the sample take and hears it, again
 * and again, until it is pressed once more. No microphone, no second button.
 */
async function toggleDemo() {
  if (looping) {
    stopListening();
    setStatus("demo_stopped");
    return;
  }
  await loadDemo();
  const notes = stream ? stream.seg.notes : [];
  if (!notes.length) return;
  looping = true;
  updateDemoLabel();
  // Voice + piano: the loop exists to judge the balance by ear.
  startListening("mix");
}

/**
 * The tutorial's own way into the demo, from its first step.
 *
 * Step one says "sing", which a newcomer cannot do before anything has been
 * heard — and the transport button is a whole section further down. So the step
 * carries a second door to the SAME gesture, and closes the sheet on the way so
 * the roll and the notes the demo produces are on screen instead of below a
 * folded tutorial. It never stops the loop: the transport button and Space own
 * that gesture, and a control that says "hear an example" must not silence it.
 */
async function hearExample() {
  document.querySelector("details.guide").open = false;
  if (looping) return; // already sounding: closing the sheet is the whole gesture
  await toggleDemo();
}

/**
 * The paper copy: the sheet is printed WITH its tutorial, which the screen
 * keeps folded.
 *
 * A printer has no pointer, so the sheet is opened for the print and put back
 * afterwards — including when the dialog is cancelled, which is why the state
 * is remembered instead of assumed. `beforeprint`/`afterprint` are fired by
 * every engine this page targets for both Ctrl+P and `window.print()`.
 */
let guideWasOpen = false;

function listenForPrint() {
  const guide = document.querySelector("details.guide");
  window.addEventListener("beforeprint", () => {
    guideWasOpen = guide.open;
    guide.open = true;
  });
  window.addEventListener("afterprint", () => {
    guide.open = guideWasOpen;
  });
  $("print").addEventListener("click", () => window.print());
}

async function analyzeFile(file) {
  stopListening();
  setStatus("analyzing");
  $("record").disabled = true;
  $("stop").disabled = true;
  await new Promise((resolve) => setTimeout(resolve, 0)); // let the status paint
  try {
    const decoder = new AudioContext();
    const buffer = await decoder.decodeAudioData(await file.arrayBuffer());
    await decoder.close();
    resetTake(buffer.sampleRate, "file");
    const samples = buffer.getChannelData(0);
    // Own copy: the decoded buffer is disposable, this one is the take.
    rememberVoice(new Float32Array(samples));
    feedTake(samples, buffer.sampleRate);
    setStatus("ready");
  } catch {
    setStatus("error_file");
  }
  $("record").disabled = false;
  draw();
}

/**
 * A remembered choice, written through one gate.
 *
 * `localStorage`, not the tab's session: a preference that suited a room and a
 * habit should still be there next time the page is opened.
 */
function remember(key, value) {
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // Storage can be blocked (private mode). The value still holds for the
    // open page, which is all any of these choices need to work.
  }
}

function storeBalance() {
  remember(BALANCE_KEY, balance);
}

function applyBalance(smoothing = 0.02) {
  if (!voiceGainNode && !pianoGainNode) return;
  const gains = balanceGains(balance, peaks);
  const now = playbackContext ? playbackContext.currentTime : 0;
  for (const [node, value] of [
    [voiceGainNode, gains.voice],
    [pianoGainNode, gains.piano],
  ]) {
    if (!node) continue;
    if (smoothing > 0) node.gain.setTargetAtTime(value, now, smoothing);
    else node.gain.value = value;
  }
}

function updateClock(position) {
  const separator = lang === "fr" ? "," : ".";
  $("clock-pos").textContent = formatClock(position, separator);
  $("clock-total").textContent = formatClock(playheadDuration, separator);
  $("clock").hidden = false;
}

function updateDemoLabel() {
  $("demo").textContent = t(looping ? "demo_stop" : "demo");
}

function updateListenLabels() {
  $("listen").textContent = t(listening === "piano" ? "listen_stop" : "listen");
  $("voice").textContent = t(listening === "mix" ? "listen_stop" : "listen_voice");
}

function stopListening() {
  // Any stop ends the demo loop: the loop is a playback decision, and no
  // caller of a stop expects sound to come back a second later.
  looping = false;
  for (const source of [player, voicePlayer]) {
    if (!source) continue;
    source.onended = null;
    try {
      source.stop();
    } catch {
      // already stopped: the buffer ended on its own
    }
  }
  player = null;
  voicePlayer = null;
  voiceGainNode = null;
  pianoGainNode = null;
  limiter = null;
  if (playbackContext) {
    playbackContext.close();
    playbackContext = null;
  }
  listening = null;
  playhead = null;
  $("clock").hidden = true;
  setPlayingIndex(-1);
  updateListenLabels();
  updateDemoLabel();
}

async function startListening(mode, offset = 0) {
  const notes = stream ? stream.seg.notes : [];
  const voice = mode === "mix" ? takeVoice() : null;
  if (!notes.length || listening || (mode === "mix" && !voice)) return;
  listening = mode;
  lastMode = mode;
  remember(MODE_KEY, mode);
  updateListenLabels();
  setStatus("preparing");
  // Let the status paint: rendering a long take is one synchronous pass
  // (six partials per note), which would otherwise freeze a silent page.
      await new Promise((resolve) => setTimeout(resolve, 0));
    // A stop during the paint must not start a take the singer already ended.
    if (listening !== mode) return;
  try {
    // The piano is rendered at the RECORDING's rate when both are played:
    // resampling one side would shift its timing against the other.
    const sr = voice ? stream.sr : 44100;
    const piano = renderNotes(notes, { sr });
    const normalized = voice ? normalizeVoice(voice) : null;
    const voiceSamples = normalized ? normalized.samples : null;
    const duration = Math.max(piano.length, voiceSamples ? voiceSamples.length : 0) / sr;
    // A click in the roll starts the SAME buffer part-way: the take is not
    // re-rendered per seek, and the playhead resumes where the gesture is.
    const startAt = Math.max(0, Math.min(offset, Math.max(0, duration - 0.001)));
    playbackContext = new AudioContext();
    if (playbackContext.state === "suspended") await playbackContext.resume();

    // Two sources, one limiter. The limiter replaces the whole-sum scaling a
    // baked mix could do: live faders cannot know the sum, so peaks that would
    // have clipped are tamed instead — below its threshold the graph is a wire.
    limiter = playbackContext.createDynamicsCompressor();
    limiter.threshold.value = -3;
    limiter.knee.value = 0;
    limiter.ratio.value = 12;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.2;
    limiter.connect(playbackContext.destination);

    const sourceFor = (samples) => {
      const buffer = playbackContext.createBuffer(1, Math.max(1, samples.length), sr);
      buffer.copyToChannel(samples, 0);
      const source = playbackContext.createBufferSource();
      source.buffer = buffer;
      return source;
    };

    pianoGainNode = playbackContext.createGain();
    pianoGainNode.connect(limiter);
    player = sourceFor(piano);
    player.connect(pianoGainNode);
    if (voiceSamples) {
      voiceGainNode = playbackContext.createGain();
      voiceGainNode.connect(limiter);
      voicePlayer = sourceFor(voiceSamples);
      voicePlayer.connect(voiceGainNode);
    }
    peaks = { voice: normalized ? normalized.peak : 0, piano: peakOf(piano) };
    applyBalance(0); // the take starts at the balance the slider shows

    const longest = voiceSamples && voiceSamples.length > piano.length ? voicePlayer : player;
    longest.onended = () => {
      if (!listening) return; // an explicit stop already cleaned up
      const again = demoRepeats({ looping, mode, notes });
      stopListening();
      if (again) {
        // Start the same take from its first second: a loop the singer has to
        // restart by hand is not a loop.
        looping = true;
        updateDemoLabel();
        startListening("mix", 0);
        return;
      }
      setStatus("listened");
    };
    player.start(0, startAt);
    if (voicePlayer) voicePlayer.start(0, startAt);
    playheadDuration = duration;
    // the clock is offset by the seek, so `playhead` stays true for the
    // whole playback instead of jumping to the start
    playheadStartedAt = playbackContext.currentTime - startAt;
    playhead = startAt;
    updateClock(startAt);
    requestAnimationFrame(tickPlayhead);
    setStatus("playing");
  } catch {
    stopListening();
    setStatus("error_play");
  }
}

/** A click in the roll hears the take from that second (see timeAtX). */
function seekFromRoll(event) {
  if (!stream || !stream.seg.notes.length || node !== null) return;
  const rect = $("roll").getBoundingClientRect();
  const [t0, t1] = viewRange();
  const seconds = timeAtX(event.clientX - rect.left, t0, t1, rect.width);
  const mode = resumeMode({ stored: lastMode, hasVoice: takeChunks.length > 0 });
  // A seek repositions the demo loop instead of ending it, so the passage can
  // be re-heard while the balance is adjusted by ear.
  const keepLooping = looping;
  if (listening) stopListening();
  startListening(mode, seconds);
  if (demoLoopsAfterSeek({ looping: keepLooping, playing: Boolean(listening) })) {
    looping = true;
    updateDemoLabel();
  }
}

function toggleListening(mode) {
  if (listening) {
    stopListening();
    setStatus("stopped");
  } else {
    startListening(mode);
  }
}

function download(bytes, name, type) {
  const blob = new Blob([bytes], { type });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

// One clock for every exported name: two files of the same take must sort
// together, and a reader must be able to tell which came first.
const stamp = () => new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");

function exportMidi() {
  const notes = stream ? stream.seg.notes : [];
  if (!notes.length) return;
  download(writeMidi(notes), `prise-web-${stamp()}.mid`, "audio/midi");
}

/**
 * Everything the page holds about the take, in one archive.
 *
 * The MIDI says what should be played; only the audio says what was sung, and
 * only the JSON says how it was measured. Someone who wants to re-run the
 * analysis, compare two takes or improve the algorithm needs all three, plus a
 * notice they can read without knowing the format.
 */
async function exportBundle() {
  const notes = stream ? stream.seg.notes : [];
  const samples = takeVoice();
  if (!notes.length || !samples) return;
  const readme = [
    t("bundle_readme_intro"),
    "",
    t("bundle_readme_audio"),
    t("bundle_readme_notes"),
    t("bundle_readme_midi"),
  ].join("\n");
  try {
    // The archive is packed on the way out, so this waits for the browser's
    // deflate; a refused stream is not fatal, `zip.js` then stores the entries.
    const bundle = await buildBundle({
      notes,
      samples,
      sr: stream.sr,
      midi: writeMidi(notes),
      readme: `${readme}\n`,
      readmeName: lang === "fr" ? "LISEZ-MOI.txt" : "README.txt",
      source: takeSource,
      language: lang,
      stamp: stamp(),
    });
    download(bundle.bytes, bundle.name, "application/zip");
  } catch {
    setStatus("error_bundle");
  }
}

async function init() {
  try {
    const response = await fetch("strings.json");
    strings = await response.json();
  } catch {
    // A missing strings file must not blank the page: the keys themselves
    // stay readable, which is enough to see the controls.
  }
  // The remembered choices are read BEFORE the first paint, so the page opens
  // in the language and the balance the singer left it in.
  try {
    lang = readStoredChoice(localStorage.getItem(LANG_KEY), ["fr", "en"]) ?? lang;
    lastMode = readStoredChoice(localStorage.getItem(MODE_KEY), ["piano", "mix"]) ?? lastMode;
    const storedBalance = readStoredBalance(localStorage.getItem(BALANCE_KEY));
    if (storedBalance !== null) balance = storedBalance;
  } catch {
    // no storage: the defaults are still a usable page
  }
  applyLanguage();
  $("balance").value = String(Math.round(balance * 100));
  $("record").addEventListener("click", startCapture);
  $("demo").addEventListener("click", toggleDemo);
  $("guide-demo").addEventListener("click", hearExample);
  listenForPrint();
  $("stop").addEventListener("click", () => stopCapture());
  $("export").addEventListener("click", exportMidi);
  $("bundle").addEventListener("click", exportBundle);
  $("listen").addEventListener("click", () => toggleListening("piano"));
  $("voice").addEventListener("click", () => toggleListening("mix"));
  $("roll").addEventListener("click", seekFromRoll);
  $("balance").addEventListener("input", () => {
    balance = Number($("balance").value) / 100;
    // Live faders: the take keeps sounding, no render and no seek, because
    // the sources are untouched — only two gains move (smoothed, so a fast
    // drag does not zipper).
    applyBalance();
    storeBalance();
  });
  $("theme").addEventListener("click", toggleTheme);
  // The system is followed only while the page has no choice of its own: the
  // moment the singer picks one, their preference outranks the operating
  // system for good.
  prefersDark?.addEventListener("change", (event) => {
    if (readStoredTheme() === null) setTheme(event.matches ? "dark" : "light");
  });
  $("balance-reset").addEventListener("click", () => {
    // The reference the page shipped with, in one gesture: one press instead
    // of aiming a 5-cent step at exactly 50.
    balance = BALANCE_DEFAULT;
    $("balance").value = String(Math.round(balance * 100));
    applyBalance();
    storeBalance();
  });
  document.addEventListener("keydown", (event) => {
    // A capture owns the page: its own buttons are disabled, and starting a
    // playback over a live take is not a gesture the singer asked for.
    if (node !== null) return;
    if (!stream || !stream.seg.notes.length) return;
    const target = event.target;
    if (
      !spaceTogglesPlayback({
        key: event.key,
        tagName: target?.tagName,
        editable: target?.isContentEditable === true,
        ctrl: event.ctrlKey,
        meta: event.metaKey,
        alt: event.altKey,
        shift: event.shiftKey,
      })
    ) {
      return;
    }
    event.preventDefault(); // the page must not scroll under the take
    toggleListening(resumeMode({ stored: lastMode, hasVoice: takeChunks.length > 0 }));
  });
  $("file").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) analyzeFile(file);
    event.target.value = "";
  });
  $("lang").addEventListener("click", () => {
    lang = lang === "fr" ? "en" : "fr";
    remember(LANG_KEY, lang);
    applyLanguage();
  });
  window.addEventListener("resize", scheduleDraw);
  draw();
}

init();
