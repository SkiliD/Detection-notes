// Conformance tests for the browser port: every case in `cases.json` was
// produced by the REAL Python segmenter (`scripts/export_web_fixtures.py`),
// so a disagreement here means the page would decide different notes from
// the desktop — the one thing a copy must not do.
//
// Run with `node --test web/test/segmenter.test.mjs` (the Pages workflow runs
// it before publishing).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { writeMidi } from "../midi.js";

import { AttackTracker } from "../attacks.js";
import { centsVerdict, formatCents } from "../cents.js";
// The page's own silence gate: the generated constant, so this chain runs at
// exactly the threshold the page uses (audio_io.GATE_RMS).
import { GATE_RMS } from "../constants.js";
import { DEMO_NOTES, DEMO_SR, demoLoopsAfterSeek, demoRepeats, demoTake } from "../demo.js";
import { balanceGains, normalizeVoice, peakOf } from "../mix.js";
import { formatClock, noteIndexAt, playheadX, timeAtX } from "../playhead.js";
import { MIN_CENTS, MIN_NOTES, summarizeIntonation } from "../intonation.js";
import { readStoredBalance, readStoredChoice, resumeMode, themeFor } from "../prefs.js";
import { spaceTogglesPlayback } from "../shortcuts.js";
import { renderNotes } from "../piano.js";
import { LiveNoteSegmenter, LivePitchStream, median, midiToName, pyRound } from "../segmenter.js";
import { YinBlock } from "../yin.js";

const payload = JSON.parse(readFileSync(new URL("./cases.json", import.meta.url), "utf8"));
const TOL = 1e-6;

for (const item of payload.cases) {
  test(`the desktop and the page agree: ${item.name}`, () => {
    const seg = new LiveNoteSegmenter(item.params);
    const events = [];
    let lastT = 0;
    for (const [t, hz, attack, strength, weak, quality] of item.frames) {
      lastT = t;
      events.push(
        seg.push(t, hz === null ? NaN : hz, {
          attack: attack !== 0,
          strength,
          weak: weak !== 0,
          quality: quality === null ? NaN : quality,
        }),
      );
    }
    if (item.flush) seg.flush(lastT + payload.frame_s);
    assert.deepEqual(events, item.events, "frame-by-frame on/off decisions diverged");
    assert.equal(seg.notes.length, item.notes.length, "note count diverged");
    seg.notes.forEach((note, index) => {
      const want = item.notes[index];
      for (const field of ["start", "end", "midi", "strength", "dev"]) {
        assert.ok(
          Math.abs(note[field] - want[field]) <= TOL,
          `${field} of note ${index}: ${note[field]} != ${want[field]}`,
        );
      }
      assert.equal(note.articulated, want.articulated, `articulated of note ${index}`);
      assert.equal(note.voiceDoubt, want.voice_doubt, `voice doubt of note ${index}`);
    });
  });
}

test("half-even rounding matches Python, not Math.round", () => {
  // Python: round(60.5) == 60, round(61.5) == 62. Math.round would answer 61
  // and 62, shifting every .5 pitch by a semitone.
  assert.equal(pyRound(60.5), 60);
  assert.equal(pyRound(61.5), 62);
  assert.equal(pyRound(-0.5), 0);
  assert.equal(pyRound(-1.5), -2);
  assert.equal(pyRound(59.4), 59);
});

test("the median of an even count is the mean of the middle two, as numpy's", () => {
  assert.equal(median([3, 1, 2, 4]), 2.5);
  assert.equal(median([2, 1]), 1.5);
  assert.equal(median([5]), 5);
});

test("note names sit on the MIDI octave numbering", () => {
  assert.equal(midiToName(60), "C4");
  assert.equal(midiToName(69), "A4");
  assert.equal(midiToName(21), "A0");
});

test("YIN reads a clean sine and keeps its clock absolute", () => {
  // The FFT correlation and the frame clock are the two things a port can
  // get subtly wrong and still look like it works: a wrong 1/N in the
  // autocorrelation halves the pitch, and a clock that restarts after a
  // buffer trim never reaches the segmenter's confirmation window.
  const sr = 44100;
  const x = new Float32Array(sr);
  for (let i = 0; i < x.length; i++) x[i] = 0.3 * Math.sin((2 * Math.PI * 220 * i) / sr);
  const yin = new YinBlock(sr, { window: 1024, hop: 512 });
  const frames = [];
  for (let offset = 0; offset + 512 <= x.length; offset += 512) {
    frames.push(...yin.pushWithQuality(x.subarray(offset, offset + 512)));
  }
  const voiced = frames.filter((frame) => Number.isFinite(frame.hz));
  assert.ok(voiced.length > 40, `only ${voiced.length} voiced frames`);
  const measured = median(voiced.map((frame) => frame.hz));
  assert.ok(Math.abs(measured - 220) / 220 < 0.01, `read ${measured.toFixed(2)} Hz`);
  assert.ok(
    frames.every((frame, index) => index === 0 || frame.t > frames[index - 1].t),
    "the frame clock is not strictly increasing",
  );
  assert.ok(frames[0].t < 0.05, `the clock starts at ${frames[0].t}`);
});

test("the piano renders the desktop's take", () => {
  // Dry render (the room's noise seed is the browser's own): the digest
  // samples the SAME signal the Python renderer produced, so a changed
  // attack, brightness exponent, damper or pedal tail moves these values.
  const piano = payload.piano;
  const rendered = renderNotes(piano.notes, { sr: piano.sr, room: false });
  assert.equal(rendered.length, piano.length, "render length diverged");
  piano.samples.forEach((want, index) => {
    const got = rendered[index * piano.stride];
    assert.ok(
      Math.abs(got - want) <= 1e-6,
      `sample ${index * piano.stride}: ${got} != ${want}`,
    );
  });
  let peak = 0;
  let energy = 0;
  for (const value of rendered) {
    peak = Math.max(peak, Math.abs(value));
    energy += value * value;
  }
  assert.ok(Math.abs(peak - piano.peak) <= 1e-6, `peak ${peak} != ${piano.peak}`);
  assert.ok(
    Math.abs(Math.sqrt(energy / rendered.length) - piano.rms) <= 1e-6,
    "RMS diverged",
  );
});

test("the recorded voice is brought to a fixed peak, and silence stays silent", () => {
  const { samples, peak } = normalizeVoice(new Float32Array([0.2, -0.1, 0.05]));
  assert.equal(peak, 0.85);
  assert.ok(Math.abs(peakOf(samples) - 0.85) < 1e-6, `peak ${peakOf(samples)}`);
  assert.ok(Math.abs(samples[0] - 0.85) < 1e-6, `first sample ${samples[0]}`);
  // A silent take must not be amplified into noise: peak 0, zeros out.
  const silent = normalizeVoice(new Float32Array([0, 0]));
  assert.equal(silent.peak, 0);
  assert.deepEqual([...silent.samples], [0, 0]);
});

test("the demo take is a signal the real chain reads as its own notes", () => {
  // The demo is not a note table: it is fed through the same chain the
  // microphone feeds, so these are the notes the page will really show.
  const samples = demoTake();
  const stream = new LivePitchStream(DEMO_SR, { gateRms: GATE_RMS });
  // The page's own block size: the attack detector's decisions depend on it,
  // and a coarser block would test a path the page never runs.
  const block = 2048;
  for (let offset = 0; offset < samples.length; offset += block) {
    const chunk = samples.subarray(offset, Math.min(samples.length, offset + block));
    stream.feed(chunk, (offset + chunk.length) / DEMO_SR);
  }
  stream.seg.flush(samples.length / DEMO_SR);
  assert.deepEqual(
    stream.seg.notes.map((note) => note.midi),
    DEMO_NOTES.map((note) => note.midi),
    "the demo phrase is not what the chain hears",
  );
  // Each onset is a real attack: the phrase is articulated, not a legato
  // smear, which is also the path that reads the detected attack's accent.
  // Every onset after the first is heard as an attack; the first has no
  // floor to rise from (the tracker's priming block IS its only history),
  // which is a property of the detector, not of the phrase.
  assert.deepEqual(
    stream.seg.notes.map((note) => note.articulated),
    DEMO_NOTES.map((_, index) => index > 0),
    "the demo onsets are no longer heard as attacks",
  );
  // The phrase exercises ALL THREE verdicts: the red one has to appear on a
  // real measurement, or the colour scale is only proven by its unit test.
  const verdicts = stream.seg.notes.map((note) => centsVerdict(note.dev));
  assert.deepEqual(
    [...new Set(verdicts)].sort(),
    ["near", "off", "ok"],
    "the demo phrase no longer spans the three intonation bands",
  );
  // ... and the LOW register is a named habit, so the register line the page
  // renders has something backed to show instead of staying hidden forever.
  const summary = summarizeIntonation(stream.seg.notes);
  assert.equal(summary.byRegister.low?.n, 4, "the demo's low band is not four notes");
  assert.ok(
    summary.byRegister.low.mean >= 12,
    `the demo's low band leans only ${summary.byRegister.low.mean} cents`,
  );
  // ... and the cents it reports are the ones the phrase was sung with.
  stream.seg.notes.forEach((note, index) => {
    assert.ok(
      Math.abs(note.dev * 100 - DEMO_NOTES[index].cents) <= 12,
      `note ${index}: ${note.dev * 100} cents instead of ${DEMO_NOTES[index].cents}`,
    );
  });
});

test("the capture path hears an attack without throwing", () => {
  // The accent of a detected attack reads a flag decided one scope above it.
  // A block-scoped declaration made every onset — every consonant of a real
  // take — raise a ReferenceError, on the one path nobody can test without a
  // microphone. This drives that path: capture-sized blocks into the stream,
  // whose attack tracker is the mic's own.
  const samples = demoTake();
  const stream = new LivePitchStream(DEMO_SR, { gateRms: GATE_RMS });
  const block = 2048; // the page's capture block, not a convenient one
  assert.doesNotThrow(() => {
    for (let offset = 0; offset < samples.length; offset += block) {
      const chunk = samples.subarray(offset, Math.min(samples.length, offset + block));
      stream.feed(chunk, (offset + chunk.length) / DEMO_SR);
    }
    stream.seg.flush(samples.length / DEMO_SR);
  }, "a detected attack threw on the capture path");
  const heard = stream.seg.notes.filter((note) => note.articulated);
  assert.ok(heard.length >= 3, `only ${heard.length} onsets were heard as attacks`);
  // The accent is what the throw prevented: a detected attack must publish it.
  assert.ok(stream.attacks.strength > 0, "the attack's accent was never computed");
});

test("the capture path reads the same take from irregular blocks", () => {
  // A device hands over what it has, and the worklet re-chunks it, but the
  // chain itself must not quietly depend on one block size: a level, an
  // attack and a frame boundary are all measured per block. Measured spread
  // against 2048-sample blocks: at most 46 ms on a note boundary, and no
  // disagreement at all on the notes or their cents.
  const run = (sizes) => {
    const samples = demoTake();
    const stream = new LivePitchStream(DEMO_SR, { gateRms: GATE_RMS });
    const ends = [];
    let offset = 0;
    let index = 0;
    while (offset < samples.length) {
      const size = sizes[index++ % sizes.length];
      const chunk = samples.subarray(offset, Math.min(samples.length, offset + size));
      const end = (offset + chunk.length) / DEMO_SR;
      ends.push(end);
      stream.feed(chunk, end);
      offset += chunk.length;
    }
    stream.seg.flush(samples.length / DEMO_SR);
    return { notes: stream.seg.notes, attacks: [...stream.attacks.recent], ends };
  };
  const regular = run([2048]).notes;
  // Small blocks, hop-sized blocks, blocks larger than the analysis window,
  // and a size that never divides the others.
  const irregularRun = run([147, 1024, 733, 2048, 512, 1861, 320]);
  const irregular = irregularRun.notes;
  assert.equal(regular.length, DEMO_NOTES.length);
  assert.deepEqual(
    irregular.map((note) => note.midi),
    regular.map((note) => note.midi),
    "the block size changed which notes were heard",
  );
  irregular.forEach((note, index) => {
    const want = regular[index];
    assert.ok(Math.abs(note.start - want.start) <= 0.07, `start ${index}: ${note.start} vs ${want.start}`);
    assert.ok(Math.abs(note.end - want.end) <= 0.07, `end ${index}: ${note.end} vs ${want.end}`);
    assert.ok(Math.abs(note.dev - want.dev) <= 0.005, `cents ${index}: ${note.dev} vs ${want.dev}`);
  });
  // An attack carries the ABSOLUTE time of the block that found it. Deriving
  // that time from the block's length instead shifts every attack on a small
  // block by tens of milliseconds — which no note comparison above notices,
  // because the shift lands inside the boundary tolerance.
  assert.ok(irregularRun.attacks.length > 0, "the irregular run detected no attack");
  for (const attackT of irregularRun.attacks) {
    assert.ok(
      irregularRun.ends.some((end) => Math.abs(end - attackT) < 1e-9),
      `the attack at ${attackT} is not one of the captured blocks' own times`,
    );
  }
});

test("a note's distance from its key reads in signed whole cents", () => {
  assert.equal(formatCents(0.07), "+7", "a sharp note says so");
  assert.equal(formatCents(-0.12), "\u221212", "and a flat one, with a real minus");
  assert.equal(formatCents(0), "0", "on the key is not a signed reading");
  assert.equal(formatCents(-0.004), "0", "a rounding to zero must not print \u22120");
  assert.equal(formatCents(0.495), "+50");
  assert.equal(formatCents(-0.5), "\u221250");
  // no measurement, no number: the page supplies its own placeholder
  assert.equal(formatCents(NaN), null);
  assert.equal(formatCents(undefined), null);
});

test("the cents verdict uses the desktop's two bands", () => {
  assert.equal(centsVerdict(0), "ok");
  assert.equal(centsVerdict(0.09), "ok", "under ten cents is in tune");
  assert.equal(centsVerdict(-0.1), "ok", "the band is inclusive");
  assert.equal(centsVerdict(0.12), "near");
  assert.equal(centsVerdict(-0.25), "near", "the second band is inclusive too");
  assert.equal(centsVerdict(0.26), "off");
  assert.equal(centsVerdict(0.5), "off", "the plateau's clamp is still 'off'");
  // The verdict reads the SAME rounded cents the column prints: a value that
  // shows 10 cannot be painted as "near".
  assert.equal(formatCents(0.104), "+10");
  assert.equal(centsVerdict(0.104), "ok");
  assert.equal(centsVerdict(NaN), null);
  assert.equal(centsVerdict(undefined), null);
});

test("the playhead names the note that sounds, and a gap names nothing", () => {
  const notes = [
    { start: 0, end: 0.5, midi: 60 },
    { start: 0.6, end: 1.0, midi: 62 },
  ];
  assert.equal(noteIndexAt(notes, 0.2), 0);
  assert.equal(noteIndexAt(notes, 0.5), -1, "the end is silent: half-open lookup");
  assert.equal(noteIndexAt(notes, 0.75), 1);
  assert.equal(noteIndexAt(notes, 1.0), -1);
});

test("the sounding note has ONE index, shared by the roll and the table", () => {
  const notes = [{ start: 0, end: 0.5 }, { start: 0.6, end: 1.0 }, { start: 0.6, end: 0.7 }];
  assert.equal(noteIndexAt(notes, 0.1), 0);
  assert.equal(noteIndexAt(notes, 0.55), -1, "a gap has no row to light");
  assert.equal(noteIndexAt(notes, 0.65), 1, "the FIRST note starting there wins");
  assert.equal(noteIndexAt(notes, -1), -1);
  assert.equal(noteIndexAt(notes, 1.0), -1, "the end is silent");
});

test("the playhead stays inside the roll, whatever the clock says", () => {
  assert.equal(playheadX(0, 0, 10, 100), 0);
  assert.equal(playheadX(5, 0, 10, 100), 50);
  assert.equal(playheadX(-1, 0, 10, 100), 0, "a reading before the window clamps");
  assert.equal(playheadX(12, 0, 10, 100), 100, "a late reading clamps to the edge");
  assert.equal(playheadX(1, 3, 3, 100), 0, "a zero-width window must not divide by zero");
});

test("a click in the roll seeks to that second, and back", () => {
  assert.equal(timeAtX(0, 0, 10, 100), 0);
  assert.equal(timeAtX(50, 0, 10, 100), 5);
  assert.equal(timeAtX(120, 0, 10, 100), 10, "a click past the edge clamps");
  assert.equal(timeAtX(-20, 0, 10, 100), 0);
  assert.equal(timeAtX(10, 3, 3, 100), 3, "a zero-width window is not a seek");
  // The click and the playhead must agree: the pixel the playhead is drawn at
  // reads back as the second it shows, so a seek never lands a frame off.
  for (const seconds of [0, 1.25, 4, 9.9, 10]) {
    const px = playheadX(seconds, 0, 10, 640);
    assert.ok(Math.abs(timeAtX(px, 0, 10, 640) - seconds) < 1e-9, `round trip ${seconds}`);
  }
});

test("Space toggles playback, but never steals a control's own Space", () => {
  assert.equal(spaceTogglesPlayback({ key: " " }), true);
  assert.equal(spaceTogglesPlayback({ key: "Spacebar" }), true, "old browsers");
  assert.equal(spaceTogglesPlayback({ key: "Enter" }), false);
  // a focused control already acts on Space: taking it would fire twice
  for (const tagName of ["button", "INPUT", "TEXTAREA", "select", "A"]) {
    assert.equal(spaceTogglesPlayback({ key: " ", tagName }), false, tagName);
  }
  assert.equal(spaceTogglesPlayback({ key: " ", editable: true }), false);
  // modifier Space is a browser gesture (scroll, menu, tab switch)
  for (const modifier of ["ctrl", "meta", "alt", "shift"]) {
    assert.equal(spaceTogglesPlayback({ key: " ", [modifier]: true }), false, modifier);
  }
});

test("the playback clock reads m:ss.s and carries its minute exactly", () => {
  assert.equal(formatClock(0), "0:00.0");
  assert.equal(formatClock(3.2), "0:03.2");
  assert.equal(formatClock(12), "0:12.0");
  assert.equal(formatClock(59.96), "1:00.0", "a tenth must not print 0:60.0");
  assert.equal(formatClock(61.5), "1:01.5");
  assert.equal(formatClock(-2), "0:00.0");
  assert.equal(formatClock(NaN), "0:00.0");
  assert.equal(formatClock(3.2, ","), "0:03,2", "French writes a comma");
});

test("the balance moves two live faders as a crossfade", () => {
  const peaks = { voicePeak: 0.85, pianoPeak: 0.2 };
  const at = (value) => balanceGains(value, peaks);
  // gains are ratios against each peak, so peak x ratio is the target level
  assert.equal(at(0).voice, 0, "piano alone silences the voice");
  assert.ok(Math.abs(0.2 * at(0).piano - 0.55) < 1e-6, `piano: ${0.2 * at(0).piano}`);
  assert.equal(at(1).piano, 0, "voice alone silences the piano");
  assert.ok(Math.abs(0.85 * at(1).voice - 0.85) < 1e-6);
  // the middle reproduces the balance the page shipped with
  assert.ok(Math.abs(0.85 * at(0.5).voice - 0.85) < 1e-6);
  assert.ok(Math.abs(0.2 * at(0.5).piano - 0.55) < 1e-6);
  assert.ok(Math.abs(0.85 * at(0.25).voice - 0.425) < 1e-6);
  assert.ok(Math.abs(0.2 * at(0.25).piano - 0.55) < 1e-6);
  // out-of-range values clamp instead of inverting the sources
  assert.deepEqual(at(2), at(1));
  assert.deepEqual(at(-1), at(0));
  assert.deepEqual(at(NaN), at(0.5));
});

test("only the demo loops, and only while it is the thing playing", () => {
  const notes = [{ midi: 60 }];
  assert.equal(demoRepeats({ looping: true, mode: "mix", notes }), true);
  // An ordinary take ends where it ends: the loop is the demo button's.
  assert.equal(demoRepeats({ looping: false, mode: "mix", notes }), false);
  // The piano alone has no voice to judge the balance against.
  assert.equal(demoRepeats({ looping: true, mode: "piano", notes }), false);
  // A take that produced no notes would restart silence forever.
  assert.equal(demoRepeats({ looping: true, mode: "mix", notes: [] }), false);
});

test("a stored balance is trusted only when it is one", () => {
  assert.equal(readStoredBalance("0.2"), 0.2);
  assert.equal(readStoredBalance("0"), 0, "piano only is a real setting");
  assert.equal(readStoredBalance("1"), 1);
  // absent or unusable: the caller keeps its default (a bare `Number()` would
  // read null and "" as 0 and pin the mix to the piano)
  assert.equal(readStoredBalance(null), null);
  assert.equal(readStoredBalance(undefined), null);
  assert.equal(readStoredBalance(""), null);
  assert.equal(readStoredBalance("abc"), null);
  assert.equal(readStoredBalance("1.5"), null);
  assert.equal(readStoredBalance("-0.2"), null);
  assert.equal(readStoredBalance("NaN"), null);
});

test("a seek repositions the demo loop instead of ending it", () => {
  // A click in the roll is how a singer returns to the passage they are
  // judging: it must survive the stop that the restart goes through.
  assert.equal(demoLoopsAfterSeek({ looping: true, playing: true }), true);
  assert.equal(demoLoopsAfterSeek({ looping: false, playing: true }), false);
  // A seek that started nothing (no notes to play) must not leave the loop
  // armed with no sound behind it.
  assert.equal(demoLoopsAfterSeek({ looping: true, playing: false }), false);
});

test("the intonation line counts the notes the column paints as in tune", () => {
  const note = (dev) => ({ dev });
  // Printed cents 10, 17, 0, 13, 34: two are on their key (<= 10).
  const take = summarizeIntonation([0.095, 0.169, 0, 0.133, 0.338].map(note));
  assert.equal(take.n, 5);
  assert.equal(take.inTune, 2);
  assert.equal(take.sharp, 3);
  assert.equal(take.flat, 0);
  assert.ok(Math.abs(take.share - 0.4) < 1e-9);
  // Mean 14.8 cents over five notes: a named lean, because the desktop's
  // rule (4 notes, 12 cents) is met.
  assert.equal(take.tendency, "sharp");
  // A flat mean of the same size names the other direction.
  assert.equal(summarizeIntonation([-0.2, -0.15, -0.1, -0.18].map(note)).tendency, "flat");
  // Too few notes, or too small a mean: no habit is named.
  assert.equal(summarizeIntonation([0.3, 0.3, 0.3].map(note)).tendency, null);
  assert.equal(MIN_NOTES, 4);
  assert.equal(summarizeIntonation([0.02, -0.01, 0.03, 0, 0.01].map(note)).tendency, "even");
  assert.equal(MIN_CENTS, 12);
  // No notes, no report; a note with no measurement is not counted as tuned.
  assert.equal(summarizeIntonation([]).n, 0);
  assert.equal(summarizeIntonation([note(NaN), note(undefined)]).n, 0);
  assert.equal(summarizeIntonation([note(NaN), note(0)]).inTune, 1);
});

test("register tendencies name a band only when the desktop's rule is met", () => {
  const note = (midi, dev) => ({ midi, dev });
  // Extent 60..72, divided in EQUAL-WIDTH bands, never by note count:
  // 60-62 land low, 66 lands mid, 72 lands high.
  const take = summarizeIntonation([
    note(60, -0.25), note(61, -0.25), note(62, -0.25), note(60, -0.25),
    note(66, 0.02), note(66, 0.02), note(66, 0.02),
    note(72, 0.2), note(72, 0.2), note(72, 0.2), note(72, 0.2),
  ]);
  assert.deepEqual(take.byRegister.low, { n: 4, mean: -25 });
  assert.equal(take.byRegister.mid, null, "three notes are not a habit");
  assert.equal(take.byRegister.high.n, 4);
  assert.ok(take.byRegister.high.mean > 0, "the high band leans sharp");
  // A take on ONE pitch has no extent to divide, so no band is named.
  const single = summarizeIntonation([60, 60, 60, 60].map((midi) => note(midi, 0.3)));
  assert.deepEqual(single.byRegister, { low: null, mid: null, high: null });
  // A band follows the SAME rule as the take: three notes are not a habit.
  const thin = summarizeIntonation([note(60, -0.2), note(72, 0.2), note(72, 0.2), note(72, 0.2)]);
  assert.equal(thin.byRegister.low, null);
  assert.equal(thin.byRegister.high, null);
  // ... and neither is a band whose notes carry no measurement.
  const empty = summarizeIntonation([note(60, NaN), note(72, NaN)]);
  assert.equal(empty.byRegister.high, null);
  assert.equal(summarizeIntonation([]).byRegister.high, null);
});

test("position halves name a drift the take-wide mean washes out", () => {
  const note = (midi, dev) => ({ midi, dev });
  // The desktop's own drifting case: six notes 30 cents flat, then six sharp.
  const drift = summarizeIntonation([
    ...Array.from({ length: 6 }, () => note(60, -0.3)),
    ...Array.from({ length: 6 }, () => note(64, 0.3)),
  ]);
  assert.equal(drift.tendency, "even", "the flat and sharp halves cancel out");
  assert.deepEqual(drift.byPosition.start, { n: 6, mean: -30 });
  assert.deepEqual(drift.byPosition.end, { n: 6, mean: 30 });
  // Halves are counted in NOTES, never in seconds: the same notes split the
  // same way whatever their durations.
  assert.equal(drift.byPosition.start.n + drift.byPosition.end.n, 12);
  // An odd count gives its extra note to the END half; three notes are still
  // not a habit, so only the end is named here.
  const odd = summarizeIntonation([60, 60, 60, 64, 64, 64, 64].map((midi) => note(midi, 0.3)));
  assert.equal(odd.byPosition.start, null);
  assert.equal(odd.byPosition.end.n, 4);
  // A flat take is flat everywhere: the position groups agree with it.
  const even = summarizeIntonation(Array.from({ length: 8 }, () => note(60, -0.25)));
  assert.deepEqual(even.byPosition.start, { n: 4, mean: -25 });
  assert.deepEqual(even.byPosition.end, { n: 4, mean: -25 });
  assert.deepEqual(summarizeIntonation([]).byPosition, { start: null, end: null });
});

test("the page opens in the remembered theme, or in the system's", () => {
  // A chosen theme outlives the visit AND the operating system: someone who
  // turned the lights down does not want the page to follow the sun back up.
  assert.equal(themeFor({ stored: "dark", prefersDark: false }), "dark");
  assert.equal(themeFor({ stored: "light", prefersDark: true }), "light");
  // Nothing remembered: the system is the only evidence of the room.
  assert.equal(themeFor({ stored: null, prefersDark: true }), "dark");
  assert.equal(themeFor({ stored: null, prefersDark: false }), "light");
  // A value the page cannot paint is not a choice.
  assert.equal(themeFor({ stored: "sepia", prefersDark: true }), "dark");
  assert.equal(themeFor({ stored: undefined, prefersDark: false }), "light");
});

test("the remembered listen mode is the one a gesture resumes", () => {
  // No demo and no page here: this is the decision Space and a roll click
  // make, read from the stored value the last visit left.
  assert.equal(resumeMode({ stored: "mix", hasVoice: true }), "mix");
  assert.equal(resumeMode({ stored: "piano", hasVoice: true }), "piano");
  assert.equal(resumeMode({ stored: "piano", hasVoice: false }), "piano");
  // A remembered mix with nothing recorded falls back to the piano: resuming
  // it would start a playback that never sounds.
  assert.equal(resumeMode({ stored: "mix", hasVoice: false }), "piano");
  // Nothing remembered, or something the page cannot act on: the default.
  assert.equal(resumeMode({ stored: null, hasVoice: true }), "piano");
  assert.equal(resumeMode({ stored: "chord", hasVoice: true }), "piano");
  assert.equal(resumeMode({ stored: undefined, hasVoice: true }), "piano");
});

test("a remembered choice is only trusted when the page can act on it", () => {
  assert.equal(readStoredChoice("en", ["fr", "en"]), "en");
  assert.equal(readStoredChoice("mix", ["piano", "mix"]), "mix");
  // A stale value outlives the code that wrote it: fall back, never act on it.
  assert.equal(readStoredChoice("de", ["fr", "en"]), null);
  assert.equal(readStoredChoice("", ["fr", "en"]), null);
  assert.equal(readStoredChoice(null, ["fr", "en"]), null);
  assert.equal(readStoredChoice(undefined, ["fr", "en"]), null);
  assert.equal(readStoredChoice("piano", ["piano", "mix"]), "piano");
});

test("moving the balance never lifts the whole take", () => {
  const peaks = { voicePeak: 0.85, pianoPeak: 0.2 };
  let voice = 0;
  let piano = Infinity; // no upper bound yet: only the trend matters below
  for (let step = 0; step <= 10; step++) {
    const value = step / 10;
    const gains = balanceGains(value, peaks);
    const voiceLevel = 0.85 * gains.voice;
    const pianoLevel = 0.2 * gains.piano;
    // the crossfade peaks at its middle: no position is louder than the sum
    // of both references
    assert.ok(voiceLevel + pianoLevel <= 1.4 + 1e-9, `sum grew to ${voiceLevel + pianoLevel} at ${value}`);
    // ... and each side only rises or only falls, so dragging is predictable
    assert.ok(voiceLevel >= voice - 1e-9, `the voice fell at ${value}`);
    assert.ok(pianoLevel <= piano + 1e-9, `the piano rose at ${value}`);
    voice = voiceLevel;
    piano = pianoLevel;
  }
});

test("the MIDI writer emits a format-0 file with both events per note", () => {
  const bytes = writeMidi([{ start: 0, end: 0.5, midi: 60, strength: 0.5 }]);
  assert.equal(String.fromCharCode(...bytes.slice(0, 4)), "MThd");
  assert.equal(String.fromCharCode(...bytes.slice(14, 18)), "MTrk");
  const text = Array.from(bytes);
  assert.ok(text.includes(0x90), "no note-on");
  assert.ok(text.includes(0x80), "no note-off");
  assert.deepEqual(text.slice(-3), [0xff, 0x2f, 0x00], "no end-of-track");
});
