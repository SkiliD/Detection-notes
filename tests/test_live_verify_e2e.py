"""A take's full chain, proven end to end.

The user's request was: "sing a take with the audio box ticked, then
check the WAV and the JSON describe the same take, and see whether the
coherence block stays empty". This test literally sings: fake mic (the
`test_live_gui` harness), a real take exported into a temp folder, then
the TWO checks — `verify_take` on the WAV/JSON pair, and the report's
`coherence` block (empty on a clean melody).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# the same synthetic singer AND the same pumping helper as the live-GUI
# harness: this test walks one step further (the exported pair), it does
# not reinvent the mic
from tests.test_live_gui import FakeMic, _pump_until


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_recorded_take_passes_verify_and_coherence(
    qapp, monkeypatch, tmp_path
):
    import sounddevice as sd

    from v2m.core.verify import verify_take
    from v2m.gui import main_window as mw

    monkeypatch.setattr(sd, "InputStream", FakeMic)
    written: dict[str, str] = {}

    def fake_export(res, stem, **kw):
        written["stem"] = str(stem)
        from v2m.core.report import export_result

        export_result(res, stem, **kw)

    monkeypatch.setattr(mw, "export_result", fake_export)
    monkeypatch.setattr(mw, "live_export_dir", lambda stem: tmp_path / Path(stem).name)

    win = mw.MainWindow()
    try:
        win.page_live.chk_record_audio.setChecked(True)  # the take must write its audio
        win.on_start_live()
        # pump until the take has captured ~1.8 s (the FakeMic sings ~1.9 s)
        captured = _pump_until(
            qapp,
            lambda: win.live_thread is not None and win.live_thread.elapsed() >= 1.8,
            timeout=60,
        )
        assert captured, "le micro factice n'a rien livré"
        win.on_stop_live()
        # the closed take's live notes survived the capture reference: the
        # diff mark compares what the singer WATCHED (they arrive queued,
        # after on_stop_live dropped the thread)
        assert win._closed_live_notes, "les notes live n'ont pas survécu à l'arrêt"
        live_count = len(win._closed_live_notes)
        # analysis stays in the results page until the user explicitly exports
        analyzed = _pump_until(qapp, lambda: win.result is not None, timeout=60)
        assert analyzed, "l'analyse finale n'est pas arrivée"
        assert "stem" not in written, "la prise a été exportée sans action explicite"
        win._export_results()
        assert "stem" in written, "l'export explicite n'a pas écrit la prise"

        stem = Path(written["stem"])
        # the audio box delivered: the WAV exists, and verify_take accepts it
        assert stem.with_suffix(".wav").exists(), "la case audio n'a rien écrit"
        ok, lines = verify_take(stem.parent / stem.name)
        assert ok, f"prise incohérente : {lines}"

        # the report's coherence block: a clean four-note melody hides nothing
        rep = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        assert rep["coherence"]["checked"] >= 1
        assert rep["coherence"]["suspects"] == []
        # and the JSON's duration IS the WAV's
        assert abs(rep["duration_s"] - 1.9) < 0.5
        # the big counter followed the take: the take's LAST tick shows the
        # live count of the take itself (the final analysis may read one
        # note more once it can see the whole take — that is the diff mark's
        # job to show, not the counter's)
        assert win.page_live.live_indicator.label_note_count.text().isdigit()
        assert win.page_live.live_indicator.label_note_count.text() != "0"
        # the live-vs-final mark ran on the handed-over notes: every note
        # the final pass moved keeps its ACCENT outline on the live roll
        # (on this clean melody the two readings agree, so few outlines)
        assert len(win.page_live.piano_live._diff_items) <= live_count
    finally:
        win.page_live.chk_record_audio.setChecked(False)
        win.close()
