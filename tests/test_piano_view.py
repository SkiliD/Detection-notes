"""Tests for the PianoView keyboard, the PianoRoll live follow and the waveform."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from v2m.core.liveseg import LiveNote
from v2m.core.music import PIANO_MAX_MIDI, PIANO_MIN_MIDI
from v2m.core.notes import Note
from v2m.gui.widgets import PianoRoll, PianoView, PitchAxis, WaveformView, figure_color


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _notes():
    return [
        Note(start=0.0, end=0.5, midi=60, confidence=0.9),   # C4 (white)
        Note(start=0.5, end=1.0, midi=64, confidence=0.9),   # E4 (white)
        Note(start=1.0, end=1.5, midi=61, confidence=0.9),   # C#4 (black)
    ]


def test_set_notes_indexes_spans(qapp):
    pv = PianoView()
    pv.set_notes(_notes())
    assert set(pv._spans) == {60, 64, 61}
    assert pv._spans[60] == [(0.0, 0.5)]


def test_playhead_lights_sounding_keys_only(qapp):
    pv = PianoView()
    pv.set_notes(_notes())
    pv.set_playhead(0.2)
    assert pv._active == {60}
    pv.set_playhead(0.7)
    assert pv._active == {64}
    pv.set_playhead(1.2)
    assert pv._active == {61}
    # after the last note: nothing lit
    pv.set_playhead(1.6)
    assert pv._active == set()


def test_playhead_without_notes_is_safe(qapp):
    pv = PianoView()
    pv.set_playhead(1.0)  # no crash, nothing lit
    assert pv._active == set()


def test_the_keyboard_range_is_a_parameter(qapp):
    """Le chercheur de note affiche les 88 touches d'un piano classique."""
    full = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
    assert len(full._whites) == 52         # 88 touches dont 36 noires
    assert full._white_midi(0) == 21 and full._white_midi(51) == 108
    assert not full._has_black_after(108)  # rien après C8
    # la vue d'analyse couvre 48..84 : 22 blanches, C6 comprise (la touche 84
    # était invisible — une note au bord haut ne pouvait pas s'allumer)
    assert len(PianoView()._whites) == 22


def test_every_black_key_of_the_piano_is_drawn(qapp):
    """Régression : le clavier ne dessinait qu'un Do# et un Fa# par octave
    (la règle testait les mauvaises classes de hauteur), donc Ré#, Sol# et
    La# n'existaient pas — aucune note noire ailleurs ne pouvait s'allumer."""
    pv = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
    drawn = {
        pv._white_midi(i) + 1
        for i in range(len(pv._whites))
        if pv._has_black_after(pv._white_midi(i))
    }
    black_pcs = (1, 3, 6, 8, 10)
    expected = {m for m in range(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1) if m % 12 in black_pcs}
    assert drawn == expected and len(drawn) == 36


def test_pressed_keys_are_lit_directly(qapp):
    """`set_pressed` allume la touche entendue, noire comprise."""
    pv = PianoView(min_midi=PIANO_MIN_MIDI, max_midi=PIANO_MAX_MIDI)
    pv.set_pressed({63})  # Ré#4 : une touche noire qui n'était pas dessinée
    assert pv._active == {63}
    assert not pv.grab().isNull()
    pv.set_pressed()
    assert pv._active == set()


def test_the_pitch_axis_names_notes_instead_of_numbers(qapp):
    """L'axe du roll se lit comme un clavier, pas comme des numéros MIDI."""
    axis = PitchAxis(orientation="left")
    assert axis.tickStrings([21, 60, 61, 108], 1.0, 1.0) == ["A0", "C4", "C#4", "C8"]


# ----------------------------------------------------------------- piano roll


def _xrange(roll: PianoRoll) -> tuple[float, float]:
    (x0, x1), _ = roll.getPlotItem().vb.viewRange()
    return float(x0), float(x1)


def _yrange(roll: PianoRoll) -> tuple[float, float]:
    _, (y0, y1) = roll.getPlotItem().vb.viewRange()
    return float(y0), float(y1)


def test_live_pitch_band_frames_what_is_sung(qapp):
    """Une bande stable évite qu'une erreur d'octave écrase le roll."""
    roll = PianoRoll()
    roll.begin_live()
    assert _yrange(roll)[1] - _yrange(roll)[0] >= roll.LIVE_Y_MIN
    roll.set_live_pitch_range([72])  # une seule note : bande minimale centrée
    y0, y1 = _yrange(roll)
    assert y1 - y0 == pytest.approx(roll.LIVE_Y_MIN)
    assert y0 < 72 < y1
    roll.set_live_pitch_range([60, 64, 67])
    y0, y1 = _yrange(roll)
    assert y0 <= 60 and y1 >= 67
    roll.set_live_pitch_range([])  # rien de chanté : la bande ne bouge pas
    assert _yrange(roll) == pytest.approx((y0, y1))


def test_begin_live_empties_the_roll_and_fixes_the_window(qapp):
    roll = PianoRoll()
    roll.set_notes(_notes())
    assert roll._bar_items
    roll.begin_live()
    assert not roll._bar_items
    assert _xrange(roll) == pytest.approx((0.0, roll.LIVE_SPAN))


def test_live_cursor_scrolls_a_constant_width_window(qapp):
    """Le curseur live avance et la fenêtre suit sans jamais s'élargir."""
    roll = PianoRoll()
    roll.begin_live()
    for t in (5.0, 40.0, 120.0):
        roll.set_live_cursor(t)
        x0, x1 = _xrange(roll)
        assert x0 <= t <= x1, f"curseur {t} hors fenêtre [{x0}, {x1}]"
        assert x1 - x0 <= roll.LIVE_SPAN + 1e-9  # jamais plus large que la fenêtre
    # au démarrage la fenêtre est plus courte (pas de temps négatif)…
    roll.begin_live()
    roll.set_live_cursor(1.0)
    assert _xrange(roll)[0] == 0.0
    # …puis elle glisse à largeur constante
    assert _xrange(roll)[1] - _xrange(roll)[0] < roll.LIVE_SPAN
    roll.set_live_cursor(120.0)
    x0, x1 = _xrange(roll)
    assert (x0, x1) == pytest.approx((120.0 - roll.LIVE_SPAN * 0.88, 120.0 + roll.LIVE_SPAN * 0.12))


def test_live_preview_follows_the_held_note(qapp):
    roll = PianoRoll()
    roll.set_live_preview(LiveNote(start=1.0, end=1.2, midi=64))
    preview = roll._live_preview
    assert preview is not None
    assert preview.opts["x1"] == [1.2]
    roll.set_live_preview(LiveNote(start=1.0, end=1.9, midi=64))  # tenue
    assert preview.opts["x1"] == [1.9]
    roll.set_live_preview(LiveNote(start=1.0, end=1.9, midi=67))  # changement de note
    assert preview.opts["y0"] == pytest.approx([66.6])
    roll.set_live_preview(None)
    assert not preview.isVisible()


def test_notation_toggle_recolors_existing_bars(qapp):
    """Basculer le mode notation doit repeindre les barres déjà posées."""
    roll = PianoRoll()
    notes = _notes()
    notes[0].figure = "noire"
    roll.set_notes(notes)
    assert roll._bar_items[0].opts["brush"] == roll._color_for_conf(0.9)
    roll.set_notation_mode(True)
    assert roll._bar_items[0].opts["brush"] == figure_color("noire")
    assert len(roll._labels) == 1  # 0,5 s : assez large pour l'étiquette N


def test_in_place_figure_labeling_refreshes_the_bars(qapp):
    """En live, label_notes mute les notes après coup : la barre doit suivre."""
    roll = PianoRoll()
    roll.set_notation_mode(True)
    notes = _notes()
    roll.set_notes(notes)
    assert roll._labels == []
    notes[0].figure = "croche"  # étiquetage live, en place
    roll.set_notes(notes)
    assert len(roll._labels) == 1
    assert roll._bar_items[0].opts["brush"] == figure_color("croche")


def test_notes_that_shrink_fall_back_to_a_full_rebuild(qapp):
    """Le raccourci d'ajout ne doit pas garder des barres fantômes."""
    roll = PianoRoll()
    notes = _notes()
    roll.set_notes(notes)
    roll.set_notes(notes[:2])
    assert len(roll._bar_items) == 2


def test_seek_moves_the_playhead_without_yanking_the_view(qapp):
    """`on_seek` demande follow=False : un double-clic ne doit pas faire
    sauter la vue loin de l'endroit qu'on vient de cliquer."""
    roll = PianoRoll()
    roll.set_notes(_notes())
    roll.getPlotItem().vb.setXRange(0.0, 2.0, padding=0)
    roll.set_playhead(30.0, follow=False)
    assert _xrange(roll) == pytest.approx((0.0, 2.0))
    roll.set_playhead(30.0)  # pendant la lecture, la vue suit le curseur
    assert _xrange(roll)[0] > 0.0


# ---------------------------------------------------------------- waveform


def _envelope(wf: WaveformView) -> tuple[float, float, int]:
    """(min, max, points) actually handed to the plot (lo/hi curves)."""
    lo_x, lo = wf._curves[0].getData()
    _, hi = wf._curves[1].getData()
    return float(np.min(lo)), float(np.max(hi)), len(lo_x)


def test_waveform_envelope_keeps_the_true_peaks(qapp):
    """Régression : l'aperçu décimait un point sur N, donc les crêtes isolées
    disparaissaient et une longue prise paraissait plate."""
    wf = WaveformView()
    y = np.zeros(600_000, dtype=np.float32)  # décimation x12 à l'affichage
    y[5] = 1.0                               # pic jamais atteint par y[::12]
    y[500_000] = -0.8
    wf.set_signal(y, 1000)

    lo, hi, n_points = _envelope(wf)
    assert n_points <= WaveformView.MAX_POINTS
    assert (lo, hi) == pytest.approx((-0.8, 1.0))


def test_waveform_short_signal_is_plotted_as_is(qapp):
    wf = WaveformView()
    y = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    wf.set_signal(y, 100)
    lo, hi, n_points = _envelope(wf)
    assert n_points == 100
    assert (lo, hi) == pytest.approx((-0.5, 0.5))


def test_paint_does_not_crash_offscreen(qapp):
    from PyQt6.QtWidgets import QWidget

    host = QWidget()
    lay = host  # minimal parent
    pv = PianoView(host)
    pv.resize(400, 80)
    host.show()
    pv.set_notes(_notes())
    pv.set_playhead(0.2)
    pix = pv.grab()  # forces a paintEvent pass offscreen
    assert not pix.isNull()
    assert pix.width() == 400
