"""Tests rests : respirations entre phrases et figures de silence.

`rests.py` n'avait pas de test direct (il était couvert indirectement par la
justesse vocale, la Portée et le MusicXML). Ce fichier-ci fixe son contrat :
ce qu'un écart devient, et sous quel nom.

Garde-fous déjà portés par le code, et testés ici parce qu'ils sont le
contrat du module : notes désordonnées ou qui se chevauchent (le curseur ne
recule jamais), écart sous le seuil de couture, fin de prise qui n'est pas un
silence, tempo absent (pas de figure devinée).
"""

from v2m.core.grid import BeatGrid
from v2m.core.notation import layout_staff
from v2m.core.notes import Note
from v2m.core.rests import MIN_REST_S, Rest, classify_rest, detect_rests

GRID = BeatGrid(120.0)


def _note(midi, start, dur_s, figure=None):
    return Note(start=start, end=start + dur_s, midi=midi, confidence=0.9,
                figure=figure)


# ------------------------------------------------------------------- figures
def test_classify_rest_speaks_rest_not_note():
    """Le classifieur est celui des notes, le lexique est celui des silences :
    1 temps est une « noire » chantée, un « soupir » écrit."""
    assert classify_rest(1.0) == "soupir"
    assert classify_rest(2.0) == "demi-pause"
    assert classify_rest(4.0) == "pause"
    assert classify_rest(0.5) == "demi-soupir"


def test_classify_rest_never_labels_a_triplet():
    """Mesuré sur chant.wav : une respiration de 1,17 temps sortait en
    « triolet de blanche » (13,5 % d'erreur) devant la noire (16,9 %), ce qui
    gravait un silence absurde. Un silence n'est jamais sous un crochet."""
    for beats in (0.33, 0.67, 1.17, 1.33, 2.67):
        fig = classify_rest(beats)
        assert fig is None or "triolet" not in fig, (beats, fig)


def test_classify_rest_rejects_a_non_positive_duration():
    assert classify_rest(0.0) is None
    assert classify_rest(-2.0) is None


def test_rest_figures_are_translatable():
    """Figures stockées en français (convention du cœur), traduites à
    l'affichage : « demi-pause » ne doit pas sortir en « demi-whole rest »."""
    from v2m.core.i18n import set_language, tr_figure

    set_language("en")
    try:
        assert tr_figure("demi-pause") == "half rest"
        assert tr_figure("quart de soupir") == "sixteenth rest"
    finally:
        set_language("fr")


# ----------------------------------------------------------------- détection
def test_lead_in_and_breaths_both_become_rests():
    notes = [_note(60, 0.41, 0.2), _note(62, 1.0, 0.2)]
    rests = detect_rests(notes, GRID)
    assert [(round(r.start, 2), round(r.end, 2)) for r in rests] == [
        (0.0, 0.41), (0.61, 1.0)
    ]


def test_the_tail_of_the_take_is_not_a_rest():
    """Une prise s'arrête où l'ingénieur a coupé, pas où la chanson finit."""
    rests = detect_rests([_note(60, 0.0, 0.5), _note(62, 1.0, 0.5)], GRID)
    assert max(r.end for r in rests) <= 1.0


def test_lead_in_can_be_excluded():
    notes = [_note(60, 0.41, 0.2), _note(62, 1.0, 0.2)]
    rests = detect_rests(notes, GRID, include_lead_in=False)
    assert [(round(r.start, 2), round(r.end, 2)) for r in rests] == [(0.61, 1.0)]


def test_a_seam_is_not_a_breath():
    """Sous le seuil, l'écart est une couture de segmentation (le
    remplisseur d'interpolation travaille dans cette zone)."""
    notes = [_note(60, 0.0, 0.5), _note(62, 0.53, 0.5)]
    assert detect_rests(notes, GRID, include_lead_in=False) == []
    assert MIN_REST_S > 0.03


def test_unordered_and_overlapping_notes_never_go_backwards():
    notes = [
        _note(62, 1.0, 0.5),
        _note(60, 0.0, 0.8),   # chevauche le suivant
        _note(64, 0.4, 0.4),
        _note(65, 2.0, 0.4),
    ]
    rests = detect_rests(notes, GRID)
    assert all(r.duration > 0 for r in rests)
    assert [r.start for r in rests] == sorted(r.start for r in rests)


def test_figure_follows_the_grid_tempo():
    notes = [_note(60, 0.0, 0.2), _note(62, 1.2, 0.2)]   # 1 s de respiration
    at_120 = detect_rests(notes, GRID, include_lead_in=False)[0]
    at_60 = detect_rests(notes, BeatGrid(60.0), include_lead_in=False)[0]
    assert at_120.figure == "demi-pause"   # 2 temps à 120
    assert at_60.figure == "soupir"        # 1 temps à 60


def test_without_a_grid_the_rests_stay_unlabeled():
    """Sans Grille on ne devine pas de figure : la Portée grave d'après la
    durée réelle, ce qui est honnête."""
    notes = [_note(60, 0.0, 0.2), _note(62, 1.2, 0.2)]
    rests = detect_rests(notes)
    assert rests and all(r.figure is None for r in rests)


# -------------------------------------------------------- Portée et MusicXML
def test_layout_engraves_a_labeled_breath_in_the_gap():
    notes = [_note(60, 0.0, 0.5, figure="noire"), _note(62, 1.0, 0.5, figure="noire")]
    rests = [Rest(start=0.5, end=1.0, figure="soupir")]
    lay = layout_staff(notes, grid=GRID, rests=rests)
    assert len(lay.rests) == 1
    assert (lay.rests[0].shape, lay.rests[0].onset) == ("soupir", 1.0)


def test_a_missing_tempo_never_raises_mid_engraving():
    """Garde-fou : la mise en page et l'export divisent tous deux par le tempo,
    un tempo absent ne doit pas faire lever une analyse déjà terminée."""
    from v2m.core.musicxml import build_score_xml

    notes = [_note(60, 0.0, 0.5, figure="noire"), _note(62, 1.0, 0.5, figure="noire")]
    for bpm in (None, 0):
        grid = BeatGrid(bpm)
        lay = layout_staff(notes, grid=grid, rests=[Rest(start=0.5, end=1.0)])
        assert len(lay.notes) == 2 and lay.total_beats > 0, bpm
        assert "<rest/>" in build_score_xml(notes, grid, None), bpm


def test_musicxml_writes_the_written_duration_not_a_wider_label():
    """Un <type> qui contredit la <duration> rend le fichier bancal : c'est la
    durée qui remplit la mesure, donc une demi-pause détectée dans un temps
    écrit reste une figure d'un temps."""
    from v2m.core.musicxml import build_score_xml

    notes = [_note(60, 0.0, 0.5), _note(62, 1.0, 0.5)]
    rests = [Rest(start=0.5, end=1.0, figure="demi-pause")]
    xml = build_score_xml(notes, BeatGrid(120.0), None, snap=False, rests=rests)
    assert "<rest/>" in xml and "<type>quarter</type>" in xml
    assert "<type>half</type>" not in xml


def test_musicxml_fuses_a_silent_run_into_a_real_figure():
    """Une respiration de deux temps s'écrit en UNE demi-pause de silence, pas
    en deux noires de remplissage."""
    from v2m.core.musicxml import build_score_xml

    notes = [_note(60, 0.0, 0.5), _note(62, 1.5, 0.5)]
    rests = detect_rests(notes, GRID, include_lead_in=False)
    assert rests and rests[0].figure == "demi-pause"
    xml = build_score_xml(notes, GRID, None, snap=False, rests=rests)
    assert "<type>half</type>" in xml
