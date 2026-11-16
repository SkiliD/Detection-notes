"""Tests for French/English note naming and the user-facing error types."""

import pytest

from v2m.core.audio_io import AudioLoadError, load_file
from v2m.core.music import (
    NOTE_NAMES,
    NOTE_NAMES_FR,
    PIANO_MAX_MIDI,
    PIANO_MIN_MIDI,
    format_note_fr,
    midi_to_name,
    midi_to_name_fr,
    on_piano,
)


def test_only_the_88_keys_of_a_classical_piano_are_notes():
    """A0 (21) to C8 (108): nothing outside these keys is a piano note."""
    assert (PIANO_MIN_MIDI, PIANO_MAX_MIDI) == (21, 108)
    assert len(range(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1)) == 88
    assert on_piano(21) and on_piano(108)  # both extreme keys
    assert on_piano(21.4) and on_piano(107.9)  # a pitch between two keys
    assert not on_piano(20.9) and not on_piano(108.1)  # just outside
    # undecidable is not a key either: the guarantee, not an approximation
    assert not on_piano(float("nan")) and not on_piano(None) and not on_piano("so")


def test_note_names_are_bilingual_aligned_and_stable():
    """Both tables cover the 88 keys with matching octaves, and the
    well-known reference spellings never move."""
    # English reference points
    assert midi_to_name(69) == "A4"
    assert midi_to_name(60) == "C4"
    assert midi_to_name(61) == "C#4"
    assert midi_to_name(21) == "A0"
    assert midi_to_name(108) == "C8"
    # the two tables are aligned, note by note
    assert len(NOTE_NAMES) == len(NOTE_NAMES_FR) == 12
    for midi in range(21, 109):
        en = midi_to_name(midi)
        fr = midi_to_name_fr(midi)
        # English letter matches the French initial (C/Do, D/Re, E/Mi...)
        assert en[0] in "CDEFGAB"
        assert fr[0] in "CDMFLRS"  # Do Ré Mi Fa Sol La Si initials
        assert fr[-1] == en[-1]  # the octave numbering agrees
    # French reference points
    assert midi_to_name_fr(69) == "La4"
    assert midi_to_name_fr(60) == "Do4"
    assert midi_to_name_fr(61) == "Do#4"
    assert midi_to_name_fr(62) == "Ré4"
    assert midi_to_name_fr(64) == "Mi4"
    assert midi_to_name_fr(65) == "Fa4"
    assert midi_to_name_fr(67) == "Sol4"
    assert midi_to_name_fr(71) == "Si4"
    # the combined display form shows both spellings
    assert format_note_fr(69) == "La4 / A4"
    assert format_note_fr(60) == "Do4 / C4"


def test_load_file_reports_missing_empty_and_undecodable_audio(tmp_path):
    """Three bad inputs, one contract: a typed AudioLoadError, never a guess."""
    with pytest.raises(AudioLoadError, match="introuvable"):
        load_file(str(tmp_path / "nope.wav"))

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioLoadError):
        load_file(str(empty))

    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"this is not a wav file" * 10)
    with pytest.raises(AudioLoadError, match="décoder"):
        load_file(str(junk))
