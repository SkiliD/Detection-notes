"""The lots manifest: which files travel together, and what proves each lot.

`scripts/lots.py` answers two questions that used to be hand-run commands —
which files belong to one commit, and whether that set is green on its own,
from a frozen export of HEAD. These pins hold the ANSWERS: every path a lot
names exists, every test has a runner chosen by its own suffix, the lots this
repository actually uses are in the manifest, and the command line refuses the
ways a lot would stop meaning anything (two lots in one index, a path that is
not there).
"""

from __future__ import annotations

import pytest

from scripts import lots


def test_every_lot_names_files_and_tests_that_exist():
    manifest = lots.load_lots()
    assert set(manifest) >= {"midi-writer", "gui-rings"}, "a lot this repository uses is gone"
    for name, lot in manifest.items():
        missing = lots.absent(lot)
        assert not missing, f"{name} names paths that are not on disk: {missing}"


def test_every_test_a_lot_names_has_its_own_runner():
    for name, lot in lots.load_lots().items():
        for test in lot["tests"]:
            assert lots.plan(test), f"{name}: {test} has no command"
    assert "pytest" in lots.plan("tests/test_midi.py")
    assert lots.plan("web/test/midi.test.mjs") == "node --test web/test/midi.test.mjs"
    with pytest.raises(lots.LotError):
        lots.plan("tests/data/parity.wav")


def test_the_manifest_prints_the_files_of_every_lot(capsys):
    assert lots.main(["--list"]) == 0
    printed = capsys.readouterr().out
    assert "midi-writer" in printed and "web/midi.js" in printed
    assert "gui-rings" in printed and "src/v2m/gui/widgets.py" in printed


def test_a_verification_exports_head_and_runs_the_lot_s_own_tests(capsys):
    assert lots.main(["--dry-run", "--lot", "gui-rings"]) == 0
    printed = capsys.readouterr().out
    assert "exporter HEAD" in printed, "a lot would be verified against the working tree"
    assert "tests/test_gui_craft_floor.py -q -n 0 -rs" in printed
    assert "node --test" not in printed, "this lot has no page test to run"


def test_two_lots_in_one_index_are_refused(capsys):
    assert lots.main(["--stage", "--lot", "midi-writer", "--lot", "gui-rings"]) == 2
    assert "UN lot" in capsys.readouterr().err


def test_a_lot_naming_a_path_that_is_not_there_is_refused(tmp_path, capsys):
    manifest = tmp_path / "lots.toml"
    manifest.write_text(
        '[lots.x]\nfiles = ["src/v2m/nowhere.py"]\ntests = ["tests/test_midi.py"]\n',
        encoding="utf-8",
    )
    assert lots.main(["--manifest", str(manifest), "--dry-run"]) == 2
    assert "absents" in capsys.readouterr().err
