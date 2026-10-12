"""Shared assertions for the export contract — the module and both adapters.

The receipt `core.take_export` returns is the single source of "what was
written"; these helpers pin the three facts every caller must agree on, so
the module test, the GUI adapter test and the CLI adapter test read the SAME
rules instead of re-spelling them:

1. **the announcement names EXACTLY the stem's files on disk** — a
   half-written export cannot read as complete, and a conditional file that
   was not written is not announced;
2. **a missing/failed PDF is never announced and never on disk**;
3. **the write ORDER is the take's WAV (when asked), the artifacts, then the
   PDF** — the rawest data first, the external tool last.

The stem's FOLDER must hold only the take's own files for the prefix match
below to be meaningful: the callers export under a dedicated directory.
"""

from __future__ import annotations

from pathlib import Path


def suffixes_on_disk(stem) -> set[str]:
    """The suffixes of the files sharing the stem's prefix, in its folder."""
    stem = Path(stem)
    prefix = stem.name
    return {
        p.name[len(prefix):]
        for p in stem.parent.iterdir()
        if p.is_file() and p.name.startswith(prefix)
    }


def assert_announcement_matches_disk(announced, stem) -> None:
    """`announced` names exactly the stem's files on disk — no more, no less.

    Names may be base names (the GUI status line) or full paths (the CLI
    summary): only the file name is compared.
    """
    stem = Path(stem)
    expected = {f"{stem.name}{suffix}" for suffix in suffixes_on_disk(stem)}
    got = {Path(name).name for name in announced}
    assert got == expected, f"announced {sorted(got)} != disk {sorted(expected)}"
    for name in announced:
        assert (stem.parent / Path(name).name).exists(), f"announced but absent: {name}"


def assert_no_pdf(announced, stem) -> None:
    """A skipped or failed PDF is neither announced nor on disk."""
    stem = Path(stem)
    assert f"{stem.name}.pdf" not in {Path(n).name for n in announced}
    assert not Path(f"{stem}.pdf").exists()


def instrument_write_order(monkeypatch, module) -> list[str]:
    """Record the export module's writer order: 'wav', 'artifacts', 'pdf'.

    The three writers are patched on the MODULE (its names are the seam both
    adapters go through), so the same instrumentation observes the GUI and the
    CLI without either adapter knowing about it. Each wrapper CALLS THROUGH to
    the real writer, so the artifacts still reach disk and an adapter that
    renders the real receipt (the CLI summary) still works; the PDF wrapper
    creates its file so a written PDF is announced too.
    """
    order: list[str] = []
    real_save = module.audio_io.save_file
    real_export = module.export_result

    def record(name: str, fn):
        def _call(*args, **kwargs):
            order.append(name)
            return fn(*args, **kwargs)

        return _call

    def fake_pdf(stem) -> str:
        order.append("pdf")
        Path(f"{stem}.pdf").write_bytes(b"%PDF-1.4")
        return str(stem)

    monkeypatch.setattr(module.audio_io, "save_file", record("wav", real_save))
    monkeypatch.setattr(module, "export_result", record("artifacts", real_export))
    monkeypatch.setattr(module, "engrave_pdf", fake_pdf)
    return order
