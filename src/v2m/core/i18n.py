"""User-facing language selection (French / English).

`set_language("fr" | "en")` switches every user-visible string at runtime via
`tr()`, including the note-name format (english A B C vs french Do Ré Mi).
Code identifiers and the figure names stored on notes stay canonical French
internally (`noire`, `triolet de croche`...) — `tr_figure` translates them
for display only.
"""

from __future__ import annotations

from pathlib import Path

from .music import midi_to_name, midi_to_name_fr

LANG_FILE = Path.home() / ".v2m_lang"

SUPPORTED_LANGUAGES = ("fr", "en")

# User-facing strings. English defaults to the key itself ("key tempo" -> "Tempo").
STRINGS = {
    "fr": {
        "key tempo": "Tempo",
        "key duration": "Durée",
        "key meter": "Mesure",
        "key key": "Tonalité",
        "key key score": "Score tonalité",
        "key alternatives": "Alternatives",
        "key chord": "Accord",
        "key notes count": "Nb de notes",
        "key measures count": "Nb de mesures",
        "key figures": "Figures",
        "key source": "Source",
        "key results title": "Résultats de l'analyse",
        "key notes table": "Notes",
        "key chord segments": "Accords par segment",
        "key time col": "Temps",
        "key note col": "Note",
        "key figure col": "Figure",
        "key span col": "Intervalle",
        "key default meter": "4/4 (défaut)",
        "key level": "niveau",
        "key detect meter": "non détectée (4/4 par défaut)",
        "key chord label": "Accord   ",
        "key key label": "Tonalité ",
        "key notes label": "Notes    ",
        "key export label": "Export   ",
        "key final analysis": "Analyse finale (tempo, tonalité, accords, notes)…",
        "key candidate": "candidat",
        "cli analyzing": "Analyse de {path} …",
        "cli interrupted": "Interrompu.",
        "cli stop capture": "\nArrêt de la capture.",
        "cli too short": "Enregistrement trop court : rien à exporter.",
        "cli live banner": "🎤 Mode temps réel — chantez ! (Ctrl+C pour arrêter et exporter)",
        "err file not found": "Fichier introuvable : {path}",
        "err bad format": "Format non pris en charge : {ext} (attendu : {fmts})",
        "err minconf": "--min-conf doit être dans [0, 1] (reçu : {v})",
        "err analyze failed": "Erreur pendant l'analyse : {e}",
        "err live failed": "Erreur en mode live : {e}",
        "err empty file": "Fichier audio vide (0 échantillon) : {name}",
        "err undecodable": "Impossible de décoder « {name} » : {e}",
        "err missing file": "Fichier introuvable : {path}",
        "err ui unsupported format": "Format non pris en charge : « {ext} ». Formats acceptés : {fmts}.",
        "err ui unreadable": "Fichier illisible : {e}",
        "err ui empty": "Fichier vide : {name}",
        "err ui missing": "Fichier introuvable : {path}",
        "err ui too short": "Enregistrement trop court (0,5 s minimum) pour être analysé.",
        "err ui export failed": "Export impossible : {e}",
        "err ui pdf": "Export PDF : {e}",
        "err ui mic": "Micro indisponible : {e}",
        "ui window title": "Voix → MIDI — notes, tempo, tonalité et accords",
        "ui app subtitle": "Chantez : l'application détecte les notes, le tempo, la tonalité et les accords,\npuis exporte un fichier MIDI, une partition MusicXML et un rapport JSON/CSV.",
        "ui open file": "📁  Analyser un fichier audio",
        "ui sing live": "🎤  Chanter au micro (temps réel)",
        "ui finder button": "🎹  Trouver la note que je chante",
        "ui finder start": "▶  Écouter le micro",
        "ui finder stop": "■  Arrêter d'écouter",
        "ui finder idle": "♪  Chantez une note…",
        "ui finder live": "●  Micro ouvert — chantez une note",
        "ui finder off": "Micro fermé",
        "ui finder hint": "Chantez une note : sa touche s'allume sur le clavier du piano.\nRien d'autre — pas d'analyse, pas d'export, rien n'est conservé.",
        "ui axis note": "Note",
        "ui axis time": "Temps",
        "ui drop hint": "Astuce : glissez-déposez un fichier audio dans cette fenêtre.",
        "ui lang button": "🌐  Langue : Français",
        "ui back home": "← Accueil",
        "ui no file": "Aucun fichier",
        "ui export options": "Options d'export",
        "ui quantize": "Quantifier sur la grille 1/16",
        "ui transpose": "Transposition (demi-tons)",
        "ui notation mode": "Mode notation (figures colorées)",
        "ui pdf option": "Exporter aussi en PDF",
        "ui pdf tooltip": "Installez MuseScore ou LilyPond pour activer l'export PDF.",
        "ui re-export": "💾  Ré-exporter avec ces options",
        "ui open another": "📁  Ouvrir un autre fichier",
        "ui stop live": "← Arrêter / Accueil",
        "ui mic preparing": "Préparation du micro…",
        "ui live progress": "🎤 {t} · {notes} notes",
        "ui staff tooltip": "Portée (clé de sol) : les notes détectées, écrites en notation musicale (figures, altérations, mesures).",
        "key bpm unit": "BPM",
        "ui stopping": "Arrêt…",
        "ui analyzing": "Analyse finale…",
        "ui live file": "Enregistrement live",
        "ui choose file": "Choisir un fichier audio",
        "ui audio filter": "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;Tous les fichiers (*)",
        "ui status exported": "Exporté : {files}",
        "ui status live done": "Tempo {bpm:.0f} BPM · {key} · {n} notes · fichiers exportés dans le dossier courant",
        "ui play tooltip": "Écouter l'analyse (synthé local) — double-clic sur le piano roll pour démarrer ailleurs",
        "ui play pos": "prêt",
        "ui steps": "1 · Chargez un fichier ou chantez au micro\n2 · L'analyse détecte notes, tempo, tonalité et accords\n3 · Écoutez ▶ puis exportez MIDI · partition · PDF",
        "ui pdf badge": "PDF indisponible — installez MuseScore ou LilyPond",
        "ui exported ok": "✓ Exporté",
        "ui analyzing file": "Analyse…",
        "ui panel waiting": "En attente d'une analyse —\nchargez un fichier ou chantez au micro.",
    },
    "en": {
        "key tempo": "Tempo",
        "key duration": "Duration",
        "key meter": "Meter",
        "key key": "Key",
        "key key score": "Key score",
        "key alternatives": "Alternatives",
        "key chord": "Chord",
        "key notes count": "Note count",
        "key measures count": "Bars",
        "key figures": "Figures",
        "key source": "Source",
        "key results title": "Analysis results",
        "key notes table": "Notes",
        "key chord segments": "Chords per segment",
        "key time col": "Time",
        "key note col": "Note",
        "key figure col": "Figure",
        "key span col": "Span",
        "key default meter": "4/4 (default)",
        "key level": "level",
        "key detect meter": "not detected (4/4 default)",
        "key chord label": "Chord    ",
        "key key label": "Key      ",
        "key notes label": "Notes    ",
        "key export label": "Export   ",
        "key final analysis": "Final analysis (tempo, key, chords, notes)…",
        "key candidate": "candidate",
        "cli analyzing": "Analyzing {path} ...",
        "cli interrupted": "Interrupted.",
        "cli stop capture": "\nCapture stopped.",
        "cli too short": "Recording too short: nothing to export.",
        "cli live banner": "🎤 Real-time mode — sing! (Ctrl+C to stop and export)",
        "err file not found": "File not found: {path}",
        "err bad format": "Unsupported format: {ext} (expected: {fmts})",
        "err minconf": "--min-conf must be in [0, 1] (got: {v})",
        "err analyze failed": "Analysis error: {e}",
        "err live failed": "Live mode error: {e}",
        "err empty file": "Empty audio file (0 samples): {name}",
        "err undecodable": "Could not decode '{name}': {e}",
        "err missing file": "File not found: {path}",
        "err ui unsupported format": "Unsupported format: '{ext}'. Accepted: {fmts}.",
        "err ui unreadable": "Unreadable file: {e}",
        "err ui empty": "Empty file: {name}",
        "err ui missing": "File not found: {path}",
        "err ui too short": "Recording too short (0.5 s minimum) to be analyzed.",
        "err ui export failed": "Export failed: {e}",
        "err ui pdf": "PDF export: {e}",
        "err ui mic": "Microphone unavailable: {e}",
        "ui window title": "Voix → MIDI — notes, tempo, key and chords",
        "ui app subtitle": "Sing: the app detects notes, tempo, key and chords,\nthen exports a MIDI file, a MusicXML score and a JSON/CSV report.",
        "ui open file": "📁  Analyze an audio file",
        "ui sing live": "🎤  Sing into the mic (real-time)",
        "ui finder button": "🎹  Find the note I am singing",
        "ui finder start": "▶  Listen to the mic",
        "ui finder stop": "■  Stop listening",
        "ui finder idle": "♪  Sing a note…",
        "ui finder live": "●  Mic open — sing a note",
        "ui finder off": "Mic closed",
        "ui finder hint": "Sing a note: its key lights up on the piano keyboard.\nNothing else — no analysis, no export, nothing is kept.",
        "ui axis note": "Note",
        "ui axis time": "Time",
        "ui drop hint": "Tip: drag & drop an audio file onto this window.",
        "ui lang button": "🌐  Language: English",
        "ui back home": "← Home",
        "ui no file": "No file",
        "ui export options": "Export options",
        "ui quantize": "Quantize to the 1/16 grid",
        "ui transpose": "Transpose (semitones)",
        "ui notation mode": "Notation mode (colored figures)",
        "ui pdf option": "Also export as PDF",
        "ui pdf tooltip": "Install MuseScore or LilyPond to enable PDF export.",
        "ui re-export": "💾  Re-export with these options",
        "ui open another": "📁  Open another file",
        "ui stop live": "← Stop / Home",
        "ui mic preparing": "Preparing the microphone…",
        "ui live progress": "🎤 {t} · {notes} notes",
        "ui staff tooltip": "Staff (treble clef): detected notes written in musical notation (figures, accidentals, bars).",
        "key bpm unit": "BPM",
        "ui stopping": "Stopping…",
        "ui analyzing": "Final analysis…",
        "ui live file": "Live recording",
        "ui choose file": "Choose an audio file",
        "ui audio filter": "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;All files (*)",
        "ui status exported": "Exported: {files}",
        "ui status live done": "{bpm:.0f} BPM · {key} · {n} notes · files exported to the current folder",
        "ui play tooltip": "Play the analysis (local synth) — double-click the piano roll to start elsewhere",
        "ui play pos": "ready",
        "ui steps": "1 · Load a file or sing into the mic\n2 · The analysis detects notes, tempo, key and chords\n3 · Listen ▶ then export MIDI · score · PDF",
        "ui pdf badge": "PDF unavailable — install MuseScore or LilyPond",
        "ui exported ok": "✓ Exported",
        "ui analyzing file": "Analyzing…",
        "ui panel waiting": "Waiting for an analysis —\nload a file or sing into the mic.",
    },
}

_current = "fr"


def available_languages() -> tuple[str, ...]:
    return SUPPORTED_LANGUAGES


def set_language(lang: str) -> None:
    """Switch the whole UI language ('fr' or 'en'); persists the choice."""
    global _current
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Langue non supportée : {lang} (choix : {SUPPORTED_LANGUAGES})")
    _current = lang
    try:
        LANG_FILE.write_text(lang, encoding="utf-8")
    except OSError:
        pass  # persistence is best-effort; the runtime switch still works


def get_language() -> str:
    """Current UI language ('fr' by default, or restored from disk)."""
    global _current
    if not LANG_FILE.exists():
        return _current
    try:
        stored = LANG_FILE.read_text(encoding="utf-8").strip()
        if stored in SUPPORTED_LANGUAGES:
            _current = stored
    except OSError:
        pass
    return _current


def get_language_display() -> str:
    return "Français" if get_language() == "fr" else "English"


def toggle_language() -> str:
    set_language("en" if get_language() == "fr" else "fr")
    return get_language()


def tr(key: str) -> str:
    """Translate a user-facing string in the current language."""
    return STRINGS[_current].get(key, key)


def note_display(midi: int) -> str:
    """Note name in the current language's format: 'La4' or 'A4'."""
    if _current == "en":
        return midi_to_name(midi)
    return midi_to_name_fr(midi)


def note_display_pair(midi: int) -> str:
    """Both notations, current first: 'La4 / A4' or 'A4 / La4'."""
    if _current == "en":
        return f"{midi_to_name(midi)} / {midi_to_name_fr(midi)}"
    return f"{midi_to_name_fr(midi)} / {midi_to_name(midi)}"


def tr_figure(figure: str | None) -> str:
    """Display a canonical French figure name ('noire') in the UI language."""
    if not figure:
        return ""
    if _current == "fr":
        return figure
    en = {
        "ronde": "whole",
        "blanche": "half",
        "noire": "quarter",
        "croche": "eighth",
        "double-croche": "sixteenth",
        "pointée": "dotted",
        "triolet de ": "triplet of ",
    }
    # longest first: "demi-pause" contains "pause", "quart de soupir"
    # contains "soupir" — a naive order would half-translate the label.
    rest_en = [
        ("quart de soupir", "sixteenth rest"),
        ("demi-soupir", "eighth rest"),
        ("soupir", "quarter rest"),
        ("demi-pause", "half rest"),
        ("pause", "whole rest"),
    ]
    out = figure
    for fr_word, en_word in en.items():
        out = out.replace(fr_word, en_word)
    for fr_word, en_word in rest_en:
        out = out.replace(fr_word, en_word)
    return out


def tr_key_label(label: str) -> str:
    """Translate a key label ('C major') into the UI language."""
    tonic, _, mode = label.partition(" ")
    if _current == "en":
        return label
    fr_mode = {"major": "majeur", "minor": "mineur"}.get(mode, mode)
    return f"{tonic} {fr_mode}".strip()


def tr_chord_quality(quality: str) -> str:
    """Optional French expansion of chord qualities ('m' -> 'mineur')."""
    if _current == "en" or not quality:
        return quality
    fr = {"m": "mineur", "dim": "dim", "aug": "aug", "7": "7", "maj7": "maj7", "m7": "m7"}
    return fr.get(quality, quality)


__all__ = [
    "available_languages",
    "get_language",
    "get_language_display",
    "note_display",
    "note_display_pair",
    "set_language",
    "toggle_language",
    "tr",
    "tr_chord_quality",
    "tr_figure",
    "tr_key_label",
]
