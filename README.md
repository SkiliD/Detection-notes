# 🎤 Voix → MIDI

Application desktop qui **détecte les notes d'une voix**, le **tempo** et la **tonalité**, puis exporte un **fichier MIDI** (notes + tempo + armure) accompagné d'un **rapport JSON/CSV**. Fonctionne sur des **fichiers audio** (WAV/MP3/FLAC/OGG) ou **en temps réel au micro**.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)

## Fonctionnalités

- **Détection de hauteur** : pYIN (librosa) hors-ligne, YIN incrémental en temps réel
- **Segmentation en notes** : hystérésis avec référence glissante + confirmation 3 frames, résiste au vibrato et aux artefacts de transition
- **Tempo** : beat tracking librosa avec repli ×2 / ÷2 (range 60–180 BPM), puis **raffinement sur les attaques de notes** (grille par moindres carrés → BPM exact)
- **Tonalité** : profils Krumhansl-Schmuckler (24 candidatures majeur/mineur) + 2 candidates alternatives
- **Figures de notes** : chaque note est étiquetée (ronde → double-croche, pointées, triolets) selon le tempo détecté, avec tolérance sur l'interprétation humaine — champ `figure` dans le JSON et le CSV
- **Export MusicXML** : partition lisible (mesures 4/4, armure, tempo, figures, liaisons à travers les barres, triolets) ouvrable dans MuseScore, Finale, Sibelius…
- **Export MIDI** : meta-events `set_tempo` + `key_signature`, vélocité proportionnelle à la confiance, quantification 1/16 optionnelle, transposition ±48 demi-tons
- **Rapport JSON** (tout, y compris la courbe de hauteur) + **CSV** (liste de notes avec figure)
- **GUI PyQt6** : piano roll, waveform, panneau de résultats, drag & drop, **mode notation** (couleur par figure) + récapitulatif des figures
- **Mode temps réel** : indicateur note/Hz/niveau, piano roll défilant, export à l'arrêt

## Installation

Python ≥ 3.10 (3.12 recommandé) :

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate              # Windows
pip install -e ".[dev]"
```

## Utilisation

### GUI (recommandé)

```bash
v2m gui            # ou : python -m v2m gui
```

- **Analyser un fichier** : bouton ou glisser-déposer d'un WAV/MP3/FLAC dans la fenêtre
- **Chanter au micro** : mode temps réel, l'analyse finale (tempo/tonalité) se fait à l'arrêt
- Exports écrits à côté du fichier source (`.mid`, `.musicxml`, `.json`, `.csv`)
- **Mode notation** : case à cocher au-dessus du piano roll — chaque note est colorée selon sa figure (ronde, blanche, noire, croche…), étiquette courte affichée sur les notes assez larges

### Ligne de commande

```bash
# analyse complète avec quantification 1/16 selon le tempo détecté
v2m analyze chant.wav --quantize

# options
v2m analyze chant.wav --min-conf 0.7 --min-note 80 --transpose -12 -o sortie

# temps réel (Ctrl+C pour arrêter et exporter)
v2m live
```

La commande `analyze` écrit `sortie.mid`, `sortie.musicxml`, `sortie.json` et `sortie.csv` ; le JSON contient pour chaque note son champ `figure` (ex. `"noire"`, `"croche pointée"`) et un récapitulatif global `figures`.

### Bibliothèque

```python
from v2m.core.pipeline import analyze_file
from v2m.core.midi_export import build_midi
from v2m.core.musicxml import build_score_xml

res = analyze_file("chant.wav")
print(res.tempo.bpm, res.key.best.label, len(res.notes))
print(res.notes[0].figure)          # ex. "noire"
mid = build_midi(res.notes, bpm=res.tempo.bpm, key=res.key)
mid.save("out.mid")
xml = build_score_xml(res.notes, bpm=res.tempo.bpm, key=res.key, title="Ma prise")
open("out.musicxml", "w", encoding="utf-8").write(xml)
```

## Architecture

```
src/v2m/
├── cli.py               # sous-commandes analyze / live / gui
├── live.py              # session temps réel CLI
├── core/
│   ├── audio_io.py      # chargement fichier / capture micro (sounddevice)
│   ├── pitch.py         # pYIN (offline) + YIN par bloc (temps réel)
│   ├── notes.py         # nettoyage (médian + anti-vibrato) et segmentation
│   ├── tempo.py         # BPM + beats (librosa.beat, repli ×2/÷2)
│   ├── keys.py          # Krumhansl-Schmuckler, 24 candidatures
│   ├── liveseg.py       # segmenteur temps réel (hystérésis confirm/realease)
│   ├── midi_export.py   # écriture MIDI (tempo, armure, notes)
│   ├── report.py        # rapports JSON + CSV
│   └── pipeline.py      # orchestration analyse_signal / analyze_file
└── gui/
    ├── main_window.py   # fenêtre, threads, drag & drop, exports
    ├── workers.py       # workers temps réel (back-up hors thread GUI)
    └── widgets.py       # piano roll, waveform, panneau, indicateur live
```

## Qualité & limites

- **Voix monodique** (une note à la fois) — la transcription polyphonique est hors périmètre
- pYIN ≈ 0,3× la durée réelle du fichier
- Le tempo détecté est celui de la performance chantée (humaine) ; la quantification reste optionnelle
- En temps réel, la segmentation live (hystérésis) peut différer légèrement de l'analyse fine finale

## Tests

```bash
pytest tests/ -v
```

24 tests : pitch (YIN/pYIN sur sinus connus), segmentation (mélodies synthétiques), tempo (clics 90/120 BPM), tonalité (gammes Do majeur/La mineur), export MIDI (meta-events relus), segmenteur live, pipeline E2E WAV → MIDI/JSON/CSV.

## Générer l'audio de démonstration

```bash
python scripts/make_demo.py     # demo.wav : gamme de Do majeur à 120 BPM
v2m analyze demo.wav --quantize
```
