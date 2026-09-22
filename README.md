# 🎤 Voix → MIDI

Application desktop qui **détecte les notes d'une voix** (hauteur, figures, mesure), le **tempo**, la **tonalité** et les **accords**, puis exporte un **fichier MIDI**, une **partition MusicXML** (PDF optionnel) et un **rapport JSON/CSV**. Fonctionne sur des **fichiers audio** (WAV/MP3/FLAC/OGG) ou **en temps réel au micro**.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)

## Fonctionnalités

- **Détection de hauteur** : pYIN (librosa) hors-ligne, YIN incrémental en temps réel
- **Segmentation en notes** : hystérésis avec référence glissante + confirmation 3 frames, **guidée par les attaques** (consonnes, coups de glotte, changements de voyelle) : un glissando legato n'est pas une note, et les micro-notes qu'il fabriquait (6 notes chantées → 10 détectées) sont absorbées dans la note d'arrivée, à sa hauteur. Résiste au vibrato (une tenue avec vibrato déclenchait 19 « attaques »)
- **Tempo** : beat tracking librosa avec repli ×2 / ÷2 (range 60–180 BPM), puis **raffinement sur les attaques de notes** (grille par moindres carrés → BPM exact)
- **Tonalité** : profils Krumhansl-Schmuckler (24 candidatures majeur/mineur) + 2 candidates alternatives
- **Accords** : reconnaissance globale, par segments jointifs de 2 s et **par mesure** (majeure, mineure, dim, aug, 7, maj7, m7) via histogramme pondéré — affichés dans la GUI et le JSON
- **Progression d'accords** : un accord par **mesure** (aligné sur la mesure détectée, mesures identiques fusionnées) — se lit comme une grille : `Dm7 | Cmaj7 | G7 | C`
- **Noms de notes bilingues** : anglaise (`A4`) **et** française (`La4`) côte à côte dans la GUI, le JSON (`name` / `name_fr`) et le CSV
- **Interface fr / en** : langue choisie dans la GUI (bouton 🌐) ou `v2m --lang en` — traduit tous les libellés **et** le format des notes (`Do Ré Mi` ↔ `A B C`), choix persisté dans `~/.v2m_lang`
- **Figures de notes** : chaque note est étiquetée (ronde → double-croche, pointées, triolets) selon le tempo détecté, avec tolérance sur l'interprétation humaine — champ `figure` dans le JSON et le CSV, **aussi en temps réel** sur le piano roll du mode micro
- **Portée (notation)** : les notes et silences écrits sur une **vraie portée** — clé de sol, armure, chiffrage, figures (ronde → double-croche, pointées, triolets), soupirs et pauses avec hampes, crochets, points et altérations, lignes supplémentaires, barres de mesure et cartouche tempo · mesure · tonalité. Affichée **après l'analyse d'un fichier** et **en direct pendant la capture** (portée qui défile, la note en cours étant surlignée). Gravure vectorielle intégrée : aucun logiciel externe requis
- **Mesure** : détection automatique — 3/4 vs 4/4 par les **accents d'attaque** (forces relatives RMS des notes, contrast requis), 6/8 par la grille ternaire des contretemps ; sans accent clair, 4/4 par défaut assumé. La partition et le rapport utilisent la mesure détectée
- **Export MusicXML** : partition lisible (mesure détectée, armure, tempo, figures, liaisons à travers les barres, triolets) ouvrable dans MuseScore, Finale, Sibelius…
- **Toutes les détections sur la partition** : accords par mesure en **chiffrage harmonique** au-dessus de la portée, noms de notes en **sol-fège** (Do, Ré, Mi…) sous chaque note, **têtes de note colorées selon la confiance** (vert = sûre → rouge = douteuse), cartouche récapitulatif (tempo · tonalité · mesure · accords) dans le titre
- **Export MIDI** : meta-events `set_tempo` + `key_signature`, vélocité proportionnelle à la confiance, quantification 1/16 optionnelle, transposition ±48 demi-tons
- **Export PDF** : gravure de la partition via MuseScore ou LilyPond (détection automatique, message d'installation sinon) — `--pdf` en CLI, case à cocher en GUI
- **Rapport JSON** (tout, y compris la courbe de hauteur, la mesure, le nombre de mesures, la position mesure:temps de chaque note, l'accord global et les segments d'accords) + **CSV** (liste de notes avec figure et position)
- **GUI PyQt6** : interface sombre soignée, piano roll, waveform, panneau de résultats avec tableaux par note (position, figure) et par segment d'accord, drag & drop, **mode notation** (couleur par figure), **bannière d'erreur rouge intégrée** (pas de popups bloquants)
- **Mode temps réel** : indicateur note/Hz/niveau, **piano roll qui suit la capture** (fenêtre glissante de 10 s, curseur de temps, horloge `mm:ss`, compteur de notes et **mesure détectée** — 3/4, 4/4, 6/8 — affichée en direct), comparaison avec une partition déjà analysée (touche verte/rouge), la note tenue s'affiche **pendant** qu'on la chante, figures étiquetées en direct, export à l'arrêt — le suivi de hauteur tourne **hors du callback audio** (thread dédié), donc le pilote ne perd pas de blocs
- **Trouveur de note** : reconnaissance isolée au micro, clavier de 88 touches et historique local des six dernières notes ; aucune analyse ni export
- **Écoute de contrôle** : bouton ▶ (synthé local intégré, aucun logiciel externe) avec **curseur de lecture synchronisé** sur le piano roll et la waveform ; **double-clic** sur le piano roll pour démarrer la lecture à cet endroit

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
- **Voir la partition** : sous le piano roll, la portée affiche les notes réellement détectées (avec leurs figures) — elle s'adapte à la taille de la fenêtre et découpe la musique en systèmes avec numéros de mesure
- **Chanter au micro** : piano roll qui défile tout seul — le curseur de temps avance avec la capture, la barre de la note en cours grandit pendant qu'on la tient, et l'étiquette de figure apparaît dès que la note se termine ; **la portée défile en dessous** et écrit la mélodie au fur et à mesure ; à l'arrêt, analyse finale (tempo, tonalité, mesure, accords) et export
- Exports écrits à côté du fichier source (`.mid`, `.musicxml`, `.json`, `.csv`)
- **Mode notation** : case à cocher au-dessus du piano roll — chaque note est colorée selon sa figure (ronde, blanche, noire, croche…), étiquette courte affichée sur les notes assez larges
- **Écoute de contrôle** : bouton ▶ + curseur synchronisé ; un **clavier de piano** sous le piano roll allume les touches des notes qui sonnent pendant la lecture

### Ligne de commande

```bash
# analyse complète avec quantification 1/16 selon le tempo détecté
v2m analyze chant.wav --quantize

# interface et noms de notes en anglais (fr par défaut ; choix persisté)
v2m --lang en analyze chant.wav --quantize

# options
v2m analyze chant.wav --min-conf 0.7 --min-note 80 --transpose -12 -o sortie

# temps réel (Ctrl+C pour arrêter et exporter)
v2m live
```

La commande `analyze` écrit `sortie.mid`, `sortie.musicxml`, `sortie.json` et `sortie.csv` (`--pdf` ajoute `sortie.pdf` si MuseScore ou LilyPond est installé — sinon le PDF est ignoré avec un message d'installation et le code de sortie reste 0 ; si la gravure échoue, exit 2 et les quatre autres artefacts restent écrits) ; le JSON contient pour chaque note son champ `figure` (ex. `"noire"`, `"croche pointée"`), sa `position` `[mesure, temps]`, ses noms `name` / `name_fr` (anglaise et française), la `meter` détectée, `num_measures`, l'accord global `chord` et ses `chord_segments`.

```bash
# avec export PDF de la partition (best-effort : message clair si aucun outil)
v2m analyze chant.wav --quantize --pdf
```

La partition MusicXML embarque **toutes les détections** : chiffrages d'accords au-dessus de la portée (progression par mesure), noms sol-fège (Do, Ré, Mi…) sous chaque note, têtes de note colorées selon la confiance (vert = sûre → rouge = douteuse), cartouche récapitulatif dans le titre. Tout s'ouvre dans MuseScore, Finale ou Sibelius.

### Bibliothèque

```python
from v2m.core.pipeline import analyze_file
from v2m.core.report import export_result

res = analyze_file("chant.wav")
print(res.tempo.bpm, res.key.best.label, len(res.notes))
print(res.notes[0].figure)          # ex. "noire"
if res.chord:
    print(res.chord.best.label)     # ex. "Cmaj7" (None si indéterminé)

rep = export_result(res, "out")     # écrit out.mid/.musicxml/.json/.csv

# en plus fin, pour construire un artefact soi-même :
from v2m.core.midi_export import build_midi
mid = build_midi(res.notes, bpm=res.tempo.bpm, key=res.key)
mid.save("out.mid")
```

## Architecture

```
src/v2m/
├── cli.py               # sous-commandes analyze / live / gui
├── live.py              # session temps réel CLI
├── core/
│   ├── audio_io.py      # chargement fichier / capture micro (sounddevice)
│   ├── pitch.py         # pYIN (offline) + YIN par bloc (temps réel)
│   ├── notes.py         # nettoyage (médian + anti-vibrato) et segmentation (attaques, transitions)
│   ├── onsets.py        # attaques : flux spectral hors ligne, niveau en temps réel
│   ├── tempo.py         # BPM + beats (librosa.beat, repli ×2/÷2)
│   ├── keys.py          # Krumhansl-Schmuckler, 24 candidatures
│   ├── chords.py        # accords (triades + septièmes, global, segments et progression par mesure)│   ├── figures.py         # classification des figures ; catalogue dans grid.py│   ├── meter.py           # détection de mesure (accents 3/4 vs 4/4, ternaire 6/8)
│   ├── grid.py            # BeatGrid, Meter, Figure/CATALOGUE et divisions MusicXML

│   ├── playback.py      # rendu audio des notes + lecteur pour l'écoute de contrôle
│   ├── liveseg.py       # temps réel : chaîne blocs → hauteur → attaques → notes (LivePitchStream)
│   ├── notation.py      # portée : orthographe des hauteurs, armures, durées écrites, mise en page
│   ├── midi_export.py   # écriture MIDI (tempo, armure, notes)
│   ├── musicxml.py      # partition MusicXML (mesure détectée, armure, figures, ties,
│   │                    #   accords <harmony>, paroles sol-fège, couleur de confiance)
│   ├── pdf_export.py    # PDF via MuseScore/LilyPond (détection auto)
│   ├── report.py        # export_result : les artefacts + le rapport JSON unifié
│   ├── live_session.py  # LiveTake : une prise micro de bout en bout (capture → analyse)
│   └── pipeline.py      # orchestration analyse_signal / analyze_file
└── gui/
    ├── main_window.py   # état, navigation, callbacks, threads et exports
    ├── pages.py         # assemblage des pages et branchement des actions
    ├── staff.py         # portée dessinée au QPainter (clé, armure, notes, silences, barres)
    └── widgets.py       # piano roll, waveform, panneau, indicateur live

scripts/
├── gui_smoke_test.py    # parcours GUI hors écran (fenêtre, panneau, langue, clavier)
├── profile_analysis.py  # diagnostic : temps par étape d'analyse + cProfile
└── score_accuracy.py    # diagnostic : ce que la transcription a fait d'une prise réelle

tests/
└── voicetruth.py        # vérité terrain : voix de synthèse à notes connues + mesure d'écart
```

## Diagnostic (temps d'analyse)

```bash
# tableau des étapes : où passe le temps sur CETTE prise ?
./.venv/Scripts/python.exe scripts/profile_analysis.py chant.wav

# moyenne de 3 exécutions + fonctions les plus lourdes (cProfile) + dump JSON
./.venv/Scripts/python.exe scripts/profile_analysis.py chant.wav --repeat 3 --profile --json mesures.json
```

Les étapes ne sont pas recopiées dans le script : il lit celles que le pipeline annonce déjà (`progress`), donc une étape renommée ou ajoutée apparaît ici toute seule. Deux colonnes à regarder : **Part** (poids de l'étape) et **Temps réel** (durée de l'étape / durée de l'audio — 0,037× = 27× plus rapide que la lecture, le seuil à surveiller pour une longue prise). Un passage à blanc de 2 s est écarté par défaut (`--no-warmup` pour le mesurer) : sans lui, les imports librosa et la compilation numba atterrissent sur la première étape et masquent le vrai coût. `--profile` ajoute un tableau cProfile séparant **le code du projet** (ce qu'on peut optimiser) des **bibliothèques** (où va le temps appelé).

Sur `demo.wav` (8,5 s) le décodage domine ; sur une prise de 68 s (répétée 8×) pYIN pèse 53 % et `refine_bpm_with_onsets` 16 % — c'est ce dernier que le tableau montre en tête côté code du projet. Le calcul des attaques (un STFT de plus) coûte 0,12 s sur ces 68 s, soit 0,2 % du temps réel.

## Diagnostic (qualité de transcription)

```bash
# ce que la transcription a fait de CETTE prise : notes, figures, positions…
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav

# les signes de découpage douteux, et la comparaison attaques vs géométrie seule
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --json rapport.json --no-attacks

# comparaison chiffrée à des notes connues (vérité terrain, même harnais que les tests)
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --truth chant_verite.json
```

Sur une prise réelle il n'y a pas de vérité terrain : le script imprime donc d'abord ce que le pipeline a trouvé (tempo, mesure, tonalité, accords, chaque note avec sa figure et sa position), puis les **signes de découpage douteux** : attaques orphelines (une consonne sans début de note — une note est peut-être manquée), débuts non articulés (la note existe mais aucune attaque ne la date), fragments (notes plus courtes que la confirmation de 120 ms, typiques d'un glissando coupé en deux). Si `--no-attacks` donne le même nombre de notes, la preuve d'attaque ne sert à rien sur cette prise (pas de consonnes exploitables). La vérité terrain est un simple JSON `{"notes": [{"midi": 60, "start_s": 0.3, "dur_s": 0.6}, …]}`.

## Documentation projet

- **`CONTEXT.md`** — glossaire du domaine (Analyse, Note, Tempo, Tonalité, Accord, Figure, Mesure, Export, Rapport) et conventions (langues, erreurs) : la référence pour toute évolution du code.
- **`core/i18n.py`** — catalogue des chaînes utilisateur (fr/en) : toute nouvelle chaîne visible y passe par `tr()` ; les noms de notes suivent la langue (`note_display`), les figures stockées restent canoniques (`tr_figure` traduit à l'affichage).
- **`AGENTS.md`** — apprentissages techniques et **règles de session** (non-régression, mise à jour des docs, lisibilité) pour les agents outillés ; lu automatiquement si l'option est activée dans le client.

## Qualité & limites

- **Voix monodique** (une note à la fois) — la transcription polyphonique est hors périmètre
- **Accords sur voix seule** : potentiellement ambigus (une mélodie suggère plusieurs harmonisations) — le rapport fournit des alternatives ; moins de deux hauteurs distinctes → pas d'accord
- pYIN ≈ 0,3× la durée réelle du fichier
- **Portée intégrée** : gravure maison (pas MuseScore) — monodique, clé de sol, notes sans ligatures (crochets pour les croches) et avec silences ; le PDF exporté reste la version gravée « professionnelle »
- Le tempo détecté est celui de la performance chantée (humaine) ; la quantification reste optionnelle
- **Transcription validée sur voix de synthèse** (`tests/voicetruth.py`, 19 cas : legato, détaché, consonnes, vibrato, bruit, voix graves/aiguës, croches/doubles-croches, portamentos) : hors ligne la segmentation est **exacte** — toutes les notes chantées retrouvées, aucune inventée. Le harnais est synthétique : une vraie prise bruitée reste à mesurer (`scripts/score_accuracy.py`)
- En temps réel, la segmentation est volontairement plus grossière que l'analyse finale (pas de regard en arrière) : les doubles-croches (≈120 ms) peuvent être fusionnées et un portamento très long (>250 ms) décale l'attaque. Les hauteurs restent justes, et l'analyse finale — celle qui est exportée — reprend tout depuis l'audio enregistré

## Tests

```bash
pytest tests/ -v
```

360 tests : pitch (YIN/pYIN sur sinus connus, **flux YIN par blocs** : temps absolus monotones et mémoire bornée), segmentation, tempo (clics + raffinement sur attaques), tonalité, **accords** (triades, septièmes, basse, segments), **progression par mesure**, figures, **mesure par accents (3/4 vs 4/4) et 6/8**, MusicXML (+ **accords/paroles/couleurs**), noms de notes FR/EN, **rendu audio de contrôle**, **clavier PianoView**, **notation** (orthographe en armure, armures, durées écrites, mise en page mesure/système), **portée** (rendu offscreen, systèmes, défilement live), erreurs de chargement audio, export MIDI (meta-events relus), segmenteur live (+ **note en cours** pour le suivi temps réel), **suivi live du piano roll** (fenêtre glissante, preview, bande de hauteur stable, recoloration après étiquetage), **mode live de bout en bout** (micro factice → capture → analyse finale → export), **capture live hors callback** (file + thread, blocs jamais perdus, erreur d'appelant non fatale), **CLI `--pdf`** (outil absent → PDF ignoré, exit 0 ; conversion ratée → exit 2, artefacts et résumé toujours conservés), **enveloppe min/max de la waveform**, **i18n de la boîte de sélection de fichier** (et parité des clés fr/en), **script de diagnostic** (tableau des étapes aligné sur le pipeline réel, moyennes, cProfile, JSON, erreurs d'entrée), pipeline E2E WAV → artefacts, **justesse de transcription sur voix de synthèse** (19 cas, hors ligne exacts, preuve d'attaque jamais pire que la géométrie, aucune note inventée dans une note chantée, temps réel exact sur les phrases articulées), **attaques** (couverture des onsets chantés, une tenue avec vibrato ne produit plus 19 attaques, flux temps réel conservateur), **règles de transition** (glissando absorbé, vraie note courte préservée, hauteur d'arrivée), **mesure fantôme** (une ligne 4/4 n'est pas transcrite en 6/8).

Smoke test GUI headless :

```bash
QT_QPA_PLATFORM=offscreen python scripts/gui_smoke_test.py
```

## Générer l'audio de démonstration

```bash
python scripts/make_demo.py     # demo.wav : gamme de Do majeur à 120 BPM
v2m analyze demo.wav --quantize
```
