[English](README.md) | **Français**

# 🎤 Voix → MIDI

Application desktop qui **détecte les notes d'une voix** (hauteur, figures, mesure), le **tempo**, la **tonalité** et les **accords**, puis exporte un **fichier MIDI**, une **partition MusicXML** (PDF optionnel) et un **rapport JSON/CSV**. Fonctionne sur des **fichiers audio** (WAV/MP3/FLAC/OGG) ou **en temps réel au micro**.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)

## Fonctionnalités

- **Détection de hauteur** : pYIN (librosa) hors-ligne, YIN incrémental en temps réel — plage de recherche C2–C7, les notes de soprano se lisent à leur octave propre (l'ancien plafond C6 les repliait une octave plus bas)
- **Stabilisation du f0** : la dérivée de la hauteur marque toute zone qui bouge (rampe de portamento, passage de main) comme une **transition**, exclue de l'estimation de hauteur — chaque note se lit sur son **plateau stable**, le vibrato restant une ondulation par-dessus
- **Correction d'accordage** : un chanteur systématiquement bas ou haut est recentré — un offset global (déviation médiane au demi-ton le plus proche, borné à ±0,35 st) est absorbé **avant** l'arrondi, pour que chaque note arrondisse du bon côté ; la déviation résiduelle de chaque note alimente le collage à la gamme. Le panneau de résultats affiche l'offset en cents avec un conseil d'intonation (« juste » / légèrement / nettement). Le JSON porte la déviation résiduelle de chaque note (`deviation_cents`, signée) et un résumé `intonation` qui ne nomme une habitude **que si l'évidence la porte** — tendance de la prise entière, tendance par **registre** de hauteur, et première moitié contre seconde — pour ne jamais décrire du bruit comme une tendance
- **Porte anti-ronflement** : un rumble grave (brassage de micro avant la prise) ne lit plus de hauteur — des contrôles de périodicité et de HF tuent les trames hors-ligne comme en direct, la prise ne s'ouvre plus sur une fausse note
- **Collage à la gamme** : une note chantée presque à mi-chemin de deux demi-tons se résout vers le voisin vers lequel elle penche quand ce voisin appartient à la tonalité détectée — un degré mal attaqué est ré-ancré, une nuance chromatique délibérée reste une nuance
- **Segmentation en notes** : hystérésis avec référence glissante + confirmation 3 frames, **guidée par les attaques** (consonnes, coups de glotte, changements de voyelle) : un glissando legato n'est pas une note, et les micro-notes qu'il fabriquait (6 notes chantées → 10 détectées) sont absorbées dans la note d'arrivée, à sa hauteur. Résiste au vibrato (une tenue avec vibrato déclenchait 19 « attaques »), au **vibrato profond** (±0,6 demi-ton) et aux **trilles** rapides, et lit les **glissandos descendants** comme les montants (aucun fragment dupliqué, l'arrivée démarre à sa consonne) ; une note qui arrive en portamento se lit à son plateau, jamais à la médiane de la rampe ; une répétition explicite d'une même hauteur se lit comme une répétition même quand le chanteur dérive entre les deux
- **Tempo** : beat tracking librosa avec repli ×2 / ÷2 (range 60–180 BPM), puis **raffinement sur les attaques de notes** (grille par moindres carrés → BPM exact)
- **Tonalité** : profils Krumhansl-Schmuckler (24 candidatures majeur/mineur) + 2 candidates alternatives
- **Accords** : reconnaissance globale, par segments jointifs de 2 s et **par mesure** (majeure, mineure, dim, aug, 7, maj7, m7) via histogramme pondéré — affichés dans la GUI et le JSON
- **Progression d'accords** : un accord par **mesure** (aligné sur la mesure détectée, mesures identiques fusionnées) — se lit comme une grille : `Dm7 | Cmaj7 | G7 | C`. Dans l'interface française les accords sont traduits à l'affichage (`Am` s'affiche **La mineur**) — le JSON, le CSV et le MusicXML gardent l'`Am` canonique
- **Noms de notes bilingues** : anglaise (`A4`) **et** française (`La4`) côte à côte dans la GUI, le JSON (`name` / `name_fr`) et le CSV
- **Interface fr / en** : langue choisie dans la GUI (bouton 🌐) ou `v2m --lang en` — traduit tous les libellés **et** le format des notes (`Do Ré Mi` ↔ `A B C`), choix persisté dans `~/.v2m_lang`
- **Figures de notes** : chaque note est étiquetée (ronde → double-croche, pointées, **double-pointées**, triolets) selon le tempo détecté, avec tolérance sur l'interprétation humaine — champ `figure` dans le JSON et le CSV, **aussi en temps réel** sur le piano roll du mode micro. Une double-pointée ne se grave que sur une longueur délibérée (±1 % de la valeur idéale) : le timing d'un chanteur autour d'une note simple n'y atterrit jamais par accident
- **Portée (notation)** : les notes et silences écrits sur une **vraie portée** — clé de sol, armure, chiffrage, figures (ronde → double-croche, pointées, double-pointées, triolets), soupirs et pauses avec hampes, crochets, points et **altérations (dièse, bémol ET bécarre)**, lignes supplémentaires, barres de mesure et cartouche tempo · mesure · tonalité. Une note de l'armure chantée bécarre s'écrit naturelle (Fa♮ en Sol majeur, avec son propre glyphe bécarre — jamais un Mi♯ déguisé). Affichée **après l'analyse d'un fichier** et **en direct pendant la capture** (portée qui défile, la note en cours étant surlignée). Gravure vectorielle intégrée : aucun logiciel externe requis
- **Mesure** : détection automatique — un chant sur **temps pointés** est publié en **6/8** (tempo corrigé ×1,5) ; sinon les **accents d'attaque** (forces relatives RMS des notes, contraste requis) tranchent 2/4, 3/4 vs 4/4 — une alternance forte/faible comparable se lit 2/4 (marche), une hiérarchie temps 1 fort / temps 3 secondaire reste 4/4 — et la grille ternaire repêche un 6/8 sans accent ; sans accent clair, 4/4 par défaut assumé. La partition et le rapport utilisent la mesure détectée
- **Export MusicXML** : partition lisible (mesure détectée, armure, tempo, figures, liaisons à travers les barres, triolets) ouvrable dans MuseScore, Finale, Sibelius…
- **Toutes les détections sur la partition** : accords par mesure en **chiffrage harmonique** au-dessus de la portée **avec leur nom lisible** (« La mineur » — le fichier garde aussi le symbole canonique), noms de notes en **sol-fège** (Do, Ré, Mi…) sous chaque note, **têtes de note colorées selon la confiance** (vert = sûre → rouge = douteuse — la couleur porte désormais la vraie probabilité de voising de pYIN, une note soufflée se lit plus pâle), cartouche récapitulatif (tempo · tonalité · mesure · accords) dans le titre
- **Export MIDI** : meta-events `set_tempo` + `key_signature`, vélocité proportionnelle à la confiance, quantification 1/16 optionnelle, transposition ±48 demi-tons, **événements de pédale forte (CC64)** sur les notes tenues — les fichiers rejoués ailleurs résonnent comme le piano intégré
- **Export PDF** : gravure de la partition via MuseScore ou LilyPond (détection automatique, message d'installation sinon) — `--pdf` en CLI, case à cocher en GUI
- **Rapport JSON** (tout, y compris la courbe de hauteur, la mesure, le nombre de mesures, la position mesure:temps de chaque note, l'accord global et les segments d'accords) + **CSV** (liste de notes avec figure et position)
- **GUI PyQt6** : interface sombre soignée, piano roll, waveform, panneau de résultats avec tableaux par note (position, figure), par segment d'accord et **ligne d'accordage**, drag & drop, **mode notation** (couleur par figure), **bannière d'erreur rouge intégrée** (pas de popups bloquants)
- **Mode temps réel** : indicateur note/Hz/niveau, **piano roll qui suit la capture** (fenêtre glissante de 10 s, curseur de temps, horloge `mm:ss`, compteur de notes et **mesure détectée** — 3/4, 4/4, 6/8 — affichée en direct), comparaison avec une partition déjà analysée (touche verte/rouge), la note tenue s'affiche **pendant** qu'on la chante (~125 ms de latence d'affichage), figures étiquetées en direct, export à l'arrêt — le suivi de hauteur tourne **hors du callback audio** (thread dédié), donc le pilote ne perd pas de blocs. Une case **« Enregistrer l'audio de la prise »** (décochée par défaut) écrit en plus la prise en WAV 16 bits (`.wav`) dans son dossier `prises_live/<stem>/`, avant les autres artefacts — en live uniquement, une analyse de fichier ayant déjà son audio
- **Trouveur de note** : reconnaissance isolée au micro, clavier de 88 touches et historique local des six dernières notes ; une touche ne s'allume que sur une **note confirmée** (jamais sur un brassage de micro), et aucune analyse ni export
- **Écoute de contrôle** : bouton ▶ (synthé local intégré, aucun logiciel externe) avec **curseur de lecture synchronisé** sur le piano roll et la waveform ; **double-clic** sur le piano roll pour démarrer la lecture à cet endroit. Le piano suit la dynamique de la prise (un accent fort frappe plus fort qu'une note douce), frappe les touches douces plus lentement, respire de quelques millisecondes autour de la grille et sonne dans une pièce légère — déterministe, un replay sonore est identique

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

La commande `analyze` écrit `sortie.mid`, `sortie.musicxml`, `sortie.json` et `sortie.csv` (`--pdf` ajoute `sortie.pdf` si MuseScore ou LilyPond est installé — sinon le PDF est ignoré avec un message d'installation et le code de sortie reste 0 ; si la gravure échoue, exit 2 et les quatre autres artefacts restent écrits) ; le JSON contient pour chaque note son champ `figure` (ex. `"noire"`, `"croche pointée"`), sa `position` `[mesure, temps]`, sa `deviation_cents` (déviation résiduelle signée, une fois l'offset global absorbé), ses noms `name` / `name_fr` (anglaise et française), la `meter` détectée, `num_measures`, l'accord global `chord` et ses `chord_segments`, et le résumé `intonation` construit sur ces déviations (moyenne/médiane, parts juste/dièse/bémol, par registre, par moitié).

```bash
# avec export PDF de la partition (best-effort : message clair si aucun outil)
v2m analyze chant.wav --quantize --pdf
```

La partition MusicXML embarque **toutes les détections** : chiffrages d'accords au-dessus de la portée (progression par mesure) avec leur nom lisible (« La mineur »), noms sol-fège (Do, Ré, Mi…) sous chaque note, têtes de note colorées selon la confiance (vert = sûre → rouge = douteuse), cartouche récapitulatif dans le titre. Tout s'ouvre dans MuseScore, Finale ou Sibelius.

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
│   ├── chords.py        # accords (triades + septièmes, global, segments et progression par mesure)
│   ├── figures.py       # classification des figures ; catalogue dans grid.py
│   ├── meter.py         # détection de mesure (temps pointés 6/8, accents 2/4, 3/4 vs 4/4)
│   ├── grid.py          # BeatGrid, Meter, Figure/CATALOGUE et divisions MusicXML
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
├── score_accuracy.py    # diagnostic : ce que la transcription a fait d'une prise réelle
├── confirm_s_sweep.py   # diagnostic : balayage du seuil confirm_s live, multi-graines
├── liveseg_sweep.py     # diagnostic : balayage HOLD_ST x REARTIC_ST live, multi-graines
├── guide_capture.py     # enregistrement guidé note par note d'une prise vérifiée
├── coherence_summary.py # diagnostic : agréger les coherence.csv des prises, par session
├── replay_takes.py      # diagnostic : rejouer les WAV des prises via la chaîne GUI
│                        #   (micro factice temps réel) et archiver l'écart live/final
├── engine_drift_watch.py # diagnostic : re-dérive les audits périmés et écrit un rapport
│                        #   de non-dérive en Markdown (code 1 si une lecture bouge)
└── reinterpret.py       # diagnostic : réinterpréter une observation sauvegardée, sans pYIN

prises_verite/          # prises vérifiées à la main : <nom>.wav + <nom>.truth.json (voir son README)
tests/
└── voicetruth.py        # vérité terrain : voix de synthèse à notes connues + mesure d'écart
```

## Diagnostic (temps d'analyse)

```bash
# tableau des étapes : où passe le temps sur CETTE prise ?
./.venv/Scripts/python.exe scripts/profile_analysis.py chant.wav

# moyenne de 3 exécutions + fonctions les plus lourdes (cProfile) + dump JSON
./.venv/Scripts/python.exe scripts/profile_analysis.py chant.wav --repeat 3 --profile --top 20 --json mesures.json --output artefacts
```

Les étapes ne sont pas recopiées dans le script : il lit celles que le pipeline annonce déjà (`progress`), donc une étape renommée ou ajoutée apparaît ici toute seule. Deux colonnes à regarder : **Part** (poids de l'étape) et **Temps réel** (durée de l'étape / durée de l'audio — 0,037× = 27× plus rapide que la lecture, le seuil à surveiller pour une longue prise). Un passage à blanc de 2 s est écarté par défaut (`--no-warmup` pour le mesurer) : sans lui, les imports librosa et la compilation numba atterrissent sur la première étape et masquent le vrai coût. `--repeat N` moyenne N exécutions (défaut : 1) ; `--top N` limite le tableau cProfile à ses N lignes les plus lourdes ; `--json fichier` écrit les mesures dans un fichier JSON ; `--output dossier` garde les artefacts exportés dans ce dossier (sinon : dossier temporaire). `--profile` ajoute un tableau cProfile séparant **le code du projet** (ce qu'on peut optimiser) des **bibliothèques** (où va le temps appelé).

Sur `demo.wav` (8,5 s) le décodage domine ; sur une prise de 68 s (répétée 8×) pYIN pèse 53 % et `refine_bpm_with_onsets` 16 % — c'est ce dernier que le tableau montre en tête côté code du projet. Le calcul des attaques (un STFT de plus) coûte 0,12 s sur ces 68 s, soit 0,2 % du temps réel.

## Diagnostic (qualité de transcription)

```bash
# ce que la transcription a fait de CETTE prise : notes, figures, positions…
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav

# les signes de découpage douteux, et la comparaison attaques vs géométrie seule
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --json rapport.json --no-attacks

# comparaison chiffrée à des notes connues (vérité terrain, même harnais que les tests)
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --truth chant_verite.json

# l'analyse rejouée deux fois donne-t-elle les mêmes chiffres ? (dérive => exit 1)
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --compare

# empreinte dorée : la prise est comparée à SA référence segmentée (<audio>.golden.json,
# créée au premier passage) — la dérive est vue même si deux rejeux s'accordent entre eux
./.venv/Scripts/python.exe scripts/score_accuracy.py chant.wav --golden
```

Sur une prise réelle il n'y a pas de vérité terrain : le script imprime donc d'abord ce que le pipeline a trouvé (tempo, mesure, tonalité, accords, chaque note avec sa figure et sa position), puis les **signes de découpage douteux** : attaques orphelines (une consonne sans début de note — une note est peut-être manquée), débuts non articulés (la note existe mais aucune attaque ne la date), fragments (notes plus courtes que la confirmation de 120 ms, typiques d'un glissando coupé en deux). Si `--no-attacks` donne le même nombre de notes, la preuve d'attaque ne sert à rien sur cette prise (pas de consonnes exploitables). La vérité terrain est un simple JSON `{"notes": [{"midi": 60, "start_s": 0.3, "dur_s": 0.6}, …]}` ; `--truth fichier.json` chiffre la transcription contre ces notes connues, `--json fichier.json` écrit le rapport d'audit, et `--output dossier` garde les artefacts exportés (sinon : dossier temporaire).

Deux modes protègent la segmentation **dans le temps** : `--compare` rejoue l'analyse complète (re-décodage compris) et compare l'empreinte de segmentation (tempo, mesure, tonalité, notes, silences) ; `--golden` compare à l'empreinte enregistrée à côté de la prise (`<audio>.golden.json`, créée au premier passage, ignorée par git comme tout `.json`) — une régression de segmentation sur une vraie prise est vue même quand deux rejeux s'accordent entre eux. Les deux impriment les différences champ par champ et quittent avec le code 1 si l'empreinte a bougé.

`--diagnose` audite les **seuils que la prise a réellement exercés** — ne jamais recalibrer un réglage sur une prise qui ne le teste pas : plage de hauteur de la courbe face aux bornes de lecture (65 Hz–2100 Hz), décalage d'accordage absorbé (cents), forces d'attaque normalisées (déciles, face à leur plancher), déviations de hauteur résiduelles qui décident du collage à la gamme, et chemin live rejoué en flux (latence d'affichage réelle, `confirm_s`).

L'analyse est scindée en **deux couches** : une `VoiceObservation` (la courbe de hauteur nettoyée, les attaques, l'horloge, l'enveloppe d'accents, l'offset d'accordage de la prise) et son **interprétation** (notes, tonalité, accords, grille, figures). L'évidence peut être sauvegardée à côté de la prise et rejouée **sans redécoder l'audio ni relancer pYIN** — c'est le rôle de `scripts/reinterpret.py`. `--save` écrit l'observation dans `<prise>.observation.npz` ; `--observation fichier.npz` en relit une au lieu de l'audio ; `--semitones`, `--min-conf` et `--min-note` (en millisecondes) changent les seuils d'interprétation sans toucher à l'évidence ; `--sweep` et `--sweep-note` balaient plusieurs valeurs de chacun sur la MÊME observation et impriment une ligne par lecture (nombre de notes, tempo), le calibrage courant étoilé — on choisit un seuil sur le tableau, pas à l'instinct ; `--against autre.npz` diffuse DEUX observations, l'évidence d'abord (courbe échantillon par échantillon, attaques, horloge — là où un moteur change vraiment), puis la segmentation ; `-o`/`--output dossier` réexporte les artefacts et rien n'est écrit sans lui.

`v2m analyze` et le GUI réutilisent automatiquement le `<prise>.observation.npz` de la prise : chaque analyse de fichier ÉCRIT l'observation qu'elle vient de construire (le coût pYIN est payé une fois par prise, pas une fois par analyse), et l'analyse suivante la relit quand l'audio n'a pas changé (taille + mtime enregistrés dans le fichier). La barre d'état explique alors pourquoi c'était instantané (« évidence reprise du cache — pYIN non relancé »). Un cache périmé, illisible ou d'une autre mise en page est ignoré silencieusement — le repli est la construction complète de l'évidence, jamais une lecture différente.

```bash
# sauvegarder l'évidence une fois, puis la réinterpréter avec d'autres seuils
./.venv/Scripts/python.exe scripts/reinterpret.py chant.wav --save
./.venv/Scripts/python.exe scripts/reinterpret.py chant.wav --observation chant.observation.npz --semitones 0.35 --min-note 80
```

Deux outils complètent le travail de corpus :

```bash
# balayage de confirm_s : les 18 cas de vérité live x 5 graines x 0,05..0,10 s —
# l'archive des valeurs mesurées vit dans AGENTS.md
./.venv/Scripts/python.exe scripts/confirm_s_sweep.py

# balayage du segmenteur live : HOLD_ST x REARTIC_ST sur les mêmes cas plus
# deux cas de vibrato débutant (+/-0,5 st), plusieurs graines par cellule —
# décide si le vibrato d'un débutant mérite une tenue plus large (verdict
# mesuré archivé dans AGENTS.md)
./.venv/Scripts/python.exe scripts/liveseg_sweep.py

# enregistrement guidé d'une prise vérifiée : chanter, annoter note par note,
# les bornes sont raffinées sur la courbe de hauteur, écrit sous prises_verite/
./.venv/Scripts/python.exe scripts/guide_capture.py prise_verite.wav

# l'auditer : validation du JSON + comparaison note à note humain vs machine
./.venv/Scripts/python.exe scripts/guide_capture.py prise_verite.wav --audit
```

`--values` du balayage restreint les valeurs de seuil scannées (séparées par des virgules) et `--seeds` les graines de synthèse (un cas peut n'échouer que sur UNE graine : le grain de bruit remplit la fenêtre de tenue quelques ms autrement) ; `--holds`/`--reartics` du balayage du segmenteur restreignent ses deux axes de la même façon ; `--audit` de la capture relit une prise existante au lieu d'en enregistrer une.

Un dernier outil suit les sessions dans le temps :

```bash
# l'audit des phrases avalées, agrégé : une ligne par session, une par
# prise — les phrases avalées diminuent-elles ? (un écart de 30 min ouvre
# une session ; un sous-total par jour garde le calendrier lisible)
./.venv/Scripts/python.exe -X utf8 scripts/coherence_summary.py
```

Le script balaie `<dossier>/*/*.coherence.csv` (défaut `prises_live`, passez un autre dossier en argument). Par session il imprime le nombre de prises, les notes contrôlées (lues dans le `<prise>.json` voisin ; affichées `-` quand un JSON manque ou est illisible — jamais devinées), les suspects, le **taux de suspects pour 100 notes contrôlées** avec son **delta face à la session précédente** (la baisse se lit sans comparer des effectifs), le **taux de contours** moyen (l'indicateur de santé live/final, lu des blocs `live_vs_final` des JSON), la portée moyenne chantée dans les suspects (st) et le plus long plateau secondaire (s) ; un second tableau garde le détail par prise — chaque suspect portant le plateau qu'il cache (son temps dans la note, sa hauteur et l'intervalle **signé** vers l'étiquette). Chaque ligne de session porte aussi une **sparkline de tendance** (▁▂▃▄▅▆▇█ — l'historique du taux de suspects jusqu'à cette session, haut = plus de suspects), et la **tendance de santé du temps réel** sur plusieurs semaines ferme le rapport : une ligne par semaine ISO avec le taux de contours moyen et les effectifs live/final, lus du journal pour qu'une prise supprimée garde l'historique de sa semaine. La GUI dessine cette tendance en **courbe pyqtgraph** (un point par semaine, plus bas = mieux) : cliquer une semaine liste SES prises rejouées, et une dégradation mesurée (plus de 10 points de taux de contours contre la semaine précédente, appuyée par plus d'une prise) déclenche un avertissement sur la page live — le seul chiffre qui dise « la vue directe dérive de la passe finale ». Chaque prise est d'abord vérifiée (`verify_take`) : celle dont le WAV et le JSON divergent est signalée dans le tableau et ses compteurs ne nourrissent PAS les taux. L'agrégation vit dans `core.coherence` (le bouton de suivi de la GUI lit le même module et affiche les mêmes tendances) ; un journal `coherence_journal.json` persiste l'audit de chaque prise à côté de ses artefacts, pour que le ménage des dossiers ne réécrive pas l'historique. Les libellés suivent la langue de l'application (`v2m --lang`) ; un dossier sans rapport explique où il a cherché et quoi faire ensuite (code 0) tandis qu'un dossier absent quitte avec le code 2, en nommant le dossier attendu.

Et un outil mesure la reconnaissance temps réel elle-même, prise par prise :

```bash
# rejouer les WAV des prises via la chaîne GUI (micro factice à vitesse
# réelle — livrer plus vite est mesuré comme mensonge) et archiver
# l'écart live/final : contours, famille ±1, vrais déplacements, taux
./.venv/Scripts/python.exe -X utf8 scripts/replay_takes.py [--out rapport.json] [--take live_...] [--changed]
```

Pour chaque prise possédant un `<stem>.wav`, le script nourrit la vraie `LiveCapture` à l'horloge réelle, lance l'analyse finale et classe chaque note live par la politique de comparaison UNIQUE (`core.compare` — le même filet que le cerclage GUI et le bloc `live_vs_final` du JSON) : identique, voisin ±1, déplacée ; le taux de contours affiché est l'indicateur de santé du temps réel. Rien n'est exporté (un rejeu n'écrase jamais les artefacts d'une prise) ; `--out fichier.json` archive le tableau ; `--take <stem>` restreint le rejeu à une prise ; `--changed` ne re-dérive que les prises dont l'audit archivé a été mesuré par un moteur ANTIÉRIEUR (le hash du code du moteur voyage dans le sidecar `<prise>.live_vs_final.json` — après une modification du moteur, re-mesurez les verdicts qu'il a pu changer). Le moteur vit dans `core.replay` (le flux est injecté — un rejeu ne remplace jamais le `sd.InputStream` global que lit le vrai micro) ; la GUI le réutilise : chaque prise est rejouée automatiquement à l'arrêt de son enregistrement, en arrière-plan, et l'audit remplace le sidecar `live_vs_final.json` de la prise. La lecture hors ligne du rejeu réutilise l'évidence en cache (`<prise>.observation.npz`, semée par le premier rejeu puis relue) pour que pYIN — l'étape la plus longue — ne tourne qu'une fois par prise ; un cache périmé ou corrompu retombe sur l'analyse complète et est re-semé, jamais cru.

Le suivi de `--changed` est automatisé par un script : `engine_drift_watch.py` re-dérive tous les audits périmés, compare chaque ligne fraîche au sidecar qu'elle remplace (deltas par métrique ; tout mouvement d'effectif est une dérive, une variation de taux sous 0,15 est le bruit de session honnête du moteur) et écrit un rapport de non-dérive en Markdown — code 1 quand une lecture a bougé, pour qu'un CI ou un hook pré-commit échoue bruyamment quand la dernière modification du moteur a déplacé un verdict mesuré. `--root <dossier>` vise un autre dossier de prises, `--report <fichier>` nomme le rapport (défaut `docs/engine-drift.md`) :

```bash
# après avoir touché le moteur : les verdicts archivés sont-ils toujours vrais ?
./.venv/Scripts/python.exe -X utf8 scripts/engine_drift_watch.py [--report docs/engine-drift.md]
```

## Documentation projet

- **`v2m weekly`** — le récapitulatif hebdomadaire en texte (`--pdf` le grave avec le même outil que la partition) : la santé du temps réel sur plusieurs semaines, QUAND vous chantez le mieux (matin/après-midi/soir), les sous-totaux par jour. La même vue existe sur la page cohérence ; la rétention (`prises_live/` maintenue sous un budget de taille documenté, les prises les plus vieilles taillées en entier, l'historique conservé dans le journal) s'exécute quand le récapitulatif ou un export touche le dossier.
- **`CONTEXT.md`** — glossaire du domaine (Analyse, Note, Tempo, Tonalité, Accord, Figure, Mesure, Export, Rapport) et conventions (langues, erreurs) : la référence pour toute évolution du code.
- **`core/i18n.py`** — catalogue des chaînes utilisateur (fr/en) : toute nouvelle chaîne visible y passe par `tr()` ; les noms de notes suivent la langue (`note_display`), les figures stockées restent canoniques (`tr_figure` traduit à l'affichage).
- **`AGENTS.md`** — apprentissages techniques et **règles de session** (non-régression, mise à jour des docs, lisibilité) pour les agents outillés ; lu automatiquement si l'option est activée dans le client.

## Qualité & limites

- Chaque rapport porte un bloc **`coherence`** : chaque note longue est auditée à la recherche d'un plateau secondaire caché (> 120 ms tenus à plus d'un demi-ton de l'étiquette de la note) — une phrase rechantée collée en une seule continuité y apparaît au lieu de passer en silence. Le même audit est écrit à côté de chaque prise en `<prise>.coherence.csv` (une ligne par suspect, entête seule si la prise est propre) pour comparer les sessions dans un tableur. `python -c "from v2m.core.verify import verify_take; print(verify_take('prises_live/<prise>/<prise>'))"` vérifie que le WAV et le JSON d'une prise décrivent le MÊME signal.

- **Pourquoi la partition finale diffère du direct** : le rapport d'une prise live porte un bloc **`live_vs_final`** qui décompose l'écart (la famille ±1 demi-ton — un arrondi, pas une erreur ; les vrais déplacements ; les glissandos que le hors-ligne absorbe), le panneau de résultats en fait une phrase honnête par cause, et chaque prise live l'archive en `<prise>.live_vs_final.json` à côté de ses artefacts (le contre-choc du rejeu la met à jour avec la lecture re-dérivée).

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

720+ tests : pitch (YIN/pYIN sur sinus connus, **flux YIN par blocs** : temps absolus monotones et mémoire bornée), segmentation, tempo (clics + raffinement sur attaques), tonalité, **accords** (triades, septièmes, basse, segments), **progression par mesure**, figures, **mesure par accents (2/4, 3/4 vs 4/4) et 6/8 temps pointés**, tempo ×1,5 corrigé live et offline, MusicXML (+ **accords/paroles/couleurs**), noms de notes FR/EN, **rendu audio de contrôle**, **clavier PianoView**, **notation** (orthographe en armure, armures, durées écrites, mise en page mesure/système), **portée** (rendu offscreen, systèmes, défilement live), erreurs de chargement audio, export MIDI (meta-events relus), segmenteur live (+ **note en cours** pour le suivi temps réel), **suivi live du piano roll** (fenêtre glissante, preview, bande de hauteur stable, recoloration après étiquetage), **mode live de bout en bout** (micro factice → capture → analyse finale → export), **capture live hors callback** (file + thread, blocs jamais perdus, erreur d'appelant non fatale), **moteur de rejeu** (chaîne réelle à l'horloge réelle, flux injecté — jamais de remplacement global —, archive du worker, annulation), **explication live/final**, **tendances de cohérence** (sparkline, tendance hebdo des contours lue du journal), **CLI `--pdf`** (outil absent → PDF ignoré, exit 0 ; conversion ratée → exit 2, artefacts et résumé toujours conservés), **enveloppe min/max de la waveform**, **i18n de la boîte de sélection de fichier** (et parité des clés fr/en), **script de diagnostic** (tableau des étapes aligné sur le pipeline réel, moyennes, cProfile, JSON, erreurs d'entrée), pipeline E2E WAV → artefacts, **justesse de transcription sur voix de synthèse** (19 cas, hors ligne exacts, preuve d'attaque jamais pire que la géométrie, aucune note inventée dans une note chantée, temps réel exact sur les phrases articulées, **respirations jamais notées** : reprises d'air, souffle isolé, tenue soufflée), **attaques** (couverture des onsets chantés, une tenue avec vibrato ne produit plus 19 attaques, flux temps réel conservateur), **règles de transition** (glissando absorbé, vraie note courte préservée, hauteur d'arrivée), **mesure fantôme** (une ligne 4/4 n'est pas transcrite en 6/8), **politique unique des seuils** (`FILE_CONF`/`LIVE_CONF`, aucun défaut réécrit localement).

Boucle rapide pendant l'itération — évite les tests marqués `slow` (les rejeux temps réel et les empreintes dorées) :

```bash
pytest tests/ -m "not slow"
```

Mesuré : ~81 s pour le run complet face au budget de 150 s de la suite, ~34 s pour la boucle rapide (environ 47 s gagnées, plus de la moitié du run). Chaque run affiche son coût (`suite cost: ...`) ; un run vert au-delà de 150 s échoue volontairement : ajouter un test lent reste une décision explicite. Les tests `slow` restent dans le run par défaut — CI lance la suite complète, la boucle rapide est pour la boucle locale.

Smoke test GUI headless :

```bash
QT_QPA_PLATFORM=offscreen python scripts/gui_smoke_test.py
```

## Générer l'audio de démonstration

```bash
python scripts/make_demo.py     # demo.wav : gamme de Do majeur à 120 BPM
v2m analyze demo.wav --quantize
```
