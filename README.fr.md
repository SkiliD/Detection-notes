**Français** | [English](README.md)

# Voix → MIDI

Transformez une mélodie chantée en notes. Chantez au micro ou ouvrez un fichier audio, corrigez la transcription, puis exportez MIDI, MusicXML, JSON, CSV — et un PDF quand un outil de partition est installé. L'audio est analysé sur votre machine et n'est jamais envoyé.

Trois entrées : l'application de bureau (`v2m gui`), la commande live (`v2m live`) et une page web statique dans `web/` qui fait tourner la moitié écoute dans le navigateur. Le bureau lit les fichiers avec pYIN et donne donc aussi le tempo, la mesure, la tonalité, les silences, la notation et le PDF ; la page n'utilise le YIN plus rapide que sur le direct, ce qui explique qu'elles peuvent diverger sur une frontière de note.

## Installation

Python 3.12 est recommandé, sous Windows, macOS ou Linux.

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Ci-dessous, `python` et `pytest` désignent cet environnement (sous Windows sans activation, utilisez `.venv/Scripts/python.exe`).

## Utiliser l’application

```bash
v2m gui
```

- **Chanter** enregistre une prise, qui reste sur la page de résultats jusqu'à l'export. Corrections, déplacements et suppressions sont une seule transaction annulable ; une édition ratée laisse le relevé et l'historique intacts.
- **Relecture** propose une action à la fois : écouter dans l'ordre les passages signalés par le contrôle de cohérence, puis exporter. Un doute n'est jamais une correction automatique, et **Passer à l'export** rend chaque étape facultative.
- **Vues** : Notes (le piano roll, où l'on corrige), Partition et Forme d'onde, une à la fois. **Afficher les détails** ajoute la portée, les diagnostics et les options d'export, et démarre replié.
- **Remplacer** : sélectionnez une plage, rechantez-la, puis gardez ou annulez le remplacement ; la prise d'avant est conservée.
- **Justesse** : la part de notes justes, une tendance nommée, et les penchants par registre (grave / médium / aigu) et par moitié (début / fin) — toujours à partir des cents mesurés. La tonalité détectée renomme une note sans réécrire sa mesure.
- **Exporter** ouvre Enregistrer sous (MIDI, MusicXML, PDF, JSON, CSV des notes) et retient le dernier dossier ; un outil PDF absent ne saute que le PDF. Une prise live garde aussi son archive d'audit complète dans `prises_live/`.
- **Cherche-note** : chantez une note ou cliquez une touche pour l'identifier. La carte de réponse réunit le nom, l'aiguille de justesse et la réception du micro ; les notes confirmées restent en pastilles écoutables.
- **Apparence** : Sombre ou Claire, enregistrée dans `~/.v2m_theme`. Les deux palettes gardent chaque mot lisible, y compris ceux que l'application peint elle-même.

### Raccourcis clavier

| Raccourci | Action |
| --- | --- |
| Tab | Parcourt les actions de la page dans l'ordre des tâches |
| Ctrl+Z / Ctrl+Y | Annule / rétablit la dernière édition (Ctrl+Maj+Z rétablit aussi) |
| Haut / Bas | Déplace la sélection d'une note |
| Début / Fin | Va à la première / dernière note |
| Espace | Fait écouter la note sélectionnée, ou l'arrête |
| Gauche / Droite | Choisit une hauteur sur le clavier focalisé (elle s'entend) ; Entrée l'applique, Échap annule |
| Suppr | Supprime la plage sélectionnée |

## Version web

`web/` est une page statique — aucun build, aucune dépendance, rien d'envoyé — qui fait tourner la moitié écoute dans le navigateur : hauteur, notes, accents, piano roll, lignes de justesse et un export MIDI de type 0. **Tout télécharger** donne la prise en une archive ZIP : `prise.wav`, `notes.json`, `prise.mid` et une courte notice, compressée quand le navigateur sait le faire et stockée sinon.

Pour la regarder en local (la page charge ses libellés depuis un fichier) :

```bash
cd web
python -m http.server 8000
```

Pour la publier, réglez **Settings → Pages → Source** sur **GitHub Actions**, puis poussez sur `main` : le workflow rejoue les cas de conformité du portage, photographie la page à 1600 px, mesure huit classes de résolution (de 320 px à 2560 px, téléphone à l'horizontale inclus), audite les deux palettes dans trois états à chacune d'elles avec axe-core, puis déploie `web/`.

**Chanter** demande le micro une fois, et quand il ne s'ouvre pas la page nomme la raison : permission refusée à ce site, aucun appareil utilisable, appareil tenu par une autre application, ou page ouverte en fichier local ou en http simple, où le navigateur n'autorise jamais la question. La console du navigateur note l'étape qui a échoué et le nom de l'erreur, pour un rapport de bug. Sans micro, **Démonstration** charge une phrase chantée lue par la même chaîne. La page joue les notes sur le piano du bureau, les suit d'une tête de lecture, et s'imprime en noir sur blanc.

## Ligne de commande

```bash
v2m analyze chant.wav --quantize
v2m analyze chant.wav --pdf
v2m live
v2m serve
v2m --lang en analyze chant.wav
```

`analyze` accepte `--min-conf`, `--min-note`, `--transpose`, `--quantize`, `--pdf` et `-o`/`--output`. Il écrit MIDI, MusicXML, JSON et CSV ; `--pdf` écrit aussi un PDF quand MuseScore ou LilyPond est installé — un outil absent ne saute que le PDF, alors qu'une conversion en échec sort en code 2. `--lang fr` et `--lang en` choisissent la langue de l'interface.

`serve` est le compagnon local facultatif que la page peut appeler, parce qu'un navigateur ne peut pas exécuter pYIN : `v2m serve` n'écoute que sur `127.0.0.1` et répond à `GET /health`, `POST /analyze` (la charge du `.json` exporté, sans écrire de fichier), `POST /render` (le MusicXML que le bureau graverait, en texte) et `POST /compare` (un `notes.json` exporté avec son audio, et la façon dont les deux lectures s'apparient). `--port` choisit le port (8765) ; `--allow-origin` (répétable) nomme une origine autorisée à l'appeler — les pages localhost et les clients non-navigateur le sont toujours, une page publiée doit être nommée.

## Diagnostics

Outils facultatifs, lancés depuis la racine du projet dans l'environnement activé. Chacun s'explique quand on l'appelle sans argument ; les drapeaux ci-dessous sont ceux qui changent ce qu'il mesure.

- `scripts/score_accuracy.py` : lire une transcription. `--truth` compare des notes connues, `--json` écrit un rapport, `--output` choisit le dossier des artefacts, `--no-attacks` omet l'évidence des attaques, `--compare` vérifie la répétabilité, `--golden` vérifie une référence enregistrée, `--diagnose` liste les seuils que la prise a exercés.
- `scripts/profile_analysis.py` : mesurer le temps d'analyse. `--repeat` fixe le nombre de passes, `--profile` ajoute un profil de fonctions, `--top` limite ce tableau, `--json` et `--output` écrivent les mesures, `--no-warmup` garde le coût de la première passe.
- `scripts/confirm_s_sweep.py` : balayer le seuil de confirmation live — `--values`, `--seeds`, `--amps`.
- `scripts/liveseg_sweep.py` : balayer la segmentation live — `--holds`, `--reartics`.
- `scripts/guide_capture.py` : enregistrer et annoter une prise vérifiée ; `--audit` vérifie une prise existante, `--annotate` écrit un brouillon machine qu'une écoute doit confirmer.
- `scripts/reinterpret.py` : relire une évidence enregistrée sans décoder l'audio ni lancer pYIN — `--save`, `--observation`, `--semitones`, `--min-conf`, `--min-note`, `--sweep`, `--sweep-note`, `--against`, `-o`/`--output`.
- `scripts/coherence_summary.py` : résumer les audits de prises par session ; `--lang` choisit les libellés français ou anglais.
- `scripts/replay_takes.py` : rejouer des prises en temps réel et enregistrer la comparaison live/final — `--out`, `--take`, `--root`, et `--changed` pour les prises mesurées par un moteur plus ancien.
- `scripts/engine_drift_watch.py` : rafraîchir les comparaisons périmées — `--root`, `--report`, `--json` ; sort en 1 quand une lecture a changé.
- `scripts/compare_takes.py` : comparer deux prises d'une même phrase ; `--json` écrit la comparaison.
- `scripts/make_demo.py` génère le `demo.wav` ignoré par git dont le test de fumée graphique a besoin.

## Vérifications

```bash
python -m pytest tests/ -q
pytest tests/ -m "not slow"
QT_QPA_PLATFORM=offscreen python -u scripts/gui_smoke_test.py
```

L'intégration continue exécute ces trois commandes dans `.github/workflows/tests.yml`, à chaque push et pull request, avec le budget de la suite relevé pour le runner partagé. La suite complète a un budget de 150 secondes et tourne sur deux processus pytest-xdist — gardez 2, car à 4 les empreintes de référence des vraies prises se disputent le cache d'évidence. La boucle rapide saute les tests marqués `slow`. Pour le travail sur le dépôt, passez par les GitHub Issues ; les notes de conception derrière le code (`AGENTS.md`, `CONTEXT.md`, `docs/adr/`) vivent dans l'arbre de travail mais ne sont pas versionnées, donc absentes d'un clone.

## Licence

MIT — voir `LICENSE`.
