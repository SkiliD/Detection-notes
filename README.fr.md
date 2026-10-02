[English](README.md) | **Français**

# Voix → MIDI

Transcrivez une mélodie chantée. Analysez un fichier audio ou chantez au micro, vérifiez et corrigez les notes, puis exportez en MIDI, MusicXML, JSON et CSV. L’export PDF est facultatif.

L’application comprend aussi un cherche-note : chantez pour identifier une note ou cliquez une touche du piano pour l’entendre.

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

La suite de ce fichier suppose l'environnement activé : `python` et
`pytest` ci-dessous sont cet interpréteur (sous Windows sans activation,
utilisez `.venv/Scripts/python.exe` à la place).

## Utiliser l’application

Une nouvelle transcription live reste dans les résultats jusqu’à l’export. Rechanter une suite sélectionnée met à jour la prise : vous écoutez la phrase, puis vous la gardez ou l’annulez (la prise d’origine est aussi conservée si l’enregistrement ou la nouvelle analyse échoue). Les modifications de notes (corrections et remplacements gardés) s’annulent et se rétablissent ; une correction manuelle conserve le timing de la note et indique que le contrôle de cohérence audio n’est plus disponible pour ce résultat.

```bash
v2m gui
```

L’écran d’accueil est un menu : les trois actions (ouvrir un fichier, chanter au micro, trouver la note que je chante) tiennent dans une seule colonne centrée, avec le sélecteur de langue et le suivi de cohérence dessous.

- Ouvrez ou déposez un fichier WAV, MP3, FLAC ou OGG, ou choisissez l’enregistrement live.
- Vérifiez les notes, écoutez le piano ou la voix. Pour corriger une note, sélectionnez-la puis cliquez la bonne touche du piano. La sélection se fait aussi directement sur le graphique : cliquez la barre d’une note pour la sélectionner, faites glisser sur les barres voisines pour sélectionner une plage, cliquez le vide pour effacer. La barre d’outils affiche la sélection courante (nombre de notes et plage temporelle) ; la touche **Suppr** supprime la plage sélectionnée — en une seule annulation. Annuler et rétablir (Ctrl+Z / Ctrl+Y, ou les boutons ↶/↷) reviennent sur les modifications de notes ; le menu déroulant au-dessus des boutons d’édition liste tous les états de la prise — choisissez-en un pour y revenir directement.
- Après une prise live, sélectionnez une suite continue de notes et choisissez **Remplacer les notes sélectionnées** ; **Écouter l’original** joue la phrase sélectionnée telle qu’elle est, avant de la rechanter. Chantez le remplacement, terminez-le, écoutez la nouvelle phrase, puis choisissez **Garder le remplacement** (ou **Annuler le remplacement** pour rétablir la prise). Aucun fichier n’est écrit avant de choisir **Exporter cette prise**.
- Après l’arrêt d’une prise, les notes qu’elle a fermées restent sur le piano roll live : cliquez une barre pour la surligner et explorer la prise telle qu’elle a été chantée (lecture seule — la prise suivante repart propre).
- Les exports d’un fichier sont placés à côté de celui-ci. Les prises live vont dans `prises_live/`.

### Raccourcis clavier

| Raccourci | Action |
| --- | --- |
| Ctrl+Z | Annuler la dernière modification de notes (page résultats) |
| Ctrl+Y | Rétablir la dernière modification annulée (page résultats) |
| Suppr | Supprimer la plage de notes sélectionnée (page résultats) |

Ctrl+Y est la touche de rétablissement ; Ctrl+Shift+Z fonctionne aussi. Les deux raccourcis agissent depuis la page résultats, où vivent les modifications.

## Ligne de commande

```bash
v2m analyze chant.wav --quantize
v2m analyze chant.wav --pdf
v2m live
v2m --lang en analyze chant.wav
```

`analyze` accepte `--min-conf`, `--min-note`, `--transpose`, `--quantize`, `--pdf` et `-o`/`--output`. Il écrit MIDI, MusicXML, JSON et CSV. `--pdf` ajoute un PDF si MuseScore ou LilyPond est installé. Si l’outil manque, seul le PDF est ignoré ; un échec de conversion renvoie le code 2. `--lang fr` et `--lang en` choisissent la langue de l’interface.

## Diagnostics

Ces outils sont facultatifs. Depuis la racine du projet, dans l'environnement activé.

- `scripts/score_accuracy.py` : examiner une transcription. `--truth` la compare à des notes connues ; `--json` écrit un rapport ; `--output` choisit le dossier des artefacts ; `--no-attacks` ignore les attaques ; `--compare` vérifie la répétabilité ; `--golden` vérifie une référence enregistrée ; `--diagnose` indique les seuils testés par la prise.
- `scripts/profile_analysis.py` : mesurer le temps d’analyse. `--repeat` règle le nombre d’exécutions ; `--profile` ajoute un profil des fonctions ; `--top` limite ce tableau ; `--json` écrit les mesures ; `--output` choisit le dossier des artefacts ; `--no-warmup` inclut le coût du premier passage.
- `scripts/confirm_s_sweep.py` : balayer le seuil de confirmation des notes live. `--values`, `--seeds` et `--amps` limitent les valeurs, graines aléatoires et amplitudes testées.
- `scripts/liveseg_sweep.py` : balayer la segmentation live. `--holds` et `--reartics` limitent les seuils de tenue et de ré-articulation.
- `scripts/guide_capture.py` : enregistrer et annoter une prise vérifiée ; `--audit` vérifie une prise existante.
- `scripts/reinterpret.py` : réutiliser les hauteurs enregistrées sans décoder l’audio ni relancer pYIN. `--save` sauvegarde l’évidence ; `--observation` la charge ; `--semitones`, `--min-conf` et `--min-note` changent l’interprétation ; `--sweep` et `--sweep-note` comparent des seuils ; `--against` compare les évidences ; `-o`/`--output` exporte les résultats.
- `scripts/coherence_summary.py` : résumer les audits par session ; passez un dossier pour utiliser un emplacement différent de `prises_live/`. `--lang` choisit les libellés français ou anglais.
- `scripts/replay_takes.py` : rejouer les prises enregistrées en temps réel et enregistrer la comparaison live/finale. `--out` choisit le rapport JSON ; `--take` choisit une prise ; `--changed` ne rejoue que les prises mesurées par une ancienne version du moteur ; `--root` choisit le dossier des prises.
- `scripts/engine_drift_watch.py` : actualiser les comparaisons périmées. `--root` choisit le dossier des prises ; `--report` choisit le rapport Markdown. Renvoie 1 si une lecture a changé. `--json` enregistre les résultats pour traitement automatique.
- `scripts/compare_takes.py` : comparer deux prises d’une phrase. `--json` écrit la comparaison.

## Vérifications

```bash
python -m pytest tests/ -q
pytest tests/ -m "not slow"
QT_QPA_PLATFORM=offscreen python -u scripts/gui_smoke_test.py
```

La suite complète a un budget de 150 secondes. La boucle rapide omet les tests marqués `slow`. Le smoke test graphique utilise `demo.wav` (ignoré par git) ; créez-le avec `python scripts/make_demo.py`.
