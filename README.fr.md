[English](README.md) | **Français**

# Voix → MIDI

Transcrivez une mélodie chantée. Analysez un fichier audio ou chantez au micro, vérifiez et corrigez les notes, puis exportez en MIDI, MusicXML, JSON et CSV. L’export PDF est facultatif. Dans l’application de bureau, **Exporter** ouvre une boîte « Enregistrer sous » pour choisir le nom et le format ; le dernier dossier choisi est mémorisé entre les lancements. Les prises live gardent aussi leur archive complète dans `prises_live/`.

L'écran d'accueil propose aussi **Calibrer mon micro** : cinq notes tenues guidées, puis l'écart de l'appareil (en cents) est proposé seulement si l'évidence le porte (un ronflement éparpillé ne propose rien; chaque tour ne compte que si le chanteur a atteint SA note demandée — une note tenue notée contre la cible SUIVANTE est précisément le bruit que les portes refusent). L'offset est enregistré dans `~/.v2m_mic_cal` (best-effort) et apparaît comme contexte sur les prises suivantes ; **la reconnaissance ne le lit jamais** (un calibrage est un fait d'affichage, jamais une entrée moteur).

L’application comprend aussi un cherche-note : chantez pour identifier une note ou cliquez une touche du piano pour l’entendre. Les notes confirmées entrent dans son historique court, répétitions comprises ; une note tenue n’y apparaît qu’une fois. L’historique utilise les lectures capturées à chaque bloc audio : même si l’interface est occupée, une note déjà confirmée n’est pas perdue. Le cherche-note indique aussi si la note chantée est juste ou s’écarte, avec un indicateur en cents et un verdict nommé. Son piano répond au clavier : Tabulation jusqu’à lui, choisissez une touche avec les flèches, puis **Entrée** pour l’entendre.

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

- Dans **Trouver la note que je chante**, une note confirmée est ajoutée au court historique même si l’interface est occupée ; une note tenue n’apparaît qu’une fois, et la rechanter après une coupure crée une nouvelle entrée.
- Ouvrez ou déposez un fichier WAV, MP3, FLAC ou OGG, ou choisissez l’enregistrement live. Pendant la prise, la page live sépare **Réglages de la prise**, **Votre prise en direct** (niveau du micro, notes confirmées, roll et portée) et **Retour sur votre chant** (justesse récente et tendance) : l’arrêt et ce que le micro entend restent faciles à repérer.
- Après l’analyse, le résumé **Votre transcription** garde le nombre de notes, la durée, le tempo et la tonalité visibles au-dessus de l’action principale **Exporter cette prise**. Ouvrez **Afficher les détails** pour consulter les diagnostics et réglages complets ; le panneau est replié au départ pour laisser la partition et l’action suivante au premier plan. Toute la colonne des résultats défile indépendamment : diagnostics, éditions, options d’export et actions d’analyse restent accessibles même dans une fenêtre basse. **Relire l’observation** applique rapidement le seuil de note minimale à l’évidence en cache ; **Réanalyser complètement (pYIN)** recalcule cette évidence depuis l’audio conservé, en arrière-plan et avec la progression des étapes. L’analyse actuelle reste affichée en cas d’échec ; une analyse réussie ne remplace le résultat qu’à la fin, et l’export reste explicite. Après une relecture ou une réanalyse complète réussie, un **différentiel de lecture** s’affiche à la place : la lecture précédente apparaît en barres translucides, les notes modifiées en accent, une ligne compacte Avant → Après rejoint le panneau (identiques / modifiées / arrondis proches / disparues / ajoutées, plus les suspects de cohérence de la lecture courante) et un détail développé liste chaque changement avec ses causes (hauteur, attaque, durée). Les suspects portent leur étiquette de différence (NOUVEAU / PERSISTE / RÉSOLU) ; la confiance pYIN n’est qu’un contexte, jamais un verdict. Toute frontière ultérieure (édition manuelle, annuler/rétablir, décision de remplacement, nouvelle prise ou nouveau fichier) efface le différentiel ; l’export reste explicite et n’écrit que la lecture courante. Vérifiez les notes, écoutez le piano ou la voix. Pour corriger une note, sélectionnez-la puis cliquez la bonne touche du piano. La correction se fait aussi au clavier : focalisez le clavier du piano, choisissez la hauteur avec **gauche/droite** (chaque choix s’entend), puis validez avec **Entrée** (**Échap** annule). La sélection se fait aussi directement sur le graphique : cliquez la barre d’une note pour la sélectionner, faites glisser sur les barres voisines pour sélectionner une plage, cliquez le vide pour effacer. Sur le roll des résultats, une barre se **déplace aussi à la souris** : glissez-la pour changer son attaque et sa hauteur, ou attrapez son bord droit pour la redimensionner — le déplacement compte comme une seule annulation. Une fois une barre choisie, les **flèches gauche/droite** déplacent la sélection note à note (Maj+flèche étend la plage), pour arrêter le curseur sur une note avant de la supprimer. Le **tableau des notes** se pilote aussi au clavier : **haut/bas** avancent d’une note, **début/fin** vont à la première ou à la dernière, **Maj+flèche** y étend la plage également, et **Espace** fait écouter la capture de cette note. Pendant l’écoute, son nom et sa position s’affichent avec une action **Arrêter**. La barre d’outils affiche la sélection courante (nombre de notes et plage temporelle), la barre d’état confirme chaque édition (*« N notes supprimées »*, le nom de la note corrigée) ; la touche **Suppr** supprime la plage sélectionnée — en une seule annulation. Annuler et rétablir (Ctrl+Z / Ctrl+Y, ou les boutons ↶/↷) reviennent sur les modifications de notes ; le menu déroulant au-dessus des boutons d’édition liste tous les états de la prise — choisissez-en un pour y revenir directement.
- Le panneau des résultats embarque un **rapport de justesse par prise** : la part de notes chantées justes, une tendance nommée (un tirage systématique au-dessus ou en dessous, annoncé seulement quand les données le soutiennent) et le même verdict par registre (graves / médiums / aigus). Les respirations font partie de la partition : les silences où une prise d’air a été réellement entendue portent un **B** sur la portée et `Respiration` dans le MusicXML exporté.
- Pendant que vous chantez, la page live affiche le **même retour de justesse** : à chaque note fermée, il lit la part de notes justes jusqu’ici et — quand assez de notes le soutiennent — la tendance en cours, avec les mots et l’échelle du rapport de prise, penchants par registre compris. Relâcher une barre déplacée joue aussi la note **avant, puis après** le déplacement, pour juger le changement à l’oreille comme une correction de hauteur.
- Après une prise live, sélectionnez une suite continue de notes et choisissez **Remplacer les notes sélectionnées** ; **Écouter l’original** joue la phrase sélectionnée telle qu’elle est, avant de la rechanter. Chantez le remplacement, terminez-le, écoutez la nouvelle phrase, puis choisissez **Garder le remplacement** (ou **Annuler le remplacement** pour rétablir la prise). Aucun fichier n’est écrit avant de choisir **Exporter cette prise**.
- Après l’arrêt d’une prise, les notes qu’elle a fermées restent sur le piano roll live : cliquez une barre pour la surligner et explorer la prise telle qu’elle a été chantée (lecture seule — la prise suivante repart propre). **Espace** fait écouter la note captée surlignée à partir de l’audio de la prise, avec une action **Arrêter** à côté. Depuis la page résultats, **Notes captées** ouvre ce même roll figé — la prise telle qu’elle a été captée, la lecture finale dessinée par-dessus (les notes que l’analyse hors ligne a déplacées portent l’accent), pour qu’une différence se lise d’un coup d’œil. La case **Lecture finale superposée** affiche ou masque uniquement cette couche ; **Retour aux résultats** revient à l’analyse.
- Dans l’application, **Exporter** ouvre « Enregistrer sous » : choisissez le nom du fichier et un format (MIDI, MusicXML, PDF, JSON ou CSV des notes). La boîte rouvre dans le dernier dossier utilisé, mémorisé entre les lancements ; au premier export, elle s’ouvre à côté du fichier source (ou dans le dossier courant pour une prise live). Le PDF reste proposé sans outil de gravure ; le choisir affiche alors un avertissement et ne crée pas de PDF. L’extension choisie est appliquée au nom du fichier. Les prises live gardent leur archive complète pour l’audit dans `prises_live/` et enregistrent aussi le fichier choisi à l’emplacement demandé ; l’audit live/final de cette archive (le bloc `live_vs_final` du rapport et son sidecar) porte les compteurs d’honnêteté temps réel de la capture : blocs perdus et pire progression du drain échantillonnée pendant la prise. La CLI conserve son export complet ; l’application et `v2m live` partagent le même reçu et le contrat d’archive live.

### Raccourcis clavier

| Raccourci | Action |
| --- | --- |
| Tabulation | Parcourir les actions de la page dans l’ordre (pages live et cherche-note) |
| Ctrl+Z | Annuler la dernière modification de notes (page résultats) |
| Ctrl+Y | Rétablir la dernière modification annulée (page résultats) |
| Haut / Bas | Déplacer la sélection d’une note (page résultats) |
| Début / Fin | Aller à la première / dernière note (page résultats) |
| Espace | Écouter la note sélectionnée ; appuyer de nouveau pour arrêter (page résultats, et roll live figé) |
| Gauche / Droite | Choisir une hauteur sur le piano focalisé (elle s’entend) ; Entrée valide, Échap annule |
| Entrée | Cherche-note : entendre la touche choisie sur le piano |
| Suppr | Supprimer la plage de notes sélectionnée (page résultats) |

Ctrl+Y est la touche de rétablissement ; Ctrl+Shift+Z fonctionne aussi. Les deux raccourcis agissent depuis la page résultats, où vivent les modifications.

## Ligne de commande

```bash
v2m analyze chant.wav --quantize
v2m analyze chant.wav --pdf
v2m live
v2m --lang en analyze chant.wav
```

`analyze` accepte `--min-conf`, `--min-note`, `--transpose`, `--quantize`, `--pdf` et `-o`/`--output`. Il écrit MIDI, MusicXML, JSON et CSV. `--pdf` ajoute un PDF si MuseScore ou LilyPond est installé. Si l’outil manque, seul le PDF est ignoré ; un échec de conversion renvoie le code 2. Dans le MIDI, les notes tenues partagent une pédale sustain globale : les fenêtres qui se chevauchent ou se touchent sont fusionnées afin qu’une note ne relève pas l’étouffoir pendant la suivante. `--lang fr` et `--lang en` choisissent la langue de l’interface.

## Diagnostics

Ces outils sont facultatifs. Depuis la racine du projet, dans l'environnement activé.

- `scripts/score_accuracy.py` : examiner une transcription. `--truth` la compare à des notes connues ; `--json` écrit un rapport ; `--output` choisit le dossier des artefacts ; `--no-attacks` ignore les attaques ; `--compare` vérifie la répétabilité ; `--golden` vérifie une référence enregistrée ; `--diagnose` indique les seuils testés par la prise.
- `scripts/profile_analysis.py` : mesurer le temps d’analyse. `--repeat` règle le nombre d’exécutions ; `--profile` ajoute un profil des fonctions ; `--top` limite ce tableau ; `--json` écrit les mesures ; `--output` choisit le dossier des artefacts ; `--no-warmup` inclut le coût du premier passage. Le tableau se termine par une ligne `Rescue` : le coût propre du balayage à fenêtre courte et la durée d’audio greffée dans la courbe de hauteur (0.0 quand les gardes ont tout refusé).
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

La suite complète a un budget de 150 secondes. La boucle rapide omet les tests marqués `slow`. Le smoke test graphique utilise `demo.wav` (ignoré par git) ; créez-le avec `python scripts/make_demo.py`. Pour le travail sur le dépôt, les agents utilisent les issues GitHub et lisent les consignes de domaine dans `CONTEXT.md` et les décisions pertinentes de `docs/adr/` ; voir `docs/agents/`.
