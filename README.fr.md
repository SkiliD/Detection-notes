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

pip install -e ".[dev]" -c constraints.txt
```

Le fichier de contraintes épingle les versions de bibliothèques sur lesquelles les références audio ont été mesurées ; sans lui, un résolveur peut installer un librosa ou un numba plus récents et signaler un autre moteur comme une régression.

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
- **Langue et contrôles** : la fenêtre tient les mêmes règles que la page web. Les nombres français s'écrivent à la française (`1,0 s`, et non `1.0 s`) parce qu'un seul formateur écrit chaque décimale, repères de seconde du roll compris ; chaque contrôle mesure au moins 24 px dans les deux sens ; et chaque contrôle focalisable dessine le même anneau de focus cuivré de 2 px (3:1 contre la surface derrière lui) quand le clavier l'atteint.

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

Pour rejouer les deux mesures de la page seules, sans lire le workflow (Node 22 et un Chrome/Chromium que le banc trouve ; `CHROME` le remplace) :

```bash
mkdir -p build
node web/test/resolutions.mjs
curl -sSfL -o build/axe.min.js https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js
node web/test/axe-audit.mjs --axe build/axe.min.js
```

La ligne `curl` télécharge l'axe-core 4.10.2 épinglé, le même que le workflow, et `node web/test/axe-audit.mjs` lance l'audit lui-même, qui couvre les deux palettes dans trois états aux huit largeurs.

Pour la publier, réglez **Settings → Pages → Source** sur **GitHub Actions**, puis poussez sur `main` : le workflow rejoue les cas de conformité du portage, photographie la page à 1600 px, mesure huit classes de résolution (de 320 px à 2560 px, téléphone à l'horizontale inclus), audite les deux palettes dans trois états à chacune d'elles avec axe-core, puis déploie `web/` — et relit enfin l'URL publiée, sur le réseau : les en-têtes, les deux fichiers dont la page a besoin, la chaîne de la démonstration et le changement de langue, pour qu'une page servie depuis un sous-dossier de dépôt comme depuis la racine d'un site fonctionne pour un visiteur, et pas seulement dans un dépôt. Elle échoue aussi sur une page qui n'est pas un **contexte sécurisé** (`isSecureContext` et `https:`) : hors de là, le navigateur cache le micro entièrement, donc **Settings → Pages → Enforce HTTPS** devient une condition mesurée à chaque déploiement, et non un réglage pris sur parole. Cette lecture tourne **après** le déploiement : elle juge une page déjà en ligne, donc un run rouge signale un site cassé pour les visiteurs, et rien n'est retiré automatiquement — ni retour arrière, ni dépublication. Le remède est de corriger puis de repousser.

La publication exige aussi le job **micro** : le chemin d'enregistrement de la page est piloté sur le faux appareil de Chrome avec un son généré, et une capture rouge bloque désormais le déploiement au lieu d'avertir à côté. La seule façon de passer ce gate est la variable de dépôt `MIC_CAPTURE_WAIVED=true` — une dérogation délibérée et tracée, pour une recette de runner cassée et non pour une page cassée ; le run porte alors un avertissement nommant ce qui n'a pas été prouvé, et la dérogation se retire dès que le job est vert. Seul un mainteneur peut la poser — c'est un réglage de dépôt, pas un fichier — et c'est une commande dans chaque sens, lancée depuis ce dépôt :

```bash
gh variable set MIC_CAPTURE_WAIVED --body true
gh variable delete MIC_CAPTURE_WAIVED
```

Les deux mêmes valeurs se règlent dans **Settings → Secrets and variables → Actions → Variables** quand `gh` n'est pas sous la main. La dérogation prend effet au run suivant : la poser, relancer le déploiement en échec, puis la retirer une fois `capture` vert.

Les tests de la page pilotent **un seul** moteur, le Chrome du runner. Un défaut propre à WebKit est déjà passé comme ça : son `ReadableStream` n'a pas d'itérateur asynchrone, le compresseur de l'archive levait donc une erreur et toutes les entrées étaient stockées — un ZIP valide, 1,5× trop gros, suite verte. `.github/workflows/tests.yml` rejoue donc les cinq contrôles qui comptent (le segmenteur porté contre les cas du bureau, YIN à la géométrie live, l'archive exportée, un fichier que vous choisissez, et une courte phrase chantée relue par la chaîne live du bureau) dans Chromium, Firefox **et** WebKit. La phrase est `tests/data/parity.wav`, versionnée avec la lecture qu'en fait le bureau dans `tests/reference/parity.json` à chaque taux où un navigateur peut la décoder : le `notes.json` de la page est donc comparé au bureau note par note, et non à un fixture de trames de hauteur. Le banc vit dans `tests/web_matrix/` et emprunte un Playwright épinglé : `web/` ne garde aucune dépendance.

Le **budget mémoire** d'une prise longue se mesure, il ne se suppose pas : à la demande, la page est poussée dans une capture de **trois minutes** sur le faux appareil de Chrome, et la comptabilité du navigateur est lue avant la prise, après elle et après l'archive — la prise est gardée une fois pendant qu'elle se chante, et un export relâche la concaténation qu'il a empaquetée : l'archive n'ajoute donc pas de seconde copie — et le run échoue au-delà d'un plafond explicite. Le coût stable est de **4 octets par échantillon** (11,5 Mo par minute à 48 kHz, 23 Mo à 96 kHz) plus quelques Mo de page, et un export culmine à environ **2,5× la prise** pendant qu'il l'empaquette — mesuré à 178 Mo pour la prise de trois minutes et tenu sous un plafond de 2,8× par le run lui-même — : c'est cette marge qu'un téléphone doit supporter une fois. C'est facultatif, car trois minutes de temps réel sont trop chères à chaque push :

```bash
V2M_LONG_TAKE=1 node --test web/test/long-take.test.mjs
V2M_LONG_TAKE=1 V2M_LONG_TAKE_RATE=48000 node --test web/test/long-take.test.mjs
```

Une machine sans sortie audio saute le contrôle, et `V2M_REQUIRE_AUDIO=1` transforme ce saut en échec. `V2M_LONG_TAKE_RATE` force le taux du graphe, pour rejouer le même budget là où l'appareil annonce 48 kHz (mesuré : signal 34,6 Mo, +36,3 Mo après capture, +36,4 Mo après l'archive) au lieu de se fier à celui de 96 kHz ; un navigateur qui refuse ce taux échoue au lieu de mesurer l'ancien.

La page porte aussi sa propre **Content-Security-Policy**, déclarée dans l'en-tête : tout vient de `'self'`, plus l'unique `blob:` dont le module de l'AudioWorklet est construit — aucun CDN, aucun script en ligne, aucun `eval`, donc la politique n'a pas besoin de `'unsafe-inline'`. `web/test/csp.test.mjs` fixe les directives et pousse la page (démonstration, écoute, archive, capture réelle) pendant que la page enregistre chaque `securitypolicyviolation` et que le journal du navigateur est relu pour un refus.

La page se parcourt aussi **au clavier seul** (`web/test/keyboard.test.mjs`) : de vrais événements Tab, Entrée et Espace fixent deux fois l'ordre de focus — avant une prise, puis quand elle porte des notes —, ouvrent le sélecteur de fichier depuis son champ et exportent les deux fichiers. Espace est le seul raccourci de la page et il est refusé tant qu'un contrôle possède la touche : un seul Espace sur **Écouter la prise** la lance une fois, et un Espace sur l'équilibre ne déplace rien. Une flèche maintenue ne reprend la prise qu'une fois : ses répétitions automatiques sont ignorées et un seek en cours de redémarrage refuse le suivant, donc la touche tenue ne peut pas lancer deux départs en course jusqu'à « La lecture n'a pas pu démarrer ». Chaque contrôle qui prend le focus dessine l'anneau cuivré de 2 px de la page, et `tests/test_web_build.py` relit cet anneau dans la feuille en regard du balisage de chacun : un sélecteur renommé échoue donc à la suite ordinaire avant même que la marche au navigateur ne tourne. `web/test/craft-floor.test.mjs` mesure l'anneau que le navigateur peint réellement : 3:1 contre la surface derrière lui dans les deux palettes — les pilules du bandeau sont mesurées sur la bande de nuit, pas sur le papier — et `tests/test_web_build.py` fixe la même paire depuis les jetons (cuivre sur papier : 3,74:1 en clair, 8,19:1 en foncé).

Ce qu'un lecteur d'écran entend du bandeau est le **nom de la note**, pas le flux de trames : `#note` est la seule région live polie de la page, réécrite uniquement quand le nom change, tandis que la ligne des Hz à côté reste hors de cette région parce qu'elle bouge à chaque trame. `web/test/craft-floor.test.mjs` fixe le placement et les deux portes de changement.

Chaque nombre affiché par la page — les dixièmes de l'horloge, la ligne des Hz, les secondes du tableau, les repères de seconde du roll, et les deux comptes de notes et le pourcentage de la ligne d'intonation — passe par un seul `Intl.NumberFormat` : une page française lit `3,2 s` là où une page anglaise lit `3.2 s` ; `web/test/format.test.mjs` fixe les deux et `web/test/craft-floor.test.mjs` vérifie le tableau rendu dans chaque langue.

Une page dont le système demande moins de mouvement coupe toutes les transitions — boutons, liens et lissage de la jauge de niveau compris — et `web/test/craft-floor.test.mjs` relit le bloc et les règles qu'il nomme, puis émule le réglage du lecteur dans Chrome et exige que les transitions calculées de `button`, `.link` et `.meter-fill` valent `0s` dans les deux palettes, tandis que `tests/test_web_build.py` redérive de la feuille chaque famille animée et l'exige dans le bloc : ni un sélecteur renommé ni une nouvelle transition ne peuvent donc laisser un contrôle s'animer.

Les titres équilibrent leurs lignes et la prose courante se replie en `pretty`, et un contrôle inactif — bouton, lien texte ou curseur d'équilibre — garde son mot (ou sa forme) sur un champ discret au lieu de s'effacer tout entier.

L'enregistrement que vous choisissez est lu par le décodeur du navigateur : un fichier passe par la même chaîne qu'une prise chantée au micro, et un fichier illisible se nomme lui-même au lieu d'accuser l'appareil.

**Chanter** demande le micro une fois, et quand il ne s'ouvre pas la page nomme la raison : permission refusée à ce site, aucun appareil utilisable, appareil tenu par une autre application, ou page ouverte en fichier local ou en http simple, où le navigateur n'autorise jamais la question. La console du navigateur note l'étape qui a échoué et le nom de l'erreur, pour un rapport de bug. Sans micro, **Démonstration** charge une phrase chantée lue par la même chaîne. La page joue les notes sur le piano du bureau, les suit d'une tête de lecture, et s'imprime en noir sur blanc. Quand le navigateur ne peut pas démarrer de sortie audio — pas de carte son, service audio arrêté — **Écouter** annonce que la lecture n'a pas pu démarrer au lieu de laisser sa phrase de préparation à l'écran : la page borne ce démarrage comme celui du micro (`web/test/playback-failure.test.mjs`). Une lecture qui échoue écrit aussi sa ligne de console — l'étape en cours et l'état du graphe audio — pour qu'un rapport de bug puisse dire pourquoi sans capture d'écran, et une sortie lente qui s'ouvre au bout d'une seconde joue quand même.

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
- `scripts/coherence_summary.py` : résumer les audits de prises par session ; `--lang` choisit les libellés et les nombres français (`1,25`) ou anglais (`1.25`).
- `scripts/replay_takes.py` : rejouer des prises en temps réel et enregistrer la comparaison live/final — `--out`, `--take`, `--root`, et `--changed` pour les prises mesurées par un moteur plus ancien.
- `scripts/engine_drift_watch.py` : rafraîchir les comparaisons périmées — `--root`, `--report`, `--json` ; sort en 1 quand une lecture a changé.
- `scripts/compare_takes.py` : comparer deux prises d'une même phrase ; `--json` écrit la comparaison.
- `scripts/make_demo.py` génère le `demo.wav` ignoré par git dont le test de fumée graphique a besoin.
- `scripts/import_bench.py` reproduit la mesure d'import WSL : il copie les paquets du venv courant sur le disque Windows et donne la médiane des imports à chaud depuis ext4 et /mnt/c.
- `scripts/check_vendor_lock.py` : recalcule le hash du dossier vendorisé et sort en 1 quand `vendor-lock.json` ne correspond plus — hors ligne, sans pytest.

## Vérifications

```bash
python -m pytest tests/ -q
pytest tests/ -m "not slow"
QT_QPA_PLATFORM=offscreen python -u scripts/gui_smoke_test.py
```

L'intégration continue exécute ces trois commandes dans `.github/workflows/tests.yml`, à chaque push et pull request, avec le budget de la suite relevé pour le runner partagé. La suite complète a un budget de 150 secondes et tourne sur deux processus pytest-xdist — gardez 2, car à 4 les empreintes de référence des vraies prises se disputent le cache d'évidence. La boucle rapide saute les tests marqués `slow`.

### Lancer la suite sous WSL

La moitié Linux du garde tourne dans une distribution WSL, et son environnement virtuel vit dans le système de fichiers Linux (`~/.venvs/v2m`), jamais sous /mnt/c : le coût est mesuré, pas supposé — importer la pile de la suite (numpy, scipy, librosa, numba, soundfile, PyQt6) depuis le même `site-packages` (20 158 fichiers, 840 Mo) prend 2,86 s depuis /mnt/c contre 0,21 s sur ext4 (médiane de 7 exécutions à cache chaud, ×13,5). Le dépôt lui-même peut rester sur le disque Windows — WSL lit très bien les sources à travers le pont ; ce sont l'interpréteur et ses paquets qui doivent être locaux. Construisez-le avec le Python de la distribution et les contraintes épinglées (le paquet `python3-venv` de la distribution est ce qui fait fonctionner `python3 -m venv`), ajoutez le moteur audio (`libportaudio2`, `libasound2t64`, `libjack-jackd2-0` ; sans serveur de son PortAudio ouvre 0 appareil et la suite passe quand même) et les bibliothèques Qt dont le test de fumée GUI a besoin (`libegl1`, `libgl1`, `libglib2.0-0t64`, `libxkbcommon0`, `libdbus-1-3`, `libfontconfig1`) — `libglib2.0-0t64` est celle qu'une distribution nue n'obtient pas des autres : sans elle, tout import `PyQt6` meurt sur `libglib-2.0.so.0: cannot open shared object file`, mesuré sur une `ubuntu-base-24.04.5` propre (le suffixe `t64` est le nom de 24.04 ; 22.04 l'écrit `libglib2.0-0`), alors que `libsndfile1` n'est PAS nécessaire, car la roue `soundfile` embarque la sienne (mesuré 1.2.2) — puis lancez la suite en série. `scripts/import_bench.py` reproduit le chiffre d'import ; sa première exécution copie les paquets dans le dossier public Windows, ce qui est la partie lente :

```bash
wsl -d Ubuntu -- bash -lc 'sudo apt-get update && sudo apt-get install -y python3-venv libportaudio2 libasound2t64 libjack-jackd2-0 libegl1 libgl1 libglib2.0-0t64 libxkbcommon0 libdbus-1-3 libfontconfig1'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && python3 -m venv ~/.venvs/v2m && ~/.venvs/v2m/bin/pip install -e ".[dev]" -c constraints.txt'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && QT_QPA_PLATFORM=offscreen XDG_RUNTIME_DIR=/tmp/v2m-run LIBGL_ALWAYS_SOFTWARE=1 V2M_SUITE_BUDGET_S=900 PYTHONFAULTHANDLER=1 ~/.venvs/v2m/bin/python -m pytest tests/ -q -n 0'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && ~/.venvs/v2m/bin/python scripts/make_demo.py'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && install -d -m 700 /tmp/v2m-run && QT_QPA_PLATFORM=offscreen XDG_RUNTIME_DIR=/tmp/v2m-run LIBGL_ALWAYS_SOFTWARE=1 ~/.venvs/v2m/bin/python -u scripts/gui_smoke_test.py'
wsl -d Ubuntu -- bash -lc 'cd "/path/to/Detection notes" && ~/.venvs/v2m/bin/python scripts/import_bench.py'
```

La suite et le test de fumée GUI hors écran sont les deux moitiés du garde desktop ; la CI lance en plus les tests Node de la page. Le test de fumée lit `demo.wav` (non versionné), donc régénérez-le d'abord comme le fait la CI, et lancez-le avec les mêmes variables `QT_QPA_PLATFORM`, `XDG_RUNTIME_DIR` et `LIBGL_ALWAYS_SOFTWARE`.

`-n 0` est la forme série que l'intégration continue utilise aussi, pour qu'un plantage natif nomme son test au lieu d'emporter la sortie d'un processus ; `QT_QPA_PLATFORM=offscreen` parce que la distribution n'a pas d'affichage ; `XDG_RUNTIME_DIR` et `LIBGL_ALWAYS_SOFTWARE` parce que Qt et PortAudio veulent un dossier d'exécution et pas de GPU ; `V2M_SUITE_BUDGET_S` parce qu'une distribution WSL est plus lente que la machine sur laquelle le budget de 150 secondes a été calibré (mesuré 322 s) ; `PYTHONFAULTHANDLER` pour qu'un plantage natif imprime la pile de chaque fil. Mettez le chemin du dépôt entre guillemets (il contient en général une espace), et gardez les recettes en plusieurs étapes dans un script : `wsl.exe` consomme `$VAR` et `$?` avant que bash ne les voie.

Pour le travail sur le dépôt, passez par les GitHub Issues ; les notes de conception derrière le code (`AGENTS.md`, `CONTEXT.md`, `SKILLS.md`, `constraints.txt`, la documentation projet sous `docs/project/` et la recette matérielle `docs/recette-materielle-navigateurs.md`) sont versionnées et suivent un clone. Les décisions sous `docs/adr/`, les rapports datés (`*_YYYY-MM-DD.md` à la racine du dépôt), les captures (`page-prise-*.png`), `docs/engine-drift.md` et `docs/superpowers/` restent locaux et ne sont pas versionnés.

Un seul dossier tiers de fabrication d'interface est vendorisé à dessein, pour que chaque clone relise les mêmes consignes : `vendor/make-interfaces-feel-better/`, copié depuis jakubkrehel/make-interfaces-feel-better au commit épinglé par `vendor-lock.json`. Le dossier est identique octet pour octet à l'amont — le hash du verrou est celui de l'amont — et sa licence MIT voyage à côté, dans `vendor/make-interfaces-feel-better-LICENSE` (Copyright (c) 2026 Jakub Krehel) ; `.gitattributes` épingle le dossier et le verrou en LF parce que le verrou hache leurs octets : `scripts/check_vendor_lock.py` et `tests/test_docs_guard.py` vérifient ce même hash ainsi que le suivi du dossier, de la licence et du verrou et le LF du verrou sur le disque, et un déplacement amont est un re-vendor délibéré, jamais une mise à jour automatique. Rien d'autre sous `vendor/` ne suit un clone, et rien ici ne lit un dossier vendorisé à l'exécution.

## Licence

MIT — voir `LICENSE`.
