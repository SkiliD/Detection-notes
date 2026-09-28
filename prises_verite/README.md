# prises_verite/ — corpus de prises vérifiées à la main

Ce dossier rassemble des **prises chantées dont chaque note a été vérifiée à
l'oreille** par un humain : c'est la vérité terrain qui manque pour juger la
segmentation sur des voix réelles (le harnas synthétique `tests/voicetruth.py`
couvre les mécanismes, pas les voix).

## Contenu

- `<nom>.wav` : la prise (mono ; le moteur la ramène à 22 050 Hz).
- `<nom>.truth.json` : la vérité vérifiée, au format ci-dessous.

## Format `<nom>.truth.json`

```json
{
  "name": "ma_prise",
  "audio": "ma_prise.wav",
  "recorded": "2026-09-25",
  "checked_by": "humain",
  "notes": [
    {"midi": 64, "start": 0.52, "end": 1.10, "lyric": "la"},
    {"midi": 67, "start": 1.25, "end": 1.90}
  ]
}
```

- `midi` : hauteur de la note (entier, la plus proche — ±50 cents tolérés,
  le moteur teste l'entonnoir de lecture, pas l'oreille absolue).
- `start` / `end` : secondes, au dixième près suffit (durée d'appui de la
  note : début de l'attaque, fin du son — la respiration suivante est hors
  note).
- `lyric` (optionnel) : la syllabe chantée, pour reconnaitre la prise à la
  relecture.
- `checked_by` : toujours `"humain"` — une prise non vérifiée n'est pas une
  vérité.

Le guide d'enregistrement `scripts/guide_capture.py` écrit ce fichier pour
vous (note par note, au rythme où vous les chantez), puis le vérifie
contre la prise :

    ./.venv/Scripts/python.exe scripts/guide_capture.py ma_prise.wav
    ./.venv/Scripts/python.exe scripts/guide_capture.py ma_prise.wav --audit

## Règles

- Une prise = une phrase simple, 4 à 12 notes, articulées nettement.
- Une variante utile : la même phrase chantée DEUX FOIS de suite (les mêmes
  hauteurs, articulées) — c'est la prise qui montre si les notes rechantées
  restent séparées ou se collent en une continuité (le retour du
  2026-09-26 : « répétition parfois aléatoire »).
- Le JSON est validé (temps croissants, pas de chevauchement, midi piano)
  à chaque audit : un fichier invalide est refusé, pas corrigé.
- Les prises et vérités sont gitignorées (contenu personnel) ; seul ce
  README engage le format.
