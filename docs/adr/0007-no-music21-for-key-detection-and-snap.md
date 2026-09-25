# ADR 0007 — Pas de music21 pour la tonalité et le collage à la gamme

Date : 2026-09-25
Statut : accepté

## Contexte

La correction du collage à la gamme (demande utilisateur : « attirer les
notes ambiguës vers les degrés de la gamme détectée, music21 le fait tout
seul ») tombait sur un choix de bibliothèque : importer `music21` pour la
détection de tonalité (Krumhansl-Schmuckler) et éventuellement l'analyse
harmonique.

## Décision

Refusé : la détection de tonalité reste dans `core/keys.py`, le collage dans
`core/keys.snap_to_key`, tous deux sans dépendance.

## Raisons (mesurées)

1. **La partie utile existe déjà.** `keys.detect_key` implémente
   Krumhansl-Schmuckler (profils KS_MAJ/KS_MIN, corrélation de Pearson sur
   l'histogramme pondéré par durée) et fournit déjà les candidates de rechange
   — c'est exactement le calcul que music21 rendrait. Le collage demandé est
   un autre problème : il lit la **déviation résiduelle** de chaque note
   (ce que la correction d'accord global n'a pas absorbé), que music21 ne
   manipule pas — sa quantification de hauteur n'a pas de notion de « note à
   mi-chemin entre deux demi-tons ». L'import n'aurait porté que les
   constantes KS, à réécrire de toute façon.
2. **Coût d'image réel.** music21 tire xml.etree + argparse + un arbre
   d'import lourd ; le noyau est importé à CHAQUE analyse (CLI, GUI, live)
   et l'image slim du Docker ne transporte aucune dépendance inutile (cf.
   ADR 0006 pour la même logique sur torch). La détection doit rester
   import-light : `LiveTake` en dépend en flux.
3. **Le test prime sur le démonstrateur.** Les profils KS et le snap sont
   épinglés par `tests/test_keys.py` et la nette de référence
   (`tests/reference/artifacts.json`, 2 vraies prises) : remplacer le
   moteur n'achète pas de justesse prouvée, seulement une dépendance.

## Quand rouvrir

Si un besoin réel d'analyse harmonique avancée (démodulation de voix à
plusieurs voix, tonalités à micro-intervalles, etc.) émerge — et avec un
corpus de vérité terrain qui puisse arbitrer le moteur. L'interface
`KeyCandidate`/`snap_to_key` est conçue pour brancher un autre détecteur
sans toucher les consommateurs.

## Conséquences

- `core/keys.py` reste le seul point d'entrée (détecteur + collage) ; il ne
  doit pas dépendre de music21, numpy seul.
- La convention « tout traducteur est une table pinnée à son producteur »
  s'étend : `tr_key_label` reste pinné contre `KEY_PROFILES` (inchangé).
- Le pas de temps du collage reste une passe locale (une lecture de la
  déviation par note) : pas d'état, pas de régression possible sur le
  découpage temporel.
