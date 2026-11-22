"""User-facing language selection (French / English).

`set_language("fr" | "en")` switches every user-visible string at runtime via
`tr()`, including the note-name format (english A B C vs french Do Ré Mi).
Code identifiers and the figure names stored on notes stay canonical French
internally (`noire`, `triolet de croche`...) — `tr_figure` translates them
for display only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .music import NOTE_NAMES, NOTE_NAMES_FR, midi_to_name, midi_to_name_fr

LANG_FILE = Path.home() / ".v2m_lang"

SUPPORTED_LANGUAGES = ("fr", "en")

# User-facing strings. English defaults to the key itself ("key tempo" -> "Tempo").
STRINGS = {
    "fr": {
        "ui guide title": "Prochaine action",
        "ui guide progress": "Passage {current} sur {total}",
        "ui guide next": "Passage douteux suivant",
        "ui guide next hint": "Vérifiez ou corrigez ces notes, puis passez au doute suivant.",
        "ui guide tools": "Tous les outils",
        "ui guide hide tools": "Masquer les outils",
        "ui guide skip": "Passer à l’export",
        "ui guide listen passage": "Écouter ce passage",
        "ui guide listen take": "Écouter la prise",
        "ui guide stop": "Arrêter l’écoute",
        "ui guide doubt": "Un passage mérite une écoute : c’est un doute, pas une erreur certaine.",
        "ui guide listen": "Réécoutez votre prise avant de choisir si vous souhaitez la retoucher.",
        "ui guide review": "Vérifier les notes de ce passage",
        "ui guide review hint": "Si vous entendez un écart, ouvrez les notes pour le vérifier.",
        "ui guide export hint": "Vous pouvez exporter la lecture actuelle, avec ou sans retouche.",
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
        "key tuning offset": "Accordage",
        "key intonation": "Justesse",
        "ui intonation share": "{share} % de notes justes",
        "ui intonation tendency": "tendance : {dir}",
        "ui intonation sharp": "au-dessus ({cents:+.0f} c)",
        "ui intonation flat": "en dessous ({cents:+.0f} c)",
        "ui intonation even": "stable",
        "ui intonation registers": "registres : {rows}",
        "ui intonation position": "position : {rows}",
        "ui position start": "début",
        "ui position end": "fin",
        "ui position lean": "{half} {cents:+.0f} c",
        "ui register low": "graves",
        "ui register mid": "médiums",
        "ui register high": "aigus",
        "ui register lean": "{band} {cents:+.0f} c",
        "key coherence": "Cohérence",
        "ui tuning ok": "juste, continuez comme ça",
        "ui tuning slightly off": "à {dir} de quelques cents, à surveiller",
        "ui tuning off": "nettement {dir} ({cents:+.0f} c)",
        "ui tuning sharp": "aigu",
        "ui tuning flat": "bas",
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
        "progress tempo": "Détection du tempo…",
        "progress pitch": "Suivi de la hauteur (pYIN)…",
        "progress notes": "Segmentation en notes…",
        "progress key": "Détection de la tonalité…",
        "progress chords": "Détection des accords…",
        "progress refine": "Raffinement du tempo…",
        "progress figures": "Étiquetage des figures de notes…",
        "progress rests": "Détection des respirations…",
        "progress done": "Terminé",
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
        "err sounddevice missing": "sounddevice indisponible (pas de périphérique audio ?)",
        "err take stopped": "cette prise live est déjà arrêtée",
        "err live incomplete": "analyse live incomplète",
        "err export incomplete": "export_result exige une analyse complète (tempo + tonalité)",
        "err export needs meter": "export_result exige une mesure",
        "err pdf install hint": "Installez MuseScore (https://musescore.org) ou LilyPond (https://lilypond.org) puis vérifiez qu'il est dans le PATH.",
        "err pdf not runnable": "« {cmd} » introuvable à l'exécution. {hint}",
        "err pdf skipped": "PDF ignoré : {hint}",
        "err pdf no tool": "Aucun outil de gravure trouvé (MuseScore, LilyPond). {hint}",
        "err pdf timeout": "{name} n'a pas répondu (délai de 120 s).",
        "err pdf conversion": "Échec de la conversion via {name} : {msg}",
        "err pdf exit code": "code de sortie {code}",
        "err ui playback": "Lecture audio impossible : {e}",
        "ui window title": "Voix → MIDI — notes, tempo, tonalité et accords",
        "ui home title": "Donnez une forme à votre mélodie.",
        "ui theme label": "Apparence",
        "ui theme dark": "Sombre",
        "ui theme light": "Claire",
        "ui home microphone": "Micro pour la prochaine capture",
        "ui app subtitle": "Chantez ou ouvrez un enregistrement pour retrouver ses notes.",
        "ui open file": "Ouvrir un fichier",
        "ui sing live": "Chanter une mélodie",
        "ui finder button": "Trouver une note",
        "ui input device default": "Micro système par défaut",
        "ui input device default name": "Micro système par défaut",
        "ui input device active": "Micro utilisé : {device}",
        "ui input device next": "Prochaine capture : {device}",
        "ui input device fallback": "Micro choisi absent — micro utilisé : {device}",
        "ui input device fallback next": "Micro absent — défaut système prochain : {device}",
        "ui input device effects": (
            "Le pilote Windows publie une chaîne d'effets pour ce micro : la prise peut être "
            "filtrée avant l'analyse. Désactivez les améliorations audio dans les réglages "
            "du son pour enregistrer le signal brut."
        ),
        "ui input device tooltip": (
            "Choisissez le micro utilisé pour la calibration, le cherche-note et les prises live. "
            "Le changement s'applique à la prochaine capture."
        ),
        "ui input device missing": (
            "Le micro choisi n’est plus disponible : {device}. "
            "Le micro système par défaut sera utilisé."
        ),
        "ui input device unavailable": "Indisponible — retour au micro système : {device}",
        "ui finder start": "▶  Démarrer l'écoute",
        "ui finder stop": "■  Arrêter l'écoute",
        "ui finder idle": "♪  Chantez une note…",
        "ui finder live": "●  Écoute en cours — chantez une note",
        "ui finder off": "Écoute arrêtée",
        "ui finder waiting": "En attente d'un son…",
        "ui finder silence": "Aucun son au-dessus du seuil du micro.",
        "ui finder no note": "Son reçu. Tenez une note un instant.",
        "ui finder note heard": "Hauteur détectée.",
        "ui finder hint": "Chantez une note : sa touche s'allume sur le clavier du piano.\nCliquez une touche pour l'entendre.",
        "ui finder history": "Dernières notes entendues",
        "ui finder history empty": "Les notes confirmées apparaîtront ici, à réécouter.",
        "ui finder history tooltip": "Réécouter {note}",
        "ui finder answer": "Ce que le micro entend",
        "ui finder range": "Étendue du clavier",
        "ui finder range voice": "Voix ({lo} – {hi})",
        "ui finder range low": "Voix grave ({lo} – {hi})",
        "ui finder range all": "Tout le clavier ({lo} – {hi})",
        "ui finder range tooltip": (
            "Choisissez les touches affichées ; le clavier ne se recentre jamais."
        ),
        "ui tuning lower": "♭ plus grave",
        "ui tuning centre": "juste",
        "ui tuning higher": "plus aigu ♯",
        "ui level idle": "niveau : —",
        "ui expected none": "Aucune note attendue",
        "ui expected note": "Attendue : {note}",
        "ui live hint": "Chantez devant la partition : la touche devient verte si la note correspond, rouge sinon.",
        "ui live controls": "Réglages de la prise",
        "ui live monitor": "Votre prise en direct",
        "ui live feedback": "Retour sur votre chant",
        "ui live details": "Détails de la prise live",
        "ui show live details": "＋ Afficher les détails",
        "ui hide live details": "− Masquer les détails",
        "ui take summary": "Votre transcription",
        "ui summary facts": "{notes} notes · {duration} s · {tempo} · {key}",
        "ui show details": "＋ Afficher les détails",
        "ui hide details": "− Masquer les détails",
        "ui metronome": "Métronome",
        "ui metronome tooltip": "Clic régulier pour travailler la régularité du chant (désactivé par défaut).",
        "ui record audio": "Enregistrer l'audio de la prise",
        "ui record audio tooltip": "Écrit la prise en .wav dans le dossier de sortie du mode live (coché par défaut — une prise sans audio reste inaudible rétroactivement).",
        "ui bpm unit": "BPM",
        "ui beat count": "Temps {beat}",
        "ui accuracy title": "Justesse moyenne :",
        "ui accuracy empty": "Justesse moyenne : —",
        "ui play piano": "Piano",
        "ui play voice": "Voix",
        "ui play voice tooltip": "Écouter seulement votre voix enregistrée.",
        "ui play both": "Ensemble",
        "ui play both tooltip": "Piano par-dessus ta voix, au volume réglé — pour comparer l'interprétation.",
        "ui group listen": "Écouter",
        "ui group adjust": "Ajuster",
        "ui group export": "Exporter",
        "ui view notes": "Notes",
        "ui view notes tooltip": "Le piano roll : les notes s'y lisent, se sélectionnent et se corrigent.",
        "ui view staff": "Partition",
        "ui view wave": "Forme d'onde",
        "ui view wave tooltip": "La forme d'onde de l'audio, avec le curseur de lecture.",
        "ui volume": "Vol. piano",
        "ui volume tooltip": "Volume du piano rejoué, pour l'entendre par rapport à ta voix.",
        "ui axis note": "Note",
        "ui axis time": "Temps",
        "ui drop hint": "Astuce : glissez-déposez un fichier audio dans cette fenêtre.",
        "ui lang button": "Langue : Français",
        "ui back home": "← Accueil",
        "ui no file": "Aucun fichier",
        "ui export options": "Options d'export",
        "ui opts transcription": "Transcription",
        "ui advanced settings": "Réglages avancés",
        "ui hide advanced settings": "− Masquer les réglages",
        "ui export tooltip": (
            "Écrit le MIDI, la partition MusicXML et le rapport JSON de cette prise ; "
            "le PDF s'ajoute si sa case est cochée."
        ),
        "ui quantize": "Quantifier sur la grille 1/16",
        "ui transpose": "Transposition (demi-tons)",
        "ui notation mode": "Mode notation (figures colorées)",
        "ui pdf option": "Exporter aussi en PDF",
        "ui pdf tooltip": "Installez MuseScore ou LilyPond pour activer l'export PDF.",
        "ui re-export": "Ré-exporter avec ces options",
        "ui export": "Exporter la transcription",
        "ui replace phrase": "Remplacer les notes sélectionnées",
        "ui finish replacement": "Terminer le remplacement",
        "ui edit hint": "Sélectionnez une note ou une suite contiguë. Cliquez une touche pour corriger une note.",
        "ui edit selection": "Notes {first}–{last} sélectionnées : vous pouvez les remplacer en chantant.",
        "ui selection label": "Sélection : {n} note(s) · {t0}–{t1} s",
        "ui replacement prompt": "Chantez la phrase de remplacement, puis cliquez « Terminer le remplacement ».",
        "ui replacement analyzing": "Mise à jour de la prise…",
        "ui replacement pending": "Remplacement prêt : écoutez la phrase, puis gardez-la ou annulez.",
        "ui replacement accepted": "Remplacement gardé : {n} notes. Exportez quand vous êtes prêt.",
        "ui replacement cancelled": "Remplacement annulé : la prise d'origine est rétablie.",
        "ui undo": "↶ Annuler",
        "ui undo tooltip": "Revenir à la prise avant la dernière modification de notes (Ctrl+Z).",
        "ui redo": "↷ Rétablir",
        "ui redo tooltip": "Réappliquer la dernière modification annulée (Ctrl+Y).",
        "ui audition replacement": "▶ Écouter la phrase",
        "ui audition replacement tooltip": "Joue la phrase rechantée, telle qu'elle sonnera dans la prise.",
        "ui audition original": "▷ Écouter l'original",
        "ui audition original tooltip": "Joue la phrase d'origine des notes sélectionnées, avant de la rechanter.",
        "ui accept replacement": "✓ Garder le remplacement",
        "ui cancel replacement": "✗ Annuler le remplacement",
        "ui edit undone": "Modification annulée.",
        "ui edit redone": "Modification rétablie.",
        "ui edit jumped": "Retour à un état antérieur.",
        "ui status notes deleted": "{n} notes supprimées",
        "ui status note corrected": "Note corrigée : {note}",
        "ui live roll": "Notes captées",
        "ui live roll tooltip": "Revoir le roll live figé : les notes telles qu'elles ont été captées.",
        "ui live comparison": "Lecture finale superposée",
        "ui live comparison tooltip": "Afficher ou masquer la lecture finale sur le roll capté.",
        "ui compare archived readings": "Comparer les lectures archivées",
        "ui compare archived readings tooltip": (
            "Choisir deux versions enregistrées de cette prise et afficher leurs notes différentes."
        ),
        "ui compare archived readings title": "Comparer deux lectures archivées",
        "ui compare before": "Lecture avant :",
        "ui compare after": "Lecture après :",
        "ui compare automatic": "Automatique",
        "ui compare corrected": "Correction v{version:03d}",
        "ui compare action": "Comparer",
        "ui compare archive no coherence": (
            "Comparaison des notes uniquement : les archives ne contiennent pas l’évidence audio "
            "nécessaire au contrôle de cohérence."
        ),
        "ui compare choose two": "Choisissez deux lectures différentes.",
        "ui compare archive failed": "Ces rapports archivés n’ont pas pu être comparés.",
        "ui compare no changes": "Aucune différence de notes.",
        "ui diff archive compact": (
            "Avant → Après : {stable} identique(s), {changed} modifiée(s), "
            "{near} arrondi(s) proche(s), {missing} disparue(s), {added} ajoutée(s)."
        ),
        "ui piano audition label": "Note {i}/{n} : {note}",
        "ui piano stop audition": "■ Arrêter",
        "ui piano stop audition tooltip": "Arrêter l'écoute de cette note.",
        "ui piano keyboard help": (
            "Correction clavier : sélectionnez une note, focalisez le piano, "
            "choisissez la hauteur avec les flèches (elle s'entend) puis "
            "validez avec Entrée. Échap annule."
        ),
        "ui live roll review": "Roll capté — les notes telles qu'elles ont été capturées.",
        "ui back to results": "← Retour aux résultats",
        "ui back to results tooltip": "Revenir à l'analyse finale.",
        "correction": "avant correction",
        "replacement": "avant remplacement",
        "deletion": "avant suppression",
        "move": "avant déplacement",
        "current": "prise actuelle",
        "ui status note moved": "Note déplacée : {note}",
        "err note move failed": "Déplacement refusé : {e}",
        "ui status live ready": "Prise prête : {bpm:.0f} BPM · {key} · {n} notes. Vérifiez puis exportez.",
        "err note correction failed": "Correction de la note impossible : {e}",
        "err note deletion failed": "Suppression impossible : {e}",
        "err replacement failed": "Remplacement impossible : {e}",
        "err replacement live only": "Le remplacement chanté est disponible pour une prise live.",
        "err replacement selection": "Sélectionnez une suite continue de notes à remplacer.",
        "ui open another": "Ouvrir un autre fichier",
        "ui stop live": "← Arrêter / Accueil",
        "ui mic preparing": "Préparation du micro…",
        "ui mic waiting": "En attente de son",
        "ui mic received": "Son reçu",
        "ui mic silence": "Silence depuis {s} s",
        "ui mic low level": "Niveau faible depuis {s} s",
        "ui mic blocked": (
            "Le système bloque le micro : autorisez-le pour cette application "
            "(Réglages → Confidentialité → Microphone), puis réessayez."
        ),
        "ui mic missing": (
            "Aucun micro utilisable : branchez-en un ou choisissez-le dans les "
            "réglages du son, puis réessayez."
        ),
        "ui mic busy": (
            "Le micro est occupé par une autre application : fermez-la, puis réessayez."
        ),
        "ui live progress": "{t} · {notes} notes · {meter}",
        "ui live silence": "Aucun son reçu depuis {s} s. Vérifiez le seuil du micro.",
        "ui live low mic": "Aucun son reçu depuis {s} s. Vérifiez le niveau du micro.",
        "ui coherence sessions tooltip": "Vérifier la transcription et les écarts entre lecture live et finale.",
        "ui live notes label": "notes confirmées",
        "ui staff tooltip": "Portée (clé de sol) : les notes détectées, écrites en notation musicale (figures, altérations, mesures).",
        "key bpm unit": "BPM",
        "ui stopping": "Arrêt…",
        "ui dropped blocks": "{n} blocs perdus en temps réel : la lecture live de la fin de prise est incomplète — l'analyse finale lira la prise entière.",
        "ui analyzing": "Analyse finale…",
        "ui live file": "Enregistrement live",
        "ui choose file": "Choisir un fichier audio",
        "ui audio filter": "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;Tous les fichiers (*)",
        "ui save export": "Enregistrer l'export",
        "ui format midi": "MIDI (*.mid)",
        "ui format musicxml": "Partition MusicXML (*.musicxml)",
        "ui format pdf": "Partition PDF (*.pdf)",
        "ui format json": "Rapport JSON (*.json)",
        "ui format csv": "Notes CSV (*.csv)",
        "ui status saved file": "Fichier enregistré : {path}",
        "ui status exported": "Exporté : {files}",
        "ui status cache": "Évidence vocale reprise du cache ({ms:.0f} ms) — pYIN non relancé",
        "ui min note": "Note minimale :",
        "ui reinterpret": "↻ Ajuster la transcription",
        "ui reinterpret tooltip": "Applique le seuil de note minimale à l'analyse déjà calculée. Ne relance pas l'analyse audio ; aucun export automatique.",
        "ui full reanalyze": "⟳ Relancer l'analyse audio",
        "ui full reanalyze tooltip": "Recalcule la transcription depuis l'audio. Cela peut prendre plus de temps ; le résultat actuel reste affiché jusqu'à la fin.",
        "ui full reanalysis running": "Analyse audio en cours…",
        "ui full reanalysis progress": "Analyse audio — {stage}",
        "ui status full reanalyzed": "Analyse terminée : {n} notes. Exportez pour mettre à jour les fichiers.",
        "err full reanalysis audio": "Audio de la prise indisponible : rouvrez le fichier ou réenregistrez la prise avant de réanalyser.",
        "err full reanalysis failed": "La réanalyse complète a échoué : {e}. Le résultat précédent est conservé.",
        "ui status reinterpreted": "Transcription ajustée : {n} notes. Exportez pour écrire les fichiers.",
        "err reinterpret none": "Aucune évidence en cache : ouvrez un fichier ou chantez une prise d'abord.",
        "err reinterpret failed": "La relecture a échoué : {e}",
        "verify ok": "OK : le WAV et le JSON décrivent la même prise",
        "verify bad": "PRISE INCOHÉRENTE",
        "verify wav": "wav  : {name}  {dur:.2f} s @ {rate} Hz",
        "verify json": "json : {name}  {dur:.2f} s, {n} notes",
        "verify gap": "ECART : le WAV et le JSON ne décrivent pas la même durée ({dur_wav:.2f} s contre {dur_json:.2f} s) — artefacts de deux prises différentes ?",
        "verify none": "aucun artefact trouvé pour {stem}",
        "ui status live done": "Tempo {bpm:.0f} BPM · {key} · {n} notes · fichiers exportés dans le dossier courant",
        "ui play tooltip": "Écouter l'analyse (synthé local) — double-clic sur le piano roll pour démarrer ailleurs",
        "ui play pos": "prêt",
        "ui steps": "Chantez · Écoutez · Ajustez",
        "ui pdf badge": "PDF indisponible — installez MuseScore ou LilyPond",
        "ui exported ok": "✓ Exporté",
        "ui analyzing file": "Analyse…",
        "ui panel waiting": "En attente d'une analyse —\nchargez un fichier ou chantez au micro.",
        # --- the take's coherence verdict (results panel) ---
        "ui coherence title": "Cohérence de la prise",
        "key voice col": "Voix",
        "ui doubt voice": "douteuse",
        "ui coherence clean": "✓ propre : {checked} note(s) contrôlée(s), aucune phrase avalée",
        "ui coherence suspects": "⚠ {n} suspect(s) sur {checked} note(s) contrôlée(s) — détail dans le rapport JSON",
        "ui coherence na": "contrôle indisponible (évidence absente)",
        "ui coherence manual": "modification manuelle — contrôle audio indisponible",
        "ui coherence live pending": "modification manuelle — contrôle live/final indisponible",
        # --- the before/after reading diff (a successful re-analysis) ---
        "key reading diff": "Différentiel",
        "ui diff none": "—",
        "ui diff compact": (
            "avant → après : {stable} identique(s), {changed} modifiée(s), "
            "{near} arrondi(s) proche(s), {missing} disparue(s), "
            "{added} ajoutée(s) · {suspects} suspect(s) de cohérence"
        ),
        "ui diff legend": "barres translucides = lecture AVANT · accents = changement",
        "ui diff pair row": ("  #{i} {before} → {after} — {status} ({reasons})"),
        "ui diff missing row": "  #{i} {before} → plus présente — {status}",
        "ui diff added row": "  #{i} → {after} — {status}",
        "ui diff suspects header": "Suspects de la lecture courante ({n}) :",
        "ui diff suspect row": (
            "  midi {midi} cache midi {plateau} ({dur} s, {offset}) {tag}{conf}"
        ),
        "ui diff suspect resolved": "  RESOLU — le doute sur {midi} ({dur} s) ne se reproduit pas",
        "ui diff confidence": " · confiance pYIN {c} (contexte, pas un verdict)",
        "diff status stable": "identique",
        "diff status near": "arrondi proche",
        "diff status changed": "modifiée",
        "diff status missing": "disparue",
        "diff status added": "ajoutée",
        "diff reason pitch": "hauteur",
        "diff reason attack": "attaque",
        "diff reason duration": "durée",
        "diff suspect new": "NOUVEAU",
        "diff suspect persistent": "PERSISTE",
        "diff suspect unknown": "non associable",
        # --- coherence follow-up (page + script, shared) ---
        "coh empty title": "Aucun rapport de cohérence dans « {root} » : rien à agréger encore.",
        "coh empty hint": "Chaque prise analysée écrit <prise>.coherence.csv à côté de ses artefacts —\nchantez ou analysez une prise, puis relancez : les sessions s'agrègent ici.",
        "coh missing folder": "Dossier introuvable : {root}",
        "coh missing hint": "Indiquez le dossier qui CONTIENT les prises (un sous-dossier par prise) —\n« prises_live », celui que remplit l'application, par défaut.",
        "coh col session": "session",
        "coh col takes": "prises",
        "coh col checked": "contrôlées",
        "coh col suspects": "suspects",
        "coh col rate": "/100",
        "coh col delta": "delta",
        "coh col contours": "contours",
        "coh col pmean": "plateau moy",
        "coh col pmax": "plateau max",
        "coh col verify": "verify",
        "coh sessions title": "Suivi de cohérence par session (écart de 30 min = nouvelle session)",
        "coh days title": "Sous-totaux par jour",
        "coh takes title": "Détail par prise (ordre chronologique)",
        "coh day row": "  {date}  {sessions} session(s), {takes} prise(s), {checked} contrôlée(s), {suspects} suspect(s), {rate}/100",
        "coh take clean": "  {stem}  contrôlées {checked}  — propre",
        "coh take suspects": "  {stem}  contrôlées {checked}  {rate}/100  suspects {n}",
        "coh take divergent": "  {stem}  contrôlées {checked}  — ARTEFACTS DIVERGENTS (WAV/JSON, voir verify)",
        "coh suspect row": "    midi {midi:>3}  portée {span}st  plateau {dur}s {at}→ midi {target} ({offset})",
        "coh plateau at": "à {t}s ",
        "coh verdict clean": "Aucun suspect sur {takes} prise(s) ({notes} notes contrôlées) : aucune note longue ne cache de plateau secondaire.",
        "coh verdict clean nonotes": "Aucun suspect sur {takes} prise(s) : aucune note longue ne cache de plateau secondaire (> 120 ms tenus à plus d'un demi-ton de l'étiquette).",
        "coh verdict suspects": "{suspects} suspect(s) au total sur {takes} prise(s) ({notes} notes contrôlées) — détail par prise ci-dessus.",
        "coh gui footer": "{takes} prise(s) agrégée(s).",
        "ui coherence sessions btn": "Qualité des transcriptions",
        "ui calibrate mic button": "Calibrer mon micro",
        "ui calibrate target": (
            "Ronde {n}/{total} : chantez la note tenue {note} — "
            "l'application lit l'écart de votre micro."
        ),
        "ui calibrate heard": "micro : {note}",
        "ui calibrate silent": "micro : rien de tenu pour l'instant (l'application écoute…)",
        "ui calibrate found": "Écart mesuré : {cents} cents — enregistrez-le comme référence pour les prochaines prises.",
        "ui calibrate none": "Aucun écart systématique détecté : vos notes ne nomment pas une habitude de l'appareil (le bruit ne se calibre pas).",
        "ui calibrate counts": "tours retenus : {used} · sautés : {skipped}",
        "ui calibrate saved": "calibrage enregistré : {cents} cents (il y a {age_s} s)",
        "ui calibrate saved now": "Enregistré — il apparaîtra comme contexte sur les prochaines prises (jamais dans la reconnaissance).",
        "ui calibrate forgotten": "Calibrage supprimé : les prochaines prises repartent sans référence.",
        "ui calibrate context": "L'offset reste un CONTEXTE d'affichage : la reconnaissance ne s'en sert jamais (ADR 0032).",
        "ui calibrate stop": "Fermer",
        "ui calibrate save": "Enregistrer",
        "ui calibrate forget": "Oublier",
        "ui coherence sessions title": "Suivi de cohérence (sessions de 30 min)",
        "ui coherence running": "Agrégation…",
        "ui coherence none": "Aucune prise à agréger : chantez une prise d'abord (prises_live/).",
        "ui status coherence": "Cohérence : {summary}",
        # --- live/final explanation (why the score differs from the live view) ---
        "explain empty live": "Lecture live vide : bilan non comparable.",
        "explain empty final": (
            "Lecture finale vide : bilan non comparable, aucun taux de concordance exploitable."
        ),
        "explain none": "aucune note live à comparer : rien à expliquer",
        "explain head": "Ce que vous avez vu en direct : {live} notes — la partition finale en garde {final}.",
        "explain off1": "{n} note(s) voisine(s) d'un demi-ton : le hors-ligne recadre la hauteur avec la tonalité de la prise (un arrondi, pas une erreur).",
        "explain moved": "{n} note(s) déplacée(s) : l'analyse finale a changé la hauteur ou l'attaque (une fusion, un découpage, un glissando relu).",
        "explain absorbed": "{n} note(s) absorbée(s) : un glissando que le hors-ligne lit comme une seule note, l'affichage direct confirmant chaque plateau.",
        "explain more": "{n} note(s) cerclées d'orange sur le roll live ({rate:.0f} %) : l'indicateur de santé du temps réel — lire à côté du compteur final.",
        # --- suivi : colonnes de tendance et semaines ---
        "coh col trend": "tendance",
        "coh weeks title": "Santé du temps réel, semaine par semaine (taux de contours — plus bas = mieux)",
        "coh week row": "  {week}  {takes} prise(s)  {rate}  ({live} notes direct / {final} finales)",
        "ui coherence session verdict": "session : {suspects} suspect(s) sur {checked} note(s) contrôlée(s), {takes} prise(s)",
        "ui replay done": "Rejeu terminé : {live} notes vues en direct, {final} finales, {rate:.0f} % de contours — archivé dans le dossier de la prise.",
        "ui health chart title": "Santé du temps réel (taux de contours par semaine — plus bas = mieux). Cliquez une semaine pour ses prises.",
        "ui health chart empty": "Pas encore de données de santé : rejouez des prises avec l'audio enregistré.",
        "ui health click none": "Cette semaine n'a pas de prise rejouée.",
        "ui health click": "Semaine {week} : {n} prise(s) rejouée(s) — détail affiché dans le suivi de cohérence.",
        "ui regression warning": "Santé du temps réel en baisse : {delta:+.0f} points cette semaine ({last:.0f} % contre {prev:.0f} %). Rejouez une prise pour vérifier.",
        "err ui wav failed": "Impossible d'écrire l'audio de la prise ({e}) — la transcription est sauvegardée, seul le fichier .wav manque.",
        "time morning": "matin (6h-12h)",
        "time afternoon": "après-midi (12h-18h)",
        "time evening": "soir (18h-6h)",
        "weekly title": "Récapitulatif hebdomadaire — progrès du chanteur",
        "weekly health": "Santé du temps réel, semaine par semaine",
        "weekly windows": "Quand chantez-vous le mieux ? (taux de suspects /100, taux de contours)",
        "weekly days": "Jours",
        "weekly window row": "  {window:<22} {takes:>3} prise(s)   {rate:>6}   {contours:>5}",
        "weekly day row": "  {date}   {sessions} session(s), {takes} prise(s), {checked} notes contrôlées, {suspects} suspect(s), {rate}/100",
        "weekly no data": "Pas encore de données : chantez des prises pour remplir le récapitulatif.",
        "cli weekly pruned caches": "Nettoyage : {n} cache(s) d'évidence orphelin(s) supprimé(s).",
        "cli weekly pdf ok": "Récapitulatif gravé : {path}",
        "cli companion listening": "Compagnon en écoute sur {url} — Ctrl+C pour arrêter.",
        "cli companion stopped": "Compagnon arrêté.",
        "ui practice windows title": "Quand chantez-vous le mieux ?",
        "ui practice window row": "  {window:<22} {takes:>3} prise(s)   {rate:>6}/100   {contours:>5}",
        "ui retention pruned": "Rétention : {n} ancienne(s) prise(s) supprimée(s) ({stems}) — l'historique reste dans le journal.",
        # --- phrase-repeat mode: two takes of one phrase, side by side ---
        "cmp unreadable": "Rapport introuvable ou illisible : {path}",
        "cmp head": "Comparaison « répéter la phrase » — prise A : {a} ({n_a} notes) contre prise B : {b} ({n_b} notes)",
        "cmp tempo": "Tempo : {bpm_a:.0f} → {bpm_b:.0f} BPM ({ratio:+.1f} %) — les écarts de temps se lisent en temps de B",
        "cmp tempo drift": "⚠ tempos à plus de 5 % d'écart : les écarts notés déplacés mélangent tempo et hésitation",
        "cmp counts": "Identiques : {same} · voisins ±1 : {off1} · déplacées en temps : {shifted} · changées : {moved} · manquantes en B : {missing} · ajoutées en B : {added}",
        "cmp stable": "✓ Les deux prises se lisent comme la même phrase : même suite de notes, timing stable.",
        "cmp detail title": "Détail des différences (ordre chanté) :",
        "cmp same": "  {a} → {b}",
        "cmp off1": "  {a} → {b} : voisin d'un demi-ton (la famille d'arrondi, pas une fausse note)",
        "cmp shifted": "  {a} → {b} : même hauteur, décalée de {beats:+.2f} temps",
        "cmp moved": "  {a} → {b} : hauteur changée (décalage {beats:+.2f} temps)",
        "cmp missing title": "Notes de A sans homologue en B :",
        "cmp missing row": "  #{i} — {note} à {at:.2f} s",
        "cmp added title": "Notes de B sans homologue en A :",
        "cmp added row": "  #{i} — {note} à {at:.2f} s",
        "cmp json written": "Comparaison écrite : {path}",
    },
    "en": {
        "ui guide title": "Next action",
        "ui guide progress": "Passage {current} of {total}",
        "ui guide next": "Next uncertain passage",
        "ui guide next hint": "Check or correct these notes, then move to the next doubt.",
        "ui guide tools": "All tools",
        "ui guide hide tools": "Hide tools",
        "ui guide skip": "Skip to export",
        "ui guide listen passage": "Listen to this passage",
        "ui guide listen take": "Listen to the take",
        "ui guide stop": "Stop listening",
        "ui guide doubt": "A passage deserves a listen: this is a doubt, not a definite error.",
        "ui guide listen": "Listen back to your take before deciding whether to adjust it.",
        "ui guide review": "Review the notes in this passage",
        "ui guide review hint": "If you hear a difference, open the notes to check it.",
        "ui guide export hint": "You can export the current reading, with or without adjustments.",
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
        "key tuning offset": "Tuning",
        "key intonation": "Intonation",
        "ui intonation share": "{share}% in tune",
        "ui intonation tendency": "tendency: {dir}",
        "ui intonation sharp": "above ({cents:+.0f} c)",
        "ui intonation flat": "below ({cents:+.0f} c)",
        "ui intonation even": "steady",
        "ui intonation registers": "registers: {rows}",
        "ui intonation position": "position: {rows}",
        "ui position start": "start",
        "ui position end": "end",
        "ui position lean": "{half} {cents:+.0f} c",
        "ui register low": "low",
        "ui register mid": "mid",
        "ui register high": "high",
        "ui register lean": "{band} {cents:+.0f} c",
        "key coherence": "Coherence",
        "ui tuning ok": "in tune, keep it up",
        "ui tuning slightly off": "a bit {dir}, watch it",
        "ui tuning off": "clearly {dir} ({cents:+.0f} c)",
        "ui tuning sharp": "sharp",
        "ui tuning flat": "flat",
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
        "progress tempo": "Detecting tempo…",
        "progress pitch": "Tracking pitch (pYIN)…",
        "progress notes": "Segmenting notes…",
        "progress key": "Detecting key…",
        "progress chords": "Detecting chords…",
        "progress refine": "Refining tempo…",
        "progress figures": "Labeling note figures…",
        "progress rests": "Detecting breaths…",
        "progress done": "Done",
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
        "err sounddevice missing": "sounddevice unavailable (no audio device?)",
        "err take stopped": "this live take is already stopped",
        "err live incomplete": "live analysis incomplete",
        "err export incomplete": "export_result requires a complete analysis (tempo + key)",
        "err export needs meter": "export_result requires a meter",
        "err pdf install hint": "Install MuseScore (https://musescore.org) or LilyPond (https://lilypond.org), then make sure it is on the PATH.",
        "err pdf not runnable": "'{cmd}' not found at run time. {hint}",
        "err pdf skipped": "PDF skipped: {hint}",
        "err pdf no tool": "No engraving tool found (MuseScore, LilyPond). {hint}",
        "err pdf timeout": "{name} did not respond (timeout 120 s).",
        "err pdf conversion": "Conversion via {name} failed: {msg}",
        "err pdf exit code": "exit code {code}",
        "err ui playback": "Audio playback failed: {e}",
        "ui window title": "Voix → MIDI — notes, tempo, key and chords",
        "ui home title": "Give your melody a shape.",
        "ui theme label": "Appearance",
        "ui theme dark": "Dark",
        "ui theme light": "Light",
        "ui home microphone": "Microphone for the next capture",
        "ui app subtitle": "Sing or open a recording to discover its notes.",
        "ui open file": "Open a file",
        "ui sing live": "Sing a melody",
        "ui finder button": "Find a note",
        "ui input device default": "System default microphone",
        "ui input device default name": "System default microphone",
        "ui input device active": "Microphone in use: {device}",
        "ui input device next": "Next capture: {device}",
        "ui input device fallback": "Selected mic unavailable — microphone in use: {device}",
        "ui input device fallback next": "Mic unavailable — system default next: {device}",
        "ui input device effects": (
            "The Windows driver publishes an effects chain for this microphone: the capture "
            "may be filtered before analysis. Turn off audio enhancements in the sound "
            "settings to record the raw signal."
        ),
        "ui input device tooltip": (
            "Choose the microphone for calibration, note finder and live takes. "
            "Changes apply to the next capture."
        ),
        "ui input device missing": (
            "The selected microphone is unavailable: {device}. The system default will be used."
        ),
        "ui input device unavailable": "Unavailable — using system default: {device}",
        "ui finder start": "▶  Start listening",
        "ui finder stop": "■  Stop listening",
        "ui finder idle": "♪  Sing a note…",
        "ui finder live": "●  Listening — sing a note",
        "ui finder off": "Listening stopped",
        "ui finder waiting": "Waiting for sound…",
        "ui finder silence": "No sound above the mic threshold.",
        "ui finder no note": "Sound received. Hold a note for a moment.",
        "ui finder note heard": "Pitch detected.",
        "ui finder hint": "Sing a note: its key lights up on the piano keyboard.\nClick a key to hear it.",
        "ui finder history": "Recently heard notes",
        "ui finder history empty": "Confirmed notes appear here, ready to hear.",
        "ui finder history tooltip": "Hear {note} again",
        "ui finder answer": "What the mic hears",
        "ui finder range": "Keyboard range",
        "ui finder range voice": "Voice ({lo} – {hi})",
        "ui finder range low": "Low voice ({lo} – {hi})",
        "ui finder range all": "Whole keyboard ({lo} – {hi})",
        "ui finder range tooltip": (
            "Choose which keys are shown. The keyboard never recentres on the note you sing."
        ),
        "ui tuning lower": "♭ lower",
        "ui tuning centre": "in tune",
        "ui tuning higher": "higher ♯",
        "ui level idle": "level: —",
        "ui expected none": "No expected note",
        "ui expected note": "Expected: {note}",
        "ui live hint": "Sing along with the score: the key turns green when it matches, red otherwise.",
        "ui live controls": "Take settings",
        "ui live monitor": "Your live take",
        "ui live feedback": "Singing feedback",
        "ui live details": "Live take details",
        "ui show live details": "＋ Show details",
        "ui hide live details": "− Hide details",
        "ui take summary": "Your transcription",
        "ui summary facts": "{notes} notes · {duration} s · {tempo} · {key}",
        "ui show details": "＋ Show details",
        "ui hide details": "− Hide details",
        "ui metronome": "Metronome",
        "ui metronome tooltip": "Steady click to practice singing in time (off by default).",
        "ui record audio": "Save the take's audio",
        "ui record audio tooltip": "Writes the take to .wav in the live mode's output folder (on by default — a take without audio cannot be audited later).",
        "ui bpm unit": "BPM",
        "ui beat count": "Beat {beat}",
        "ui accuracy title": "Average tuning:",
        "ui accuracy empty": "Average tuning: —",
        "ui play piano": "Piano",
        "ui play voice": "Voice",
        "ui play voice tooltip": "Listen to your recorded voice only.",
        "ui play both": "Together",
        "ui play both tooltip": "Piano over your voice, at the set volume — to compare the takes.",
        "ui group listen": "Listen",
        "ui group adjust": "Adjust",
        "ui group export": "Export",
        "ui view notes": "Notes",
        "ui view notes tooltip": "The piano roll: notes are read, selected and corrected there.",
        "ui view staff": "Score",
        "ui view wave": "Waveform",
        "ui view wave tooltip": "The audio's waveform, with the playback cursor.",
        "ui volume": "Piano vol.",
        "ui volume tooltip": "Rendered piano volume, to hear it against your voice.",
        "ui axis note": "Note",
        "ui axis time": "Time",
        "ui drop hint": "Tip: drag & drop an audio file onto this window.",
        "ui lang button": "Language: English",
        "ui back home": "← Home",
        "ui no file": "No file",
        "ui export options": "Export options",
        "ui opts transcription": "Transcription",
        "ui advanced settings": "Advanced settings",
        "ui hide advanced settings": "− Hide settings",
        "ui export tooltip": (
            "Writes this take's MIDI, MusicXML score and JSON report; the PDF is "
            "added when its box is checked."
        ),
        "ui quantize": "Quantize to the 1/16 grid",
        "ui transpose": "Transpose (semitones)",
        "ui notation mode": "Notation mode (colored figures)",
        "ui pdf option": "Also export as PDF",
        "ui pdf tooltip": "Install MuseScore or LilyPond to enable PDF export.",
        "ui re-export": "Re-export with these options",
        "ui export": "Export the transcription",
        "ui replace phrase": "Replace selected notes",
        "ui finish replacement": "Finish replacement",
        "ui edit hint": "Select one note or a continuous range. Click a key to correct one note.",
        "ui edit selection": "Notes {first}–{last} selected: you can replace them by singing.",
        "ui selection label": "Selection: {n} note(s) · {t0}–{t1} s",
        "ui replacement prompt": "Sing the replacement phrase, then click “Finish replacement”.",
        "ui replacement analyzing": "Updating the take…",
        "ui replacement pending": "Replacement ready: audition the phrase, then keep it or cancel.",
        "ui replacement accepted": "Replacement kept: {n} notes. Export when ready.",
        "ui replacement cancelled": "Replacement cancelled: the original take is restored.",
        "ui undo": "↶ Undo",
        "ui undo tooltip": "Revert the take to before the last note edit (Ctrl+Z).",
        "ui redo": "↷ Redo",
        "ui redo tooltip": "Reapply the last undone edit (Ctrl+Y).",
        "ui audition replacement": "▶ Listen to the phrase",
        "ui audition replacement tooltip": "Plays the re-sung phrase as it will sound in the take.",
        "ui audition original": "▷ Listen to the original",
        "ui audition original tooltip": "Plays the selected notes' original phrase, before re-singing it.",
        "ui accept replacement": "✓ Keep replacement",
        "ui cancel replacement": "✗ Cancel replacement",
        "ui edit undone": "Edit undone.",
        "ui edit redone": "Edit redone.",
        "ui edit jumped": "Jumped to an earlier state.",
        "ui status notes deleted": "{n} notes deleted",
        "ui status note corrected": "Note corrected: {note}",
        "ui live roll": "Captured notes",
        "ui live roll tooltip": "Review the frozen live roll: the notes as they were captured.",
        "ui live comparison": "Show final reading",
        "ui live comparison tooltip": "Show or hide the final reading on the captured roll.",
        "ui compare archived readings": "Compare archived readings",
        "ui compare archived readings tooltip": (
            "Choose two saved versions of this take and show how their notes differ."
        ),
        "ui compare archived readings title": "Compare two archived readings",
        "ui compare before": "Earlier reading:",
        "ui compare after": "Later reading:",
        "ui compare automatic": "Automatic",
        "ui compare corrected": "Correction v{version:03d}",
        "ui compare action": "Compare",
        "ui compare archive no coherence": (
            "Notes only: these archives do not contain the audio evidence needed for a coherence check."
        ),
        "ui compare choose two": "Choose two different readings.",
        "ui compare archive failed": "These archived reports could not be compared.",
        "ui compare no changes": "No note differences.",
        "ui diff archive compact": (
            "Before → After: {stable} identical, {changed} changed, {near} near-rounding, "
            "{missing} missing, {added} added."
        ),
        "ui piano audition label": "Note {i}/{n}: {note}",
        "ui piano stop audition": "■ Stop",
        "ui piano stop audition tooltip": "Stop listening to this note.",
        "ui piano keyboard help": (
            "Keyboard correction: select a note, focus the piano, use arrows "
            "to choose a pitch (it plays), then press Enter. Escape cancels."
        ),
        "ui live roll review": "Captured roll — the notes as they were recorded.",
        "ui back to results": "← Back to results",
        "ui back to results tooltip": "Return to the final analysis.",
        "correction": "before correction",
        "replacement": "before replacement",
        "deletion": "before deletion",
        "move": "before move",
        "current": "current take",
        "ui status note moved": "Note moved: {note}",
        "err note move failed": "Move refused: {e}",
        "ui status live ready": "Take ready: {bpm:.0f} BPM · {key} · {n} notes. Review and export.",
        "err note correction failed": "Could not correct the note: {e}",
        "err note deletion failed": "Could not delete the notes: {e}",
        "err replacement failed": "Could not replace the phrase: {e}",
        "err replacement live only": "Sung replacement is available for a live take.",
        "err replacement selection": "Select a continuous range of notes to replace.",
        "ui open another": "Open another file",
        "ui stop live": "← Stop / Home",
        "ui mic preparing": "Preparing the microphone…",
        "ui mic waiting": "Waiting for sound",
        "ui mic received": "Sound received",
        "ui mic silence": "Silent for {s} s",
        "ui mic low level": "Low level for {s} s",
        "ui mic blocked": (
            "The system is blocking the microphone: allow it for this application "
            "(Settings → Privacy → Microphone), then try again."
        ),
        "ui mic missing": (
            "No usable microphone: plug one in or choose it in the sound settings, "
            "then try again."
        ),
        "ui mic busy": (
            "The microphone is busy in another application: close it, then try again."
        ),
        "ui live progress": "{t} · {notes} notes · {meter}",
        "ui live silence": "No sound received for {s} s. Check the mic threshold.",
        "ui live low mic": "No sound received for {s} s. Check the microphone level.",
        "ui coherence sessions tooltip": "Review transcription quality and differences between live and final readings.",
        "ui live notes label": "notes confirmed",
        "ui staff tooltip": "Staff (treble clef): detected notes written in musical notation (figures, accidentals, bars).",
        "key bpm unit": "BPM",
        "ui stopping": "Stopping…",
        "ui dropped blocks": "{n} blocks lost in real time: the live reading of the take's end is incomplete — the final analysis will read the take whole.",
        "ui analyzing": "Final analysis…",
        "ui live file": "Live recording",
        "ui choose file": "Choose an audio file",
        "ui audio filter": "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;All files (*)",
        "ui save export": "Save export",
        "ui format midi": "MIDI (*.mid)",
        "ui format musicxml": "MusicXML score (*.musicxml)",
        "ui format pdf": "PDF score (*.pdf)",
        "ui format json": "JSON report (*.json)",
        "ui format csv": "Notes CSV (*.csv)",
        "ui status saved file": "File saved: {path}",
        "ui status exported": "Exported: {files}",
        "ui status cache": "Vocal evidence reloaded from cache ({ms:.0f} ms) — pYIN skipped",
        "ui min note": "Min note:",
        "ui reinterpret": "↻ Adjust transcription",
        "ui reinterpret tooltip": "Applies the minimum-note threshold to the analysis already computed. It does not re-run audio analysis; nothing is exported automatically.",
        "ui full reanalyze": "⟳ Run audio analysis again",
        "ui full reanalyze tooltip": "Rebuilds the transcription from the audio. This can take longer; the current result stays visible until it finishes.",
        "ui full reanalysis running": "Audio analysis in progress…",
        "ui full reanalysis progress": "Audio analysis — {stage}",
        "ui status full reanalyzed": "Analysis complete: {n} notes. Export to update the files.",
        "err full reanalysis audio": "The take's audio is unavailable: reopen the file or record the take again before reanalyzing.",
        "err full reanalysis failed": "Full reanalysis failed: {e}. The previous result is preserved.",
        "ui status reinterpreted": "Transcription adjusted: {n} notes. Export to write the files.",
        "err reinterpret none": "No cached evidence: open a file or sing a take first.",
        "err reinterpret failed": "The re-read failed: {e}",
        "verify ok": "OK: the WAV and the JSON describe the same take",
        "verify bad": "INCONSISTENT TAKE",
        "verify wav": "wav  : {name}  {dur:.2f} s @ {rate} Hz",
        "verify json": "json : {name}  {dur:.2f} s, {n} notes",
        "verify gap": "GAP: the WAV and the JSON do not describe the same duration ({dur_wav:.2f} s vs {dur_json:.2f} s) — artifacts of two different takes?",
        "verify none": "no artifact found for {stem}",
        "ui status live done": "{bpm:.0f} BPM · {key} · {n} notes · files exported to the current folder",
        "ui play tooltip": "Play the analysis (local synth) — double-click the piano roll to start elsewhere",
        "ui play pos": "ready",
        "ui steps": "Sing · Listen · Adjust",
        "ui pdf badge": "PDF unavailable — install MuseScore or LilyPond",
        "ui exported ok": "✓ Exported",
        "ui analyzing file": "Analyzing…",
        "ui panel waiting": "Waiting for an analysis —\nload a file or sing into the mic.",
        # --- take coherence verdict (results panel) ---
        "ui coherence title": "Take coherence",
        "key voice col": "Voice",
        "ui doubt voice": "doubtful",
        "ui coherence clean": "✓ clean: {checked} note(s) audited, no swallowed phrase",
        "ui coherence suspects": "⚠ {n} suspect(s) over {checked} audited note(s) — details in the JSON report",
        "ui coherence na": "check unavailable (no evidence)",
        "ui coherence manual": "manual edit — audio check unavailable",
        "ui coherence live pending": "manual edit — live/final comparison unavailable",
        # --- the before/after reading diff (a successful re-analysis) ---
        "key reading diff": "Reading diff",
        "ui diff none": "—",
        "ui diff compact": (
            "before → after: {stable} identical, {changed} changed, "
            "{near} near-rounding, {missing} missing, {added} added "
            "· {suspects} coherence suspect(s)"
        ),
        "ui diff legend": "translucent bars = reading BEFORE · accents = a change",
        "ui diff pair row": ("  #{i} {before} → {after} — {status} ({reasons})"),
        "ui diff missing row": "  #{i} {before} → gone — {status}",
        "ui diff added row": "  #{i} → {after} — {status}",
        "ui diff suspects header": "Current reading's suspects ({n}):",
        "ui diff suspect row": (
            "  midi {midi} hides midi {plateau} ({dur} s, {offset}) {tag}{conf}"
        ),
        "ui diff suspect resolved": "  RESOLVED — the doubt on {midi} ({dur} s) no longer shows",
        "ui diff confidence": " · pYIN confidence {c} (context, not a verdict)",
        "diff status stable": "identical",
        "diff status near": "near rounding",
        "diff status changed": "changed",
        "diff status missing": "missing",
        "diff status added": "added",
        "diff reason pitch": "pitch",
        "diff reason attack": "attack",
        "diff reason duration": "duration",
        "diff suspect new": "NEW",
        "diff suspect persistent": "PERSISTS",
        "diff suspect unknown": "unmatchable",
        # --- coherence follow-up (page + script, shared) ---
        "coh empty title": "No coherence report in '{root}' yet: nothing to aggregate.",
        "coh empty hint": "Every analyzed take writes <take>.coherence.csv next to its artifacts —\nsing or analyze a take, then run again: the sessions aggregate here.",
        "coh missing folder": "Folder not found: {root}",
        "coh missing hint": "Point at the folder that CONTAINS the takes (one sub-folder per take) —\n'prises_live', the one the application fills, by default.",
        "coh col session": "session",
        "coh col takes": "takes",
        "coh col checked": "audited",
        "coh col suspects": "suspects",
        "coh col rate": "/100",
        "coh col delta": "delta",
        "coh col contours": "contours",
        "coh col pmean": "plateau mean",
        "coh col pmax": "plateau max",
        "coh col verify": "verify",
        "coh sessions title": "Coherence follow-up by session (a 30-min gap = a new session)",
        "coh days title": "Subtotals per day",
        "coh takes title": "Per-take detail (chronological)",
        "coh day row": "  {date}  {sessions} session(s), {takes} take(s), {checked} audited, {suspects} suspect(s), {rate}/100",
        "coh take clean": "  {stem}  audited {checked}  — clean",
        "coh take suspects": "  {stem}  audited {checked}  {rate}/100  suspects {n}",
        "coh take divergent": "  {stem}  audited {checked}  — DIVERGING ARTIFACTS (WAV/JSON, see verify)",
        "coh suspect row": "    midi {midi:>3}  span {span}st  plateau {dur}s {at}→ midi {target} ({offset})",
        "coh plateau at": "at {t}s ",
        "coh verdict clean": "No suspect over {takes} take(s) ({notes} audited notes): no long note hides a secondary plateau.",
        "coh verdict clean nonotes": "No suspect over {takes} take(s): no long note hides a secondary plateau (> 120 ms held more than a semitone off the label).",
        "coh verdict suspects": "{suspects} suspect(s) total over {takes} take(s) ({notes} audited notes) — per-take detail above.",
        "coh gui footer": "{takes} take(s) aggregated.",
        "ui coherence sessions btn": "Transcription quality",
        "ui calibrate mic button": "Calibrate my mic",
        "ui calibrate target": (
            "Round {n}/{total}: hold the note {note} — the app reads your mic's offset."
        ),
        "ui calibrate heard": "mic: {note}",
        "ui calibrate silent": "mic: nothing held yet (listening…)",
        "ui calibrate found": "Measured offset: {cents} cents — save it as reference for upcoming takes.",
        "ui calibrate none": "No systematic offset detected: your rounds do not name a device habit (noise is never calibrated).",
        "ui calibrate counts": "rounds used: {used} · skipped: {skipped}",
        "ui calibrate saved": "calibration saved: {cents} cents ({age_s} s ago)",
        "ui calibrate saved now": "Saved — it will show as context on upcoming takes (never in recognition).",
        "ui calibrate forgotten": "Calibration cleared: upcoming takes start without a reference.",
        "ui calibrate context": "The offset stays a DISPLAY context: recognition never reads it (ADR 0032).",
        "ui calibrate stop": "Close",
        "ui calibrate save": "Save",
        "ui calibrate forget": "Forget",
        "ui coherence sessions title": "Coherence follow-up (30-min sessions)",
        "ui coherence running": "Aggregating…",
        "ui coherence none": "No take to aggregate: sing a take first (prises_live/).",
        "ui status coherence": "Coherence: {summary}",
        # --- live/final explanation (why the score differs from the live view) ---
        "explain empty live": "Empty live reading: not comparable.",
        "explain empty final": (
            "Empty final reading: not comparable; no meaningful agreement rate."
        ),
        "explain none": "no live notes to compare: nothing to explain",
        "explain head": "What you watched live: {live} notes — the final score keeps {final}.",
        "explain off1": "{n} note(s) one semitone off: the offline pass re-frames pitch with the take's key (a rounding, not a mistake).",
        "explain moved": "{n} note(s) moved: the final analysis changed the pitch or the onset (a merge, a split, a re-read glide).",
        "explain absorbed": "{n} note(s) absorbed: a glide the offline path reads as ONE note while the live view honestly confirms each plateau.",
        "explain more": "{n} note(s) outlined in orange on the live roll ({rate:.0f} %): the real-time health indicator — read it next to the final count.",
        # --- follow-up: trend columns and weeks ---
        "coh col trend": "trend",
        "coh weeks title": "Real-time health, week by week (contour rate — lower is better)",
        "coh week row": "  {week}  {takes} take(s)  {rate}  ({live} live / {final} final notes)",
        "ui coherence session verdict": "session: {suspects} suspect(s) over {checked} audited note(s), {takes} take(s)",
        "ui replay done": "Replay done: {live} notes seen live, {final} final, {rate:.0f}% contours — archived in the take's folder.",
        "ui health chart title": "Real-time health (contour rate per week — lower is better). Click a week to see its takes.",
        "ui health chart empty": "No health data yet: replay takes with the audio box ticked.",
        "ui health click none": "No replayed take in this week.",
        "ui health click": "Week {week}: {n} replayed take(s) — detail shown in the coherence follow-up.",
        "ui regression warning": "Real-time health dropped: {delta:+.0f} points this week ({last:.0f}% vs {prev:.0f}%). Replay a take to check.",
        "err ui wav failed": "Could not write the take's audio ({e}) — the transcription is saved, only the .wav file is missing.",
        "time morning": "morning (6am-12pm)",
        "time afternoon": "afternoon (12pm-6pm)",
        "time evening": "evening (6pm-6am)",
        "weekly title": "Weekly recap — the singer's progress",
        "weekly health": "Real-time health, week by week",
        "weekly windows": "When do you sing best? (suspects per 100, contour rate)",
        "weekly days": "Days",
        "weekly window row": "  {window:<22} {takes:>3} take(s)   {rate:>6}   {contours:>5}",
        "weekly day row": "  {date}   {sessions} session(s), {takes} take(s), {checked} audited notes, {suspects} suspect(s), {rate}/100",
        "weekly no data": "No data yet: sing some takes to fill the recap.",
        "cli weekly pruned caches": "Cleanup: {n} orphan evidence cache(s) deleted.",
        "cli weekly pdf ok": "Recap engraved: {path}",
        "cli companion listening": "Companion listening on {url} — Ctrl+C to stop.",
        "cli companion stopped": "Companion stopped.",
        "ui practice windows title": "When do you sing best?",
        "ui practice window row": "  {window:<22} {takes:>3} take(s)   {rate:>6}/100   {contours:>5}",
        "ui retention pruned": "Retention: {n} old take(s) deleted ({stems}) — history stays in the journal.",
        # --- phrase-repeat mode: two takes of one phrase, side by side ---
        "cmp unreadable": "Report missing or unreadable: {path}",
        "cmp head": (
            "Phrase-repeat comparison — take A: {a} ({n_a} notes) vs take B: {b} ({n_b} notes)"
        ),
        "cmp tempo": "Tempo: {bpm_a:.0f} → {bpm_b:.0f} BPM ({ratio:+.1f}%) — timing shifts read in B's beats",
        "cmp tempo drift": "⚠ tempos more than 5% apart: the shifted gaps mix tempo and hesitation",
        "cmp counts": "Identical: {same} · ±1 neighbours: {off1} · time-shifted: {shifted} · pitch-changed: {moved} · missing from B: {missing} · added in B: {added}",
        "cmp stable": "✓ The two takes read as the same phrase: same note sequence, steady timing.",
        "cmp detail title": "Detail of the differences (singing order):",
        "cmp same": "  {a} → {b}",
        "cmp off1": "  {a} → {b} : one semitone neighbour (the rounding family, not a wrong note)",
        "cmp shifted": "  {a} → {b} : same pitch, shifted by {beats:+.2f} beats",
        "cmp moved": "  {a} → {b} : pitch changed ({beats:+.2f}-beat shift)",
        "cmp missing title": "A notes with no counterpart in B:",
        "cmp missing row": "  #{i} — {note} at {at:.2f} s",
        "cmp added title": "B notes with no counterpart in A:",
        "cmp added row": "  #{i} — {note} at {at:.2f} s",
        "cmp json written": "Comparison written: {path}",
    },
}

_current = "fr"
_debug = os.environ.get("V2M_DEBUG_TR") == "1"


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


def set_debug(on: bool) -> None:
    """Log missing tr() keys on stderr (also arms set_language, in DEBUG)."""
    global _debug
    _debug = bool(on)


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
    """Translate a user-facing string in the current language.

    A missing key falls back to the key itself, which never raises — so a
    typo'd key ships silently and the user just reads the raw key. In DEBUG
    mode (V2M_DEBUG_TR=1 at import time, or set_debug(True)) every miss is
    logged on stderr: this is what catches the DYNAMIC call sites (a key
    built into a variable) the static AST guard cannot see.
    """
    if key not in STRINGS[_current]:
        if _debug:
            print(f"[v2m i18n] clé absente de la table « {_current} » : {key!r}", file=sys.stderr)
        return key
    return STRINGS[_current][key]


def format_number(value: float, digits: int = 0, sign: bool = False) -> str:
    """Write one displayed number in the current language.

    The desktop twin of the page's single `Intl.NumberFormat`: a Python
    f-string writes a dot in every language, so a French window printed
    "3.2 s" beside a French page reading "3,2 s". Every number carrying a
    decimal goes through this one formatter instead of a call-site format
    spec. Grouping stays OFF — these are seconds, hertz and cents inside a
    measure, never thousands — and a non-finite reading prints as 0 rather
    than "nan".
    """
    if not isinstance(value, (int, float)):
        value = 0.0
    elif value != value or value in (float("inf"), float("-inf")):  # NaN, ±inf
        value = 0.0
    text = f"{value:{'+' if sign else ''}.{digits}f}"
    return text.replace(".", ",") if _current == "fr" else text


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


# English display of every canonical figure name, keyed by the WHOLE name:
# English puts the dot BEFORE the value (« dotted half », never « half
# dotted ») and folds « double-croche » into one word — word-by-word
# replacement cannot express either. Rests follow the same dotted-first
# order; the vocabulary is closed (pinned against grid.CATALOGUE by the
# i18n tests), so there is no word-level fallback path.
_FIGURE_EN = {
    "ronde": "whole",
    "blanche": "half",
    "noire": "quarter",
    "croche": "eighth",
    "double-croche": "sixteenth",
    "trente-deuxième": "thirty-second",
    "ronde pointée": "dotted whole",
    "blanche pointée": "dotted half",
    "noire pointée": "dotted quarter",
    "croche pointée": "dotted eighth",
    "double-croche pointée": "dotted sixteenth",
    "pause": "whole rest",
    "demi-pause": "half rest",
    "soupir": "quarter rest",
    "demi-soupir": "eighth rest",
    "quart de soupir": "sixteenth rest",
    "seizième de soupir": "thirty-second rest",
    "pause pointée": "dotted whole rest",
    "demi-pause pointée": "dotted half rest",
    "soupir pointée": "dotted quarter rest",
    "demi-soupir pointée": "dotted eighth rest",
    "quart de soupir pointée": "dotted sixteenth rest",
    "noire double-pointée": "double dotted quarter",
    "croche double-pointée": "double dotted eighth",
    "soupir double-pointée": "double dotted quarter rest",
    "demi-soupir double-pointée": "double dotted eighth rest",
    "quart de soupir double-pointée": "double dotted sixteenth rest",
}


def tr_figure(figure: str | None) -> str:
    """Display a canonical French figure name ('noire') in the UI language.

    A name outside the table comes back unchanged, like tr()'s key
    fallback — half-translating unknown input is worse than showing the
    canonical name, and the vocabulary is closed.
    """
    if not figure:
        return ""
    if _current == "fr":
        return figure
    if figure.startswith("triolet de "):
        return "triplet of " + tr_figure(figure[len("triolet de ") :])
    return _FIGURE_EN.get(figure, figure)


# French display of the key modes, keyed by the WHOLE canonical mode word
# from keys.KEY_PROFILES — the same whole-name-table design as _FIGURE_EN.
# The guard tests pin this table against the detection source, so a mode
# added there without its display translation fails the i18n tests.
_MODE_FR = {
    "major": "majeur",
    "minor": "mineur",
}


def tr_key_label(label: str) -> str:
    """Translate a key label ('C major') into the UI language.

    The mode word is looked up in _MODE_FR; a mode outside the table comes
    back unchanged, like tr()'s key fallback — the vocabulary is closed and
    pinned against keys.KEY_PROFILES by the i18n tests.
    """
    tonic, _, mode = label.partition(" ")
    if _current == "en":
        return label
    return f"{tonic} {_MODE_FR.get(mode, mode)}".strip()


# Whole-label chord translation, keyed by quality, Pinned to its producer:
# the keys are exactly chords.CHORD_TEMPLATES (a quality added there without
# its display word fails the i18n guard, like tr_figure's). Replaces the
# deleted per-suffix tr_chord_quality (ADR 0004), which could only expand
# "m" and left "Am" as "Am" — the label a French reader reads whole.
_CHORD_QUALITY_FR = {
    "": "majeur",
    "m": "mineur",
    "dim": "diminué",
    "aug": "augmenté",
    "7": "septième",
    "maj7": "septième majeure",
    "m7": "septième mineure",
}


def tr_chord_label(label: str) -> str:
    """Translate a whole chord label ('Am', 'Cmaj7') into the UI language.

    Storage stays canonical ('Am' in the JSON and MusicXML); this is the
    DISPLAY edge, like tr_figure/tr_key_label. The root is PARSED against
    music.NOTE_NAMES (the English table chords.py builds labels from —
    two-char 'C#'/'F#' first, so 'C#m' is not 'C' + '#m') and DISPLAYED in
    the solfège table; the quality — every suffix after the root, one of
    the CHORD_TEMPLATES keys — is looked up in the table above. An unknown
    root or quality comes back unchanged: half-translating an unknown label
    is worse than showing the canonical name.
    """
    if _current == "en":
        return label
    for i in (2, 1):
        if i <= len(label) and label[:i] in NOTE_NAMES:
            root, quality = label[:i], label[i:]
            if quality not in _CHORD_QUALITY_FR:
                return label
            fr_root = NOTE_NAMES_FR[NOTE_NAMES.index(root)]
            return f"{fr_root} {_CHORD_QUALITY_FR[quality]}"
    return label


__all__ = [
    "available_languages",
    "get_language",
    "get_language_display",
    "note_display",
    "note_display_pair",
    "set_debug",
    "set_language",
    "toggle_language",
    "tr",
    "tr_chord_label",
    "tr_figure",
    "tr_key_label",
]
