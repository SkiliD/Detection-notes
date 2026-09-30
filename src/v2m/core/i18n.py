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
        "key coherence": "Cohérence",
        "ui tuning ok": "juste, continuez comme ça",
        "ui tuning slightly off": "à {dir} de quelques cents, à surveiller",
        "ui tuning off": "nettement {dir}, détendez avant de chanter",
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
        "ui app subtitle": "Chantez : l'application détecte les notes, le tempo, la tonalité et les accords,\npuis exporte un fichier MIDI, une partition MusicXML et un rapport JSON/CSV.",
        "ui open file": "📁  Analyser un fichier audio",
        "ui sing live": "🎤  Chanter au micro (temps réel)",
        "ui finder button": "🎹  Trouver la note que je chante",
        "ui finder start": "▶  Écouter le micro",
        "ui finder stop": "■  Arrêter d'écouter",
        "ui finder idle": "♪  Chantez une note…",
        "ui finder live": "●  Micro ouvert — chantez une note",
        "ui finder off": "Micro fermé",
        "ui finder silence": "Le micro n'entend rien — vérifiez qu'il est actif et rapprochez-vous.",
        "ui finder no note": "Le micro entend un son — tenez la note plus d'un instant.",
        "ui finder hint": "Chantez une note : sa touche s'allume sur le clavier du piano.\nRien d'autre — pas d'analyse, pas d'export, rien n'est conservé.",
        "ui finder history": "Dernières notes entendues",
        "ui finder history empty": "L'historique apparaîtra ici.",
        "ui level idle": "niveau : —",
        "ui expected none": "Aucune note attendue",
        "ui expected note": "Attendue : {note}",
        "ui live hint": "Chantez devant la partition : la touche devient verte si la note correspond, rouge sinon.",
        "ui metronome": "Métronome",
        "ui metronome tooltip": "Clic régulier pour travailler la régularité du chant (désactivé par défaut).",
        "ui record audio": "Enregistrer l'audio de la prise",
        "ui record audio tooltip": "Écrit la prise en .wav dans le dossier de sortie du mode live (coché par défaut — une prise sans audio reste inaudible rétroactivement).",
        "ui bpm unit": "BPM",
        "ui beat count": "Temps {beat}",
        "ui accuracy title": "Justesse moyenne :",
        "ui accuracy empty": "Justesse moyenne : —",
        "ui play voice": "🎙 Voix",
        "ui play both": "🎹+🎙",
        "ui play both tooltip": "Piano par-dessus ta voix, au volume réglé — pour comparer l'interprétation.",
        "ui volume": "Vol. piano",
        "ui volume tooltip": "Volume du piano rejoué, pour l'entendre par rapport à ta voix.",
        "ui axis note": "Note",
        "ui axis time": "Temps",
        "ui drop hint": "Astuce : glissez-déposez un fichier audio dans cette fenêtre.",
        "ui lang button": "🌐  Langue : Français",
        "ui back home": "← Accueil",
        "ui no file": "Aucun fichier",
        "ui export options": "Options d'export",
        "ui quantize": "Quantifier sur la grille 1/16",
        "ui transpose": "Transposition (demi-tons)",
        "ui notation mode": "Mode notation (figures colorées)",
        "ui pdf option": "Exporter aussi en PDF",
        "ui pdf tooltip": "Installez MuseScore ou LilyPond pour activer l'export PDF.",
        "ui re-export": "💾  Ré-exporter avec ces options",
        "ui open another": "📁  Ouvrir un autre fichier",
        "ui stop live": "← Arrêter / Accueil",
        "ui mic preparing": "Préparation du micro…",
        "ui live progress": "🎤 {t} · {notes} notes · {meter}",
        "ui live silence": "Silence depuis {s} s — le micro n'entend rien (échec, déplacé ou muet ?)",
        "ui live low mic": "Rien au-dessus du seuil depuis {s} s — le micro n'entend rien : vérifiez-le avant de commencer la prise.",
        "ui live notes label": "notes confirmées",
        "ui staff tooltip": "Portée (clé de sol) : les notes détectées, écrites en notation musicale (figures, altérations, mesures).",
        "key bpm unit": "BPM",
        "ui stopping": "Arrêt…",
        "ui analyzing": "Analyse finale…",
        "ui live file": "Enregistrement live",
        "ui choose file": "Choisir un fichier audio",
        "ui audio filter": "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;Tous les fichiers (*)",
        "ui status exported": "Exporté : {files}",
        "ui status cache": "Évidence vocale reprise du cache ({ms:.0f} ms) — pYIN non relancé",
        "ui min note": "Note minimale :",
        "ui reinterpret": "↻ Rejouer l'analyse",
        "ui reinterpret tooltip": "Relit l'évidence vocale en cache avec le seuil ci-dessus (pYIN non relancé, quelques ms). Aucun fichier n'est réexporté automatiquement.",
        "ui status reinterpreted": "Analyse rejouée : {n} notes (évidence inchangée). Exportez pour écrire les fichiers.",
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
        "ui steps": "1 · Chargez un fichier ou chantez au micro\n2 · L'analyse détecte notes, tempo, tonalité et accords\n3 · Écoutez ▶ puis exportez MIDI · partition · PDF",
        "ui pdf badge": "PDF indisponible — installez MuseScore ou LilyPond",
        "ui exported ok": "✓ Exporté",
        "ui analyzing file": "Analyse…",
        "ui panel waiting": "En attente d'une analyse —\nchargez un fichier ou chantez au micro.",
        # --- the take's coherence verdict (results panel) ---
        "ui coherence title": "Cohérence de la prise",
        "ui coherence clean": "✓ propre : {checked} note(s) contrôlée(s), aucune phrase avalée",
        "ui coherence suspects": "⚠ {n} suspect(s) sur {checked} note(s) contrôlée(s) — détail dans le rapport JSON",
        "ui coherence na": "contrôle indisponible (évidence absente)",
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
        "coh suspect row": "    midi {midi:>3}  portée {span:.2f}st  plateau {dur:.2f}s {at}→ midi {target} ({offset})",
        "coh plateau at": "à {t:.2f}s ",
        "coh verdict clean": "Aucun suspect sur {takes} prise(s) ({notes} notes contrôlées) : aucune note longue ne cache de plateau secondaire.",
        "coh verdict clean nonotes": "Aucun suspect sur {takes} prise(s) : aucune note longue ne cache de plateau secondaire (> 120 ms tenus à plus d'un demi-ton de l'étiquette).",
        "coh verdict suspects": "{suspects} suspect(s) au total sur {takes} prise(s) ({notes} notes contrôlées) — détail par prise ci-dessus.",
        "coh gui footer": "{takes} prise(s) agrégée(s).",
        "ui coherence sessions btn": "📊  Suivi de cohérence des prises",
        "ui coherence sessions title": "Suivi de cohérence (sessions de 30 min)",
        "ui coherence running": "Agrégation…",
        "ui coherence none": "Aucune prise à agréger : chantez une prise d'abord (prises_live/).",
        "ui status coherence": "Cohérence : {summary}",
        # --- live/final explanation (why the score differs from the live view) ---
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
        "key coherence": "Coherence",
        "ui tuning ok": "in tune, keep it up",
        "ui tuning slightly off": "a bit {dir}, watch it",
        "ui tuning off": "clearly {dir}, relax before singing on",
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
        "ui app subtitle": "Sing: the app detects notes, tempo, key and chords,\nthen exports a MIDI file, a MusicXML score and a JSON/CSV report.",
        "ui open file": "📁  Analyze an audio file",
        "ui sing live": "🎤  Sing into the mic (real-time)",
        "ui finder button": "🎹  Find the note I am singing",
        "ui finder start": "▶  Listen to the mic",
        "ui finder stop": "■  Stop listening",
        "ui finder idle": "♪  Sing a note…",
        "ui finder live": "●  Mic open — sing a note",
        "ui finder off": "Mic closed",
        "ui finder silence": "The mic hears nothing — check it is not muted and get closer.",
        "ui finder no note": "The mic hears sound — hold the note a moment longer.",
        "ui finder hint": "Sing a note: its key lights up on the piano keyboard.\nNothing else — no analysis, no export, nothing is kept.",
        "ui finder history": "Recently heard notes",
        "ui finder history empty": "Your history will appear here.",
        "ui level idle": "level: —",
        "ui expected none": "No expected note",
        "ui expected note": "Expected: {note}",
        "ui live hint": "Sing along with the score: the key turns green when it matches, red otherwise.",
        "ui metronome": "Metronome",
        "ui metronome tooltip": "Steady click to practice singing in time (off by default).",
        "ui record audio": "Save the take's audio",
        "ui record audio tooltip": "Writes the take to .wav in the live mode's output folder (on by default — a take without audio cannot be audited later).",
        "ui bpm unit": "BPM",
        "ui beat count": "Beat {beat}",
        "ui accuracy title": "Average tuning:",
        "ui accuracy empty": "Average tuning: —",
        "ui play voice": "🎙 Voice",
        "ui play both": "🎹+🎙",
        "ui play both tooltip": "Piano over your voice, at the set volume — to compare the takes.",
        "ui volume": "Piano vol.",
        "ui volume tooltip": "Rendered piano volume, to hear it against your voice.",
        "ui axis note": "Note",
        "ui axis time": "Time",
        "ui drop hint": "Tip: drag & drop an audio file onto this window.",
        "ui lang button": "🌐  Language: English",
        "ui back home": "← Home",
        "ui no file": "No file",
        "ui export options": "Export options",
        "ui quantize": "Quantize to the 1/16 grid",
        "ui transpose": "Transpose (semitones)",
        "ui notation mode": "Notation mode (colored figures)",
        "ui pdf option": "Also export as PDF",
        "ui pdf tooltip": "Install MuseScore or LilyPond to enable PDF export.",
        "ui re-export": "💾  Re-export with these options",
        "ui open another": "📁  Open another file",
        "ui stop live": "← Stop / Home",
        "ui mic preparing": "Preparing the microphone…",
        "ui live progress": "🎤 {t} · {notes} notes · {meter}",
        "ui live silence": "Silent for {s} s — the mic hears nothing (dropped, moved or muted?)",
        "ui live low mic": "Nothing above the gate for {s} s — the mic hears nothing: check it before starting the take.",
        "ui live notes label": "notes confirmed",
        "ui staff tooltip": "Staff (treble clef): detected notes written in musical notation (figures, accidentals, bars).",
        "key bpm unit": "BPM",
        "ui stopping": "Stopping…",
        "ui analyzing": "Final analysis…",
        "ui live file": "Live recording",
        "ui choose file": "Choose an audio file",
        "ui audio filter": "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;All files (*)",
        "ui status exported": "Exported: {files}",
        "ui status cache": "Vocal evidence reloaded from cache ({ms:.0f} ms) — pYIN skipped",
        "ui min note": "Min note:",
        "ui reinterpret": "↻ Re-run analysis",
        "ui reinterpret tooltip": "Re-reads the cached vocal evidence with the threshold above (pYIN skipped, a few ms). Nothing is re-exported automatically.",
        "ui status reinterpreted": "Analysis re-run: {n} notes (evidence unchanged). Export to write the files.",
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
        "ui steps": "1 · Load a file or sing into the mic\n2 · The analysis detects notes, tempo, key and chords\n3 · Listen ▶ then export MIDI · score · PDF",
        "ui pdf badge": "PDF unavailable — install MuseScore or LilyPond",
        "ui exported ok": "✓ Exported",
        "ui analyzing file": "Analyzing…",
        "ui panel waiting": "Waiting for an analysis —\nload a file or sing into the mic.",
        # --- take coherence verdict (results panel) ---
        "ui coherence title": "Take coherence",
        "ui coherence clean": "✓ clean: {checked} note(s) audited, no swallowed phrase",
        "ui coherence suspects": "⚠ {n} suspect(s) over {checked} audited note(s) — details in the JSON report",
        "ui coherence na": "check unavailable (no evidence)",
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
        "coh suspect row": "    midi {midi:>3}  span {span:.2f}st  plateau {dur:.2f}s {at}→ midi {target} ({offset})",
        "coh plateau at": "at {t:.2f}s ",
        "coh verdict clean": "No suspect over {takes} take(s) ({notes} audited notes): no long note hides a secondary plateau.",
        "coh verdict clean nonotes": "No suspect over {takes} take(s): no long note hides a secondary plateau (> 120 ms held more than a semitone off the label).",
        "coh verdict suspects": "{suspects} suspect(s) total over {takes} take(s) ({notes} audited notes) — per-take detail above.",
        "coh gui footer": "{takes} take(s) aggregated.",
        "ui coherence sessions btn": "📊  Takes coherence follow-up",
        "ui coherence sessions title": "Coherence follow-up (30-min sessions)",
        "ui coherence running": "Aggregating…",
        "ui coherence none": "No take to aggregate: sing a take first (prises_live/).",
        "ui status coherence": "Coherence: {summary}",
        # --- live/final explanation (why the score differs from the live view) ---
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
        "ui practice windows title": "When do you sing best?",
        "ui practice window row": "  {window:<22} {takes:>3} take(s)   {rate:>6}/100   {contours:>5}",
        "ui retention pruned": "Retention: {n} old take(s) deleted ({stems}) — history stays in the journal.",
        # --- phrase-repeat mode: two takes of one phrase, side by side ---
        "cmp unreadable": "Report missing or unreadable: {path}",
        "cmp head": "Phrase-repeat comparison — take A: {a} ({n_a} notes) vs take B: {b} ({n_b} notes)",
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
            print(f"[v2m i18n] clé absente de la table « {_current} » : {key!r}",
                  file=sys.stderr)
        return key
    return STRINGS[_current][key]


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
        return "triplet of " + tr_figure(figure[len("triolet de "):])
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
