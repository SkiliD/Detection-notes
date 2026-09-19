# --- Voix -> MIDI : image CLI (analyse de fichiers + conversion MIDI) ---
# Build : docker build -t v2m .
# CLI   : docker run --rm -v "%cd%:/data" v2m analyze /data/chant.wav --quantize
#         avec PDF : ajouter --pdf (MuseScore est installé dans l'image)
# GUI   : (Linux/X11 uniquement)
#         docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#           -v "$PWD:/data" v2m gui
FROM python:3.12-slim

# Dépendances système :
#  - libsndfile1   : lecture/écriture WAV/FLAC/OGG (soundfile)
#  - libgomp1      : OpenMP (numba/llvmlite utilisés par librosa)
#  - ffmpeg        : décodage MP3/M4A/AAC par librosa
#  - libgl1/libegl1/fontconfig : rendu Qt pour la GUI (X11) et pyqtgraph
#  - musescore3    : gravure PDF de la partition (--pdf dans le conteneur)
#  - xvfb + xauth  : écran virtuel requis par MuseScore même headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        ffmpeg \
        libgl1 \
        libegl1 \
        fontconfig \
        musescore3 \
        xvfb \
        xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances verrouillées d'abord (couche cache stable), puis le paquet
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && pip install --no-cache-dir --no-deps .

# Données montées dans /data (audios d'entrée, MIDI/MusicXML/JSON/CSV de sortie)
RUN mkdir /data
WORKDIR /data

# Le micro n'est PAS exposé ici : le temps réel nécessite un accès audio
# hôte (PulseAudio/PipeWire) et un vrai périphérique -> à utiliser hors Docker.
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["analyze", "/data/audio.wav"]
