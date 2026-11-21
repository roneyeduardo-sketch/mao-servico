FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# baixa os modelos no build: detector de mão (MediaPipe) e a rede de linhas (2 partes, 10 bytes de cabeçalho cada)
RUN curl -fsSL -o hand.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
 && curl -fsSL -o a.bin https://media.atomicatmedia.net/u/DOxpwc55tuUBGfPB3FEh9ojw84k1/Pictures/QbaAQi0896861.mp3 \
 && curl -fsSL -o b.bin https://media.atomicatmedia.net/u/DOxpwc55tuUBGfPB3FEh9ojw84k1/Pictures/lrJSbc1046376.mp3 \
 && (tail -c +11 a.bin; tail -c +11 b.bin) > palm.onnx && rm a.bin b.bin && ls -l palm.onnx hand.task
ENV PORT=8080
CMD gunicorn -b 0.0.0.0:$PORT -w 1 --threads 1 --timeout 180 app:app
