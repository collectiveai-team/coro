AUDIO=./audios/RNE14-agosto-13.mp3
# AUDIO=audios/SPP_RODRIGUEZ_CARLOS_78274-2023.wav

# SERVER_CMD=uv run --no-sync python custom_server.py --pcm-input --model medium --diarization --language es --port 4000

# uv run --no-sync python custom_client.py "$AUDIO" \
#     --language es \
#     --openai \
#     --url http://localhost:4000 \
#     --final-output outputs/$AUDIO/whisper-medium/http-final.json \
#     --concat-output outputs/$AUDIO/whisper-medium/http-transcript.txt

uv run --no-sync python custom_client.py "$AUDIO" \
    --language es \
    --openai \
    --url http://localhost:4000 \
    --stream \
    --intermediate-output outputs/$AUDIO/whisper-medium/sse-intermediates.jsonl \
    --final-output outputs/$AUDIO/whisper-medium/sse-final.json \
    --concat-output outputs/$AUDIO/whisper-medium/sse-transcript.txt

# uv run --no-sync python custom_client.py "$AUDIO" \
#     --language es \
#     --openai \
#     --url http://localhost:4000 \
#     --stream \
#     --intermediate-output outputs/$AUDIO/whisper-medium/nodiar-sse-intermediates.jsonl \
#     --final-output outputs/$AUDIO/whisper-medium/nodiar-sse-final.json \
#     --concat-output outputs/$AUDIO/whisper-medium/nodiar-sse-transcript.txt