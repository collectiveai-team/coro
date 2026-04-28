AUDIO=./audios/RNE14-agosto-13.mp3

# SERVER_CMD=uv run --no-sync python custom_server.py --pcm-input --model medium --diarization --language es

uv run --no-sync python custom_client.py "$AUDIO" \
    --language es \
    --openai \
    --url http://localhost:8000 \
    --final-output outputs/$AUDIO/whisper-medium/http-final.json \
    --concat-output outputs/$AUDIO/whisper-medium/http-transcript.txt

uv run --no-sync python custom_client.py "$AUDIO" \
    --language es \
    --openai \
    --url http://localhost:8000 \
    --stream \
    --intermediate-output outputs/$AUDIO/whisper-medium/sse-intermediates.jsonl \
    --final-output outputs/$AUDIO/whisper-medium/sse-final.json \
    --concat-output outputs/$AUDIO/whisper-medium/sse-transcript.txt