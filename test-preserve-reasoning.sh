#!/bin/bash
# Probe whether the loaded model's chat template renders reasoning_content back into the prompt.
# Usage on rocky2:  ./test-preserve-reasoning.sh [SERVER_URL]   (default http://10.0.2.2:8080)

URL="${1:-http://10.0.2.2:8080}"
TAG="$(date +%s)_$$"
MARKER="XYZZYREASONINGMARKER42"

echo "== Request 1: single user turn (baseline) =="
BASELINE=$(curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role":"user","content":"say hi"}],
  "max_tokens": 1,
  "temperature": 0,
  "logprobs": 0
}')
echo "$BASELINE" | python3 -c 'import sys,json; r=json.load(sys.stdin); print("baseline: prompt_tokens =", r["usage"]["prompt_tokens"], "| completion ok =", bool(r.get("choices")))'

echo
echo "== Request 2: same + assistant turn carrying reasoning_content with unique marker =="
WITH_REASONING=$(curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"qwen3.8-27b\",
  \"messages\": [
    {\"role\":\"user\",\"content\":\"say hi\"},
    {\"role\":\"assistant\",\"content\":\"Hello! How can I help?\",\"reasoning_content\":\"$MARKER\"}
  ],
  \"max_tokens\": 1,
  \"temperature\": 0,
  \"logprobs\": 0
}")
echo "$WITH_REASONING" | python3 -c 'import sys,json; r=json.load(sys.stdin); print("with-reasoning: prompt_tokens =", r["usage"]["prompt_tokens"], "| completion ok =", bool(r.get("choices")))'
