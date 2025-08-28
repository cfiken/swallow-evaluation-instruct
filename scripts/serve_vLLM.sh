PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export VLLM_FLASH_ATTN_VERSION=3
export CUDA_VISIBLE_DEVICES=1
export NUM_GPUS=1
MODEL="openai/gpt-oss-20b"
# SYSTEM_PROMPT="Reasoning: high"
MAX_MODEL_LENGTH=16384
# TEMPERATURE=0.0
# TOP_P=1.0

HOST=192.168.1.103
PORT=9001
BASE_URL="http://$HOST:$PORT/v1"
API_KEY="dummy"

SANITIZED_NAME=$(echo "$MODEL" | tr '/:' '__')

vllm serve "$MODEL" \
--tensor-parallel-size=$NUM_GPUS \
--host "$HOST" \
--port "$PORT"
# --max-model-len ${MAX_MODEL_LENGTH} \