#!/bin/bash

# litellm backend で reasoning parser を利用可能にする，なおかつ文字列ベースのreasoning_parserを動作確認するためのスクリプト
# 想定するモデルは Llama-3.*-Nemotronのような <think>タグを語彙に持たない推論型モデル．
set -e

export OPENAI_API_KEY="sk-proj-5d2bTVHUfUxh37Vc3zZRT3BlbkFJhyUowK2sFi6ECkyxvBib"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=4,5
export NUM_GPUS=2

export REQUEST_TIMEOUT=600000 # litellm default timeout [sec]
export LITELLM_CONCURRENT_CALLS=50

HOST=192.168.1.108
PORT=9001
BASE_URL="http://$HOST:$PORT/v1"
API_KEY="dummy"

# SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=16384
MAX_NUM_SEQS=32
TEMPERATURE=0.0
TOP_P=1.0

#―― ここに評価したいモデルを並べる ―――――――――――――――――――――――――
# 左から順に HF ID, reasoning_parser, system_message, max-model-len である．
ENTRIES=(
  # "Qwen/Qwen3-8B,qwen3,,"
  # "google/gemma-3-12b-it,,,"
  "microsoft/phi-4-reasoning-plus,deepseek_r1,,"
  # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B,deepseek_r1,,"
  # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B,,,"
  # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1,deepseek_r1_markup,thinking mode on,"
  # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1,,thinking mode off,"
  # "Qwen/Qwen2.5-7B-Instruct,,,"
  # "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5,,あなたは誠実で優秀な日本人のアシスタントです。,8192"
)

# 左から順に HF ID, reasoning_parser, system_message, max-model-len である．
for ENTRY in "${ENTRIES[@]}"; do
    MODEL=$(echo "$ENTRY" | cut -d',' -f1)
    REASONING_PARSER=$(echo "$ENTRY" | cut -d',' -f2)
    SYSTEM_MESSAGE=$(echo "$ENTRY" | cut -d',' -f3)
    ENTRY_MAX_MODEL_LEN=$(echo "$ENTRY" | cut -d',' -f4)
    # 出力ディレクトリをモデルごとに分ける（/ や : を _ に変換）
    SANITIZED_NAME=$(echo "$MODEL" | tr '/:' '__')
    OUTPUT_DIR="data/evals/"

    echo "▶︎ Evaluating $MODEL …"

    # vllm serveをバックグラウンドで起動
    
    # max-model-lenをENTRYごとに切り替え
    MAX_MODEL_LEN_ARG="--max-model-len ${MAX_MODEL_LENGTH}"
    if [[ -n "$ENTRY_MAX_MODEL_LEN" ]]; then
        MAX_MODEL_LEN_ARG="--max-model-len ${ENTRY_MAX_MODEL_LEN}"
    fi
    REASONING_PARSER_ARG=""
    if [[ -n "$REASONING_PARSER" ]]; then
        REASONING_PARSER_ARG="--reasoning-parser $REASONING_PARSER"
    fi

    vllm serve "$MODEL" \
      --tensor-parallel-size=$NUM_GPUS \
      $MAX_MODEL_LEN_ARG \
      --host "$HOST" \
      --port "$PORT" \
      $REASONING_PARSER_ARG \
      > "vllm_${SANITIZED_NAME}.log" 2>&1 &      
      # --max-num-seqs ${MAX_NUM_SEQS} \
    VLLM_PID=$!

    # ポートが開くまで最大300秒待機（10秒間隔で30回）
    for i in {1..60}; do
        if nc -z "$HOST" "$PORT"; then
            echo "vllm serve is up."
            break
        fi
        echo "Waiting for vllm serve to start... ($i/60)"
        sleep 10
    done

    if ! nc -z "$HOST" "$PORT"; then
        echo "Error: vllm serve did not start on $HOST:$PORT"
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null
        exit 1
    fi

    # lighteval実行
    SYSTEM_MESSAGE_ARG=()
    if [[ -n "$SYSTEM_MESSAGE" ]]; then
        SYSTEM_MESSAGE_ARG=(--system-prompt "$SYSTEM_MESSAGE")
    fi    
    echo ${SYSTEM_MESSAGE_ARG[@]}

    set +e
    lighteval endpoint litellm \
    "model=hosted_vllm/$MODEL,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={temperature:$TEMPERATURE,top_p:$TOP_P}" \
    "swallow|japanese_mt_bench|0|0" \
    "${SYSTEM_MESSAGE_ARG[@]}" \
    --save-details \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR" \
    --max-samples 3
    EXIT_CODE=$?
    set -e

    # vllm serveを必ず終了
    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "lighteval failed (exit code $EXIT_CODE)"
        exit $EXIT_CODE
    fi

    echo "Finished evaluating $MODEL"
done
