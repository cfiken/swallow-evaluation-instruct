#!/bin/bash

# litellm backend で reasoning parser を利用可能にする，なおかつ文字列ベースのreasoning_parserを動作確認するためのスクリプト
# 想定するモデルは Llama-3.*-Nemotronのような <think>タグを語彙に持たない推論型モデル．
set -e

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=2,3
export NUM_GPUS=2

export REQUEST_TIMEOUT=600000 # litellm default timeout [sec]
export LITELLM_CONCURRENT_CALLS=50

HOST=192.168.1.108
PORT=9001
BASE_URL="http://$HOST:$PORT/v1"
API_KEY="dummy"
CHAT_TEMPLATE="./resources/chat_template_base_model.jinja"

# SYSTEM_PROMPT="あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MODEL_LENGTH=8192
MAX_NUM_SEQS=32
TEMPERATURE=0.2
TOP_P=0.95

BENCHMARK="swallow|swallow_jhumaneval|0|0"

#―― ここに評価したいモデルを並べる ―――――――――――――――――――――――――
# 左から順に HF ID, max-model-len, stop_tokens である．
ENTRIES=(
  "google/gemma-2-9b,,"
  "google/gemma-3-12b-pt,,"
  "meta-llama/Meta-Llama-3.1-8B,,"
  "Qwen/Qwen2.5-1.5B,,"
  "Qwen/Qwen2.5-7B,,"
  "Qwen/Qwen3-8B-Base,,"
  "tokyotech-llm/Llama-3.1-Swallow-8B-v0.5,,"
)

# 左から順に HF ID, reasoning_parser, system_message, max-model-len である．
for ENTRY in "${ENTRIES[@]}"; do
    MODEL=$(echo "$ENTRY" | cut -d',' -f1)
    ENTRY_MAX_MODEL_LEN=$(echo "$ENTRY" | cut -d',' -f2)
    # STOP_SEQUENCESをカンマ区切り3番目以降すべて取得
    STOP_SEQUENCES=$(echo "$ENTRY" | awk -F',' '{OFS=","; for(i=3;i<=NF;i++) printf "%s%s", $i, (i<NF?",":""); print ""}')
    
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

    vllm serve "$MODEL" \
      --tensor-parallel-size=$NUM_GPUS \
      $MAX_MODEL_LEN_ARG \
      --chat-template $CHAT_TEMPLATE \
      --host "$HOST" \
      --port "$PORT" \
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
    echo ${STOP_SEQUENCES}

    # generation_parametersの組み立て
    GEN_PARAMS="temperature:$TEMPERATURE,top_p:$TOP_P"
    if [[ -n "$STOP_SEQUENCES" ]]; then
        GEN_PARAMS="$GEN_PARAMS,stop_token:\\\"$STOP_SEQUENCES\\\""
    fi

    set +e
    lighteval endpoint litellm \
    "model=hosted_vllm/$MODEL,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={$GEN_PARAMS}" \
    ${BENCHMARK} \
    --save-details \
    --output-dir "$OUTPUT_DIR" \
    --use-chat-template \
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
