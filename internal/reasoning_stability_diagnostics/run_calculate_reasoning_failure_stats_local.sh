#!/bin/bash
# 推論過程の安定性分析プログラムの実行例

# スクリプトのディレクトリに移動
# cd "$(dirname "$0")"

# デフォルト値
LIGHTEVAL_OUTPUT_DIR="/home/sakae/temp/lighteval-outputs"
MODEL_ID="Qwen/Qwen3-8B/reasoning"
REASONING_STARTER="<think>"
LLM_PROVIDER="hosted_vllm"

python calculate_reasoning_failure_stats.py \
    --model_id "$MODEL_ID" \
    --lighteval-output-dir "$LIGHTEVAL_OUTPUT_DIR" \
    --reasoning_starter "$REASONING_STARTER" \
    --provider "$LLM_PROVIDER"
