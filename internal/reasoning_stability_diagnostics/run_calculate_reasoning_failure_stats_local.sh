#!/bin/bash
# 推論過程の安定性分析プログラムの実行例

# スクリプトのディレクトリに移動
# cd "$(dirname "$0")"

# Qwen3 の例
LIGHTEVAL_OUTPUT_DIR="/home/sakae/temp/lighteval-outputs"
MODEL_ID="Qwen/Qwen3-8B/reasoning"
REASONING_STARTER="<think>"

# GPT-OSS-Swallow の例
: '
LIGHTEVAL_OUTPUT_DIR="/home/sakae/large_language_models/swallow-evaluation-instruct-private/data/evals"
MODEL_ID="tokyotech-llm/GPT-OSS-Swallow-v0.1-ablation-LR-3.0E-5-iter0012500"
REASONING_STARTER="<think_dummy>"
LLM_PROVIDER="hosted_vllm"
'

uv run python calculate_reasoning_failure_stats.py \
    --model_id "$MODEL_ID" \
    --lighteval-output-dir "$LIGHTEVAL_OUTPUT_DIR" \
    --reasoning_starter "$REASONING_STARTER"