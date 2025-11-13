#!/bin/bash
# 推論過程の安定性分析プログラムの実行例

# スクリプトのディレクトリに移動
# cd "$(dirname "$0")"

# デフォルト値
MODEL_ID="tokyotech-llm/Qwen3-Swallow-8B-v0.1-SFT-exp11-LR1.5E-5-iter0023000"
HF_ORGANIZATION="tokyotech-llm"
REASONING_STARTER="<think>"
LLM_PROVIDER="hosted_vllm"

uv run python calculate_reasoning_failure_stats.py \
    --model_id "$MODEL_ID" \
    --hf_organization "$HF_ORGANIZATION" \
    --reasoning_starter "$REASONING_STARTER" \
    --provider "$LLM_PROVIDER"
