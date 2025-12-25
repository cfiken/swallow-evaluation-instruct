#!/bin/bash
# 推論過程の安定性分析プログラムのバッチ実行用スクリプト
# このスクリプトを実行すると，指定した複数のモデルに対して
# calculate_reasoning_failure_stats.py が順次実行される．結果はすべて OUTPUT_BASENAME で指定したファイルに追記される．
# （注意：既存ファイルがある場合は削除すること．さもないと追記されてしまう）
# 特に，ワンライナーCSV形式のファイルを回収してスプレッドシートに貼り付けると便利．

# LightEvalの --output-dir 引数で指定しているディレクトリ．このディレクトリの直下に ./details/.../*.parquet ファイルが存在している必要がある．
LIGHTEVAL_OUTPUT_DIR="/home/sakae/large_language_models/swallow-evaluation-instruct-private/data/evals"

# 出力ファイルのベースネーム
OUTPUT_BASENAME="stability_diagnosis_results"

#―― ここに評価したいモデルを並べる ―――――――――――――――――――――――――
# 左から順に MODEL ID, provider, reasoning_starter をカンマ区切りで書く．
# provider, reasoning_parser を省略した場合はデフォルト値が適用される．
ENTRIES=(
  "tokyotech-llm/Qwen-3-8B-SFT-ablation-exp1-LR1.5E-5-iter0007800,,"
  "tokyotech-llm/Qwen-3-8B-SFT-ablation-exp5-LR1.5E-5-iter0023500,,"
)

# provider, reasoning_parser のデフォルト値
DEFAULT_REASONING_STARTER="<think>"
DEFAULT_LLM_PROVIDER="hosted_vllm"

# ENTRIESの各エントリーに対してループ実行
for ENTRY in "${ENTRIES[@]}"; do
    # カンマ区切りでMODEL_ID, provider, reasoning_starterを分割
    IFS=',' read -r MODEL_ID LLM_PROVIDER REASONING_STARTER <<< "$ENTRY"
    
    # providerが空の場合はデフォルト値を使用
    if [ -z "$LLM_PROVIDER" ]; then
        LLM_PROVIDER="$DEFAULT_LLM_PROVIDER"
    fi
    
    # reasoning_starterが空の場合はデフォルト値を使用
    if [ -z "$REASONING_STARTER" ]; then
        REASONING_STARTER="$DEFAULT_REASONING_STARTER"
    fi
    
    echo "Processing: $MODEL_ID (provider: $LLM_PROVIDER, reasoning_starter: $REASONING_STARTER)"
    
    uv run python calculate_reasoning_failure_stats.py \
        --model_id "$MODEL_ID" \
        --lighteval-output-dir "$LIGHTEVAL_OUTPUT_DIR" \
        --reasoning_starter "$REASONING_STARTER" \
        --provider "$LLM_PROVIDER" \
        --output-basename "$OUTPUT_BASENAME" \
        --append
done
