# GPT-4.1 の MMLU-ProX を OpenAI API で評価する
API_KEY="sk-proj-5d2bTVHUfUxh37Vc3zZRT3BlbkFJhyUowK2sFi6ECkyxvBib"
MODEL_NAME="gpt-4.1-2025-04-14"
BASE_URL="https://api.openai.com/v1" # OpenAI API の URL
OUTPUT_DIR=data/evals/

# MMLU-ProX Japanese
lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL" \
    "swallow|mmlu_prox_japanese|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# micro average を算出
# python scripts/aggregate_results.py --model_name Qwen/Qwen3-32B --raw_outputs_dir ./data/evals/results/Qwen/Qwen3-32B/ --aggregated_outputs_dir ./
