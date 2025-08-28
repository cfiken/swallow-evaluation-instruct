# Llama 4 Maverick を DeepInfra API で評価する
API_KEY="DeepInfraのAPI Key"
API_KEY="sk-proj-5d2bTVHUfUxh37Vc3zZRT3BlbkFJhyUowK2sFi6ECkyxvBib"
MODEL_NAME="openai/gpt-4o-2024-08-06"
BASE_URL="https://api.openai.com/v1/" # OpenAI API の URL
OUTPUT_DIR=data/evals/

export LITELLM_CONCURRENT_CALLS=50

# LiveCodeBench test-run
lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={max_n:4,top_p:0.95,temperature:0.6}" \
    "swallow|lcb:codegeneration_v6|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --max-samples 1

# GPQA test-run
lighteval endpoint litellm \
"model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL" \
"swallow|gpqa:diamond|0|0" \
--save-details \
--use-chat-template \
--output-dir $OUTPUT_DIR \
--max-samples 5

# GPQA test-run with reasoning_effort
: '
lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={reasoning_effort:\"high\"}" \
    "swallow|gpqa:diamond|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --max-samples 5 \
    --push-to-hub \
    --results-org tokyotech-llm
'