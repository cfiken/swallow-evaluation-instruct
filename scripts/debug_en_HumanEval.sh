PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=7
export NUM_GPUS=1
MODEL=Qwen/Qwen2.5-1.5B-Instruct
# MODEL=tokyotech-llm/Llama-3.1-Swallow-8B-v0.5
MAX_MODEL_LENGTH=8192
TEMPERATURE=0.2
TOP_P=0.95

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.8,generation_parameters={temperature:$TEMPERATURE,top_p:$TOP_P}"
OUTPUT_DIR=data/evals/

# HumanEval
lighteval vllm $MODEL_ARGS "swallow|humanevalplus|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --use-chat-template
