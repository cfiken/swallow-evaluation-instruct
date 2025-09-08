PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
# MODEL=Qwen/Qwen2.5-1.5B
MODEL=tokyotech-llm/Llama-3.1-Swallow-8B-v0.5
MAX_MODEL_LENGTH=8192
TEMPERATURE=0.2
TOP_P=0.95

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=0.5,generation_parameters={temperature:$TEMPERATURE,top_p:$TOP_P}"
OUTPUT_DIR=data/evals/

# JHumanEval
lighteval vllm $MODEL_ARGS "swallow|swallow_jhumaneval|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details \
    --max-samples 10
