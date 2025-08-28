#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NUM_GPU=4
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
MODEL_NAME="openai/gpt-oss-120b"

# launch vLLM
vllm serve \
$MODEL_NAME \
--enable-log-requests \
--enable-log-outputs \
--return-tokens-as-token-ids \
--host 192.168.1.108 \
--port 9000 \
--tensor-parallel-size ${NUM_GPU} \
--gpu-memory-utilization 0.9
