#!/bin/bash

# Configuration
# Configuration
VLM_MODEL="PaddlePaddle/PaddleOCR-VL"
LLM_MODEL="Qwen/Qwen2.5-1.5B"
GPU_UTIL=0.45 
VLLM_PYTHON="./vllm_engine/bin/python3"

echo "--- Starting vLLM Servers (Phase 1 Automation) ---"

# Start VLM Server (Port 8000)
echo "Launching VLM Server on port 8000..."
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
    --model $VLM_MODEL \
    --port 8000 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 4096 \
    --trust-remote-code > logs/vlm.log 2>&1 &

# Start LLM Server (Port 8001)
echo "Launching LLM Server on port 8001..."
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
    --model $LLM_MODEL \
    --port 8001 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    --trust-remote-code > logs/llm.log 2>&1 &


echo "--- Servers are starting in the background ---"
echo "Use 'ps aux | grep vllm' to monitor or check logs."
