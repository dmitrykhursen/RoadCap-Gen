#!/bin/bash

# 8B QWEN tests
uv run vllm_test.py --model-name="Qwen/Qwen3-8B" --use-vllm --thinking
uv run vllm_test.py --model-name="Qwen/Qwen3-8B" --thinking
uv run vllm_test.py --model-name="Qwen/Qwen3-8B" --use-vllm 
uv run vllm_test.py --model-name="Qwen/Qwen3-8B" 

# 14B QWEN tests
uv run vllm_test.py --model-name="Qwen/Qwen3-14B" --use-vllm --thinking
uv run vllm_test.py --model-name="Qwen/Qwen3-14B" --thinking
uv run vllm_test.py --model-name="Qwen/Qwen3-14B" --use-vllm 
uv run vllm_test.py --model-name="Qwen/Qwen3-14B" 