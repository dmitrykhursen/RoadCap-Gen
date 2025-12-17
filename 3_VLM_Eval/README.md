This folder contains code for evaluating Vision–Language Models (VLMs) on geometry-related tasks using generated question–answer pairs.

Evaluation is performed in two ways:
1. DriveLM Benchmark Evaluation, where models are evaluated using the official Hugging Face test server (ground-truth answers are not accessible).
2. NLP-Based Evaluation, where the data is split into train/validation sets and standard NLP metrics (e.g., BLEU, CIDEr, ROUGE, LLaMA-based or ChatGPT-based scores) are computed.