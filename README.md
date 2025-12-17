# Caption Generation for Enhancing Road Scene Reasoning in VLMs

Generate QAs (and captions in the future) for driving scenes to fine-tune Vision–Language Models (VLMs), helping them better describe and reason about automotive scenarios.

---

## Pipeline Overview

1. **QAs_Generation**  
   Generate pseudo ground-truth question–answer pairs from driving scenes

2. **VLM_Finetuning**  
   Fine-tune VLMs in two modes:  
   - **Simple mode** – standard visual-text supervision  
   - **Extended mode** – with auxiliary latent-space loss to capture geometric information

3. **VLM_Eval**  
   Evaluate VLMs via:  
   - **DriveLM Benchmark** – tested on their HF server (GT answers not accessible)  
   - **NLP Evaluation** – split data into train/validation sets and compute metrics such as **BLEU, CIDEr, ROUGE, LLaMA/ChatGPT scores**

4. **VLM_Inference** *(optional)*  
   Run inference on new driving scenes.

---

## Pipeline Visualization

![Pipeline](assets/Pipeline%20on%20white%20board.jpg)  