# Caption Generation for Enhancing Road Scene Reasoning in VLMs

Generate QAs (and captions in the future) for driving scenes to fine-tune Vision–Language Models (VLMs), helping them better describe and reason about automotive scenarios.

---

## 🚀 Setup & Installation

Create and activate a Python virtual environment:

```bash
python3.11 -m venv roadcap-gen
source roadcap-gen/bin/activate
```

The project uses **Git Submodules** to integrate external tools like DriveLM. You must clone recursively:

```bash
# Option A: Cloning for the first time
git clone --recurse-submodules [https://github.com/dmitrykhursen/RoadCap-Gen.git](https://github.com/dmitrykhursen/RoadCap-Gen.git)
cd RoadCap-Gen

# Option B: If you already cloned normally
git submodule update --init --recursive

Install project dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
# pip install -e .
```

---

## ▶️ Run the Project

Execute the Python script using hydra config (example for lora training):

```bash
python python scripts/02_finetuning/train.py model=llava dataset=qa_dataset training=lora experiment_name=qa_train_debug
```

## Pipeline Overview (acoording to the visualization below but outdated with the code)

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