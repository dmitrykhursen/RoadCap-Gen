# RoadCap-Gen Source Library

This directory contains the core Python logic for the RoadCap-Gen project. It is structured as a package to be importable by the scripts in `../scripts/`.

## 📂 Package Structure

```text
src/
├── __init__.py                # Makes 'src' a globally importable package
│
├── utils/                     # SHARED UTILITIES
│   ├── __init__.py
│   ├── io.py                  # (TODO) Helpers for loading JSONs, Images, Videos
│   └── geometry.py            # (TODO) Helpers for parsing 3D coords & depth maps
│
├── data/                      # DATA PIPELINE (QA & CAPTIONS)
│   ├── __init__.py
│   ├── dataset.py             # RoadCapDataset: Unified loader for QA and Caption tasks
│   ├── processor.py           # PromptFormatter: Handles model-specific templates (e.g., "USER: ...")
│   └── collator.py            # DataCollator: Handles padding and batch construction
│
├── models/                    # MODEL ARCHITECTURES
│   ├── __init__.py
│   ├── builder.py             # Factory Pattern: `build_model(cfg)` initializes the correct wrapper
│   ├── losses.py              # GeometricAuxLoss: Custom loss for Extended Mode
│   │
│   # --- Model Wrappers (Adapters) ---
│   # These standardize different VLM architectures into a common interface
│   ├── base_wrapper.py        # Abstract base class (optional)
│   ├── llava_module.py        # Wrapper for LLaVA 1.6 + Geo Projection Head
│   ├── internvl_module.py     # Wrapper for Mini-InternVL + Geo Projection Head
│   └── llama_adapter_module.py# Wrapper for LLaMA-Adapter + Geo Projection Head
│
└── training/                  # TRAINING ENGINE
    ├── __init__.py
    ├── trainer.py             # RoadCapTrainer: Custom HF Trainer implementing the 'Extended Mode' loop
    └── peft_utils.py          # LoRA/QLoRA Manager: Applies adapters and 4-bit quantization
