#!!!FINE-TUNED GEMMA 2 2B CAN BE DOWNLOADED HERE: https://drive.google.com/drive/folders/1AoQbqucjrJWkpEXuA87viH8rWKXcrOwh?usp=drive_link


# Medical Data Extraction with Fine-Tuned LLMs

## 📋 Overview

This project demonstrates how to fine-tune small Large Language Models (LLMs) for the task of medical data extraction from unstructured text in Bulgarian. The focus is on overcoming linguistic and technical challenges specific to this domain and language, with training and evaluation done in a resource-constrained, local environment.

## 🎯 Objectives

- Fine-tune open-source LLMs on a domain-specific dataset of medical records written in Bulgarian.
- Evaluate performance based on extraction accuracy, using a custom metric.
- Explore translation pipelines with DeepL for optional multilingual preprocessing.
- Run all training and inference locally using CUDA-compatible GPUs.

## 🧰 Key Components

### 1. Data Import & Preparation
- Loads structured and unstructured data from local sources.
- Applies text preprocessing and formatting for model inputs.

### 2. Model Setup
- Utilizes models like `llama-2-2b-it` with PEFT (Parameter-Efficient Fine-Tuning) using LoRA.
- Applies quantization to reduce memory usage during fine-tuning.
- Environment configuration via PyTorch and Hugging Face Transformers.

### 3. Training Pipeline
- Uses the Hugging Face Trainer API for training small LLMs.
- Incorporates JSON format for input/output during prompt-response setup.
- Designed to run within 16GB VRAM constraints.

### 4. Evaluation
- Custom metric for evaluating performance (e.g., correct duration of diabetes extraction).
- Evaluates accuracy of extracted structured data against ground truth.

### 5. Translation Pipeline
- Integrates DeepL API for translation between Bulgarian and English.

## 🖥️ Local Setup

### Requirements

- Python 3.10+
- CUDA-enabled GPU
- Install dependencies:
  ```bash
  pip install torch transformers peft accelerate bitsandbytes deepl pandas numpy
  ```

### Running the Notebook

Run the notebook `MedicalExtraction.ipynb` step-by-step in a Jupyter environment. Ensure your GPU is available and CUDA is set up correctly.

## 📊 Example Use Case

Given this input text:
```
Пациентът е диагностициран с диабет преди 7 години.
```
The model is expected to extract:
```json
{
  "diabetes_duration_years": 7
}
```

## 🚧 Known Limitations

- Fine-tuning larger models (>7B) locally may exceed GPU memory.
- Translation introduces variability and may affect accuracy.
- Evaluation is currently based on a single custom metric.

## 📌 Acknowledgements

- [Google's Gemma models](https://ai.google.dev/gemma)
- Hugging Face Transformers and PEFT
- DeepL API for translation support
