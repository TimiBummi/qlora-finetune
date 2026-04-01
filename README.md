# QLoRA Fine-Tuning: Mistral-7B on GermanQuad

Fine-tunes `mistralai/Mistral-7B-Instruct-v0.3` on the [`deepset/germanquad`](https://huggingface.co/datasets/deepset/germanquad) dataset using QLoRA (4-bit quantization + LoRA adapters). Runs on a free Google Colab T4 GPU.

## Notebooks

| Notebook | Description |
|---|---|
| `qlora_finetune.ipynb` | Full training run — 1 epoch on 10k samples, saves LoRA adapter |
| `lr_sweep.ipynb` | Learning rate sweep across 5 values (50 steps each) to find the optimal LR |

## Setup

**Model:** `mistralai/Mistral-7B-Instruct-v0.3`  
**Dataset:** `deepset/germanquad` — German extractive QA (Wikipedia-based)  
**Split:** 80 / 10 / 10 (train / val / test), 10,000 samples total

## QLoRA Configuration

| Parameter | Value |
|---|---|
| Quantization | 4-bit NF4, double quantization |
| Compute dtype | float16 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

## Training

| Parameter | Value |
|---|---|
| Epochs | 1 |
| Batch size | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| Learning rate | 2e-4 |
| Max sequence length | 512 |
| Warmup steps | 10 |

## Learning Rate Sweep

Swept `[5e-5, 1e-4, 2e-4, 5e-4, 1e-3]` over 50 steps each. **2e-4 produced the lowest training loss** and was used for the full run — consistent with standard QLoRA recommendations.

## Prompt Format

```
<s>[INST] Beantworte die folgende Frage auf Deutsch:

{question} [/INST] {answer} </s>
```

## Runtime

Google Colab free tier (T4 GPU). Adapter is saved to Google Drive to survive session disconnects.

## Dependencies

```
transformers peft bitsandbytes accelerate datasets trl
```
