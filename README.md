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

Swept `[5e-5, 1e-4, 2e-4, 5e-4, 1e-3]` over 50 gradient steps each (see `lr_sweep.ipynb`). **2e-4 produced the lowest training loss** and was used for the full run.

Observed pattern: rates ≥ 5e-4 diverged within 50 steps on 512-token sequences; rates ≤ 1e-4 had not converged by step 50 and would require more steps to benefit from. 2e-4 is consistent with standard QLoRA recommendations for instruction fine-tuning on 7B-class models.

## What fine-tuning changes (and what it doesn't)

Token F1 without RAG is slightly *lower* after fine-tuning (0.109 vs 0.116 baseline). This is expected: the adapter shortens answers and removes verbose context the base model sometimes produced. Shorter predictions contain fewer incidentally matching tokens.

With RAG, Token F1 reaches 0.263 — a +141% improvement over fine-tune alone. The adapter's real contribution is answer formatting and German QA instruction following; the retrieved passage supplies the facts.

See [`german-qa-rag/evaluate/evaluate.ipynb`](../german-qa-rag/evaluate/evaluate.ipynb) for the full generation evaluation.

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
