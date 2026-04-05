# Mechanistic Interpretability — Tarun Arora

Hands-on experiments in mech interp. Running real models, 
finding real things.

## Findings

### Set-Completion Heads in Qwen1.5-1.8B
Layer 12 attention is redundant on clean runs (7.4% ablation damage) 
but causally necessary on corrupted runs (99.1% collapse). 
Heads 4 and 9 implement a structure/content dissociation — 
Head 9 tracks counting skeleton, Head 4 tracks fruit identity.
→ Full writeup: activation_patching.md

## Notebooks
- 0_Zero_Layer_Transformer.ipynb — transformer built from scratch
- Attention_Mechanism.ipynb — single and multi-head attention
- MLP.ipynb — MLP block with GELU and residual
- Activation_Patching_.ipynb — full patching pipeline

## Contact
hi@tarunarora.de
