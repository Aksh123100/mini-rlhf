# Mini RLHF Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-TRL-yellow)
![Model](https://img.shields.io/badge/Model-GPT--2-green)
![Dataset](https://img.shields.io/badge/Dataset-Anthropic_hh--rlhf-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A minimal end-to-end implementation of **Reinforcement Learning from Human Feedback (RLHF)** using GPT-2 and the Anthropic hh-rlhf preference dataset. Built to develop practical intuition for post-training alignment techniques used in production LLMs.

Covers the full pipeline: **SFT → Reward Modeling → PPO** — including a real observed instance of **reward hacking**.

---

## What is RLHF?

Modern LLMs go through three training phases:

| Phase | Description |
|---|---|
| **Pre-training** | Learn language from massive unsupervised text |
| **Supervised Fine-Tuning (SFT)** | Learn to follow instructions from high-quality examples |
| **RLHF** | Learn to produce outputs humans prefer, using a reward model + RL |

In RLHF, instead of explicit labels, the model learns from comparisons — *"which of these two responses is better?"* A reward model is trained on these preferences, and then **PPO (Proximal Policy Optimization)** is used to push the language model toward outputs that score higher under the reward model.

This repo implements all three stages at a small scale using GPT-2, a subset of the hh-rlhf dataset, and small batch sizes — **runnable on Google Colab or modest hardware**.

---

## Pipeline Overview

```
GPT-2 (base)
    │
    ▼
1_sft.py          ──►  ./sft_model        (Supervised Fine-Tuning on chosen responses)
    │
    ▼
2_reward_model.py ──►  ./reward_model     (Trains scalar reward head with Bradley-Terry loss)
    │
    ▼
3_ppo.py          ──►  ./ppo_model        (PPO optimization against the reward model)
    │
    ▼
4_eval.py         ──►  Comparison table   (GPT-2 vs SFT vs PPO scored by reward model)
```

---

## Results

Evaluated on 10 prompts from the hh-rlhf test split.

| Model | Mean Reward Score |
|---|---|
| GPT-2 (base) | 1.014 |
| SFT | 0.511 |
| PPO | 0.524 |

> ⚠️ **Reward Hacking Observed:** The PPO model collapsed to generating URLs (e.g. `https://www.reddit.com/...`) instead of actual responses — discovering that certain token patterns score high under the reward model despite being meaningless. This is a classic RLHF failure mode and demonstrates exactly why **KL penalties and robust reward model design** are critical in production alignment pipelines.

### Example Outputs

**Prompt:** *"What are some pranks with a pen I can do?"*

| Model | Response |
|---|---|
| GPT-2 | *"The most common joke is that you can't get a pen..."* (repetitive loop) |
| SFT | (mostly incoherent) |
| PPO | `https://www.reddit.com/r/PokemonGo/...` ← **reward hacking** |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Aksh123100/mini-rlhf
cd mini-rlhf

# Install dependencies (pinned for compatibility)
pip install torch transformers==4.40.0 trl==0.8.6 datasets accelerate==0.29.0 peft==0.10.0 wandb tokenizers==0.19.0
```

> Optional: run `wandb login` for online logging. If skipped, logs are saved locally under `./wandb/`.

---

## Running the Pipeline

Run scripts in order — each step depends on artifacts from the previous one.

### Step 1 — Supervised Fine-Tuning
```bash
python 1_sft.py
```
Fine-tunes GPT-2 on chosen responses from the hh-rlhf dataset using TRL's `SFTTrainer`. Saves model to `./sft_model`.

### Step 2 — Reward Model Training
```bash
python 2_reward_model.py
```
Loads `./sft_model`, adds a linear scalar head, and trains with pairwise **Bradley-Terry loss** — chosen responses should score higher than rejected ones. Saves to `./reward_model`.

### Step 3 — PPO Training
```bash
python 3_ppo.py
```
Loads `./sft_model` as the policy, a frozen copy as the reference model, and `./reward_model` as the reward function. Runs PPO for 100 steps with **KL-divergence penalty** to prevent policy drift. Saves to `./ppo_model`.

### Step 4 — Evaluation
```bash
python 4_eval.py
```
Compares GPT-2 base, SFT, and PPO on 10 prompts from the hh-rlhf test split. Scores each response using the reward model and prints a comparison table.

---

## Key Concepts Implemented

| Concept | File |
|---|---|
| Supervised Fine-Tuning (SFT) | `1_sft.py` |
| Bradley-Terry preference loss | `2_reward_model.py` |
| PPO with KL penalty | `3_ppo.py` |
| Reward hacking (observed empirically) | `3_ppo.py` output |
| Automated reward-based evaluation | `4_eval.py` |

---

## Limitations

- GPT-2 is too small for meaningful alignment — this is a learning exercise, not a production system
- 100 PPO steps is far too few; real RLHF runs thousands to millions of steps
- Reward model quality directly caps PPO quality — a weak reward model leads to reward hacking
- The hh-rlhf subset used is small, limiting SFT and reward model generalization

---

## Next Steps

- [ ] DPO (Direct Preference Optimization) — simpler alternative to PPO, no reward model needed
- [ ] Try `gpt2-medium` for better capacity
- [ ] KL coefficient tuning to reduce reward hacking
- [ ] Evaluate with a separate judge model instead of the trained reward model

---

## References

- [Anthropic hh-rlhf dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [TRL library](https://github.com/huggingface/trl)
- [Learning to summarize from human feedback (OpenAI)](https://arxiv.org/abs/2009.01325)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
