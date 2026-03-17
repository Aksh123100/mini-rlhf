# Mini RLHF Pipeline

A minimal end-to-end implementation of **Reinforcement Learning from Human Feedback (RLHF)** using GPT-2 and the Anthropic `hh-rlhf` preference dataset. Built to understand the full post-training pipeline: SFT → Reward Modeling → PPO.

---

## What is RLHF?

Modern LLMs go through three training phases:

1. **Pre-training** — learn language from massive unsupervised text
2. **Supervised Fine-Tuning (SFT)** — learn to follow instructions from high-quality examples
3. **RLHF** — learn to produce outputs that humans prefer, using a reward model + RL

In RLHF, instead of explicit labels, the model learns from **comparisons** — "which of these two responses is better?" A reward model is trained on these preferences, and then PPO (Proximal Policy Optimization) is used to push the language model toward outputs that score higher under the reward model.

This repo implements all three RLHF stages at a small scale using GPT-2, a subset of the `hh-rlhf` dataset, and small batch sizes — runnable on modest hardware or Google Colab.

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

## Installation

```bash
# Clone the repo
git clone https://github.com/Aksh123100/mini-rlhf
cd mini-rlhf

# Install dependencies (pinned for compatibility)
pip install torch transformers==4.40.0 trl==0.8.6 datasets accelerate==0.29.0 peft==0.10.0 wandb tokenizers==0.19.0

# Optional: login to Weights & Biases for online logging
wandb login
```

> If you skip `wandb login`, it runs in offline mode. Logs are saved locally under `./wandb/`.

---

## Running the Pipeline

Run scripts **in order** — each step depends on artifacts from the previous one.

### Step 1 — Supervised Fine-Tuning
```bash
python 1_sft.py
```
Fine-tunes GPT-2 on `chosen` responses from the `hh-rlhf` dataset using TRL's `SFTTrainer`. Saves model to `./sft_model`.

### Step 2 — Reward Model Training
```bash
python 2_reward_model.py
```
Loads `./sft_model`, adds a linear scalar head, and trains it with **pairwise Bradley-Terry loss** — `chosen` responses should score higher than `rejected` ones. Saves to `./reward_model`.

### Step 3 — PPO Training
```bash
python 3_ppo.py
```
Loads `./sft_model` as the policy, a frozen copy as the reference model, and `./reward_model` as the reward function. Runs PPO for 100 steps with KL-divergence penalty to prevent the policy from drifting too far from the reference. Saves to `./ppo_model`.

### Step 4 — Evaluation
```bash
python 4_eval.py
```
Compares GPT-2 base, SFT model, and PPO model on 10 prompts from the `hh-rlhf` test split. Scores each response using the reward model and prints a comparison table.

---

## Results

Evaluated on 10 prompts from the `hh-rlhf` test split.

| Model     | Mean Reward Score |
|-----------|:-----------------:|
| GPT-2     | 1.014             |
| SFT       | 0.511             |
| PPO       | 0.524             |

### Observations

**PPO barely improved over SFT.** This is expected at this scale — GPT-2 (117M params) has limited capacity, and 100 PPO steps is a very short training run.

**SFT scored lower than base GPT-2.** The base GPT-2 reward scores look higher partly because the reward model was trained on SFT-style outputs — the base model's distribution can accidentally hit high-reward token patterns without coherent reasoning.

**PPO exhibited reward hacking.** The most interesting finding: the PPO model started generating URLs (e.g., `https://www.reddit.com/...`, `https://www.icann.org/...`) instead of actual responses. This is a classic RLHF failure mode — the policy discovered that certain token patterns get high reward scores from the reward model, even though they're meaningless as responses. This phenomenon is well-documented at scale too and is part of why KL penalties and careful reward model design matter.

### Example Outputs

**Prompt:** *"What are some pranks with a pen I can do?"*

| Model  | Response |
|--------|----------|
| GPT-2  | "The most common joke is that you can't get a pen. Assistant: I'm not sure if you can get a pen..." *(repetitive loop)* |
| SFT    | *(mostly empty / incoherent)* |
| PPO    | `https://www.reddit.com/r/PokemonGo/...` *(reward hacking — outputs a URL)* |

---

## Key Concepts Implemented

| Concept | Where |
|---------|-------|
| Supervised Fine-Tuning (SFT) | `1_sft.py` |
| Bradley-Terry preference loss | `2_reward_model.py` |
| PPO with KL penalty | `3_ppo.py` |
| Reward hacking (observed) | `3_ppo.py` output |
| Automatic reward-based eval | `4_eval.py` |

---

## Limitations

- **GPT-2 is too small** for meaningful alignment — this is a learning exercise, not a production system
- **100 PPO steps** is far too few; real RLHF runs thousands to millions of steps
- **Reward model quality** directly caps PPO quality — a weak reward model leads to reward hacking
- The `hh-rlhf` subset used is small, so both SFT and reward model generalization is limited

---

## Next Steps

- [ ] DPO (Direct Preference Optimization) — simpler alternative to PPO, no reward model needed
- [ ] Try `gpt2-medium` for better capacity
- [ ] Add KL coefficient tuning to reduce reward hacking
- [ ] Evaluate with a separate judge model instead of the trained reward model

---

## References

- [Anthropic hh-rlhf dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [TRL library](https://github.com/huggingface/trl)
- [Learning to summarize from human feedback (OpenAI)](https://arxiv.org/abs/2009.01325)
- [PPO paper](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
