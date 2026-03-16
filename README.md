

Mini RLHF Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aksh123100/mini-rlhf/blob/copilot/run-repo-in-colab/mini_rlhf_colab.ipynb)

This project implements a **minimal end‑to‑end RLHF (Reinforcement Learning from Human Feedback) pipeline** using GPT‑2 and the Anthropic `hh-rlhf` preference dataset. RLHF is a training paradigm where we start from a supervised model, learn a reward model from human preferences, and then optimize the policy with reinforcement learning so that it produces outputs that humans prefer. Instead of directly learning from explicit labels, the model learns from *comparisons* between outputs (for example, "chosen" vs "rejected" responses) and is pushed towards behaviors that score higher under a learned reward function.

In practice, modern large language models are usually first pre‑trained on large unsupervised corpora, then **supervised fine‑tuned (SFT)** on high‑quality instruction‑following examples, and finally **fine‑tuned with RLHF**. The RLHF phase uses a reward model trained on human preference data and an RL algorithm such as PPO to adjust the base model so that it aligns better with human values and expectations. This repository demonstrates this flow on a much smaller scale using GPT‑2, a small dataset subset, and very small batch sizes so that it can run on modest hardware.

## Run in Google Colab

The easiest way to run this pipeline is via the provided Colab notebook.
Click the badge below (or at the top of this page) to open it directly in Colab,
then set the runtime to **GPU** and run all cells in order.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aksh123100/mini-rlhf/blob/copilot/run-repo-in-colab/mini_rlhf_colab.ipynb)

The notebook:
1. Clones this repository inside Colab.
2. Installs all required packages.
3. Optionally logs in to Weights & Biases (set `USE_WANDB = True` in the notebook cell to enable).
4. Runs each of the four pipeline steps in order.

## Installation

1. **Create and activate a virtual environment (optional but recommended)**:

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows PowerShell
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Login to Weights & Biases (wandb) if you want online logging**:

```bash
wandb login
```

If you do not log in, wandb will still run in offline mode by default, but you may not see your runs in the web UI.

## Running the Pipeline

Run the scripts **in order**, as each step depends on artifacts produced by the previous ones.

1. **Supervised Fine‑Tuning (SFT)**:

```bash
python 1_sft.py
```

This script fine‑tunes GPT‑2 on the Anthropic `hh-rlhf` dataset using only the `chosen` responses. It saves the resulting model to the `./sft_model` directory.

2. **Reward Model Training**:

```bash
python 2_reward_model.py
```

This script loads the `./sft_model` as a base, adds a linear head that outputs a single scalar score, and trains it as a preference‑based reward model using both `chosen` and `rejected` responses with a pairwise ranking (Bradley–Terry) loss. The trained reward model is saved to `./reward_model`.

3. **PPO Training**:

```bash
python 3_ppo.py
```

This script loads the `./sft_model` as the policy model, a frozen copy of the same model as the reference model, and the `./reward_model` as the reward function. It then runs a short PPO optimization loop (100 steps) where the policy generates responses, they are scored by the reward model, a KL penalty against the reference model is computed, and the policy is updated via PPO. The final policy is saved to `./ppo_model`.

4. **Evaluation**:

```bash
python 4_eval.py
```

This script compares three models:

- The original GPT‑2 base model
- The supervised fine‑tuned model (`./sft_model`)
- The PPO‑optimized model (`./ppo_model`)

It takes 10 sample prompts derived from the `hh-rlhf` test split, generates responses from each model, scores them using the reward model from `./reward_model`, and prints a comparison table to the console.

## What Each Script Does

- **`1_sft.py`**: Performs supervised fine‑tuning of GPT‑2 on the `chosen` responses from the `hh-rlhf` dataset using TRL's `SFTTrainer`. Handles tokenization, padding, truncation, and logging training loss to wandb.
- **`2_reward_model.py`**: Builds a scalar reward model on top of the SFT model by adding a linear classification head and training it with a pairwise Bradley–Terry loss so that `chosen` responses receive higher scores than `rejected` responses.
- **`3_ppo.py`**: Uses TRL's `PPOTrainer` to optimize the SFT policy against the learned reward model while regularizing it via KL‑divergence to a frozen reference copy of the SFT model.
- **`4_eval.py`**: Evaluates and compares GPT‑2, the SFT model, and the PPO‑tuned model on a small set of prompts using the reward model as an automatic evaluator, and logs summary statistics to wandb.

## Results

Fill in after running eval.

