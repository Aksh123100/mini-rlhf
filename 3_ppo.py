import os
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm.auto import tqdm
import wandb

from transformers import AutoModelForSequenceClassification


@dataclass
class PPOPipelineConfig:
    sft_model_dir: str = "./sft_model"
    reward_model_dir: str = "./reward_model"
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train[:1000]"  # small subset for speed
    max_prompt_length: int = 64
    max_generation_length: int = 64
    batch_size: int = 4
    total_ppo_steps: int = 100
    ppo_learning_rate: float = 1.41e-5
    ppo_batch_size: int = 4
    ppo_mini_batch_size: int = 4
    kl_coeff: float = 0.1
    wandb_project: str = "mini-rlhf"
    wandb_run_name: str = "ppo"


def prepare_tokenizer(config: PPOPipelineConfig):
    """Load tokenizer and ensure padding token is set."""
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_ppo_trainer(config: PPOPipelineConfig, tokenizer):
    """
    Build PPOTrainer with policy and reference models based on the SFT model.
    """
    ppo_config = PPOConfig(
        model_name=config.sft_model_dir,
        learning_rate=config.ppo_learning_rate,
        batch_size=config.ppo_batch_size,
        mini_batch_size=config.ppo_mini_batch_size,
        log_with="wandb",
        project_name=config.wandb_project,
    )

    # Policy and reference models both start from the SFT model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.sft_model_dir)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.sft_model_dir)

    # Build PPOTrainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=None,  # we will feed queries manually
    )

    return ppo_trainer


def prepare_prompts(config: PPOPipelineConfig) -> List[str]:
    """
    Build a small list of prompts from the hh-rlhf dataset.
    Since hh-rlhf only has 'chosen' / 'rejected' strings, we
    derive short prompts from the beginning of the chosen text.
    """
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    prompts: List[str] = []
    for row in dataset:
        text = row["chosen"]
        # Use a short prefix of the chosen response as a pseudo-prompt
        prefix = text[:200].replace("\n", " ")
        prompt = f"Human: {prefix}\nAssistant:"
        prompts.append(prompt)
    return prompts


def load_reward_model(config: PPOPipelineConfig, device: torch.device):
    """Load the trained reward model from disk."""
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_dir)
    reward_model.to(device)
    reward_model.eval()
    return reward_model


def compute_reward(
    reward_model: AutoModelForSequenceClassification,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute scalar rewards for (prompt, response) pairs using the reward model.
    We simply concatenate prompt and response and score the full text.
    """
    texts = [p + " " + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length < 512 else 512,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = reward_model(**enc)
        scores = outputs.logits.squeeze(-1)
    return scores


def main():
    config = PPOPipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=vars(config),
    )

    # Prepare tokenizer and PPO trainer
    tokenizer = prepare_tokenizer(config)
    ppo_trainer = build_ppo_trainer(config, tokenizer)
    ppo_trainer.model.to(device)
    ppo_trainer.ref_model.to(device)

    # Load reward model
    reward_model = load_reward_model(config, device)

    # Prepare prompts
    prompts = prepare_prompts(config)

    # Main PPO loop
    step_bar = tqdm(range(config.total_ppo_steps), desc="PPO training", unit="step")

    for step in step_bar:
        # Sample a small batch of prompts
        batch_prompts = []
        for _ in range(config.batch_size):
            idx = (step * config.batch_size + _) % len(prompts)
            batch_prompts.append(prompts[idx])

        # Tokenize prompts
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
            return_tensors="pt",
        )
        tokenized_prompts = {k: v.to(device) for k, v in tokenized_prompts.items()}

        # Generate responses from the policy model
        # We use the built-in generate from PPOTrainer's model
        with torch.no_grad():
            response_ids = ppo_trainer.model.generate(
                **tokenized_prompts,
                max_new_tokens=config.max_generation_length,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Extract only the generated continuation tokens for PPOTrainer
        query_tensors = tokenized_prompts["input_ids"]
        response_tensors = []
        responses_text: List[str] = []

        for i in range(len(batch_prompts)):
            query_len = query_tensors[i].shape[0]
            full = response_ids[i]
            # Take only the new tokens generated after the prompt
            gen = full[query_len:]
            if gen.numel() == 0:
                # Ensure at least one token
                gen = full[-1:].clone()
            response_tensors.append(gen)
            responses_text.append(tokenizer.decode(gen, skip_special_tokens=True))

        # Compute rewards using the reward model
        rewards = compute_reward(
            reward_model=reward_model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            responses=responses_text,
            device=device,
        )

        # PPOTrainer expects a list of tensors for rewards
        reward_tensors = [r for r in rewards]

        # Run a PPO optimization step
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        # Log key statistics including approximate KL and reward
        mean_reward = rewards.mean().item()
        kl = stats.get("ppo/kl", 0.0)
        wandb.log(
            {
                "ppo/step": step,
                "ppo/reward_mean": mean_reward,
                "ppo/kl": kl,
            }
        )

        step_bar.set_postfix({"reward": f"{mean_reward:.3f}", "kl": f"{kl:.3f}"})

    # Save final PPO model
    output_dir = "./ppo_model"
    os.makedirs(output_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()

