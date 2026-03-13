from dataclasses import dataclass
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from tqdm.auto import tqdm
import wandb


@dataclass
class EvalConfig:
    base_model_name: str = "gpt2"
    sft_model_dir: str = "./sft_model"
    ppo_model_dir: str = "./ppo_model"
    reward_model_dir: str = "./reward_model"
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_split: str = "test[:10]"
    max_prompt_length: int = 64
    max_generation_length: int = 64
    wandb_project: str = "mini-rlhf"
    wandb_run_name: str = "eval"


def prepare_prompts(config: EvalConfig) -> List[str]:
    """
    Build prompts from the hh-rlhf test split.
    As in PPO, we derive short prompts from the beginning of the chosen text.
    """
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    prompts: List[str] = []
    for row in dataset:
        text = row["chosen"]
        prefix = text[:200].replace("\n", " ")
        prompt = f"Human: {prefix}\nAssistant:"
        prompts.append(prompt)
    return prompts


def load_models_and_tokenizers(config: EvalConfig, device: torch.device):
    """Load GPT-2 base, SFT, PPO, and reward models plus a shared tokenizer."""
    # Base GPT-2
    base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # Use the SFT tokenizer as the canonical tokenizer for all models
    sft_tokenizer = AutoTokenizer.from_pretrained(config.sft_model_dir)
    if sft_tokenizer.pad_token is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    sft_tokenizer.padding_side = "right"

    # Models
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name).to(device)
    sft_model = AutoModelForCausalLM.from_pretrained(config.sft_model_dir).to(device)
    ppo_model = AutoModelForCausalLM.from_pretrained(config.ppo_model_dir).to(device)

    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_dir).to(device)
    reward_model.eval()

    return sft_tokenizer, base_model, sft_model, ppo_model, reward_model


def generate_responses(
    tokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    max_prompt_length: int,
    max_generation_length: int,
    device: torch.device,
) -> List[str]:
    """Generate responses for a list of prompts from a given model."""
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_generation_length,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses: List[str] = []
    input_ids = enc["input_ids"]
    for i in range(len(prompts)):
        prompt_len = input_ids[i].shape[0]
        full = outputs[i]
        gen = full[prompt_len:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        responses.append(text)

    return responses


def score_responses(
    tokenizer,
    reward_model: AutoModelForSequenceClassification,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Score responses with the reward model by concatenating prompt and response."""
    texts = [p + " " + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = reward_model(**enc)
        scores = outputs.logits.squeeze(-1)

    return scores.cpu()


def print_comparison_table(
    prompts: List[str],
    base_scores: torch.Tensor,
    sft_scores: torch.Tensor,
    ppo_scores: torch.Tensor,
):
    """Print a simple comparison table of reward scores."""
    header = f"{'Idx':<4} | {'Prompt (truncated)':<40} | {'GPT2 Score':>10} | {'SFT Score':>10} | {'PPO Score':>10}"
    print(header)
    print("-" * len(header))

    for i, prompt in enumerate(prompts):
        truncated = (prompt[:37] + "...") if len(prompt) > 40 else prompt
        print(
            f"{i:<4} | {truncated:<40} | {base_scores[i]:>10.4f} | {sft_scores[i]:>10.4f} | {ppo_scores[i]:>10.4f}"
        )


def print_example_pairs(
    prompts: List[str],
    base_responses: List[str],
    sft_responses: List[str],
    ppo_responses: List[str],
    num_examples: int = 3,
):
    """Print a few example prompt/response triplets side by side."""
    print("\nExample prompt/response comparisons:")
    print("=" * 80)
    for i in range(min(num_examples, len(prompts))):
        print(f"\n--- Example {i} ---")
        print(f"Prompt:\n{prompts[i]}\n")
        print(f"GPT-2 response:\n{base_responses[i]}\n")
        print(f"SFT response:\n{sft_responses[i]}\n")
        print(f"PPO response:\n{ppo_responses[i]}\n")
        print("=" * 80)


def main():
    config = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=vars(config),
    )

    # Prepare prompts
    prompts = prepare_prompts(config)

    # Load models and tokenizer
    tokenizer, base_model, sft_model, ppo_model, reward_model = load_models_and_tokenizers(config, device)

    # Generate responses for all models
    print("Generating responses from GPT-2, SFT, and PPO models...")
    base_responses = generate_responses(
        tokenizer, base_model, prompts, config.max_prompt_length, config.max_generation_length, device
    )
    sft_responses = generate_responses(
        tokenizer, sft_model, prompts, config.max_prompt_length, config.max_generation_length, device
    )
    ppo_responses = generate_responses(
        tokenizer, ppo_model, prompts, config.max_prompt_length, config.max_generation_length, device
    )

    # Score all responses with the reward model
    print("Scoring responses with reward model...")
    base_scores = score_responses(tokenizer, reward_model, prompts, base_responses, device)
    sft_scores = score_responses(tokenizer, reward_model, prompts, sft_responses, device)
    ppo_scores = score_responses(tokenizer, reward_model, prompts, ppo_responses, device)

    # Print comparison table
    print("\nReward score comparison:")
    print_comparison_table(prompts, base_scores, sft_scores, ppo_scores)

    # Print a few example prompt/response triplets
    print_example_pairs(prompts, base_responses, sft_responses, ppo_responses, num_examples=3)

    # Log aggregate statistics to wandb
    wandb.log(
        {
            "eval/base_mean_reward": base_scores.mean().item(),
            "eval/sft_mean_reward": sft_scores.mean().item(),
            "eval/ppo_mean_reward": ppo_scores.mean().item(),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()

