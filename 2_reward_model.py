import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from tqdm.auto import tqdm
import wandb

USE_WANDB = os.getenv("USE_WANDB", "1") == "1"


@dataclass
class RewardModelConfig:
    base_model_dir: str = "./sft_model"
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train[:5000]"
    output_dir: str = "./reward_model"
    max_length: int = 512
    batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 1e-5
    wandb_project: str = "mini-rlhf"
    wandb_run_name: str = "reward_model"


def bradley_terry_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Bradley-Terry loss:
    P(chosen > rejected) = sigmoid(chosen - rejected)
    Loss = -log(P)
    """
    diff = chosen_scores - rejected_scores
    return -torch.log(torch.sigmoid(diff) + 1e-12).mean()


def prepare_tokenizer_and_model(config: RewardModelConfig):
    """Load tokenizer and turn the SFT model into a sequence classification model with a scalar head."""
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_dir)

    # Ensure we have a padding token (should already be set in SFT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build a sequence classification config on top of the SFT model
    base_config = AutoConfig.from_pretrained(config.base_model_dir)
    base_config.num_labels = 1  # single scalar score

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_dir,
        config=base_config,
    )

    return tokenizer, model


def collate_fn(batch, tokenizer, max_length: int):
    """Tokenize chosen and rejected texts in a batch with padding and truncation."""
    chosen_texts = [item["chosen"] for item in batch]
    rejected_texts = [item["rejected"] for item in batch]

    chosen_enc = tokenizer(
        chosen_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    rejected_enc = tokenizer(
        rejected_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "chosen": chosen_enc,
        "rejected": rejected_enc,
    }


def main():
    config = RewardModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    if USE_WANDB:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Prepare tokenizer and reward model
    tokenizer, model = prepare_tokenizer_and_model(config)
    model.to(device)

    # Load dataset with both chosen and rejected columns
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    # Simple DataLoader with small batch size for limited hardware
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, config.max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    global_step = 0
    model.train()

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Reward model training (epoch {epoch + 1})", unit="batch")

        for batch in progress:
            optimizer.zero_grad()

            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}

            # Forward pass for chosen and rejected
            chosen_outputs = model(**chosen)
            rejected_outputs = model(**rejected)

            # Model returns logits of shape (batch_size, 1)
            chosen_scores = chosen_outputs.logits.squeeze(-1)
            rejected_scores = rejected_outputs.logits.squeeze(-1)

            loss = bradley_terry_loss(chosen_scores, rejected_scores)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            progress.set_postfix({"loss": loss.item()})
            wandb.log({"reward_model/loss": loss.item(), "step": global_step})

        avg_loss = epoch_loss / len(dataloader)
        wandb.log({"reward_model/epoch_loss": avg_loss, "epoch": epoch + 1})

    # Save the trained reward model
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()

