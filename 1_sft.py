import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer
from tqdm.auto import tqdm
USE_WANDB = os.getenv("USE_WANDB", "1") == "1"
import wandb


@dataclass
class SFTConfig:
    model_name: str = "gpt2"
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train[:5000]"
    output_dir: str = "./sft_model"
    max_length: int = 512
    batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 5e-5
    wandb_project: str = "mini-rlhf"
    wandb_run_name: str = "sft"


def prepare_tokenizer_and_model(config: SFTConfig):
    """Load GPT-2 tokenizer and model and handle padding token."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # GPT-2 does not have a pad token by default; set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    # Ensure model embedding size matches tokenizer vocab (in case pad token added)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def load_and_prepare_dataset(config: SFTConfig):
    """
    Load hh-rlhf dataset and keep only the 'chosen' responses.
    SFTTrainer can tokenize internally, so we keep raw text.
    """
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    # The dataset has columns: 'chosen' and 'rejected'.
    # Keep only the chosen responses and rename the column to 'text'
    # so that SFTTrainer can use its default dataset_text_field="text".
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "chosen"])
    dataset = dataset.rename_column("chosen", "text")

    return dataset


def main():
    config = SFTConfig()

    # Initialize wandb for logging
    if USE_WANDB:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
    )

    # Prepare tokenizer and model
    tokenizer, model = prepare_tokenizer_and_model(config)

    # Load dataset
    dataset = load_and_prepare_dataset(config)

    # Define Hugging Face TrainingArguments
    training_args = TrainingArguments(
                report_to=["wandb"] if USE_WANDB else [],
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=["wandb"] if USE_WANDB else [],
        bf16=False,  # keep CPU / simple GPU friendly
        fp16=False,
    )

    # Initialize SFTTrainer from TRL
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Use tqdm to show training progress
    # SFTTrainer wraps HF Trainer, so we call train() directly
    # and rely on its internal progress bar. We still wrap for clarity.
    with tqdm(total=int(config.num_epochs * len(dataset) / config.batch_size), desc="SFT training", unit="step") as pbar:

        # We cannot easily step tqdm per optimizer step from outside,
        # so we hook into the callback using a custom training loop-like wrapper.
        # The simplest reliable option is to call trainer.train() and then
        # set the bar to the total to show completion.
        trainer.train()
        pbar.n = pbar.total
        pbar.refresh()

    # Save the fine-tuned model
    os.makedirs(config.output_dir, exist_ok=True)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()

