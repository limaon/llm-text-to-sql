import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path: str = "data/spider_formatted.json") -> list:
    with open(path) as f:
        return json.load(f)


def format_for_training(entry: dict) -> dict:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    text = tokenizer.apply_chat_template(
        entry["messages"], tokenize=False, add_generation_prompt=True
    )
    return {"text": text}


def finetune_lora(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "models/lora_spider",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
):
    seed_everything(42)

    print(f"Carregando modelo base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data = load_data()
    texts = []
    for entry in data:
        text = tokenizer.apply_chat_template(
            entry["messages"], tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    dataset = Dataset.from_dict({"text": texts})

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    print("Iniciando fine-tuning...")
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapters salvos em {output_dir}")


if __name__ == "__main__":
    finetune_lora()
