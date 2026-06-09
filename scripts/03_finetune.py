import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path: str = "data/spider_train_formatted.json") -> list:
    with open(path) as f:
        return json.load(f)


def finetune_lora(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "models/lora_spider_run1",
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 512,
    run_name: str = "Run 1",
):
    seed_everything(42)

    print(f"=== {run_name} ===")
    print(f"Modelo: {model_name}")
    print(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"LR: {learning_rate} | Epochs: {num_epochs} | Batch: {batch_size}")

    print(f"Carregando modelo base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data = load_data()
    texts = []
    for entry in data:
        text = tokenizer.apply_chat_template(
            entry["messages"], tokenize=False, add_generation_prompt=False
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
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    print("Iniciando fine-tuning...")
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapters salvos em {output_dir}")

    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    seed_everything(42)

    print("=" * 60)
    print("RODADA 1: lr=2e-4, r=16, alpha=32, epochs=1")
    print("=" * 60)
    finetune_lora(
        output_dir="models/lora_spider_run1",
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32,
        num_epochs=1,
        batch_size=4,
        run_name="Run 1 (lr=2e-4, r=16, ep=1)",
    )

    print("\n" + "=" * 60)
    print("RODADA 2: lr=1e-4, r=8, alpha=16, epochs=2")
    print("=" * 60)
    finetune_lora(
        output_dir="models/lora_spider_run2",
        learning_rate=1e-4,
        lora_r=8,
        lora_alpha=16,
        num_epochs=2,
        batch_size=4,
        run_name="Run 2 (lr=1e-4, r=8, ep=2)",
    )

    print("\nAmbas as rodadas concluídas!")
