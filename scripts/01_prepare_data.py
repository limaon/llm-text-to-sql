import json
import os
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_schema(db_id: str, db_dir: str = "data/spider/database") -> str:
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    schema_parts = []
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_defs = ", ".join(
            f"{col[1]} {col[2]}" for col in columns
        )
        schema_parts.append(f"CREATE TABLE {table_name} ({col_defs})")

    conn.close()
    return "\n".join(schema_parts)


def format_spider_entry(entry: dict, db_dir: str) -> dict:
    schema = format_schema(entry["db_id"], db_dir)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert SQL assistant. Generate a valid SQL query "
                "based on the database schema and the user's question."
            ),
        },
        {
            "role": "user",
            "content": f"Database Schema:\n{schema}\n\nQuestion: {entry['question']}",
        },
    ]

    return {
        "messages": messages,
        "expected_sql": entry["query"],
        "db_id": entry["db_id"],
    }


def get_mmlu_subset(subject: str, num_samples: int = 50) -> list:
    dataset = load_dataset("cais/mmlu", subject, split="test")
    return dataset.shuffle(seed=42).select(range(num_samples))


def prepare_spider(output_path: str = "data/spider_formatted.json") -> None:
    spider_train = load_dataset("spider", split="train")

    formatted = []
    for entry in spider_train:
        formatted.append(format_spider_entry(entry, "data/spider/database"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2)

    print(f"Spider formatado salvo em {output_path} ({len(formatted)} exemplos)")


def prepare_mmlu(output_path: str = "data/mmlu_150.json") -> None:
    stem_data = get_mmlu_subset("computer_science")
    humanities_data = get_mmlu_subset("philosophy")
    social_data = get_mmlu_subset("economics")

    all_samples = []

    for item in stem_data:
        all_samples.append({
            "category": "STEM",
            "subject": "computer_science",
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
        })

    for item in humanities_data:
        all_samples.append({
            "category": "Humanities",
            "subject": "philosophy",
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
        })

    for item in social_data:
        all_samples.append({
            "category": "Social Sciences",
            "subject": "economics",
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
        })

    random.seed(42)
    random.shuffle(all_samples)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"MMLU 150 salvo em {output_path} ({len(all_samples)} questões)")


if __name__ == "__main__":
    seed_everything(42)
    prepare_spider()
    prepare_mmlu()
