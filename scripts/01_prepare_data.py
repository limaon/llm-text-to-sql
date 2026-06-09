import json
import os
import random
import shutil
import sqlite3
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def download_spider_databases(target_dir: str = "data/spider") -> None:
    """Baixa os bancos de dados oficiais do Spider do Google Drive."""
    db_dir = os.path.join(target_dir, "database")
    if os.path.exists(db_dir):
        return

    print("Baixando bancos de dados do Spider do Google Drive...")
    os.makedirs(target_dir, exist_ok=True)

    subprocess.run(["pip", "install", "-q", "gdown"], check=True)

    gdown_id = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
    zip_path = "data/spider.zip"
    subprocess.run(["gdown", gdown_id, "-O", zip_path], check=True)

    print("Extraindo arquivos...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    for root, dirs, files in os.walk(target_dir):
        if "database" in dirs:
            src = os.path.join(root, "database")
            if src != db_dir:
                if os.path.exists(db_dir):
                    shutil.rmtree(db_dir)
                shutil.move(src, db_dir)
            break

    os.remove(zip_path)

    if not os.path.exists(db_dir):
        print("AVISO: pasta 'database' não encontrada no zip. Conteúdo extraído:")
        for item in os.listdir(target_dir):
            print(f"  - {item}")
        raise FileNotFoundError("Pasta 'database' não encontrada após extração")

    print("Bancos de dados baixados com sucesso!")


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
        col_defs = ", ".join(f"{col[1]} {col[2]}" for col in columns)
        schema_parts.append(f"CREATE TABLE {table_name} ({col_defs})")

    conn.close()
    return "\n".join(schema_parts)


def format_spider_entry(entry: dict, db_dir: str, include_assistant: bool = True) -> dict:
    schema = format_schema(entry["db_id"], db_dir)

    messages = [
        {
            "role": "system",
            "content": "You are an expert SQL assistant. Generate a valid SQL query based on the database schema and the user's question."
        },
        {
            "role": "user",
            "content": f"Database Schema:\n{schema}\n\nQuestion: {entry['question']}"
        }
    ]

    # Para treino, o assistente DEVE ter a resposta dentro das mensagens.
    if include_assistant:
        messages.append({
            "role": "assistant",
            "content": entry["query"]
        })

    return {
        "messages": messages,
        "db_id": entry["db_id"],
        "expected_sql": entry["query"] # Mantido para facilitar a avaliação na Fase 2 e 4
    }


def get_mmlu_subset(subject: str, num_samples: int = 50) -> list:
    dataset = load_dataset("cais/mmlu", subject, split="test")
    return list(dataset.shuffle(seed=42).select(range(num_samples)))


def prepare_spider(db_dir: str = "data/spider/database") -> None:
    spider_train = load_dataset("spider", split="train")
    spider_dev = load_dataset("spider", split="validation")

    print("Processando Spider Train...")
    train_formatted = [format_spider_entry(entry, db_dir, include_assistant=True) for entry in spider_train]

    print("Processando Spider Dev...")
    # No dev (avaliação), o modelo precisa prever a resposta, mas mantemos o target no JSON para a métrica comparar depois
    dev_formatted = [format_spider_entry(entry, db_dir, include_assistant=False) for entry in spider_dev]

    with open("data/spider_train_formatted.json", "w") as f:
        json.dump(train_formatted, f, indent=2)

    with open("data/spider_dev_formatted.json", "w") as f:
        json.dump(dev_formatted, f, indent=2)

    print(f"Spider salvo! Treino: {len(train_formatted)} | Dev: {len(dev_formatted)} exemplos.")


def prepare_mmlu(output_path: str = "data/mmlu_150.json") -> None:
    print("Processando MMLU...")
    subjects = {
        "STEM": "computer_science",
        "Humanities": "philosophy",
        "Social Sciences": "economics"
    }

    all_samples = []

    for category, subject in subjects.items():
        data = get_mmlu_subset(subject)
        for item in data:
            all_samples.append({
                "category": category,
                "subject": subject,
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
    download_spider_databases()
    prepare_spider()
    prepare_mmlu()
