import json
import os
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_spider_data(path: str = "data/spider_formatted.json") -> list:
    with open(path) as f:
        return json.load(f)


def execute_sql(query: str, db_id: str, db_dir: str = "data/spider/database") -> list:
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        result = sorted(cursor.fetchall())
    except Exception:
        result = []
    finally:
        conn.close()
    return result


def run_baseline(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", max_samples: int = 50):
    seed_everything(42)

    print(f"Carregando modelo base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    data = load_spider_data()
    results = []

    for i, entry in enumerate(data[:max_samples]):
        messages = entry["messages"]
        expected_sql = entry["expected_sql"]
        db_id = entry["db_id"]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )

        generated_sql = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted_result = execute_sql(generated_sql, db_id)
        expected_result = execute_sql(expected_sql, db_id)
        correct = predicted_result == expected_result

        results.append({
            "question": messages[-1]["content"],
            "generated_sql": generated_sql,
            "expected_sql": expected_sql,
            "correct": correct,
            "db_id": db_id,
        })

        print(f"[{i+1}/{max_samples}] {'OK' if correct else 'FALHOU'} | db={db_id}")

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    print(f"\nAcurácia baseline (zero-shot): {accuracy:.2%}")

    output_path = "data/baseline_results.json"
    with open(output_path, "w") as f:
        json.dump({"accuracy": accuracy, "details": results}, f, indent=2)
    print(f"Resultados salvos em {output_path}")

    return accuracy


if __name__ == "__main__":
    run_baseline()
