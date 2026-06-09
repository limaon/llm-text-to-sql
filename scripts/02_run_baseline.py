import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from custom_metrics import ExecutionAccuracyMetric


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def build_few_shot_prompt(train_data: list, question_messages: list, n_shots: int = 3) -> list:
    system_msg = question_messages[0]
    user_msg = question_messages[1]

    few_shot_examples = train_data[:n_shots]
    few_shot_messages = []
    for ex in few_shot_examples:
        few_shot_messages.append({"role": "user", "content": ex["messages"][1]["content"]})
        few_shot_messages.append({"role": "assistant", "content": ex["messages"][2]["content"]})

    return [system_msg] + few_shot_messages + [user_msg]


def run_baseline(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    train_path: str = "data/spider_train_formatted.json",
    dev_path: str = "data/spider_dev_formatted.json",
    output_path: str = "data/baseline_results.json",
    max_samples: int = 50,
):
    seed_everything(42)

    print(f"Carregando modelo base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    train_data = load_json(train_path)
    dev_data = load_json(dev_path)

    metric = ExecutionAccuracyMetric()
    test_cases = []

    print(f"Gerando queries para {min(max_samples, len(dev_data))} exemplos do dev set...")
    generated_outputs = []

    for i, entry in enumerate(dev_data[:max_samples]):
        messages = build_few_shot_prompt(train_data, entry["messages"])

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_sql = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        generated_outputs.append(generated_sql)

        test_case = LLMTestCase(
            input=messages[-1]["content"],
            actual_output=generated_sql,
            expected_output=entry["expected_sql"],
            additional_metadata={"db_id": entry["db_id"]},
        )
        test_cases.append(test_case)

        print(f"[{i+1}/{max_samples}] SQL gerado | db={entry['db_id']}")

    print(f"\nAvaliando com Execution Accuracy (DeepEval)...")
    eval_results = evaluate(test_cases, [metric])

    results = []
    for i, entry in enumerate(dev_data[:max_samples]):
        results.append({
            "question": entry["messages"][1]["content"],
            "generated_sql": generated_outputs[i],
            "expected_sql": entry["expected_sql"],
            "db_id": entry["db_id"],
        })

    accuracy = eval_results.overall_score if hasattr(eval_results, "overall_score") else 0.0

    with open(output_path, "w") as f:
        json.dump({"accuracy": accuracy, "details": results}, f, indent=2)

    print(f"\nAcurácia baseline (few-shot): {accuracy:.2%}")
    print(f"Resultados salvos em {output_path}")

    return accuracy


if __name__ == "__main__":
    run_baseline()
