# Fine-Tuning LLM para Text-to-SQL

Projeto de fine-tuning de um LLM para geração de queries SQL a partir de linguagem natural, utilizando o dataset Spider e avaliando com a métrica Execution Accuracy via DeepEval.

## Estrutura

```
├── data/                  # Datasets brutos e processados (gitignored)
├── custom_metrics/        # Métrica customizada do DeepEval
│   ├── __init__.py
│   └── execution_accuracy.py
├── scripts/               # Scripts executáveis
│   ├── 01_prepare_data.py
│   ├── 02_run_baseline.py
│   └── 03_finetune.py
├── requirements.txt       # Dependências fixadas
└── README.md
```

## Setup do trabalho no google colab

### Inicio

1. Abra o notebook no Colab e certifique-se de que a GPU está ativa:

   ```python
   !nvidia-smi
   ```

2. Monte o Google Drive para persistência dos pesos:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Instale as dependências:
   ```bash
   !pip install -r requirements.txt
   ```

### Ambiente local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

### 1. Preparar dados

```bash
python scripts/01_prepare_data.py
```

### 2. Rodar baseline (zero-shot)

```bash
python scripts/02_run_baseline.py
```

### 3. Fine-tuning com LoRA

```bash
python scripts/03_finetune.py
```

## Reprodutibilidade

Todos os scripts fixam sementes via `seed_everything(42)` para garantir reprodutibilidade total.
