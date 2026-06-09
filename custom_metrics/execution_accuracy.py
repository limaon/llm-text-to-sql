import re
import sqlite3
from collections import Counter
from typing import List, Tuple

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ExecutionAccuracyMetric(BaseMetric):
    """Metrica customizada que mede acuracia de execucao de SQL.

    Compara o resultado da execucao do SQL gerado pelo modelo com o
    resultado esperado, executando ambos contra o banco de dados
    correspondente do Spider.
    """

    def __init__(self, db_dir: str = "data/spider/database"):
        super().__init__()
        self.db_dir = db_dir
        self.threshold = 1.0
        self.score = 0.0
        self.success = False
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        predicted_sql: str = test_case.actual_output
        expected_sql: str = test_case.expected_output
        db_id: str = test_case.additional_metadata.get("db_id", "")

        predicted_sql = self._extract_sql(predicted_sql)

        try:
            predicted_result, pred_has_order = self._execute_sql(predicted_sql, db_id)
            expected_result, exp_has_order = self._execute_sql(expected_sql, db_id)

            if pred_has_order or exp_has_order:
                self.score = 1.0 if predicted_result == expected_result else 0.0
            else:
                self.score = 1.0 if Counter(predicted_result) == Counter(expected_result) else 0.0
        except Exception:
            self.score = 0.0

        self.success = self.score >= self.threshold
        self.reason = None if self.success else "SQL execution result does not match expected output"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        return self.measure(test_case)

    def _extract_sql(self, text: str) -> str:
        text = text.strip()
        patterns = [
            r"```sql\s*\n?(.*?)```",
            r"```\s*\n?(.*?)```",
            r"`(.+?)`",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        lines = text.strip().split("\n")
        sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.upper().startswith(sql_keywords):
                return stripped

        return text

    def _execute_sql(self, query: str, db_id: str) -> Tuple[List[tuple], bool]:
        db_path = f"{self.db_dir}/{db_id}/{db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            has_order = bool(re.search(r'\bORDER\s+BY\b', query, re.IGNORECASE))
        except Exception:
            result = []
            has_order = False
        finally:
            conn.close()
        return result, has_order

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Execution Accuracy"
