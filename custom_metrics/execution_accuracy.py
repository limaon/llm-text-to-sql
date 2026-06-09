import sqlite3
from typing import List

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ExecutionAccuracyMetric(BaseMetric):
    """Metrica customizada que mede acuracia de execução de SQL.

    Compara o resultado da execução do SQL gerado pelo modelo com o
    resultado esperado, executando ambos contra o banco de dados
    correspondente do Spider.
    """

    def __init__(self, db_dir: str = "data/spider/database"):
        super().__init__()
        self.db_dir = db_dir
        self.threshold = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        predicted_sql: str = test_case.actual_output
        expected_sql: str = test_case.expected_output
        db_id: str = test_case.additional_metadata.get("db_id", "")

        try:
            predicted_result = self._execute_sql(predicted_sql, db_id)
            expected_result = self._execute_sql(expected_sql, db_id)

            if predicted_result == expected_result:
                self.score = 1.0
            else:
                self.score = 0.0
        except Exception:
            self.score = 0.0

        self.success = self.score >= self.threshold
        return self.score

    def _execute_sql(self, query: str, db_id: str) -> List[tuple]:
        db_path = f"{self.db_dir}/{db_id}/{db_id}.sqlite"
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

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Execution Accuracy"
