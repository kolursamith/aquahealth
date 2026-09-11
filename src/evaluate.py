"""Produces accuracy, precision, recall, macro-F1, confusion matrix, and inference time.

Owner: Student 2
"""

from dataclasses import dataclass


@dataclass
class EvaluationReport:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    avg_inference_time_ms: float


def evaluate(model, dataloader) -> EvaluationReport:
    raise NotImplementedError("Run inference over dataloader and compute metrics")
