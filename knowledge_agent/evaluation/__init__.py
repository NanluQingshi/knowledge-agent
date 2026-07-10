"""Agent 评估模块 — 检索质量与答案质量评估.

提供评估数据集管理、检索质量指标（MRR/Recall/Precision/NDCG）和
基于 RAGAS 的答案质量评估（需安装 ragas 包）。
"""

from knowledge_agent.evaluation.dataset import EvaluationDataset
from knowledge_agent.evaluation.metrics import RetrievalMetrics
from knowledge_agent.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationDataset",
    "EvaluationRunner",
    "RetrievalMetrics",
]
