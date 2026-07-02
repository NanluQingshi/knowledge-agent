"""LLM 驱动的关系抽取模块."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from knowledge_agent.config import settings


# 标准关系类型清单（用于 prompt 提示）
_STANDARD_RELATIONS = [
    "uses",            # A 使用 B
    "depends_on",      # A 依赖 B
    "part_of",         # A 是 B 的一部分
    "creates",         # A 创建 B
    "influences",      # A 影响 B
    "belongs_to",      # A 属于 B
    "related_to",      # A 与 B 相关
    "located_in",      # A 位于 B
    "works_at",        # A 在 B 工作
    "authored_by",     # A 由 B 创作
    "contains",        # A 包含 B
    "collaborates_with",  # A 与 B 协作
    "leads",           # A 领导 B
    "developed_by",    # A 由 B 开发
    "other",           # 其他关系（请在 predicate 字段自行描述）
]


class RelationExtractor:
    """使用 LLM 从文本中抽取实体间的关系.

    支持标准关系类型（uses, depends_on, part_of 等），
    也可通过 ``other`` 类型覆盖自定义关系。

    Attributes:
        client: OpenAI 客户端实例.
        model: LLM 模型名称.
    """

    def __init__(self) -> None:
        """初始化 RelationExtractor，从配置创建 OpenAI 客户端。"""
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.llm_model

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_extraction_prompt(
        self,
        text: str,
        entities: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """构建关系抽取的系统提示与用户提示。

        Args:
            text: 待分析的文本。
            entities: 可选的已知实体列表，用于辅助 LLM 聚焦。
        """
        entity_context = ""
        if entities:
            names = [e.get("name", "") for e in entities if isinstance(e, dict) and e.get("name")]
            if names:
                entity_context = (
                    f"\n已知文本中包含以下实体：{', '.join(names)}。"
                    "\n请优先分析这些实体之间的关系，"
                    "同时也可发现其他未列出的实体之间的关系。"
                )

        rel_types = "\n".join(f'- "{r}": {_RELATION_DESCRIPTIONS.get(r, "")}' for r in _STANDARD_RELATIONS)

        system_prompt = (
            "你是一个专业的关系抽取系统。请从给定的文本中识别实体之间的关系。\n\n"
            "标准关系类型：\n"
            f"{rel_types}"
            f"{entity_context}\n\n"
            "请以 JSON 数组格式返回，每个元素包含以下字段：\n"
            '- "subject": 主体实体名称\n'
            '- "predicate": 关系类型（优先使用标准类型）\n'
            '- "object": 客体实体名称\n'
            '- "confidence": 置信度 (0.0 ~ 1.0)\n'
            '- "evidence": 支持该关系存在的原文片段\n\n'
            "只返回 JSON 数组，不要包含其他说明文字。"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析以下文本中的实体关系：\n\n{text}"},
        ]

    # ------------------------------------------------------------------
    # JSON 解析
    # ------------------------------------------------------------------

    def _parse_json_response(self, response_text: str) -> list[dict[str, Any]] | None:
        """鲁棒地解析 LLM 返回的 JSON。

        处理 Markdown 代码块、尾随逗号等常见问题。
        """
        text = response_text.strip()

        # 移除 markdown 代码块
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 清理尾随逗号
        cleaned = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 尝试提取第一个 JSON 数组
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            candidate = array_match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            try:
                return json.loads(re.sub(r",\s*([\]}])", r"\1", candidate))
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        entities: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """从文本中抽取实体关系。

        Args:
            text: 输入文本。
            entities: 可选，已知的实体列表。如果未提供，将先执行实体识别。

        Returns:
            关系列表，每条关系包含 ``subject``, ``predicate``, ``object``,
            ``confidence``, ``evidence`` 字段。

        Raises:
            RuntimeError: LLM API 调用失败时抛出，附带上下文信息。
        """
        if not text or not text.strip():
            return []

        # 如果未提供实体，先执行实体识别
        if entities is None:
            from knowledge_agent.extraction.entity_extractor import EntityExtractor

            try:
                extractor = EntityExtractor()
                entities = extractor.extract(text)
            except RuntimeError:
                entities = []

        messages = self._build_extraction_prompt(text, entities)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            raise RuntimeError(f"关系抽取 API 调用失败: {e}") from e

        content = response.choices[0].message.content or ""

        parsed = self._parse_json_response(content)
        if parsed is not None and isinstance(parsed, list):
            validated: list[dict[str, Any]] = []
            for rel in parsed:
                if not isinstance(rel, dict):
                    continue
                subj = rel.get("subject")
                pred = rel.get("predicate")
                obj = rel.get("object")
                if subj and pred and obj:
                    validated.append({
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                        "confidence": float(rel.get("confidence", 0.5)),
                        "evidence": rel.get("evidence", ""),
                    })
            if validated:
                return validated

        return []


# 关系类型描述（用于 prompt）
_RELATION_DESCRIPTIONS: dict[str, str] = {
    "uses": "A 使用 B",
    "depends_on": "A 依赖 B",
    "part_of": "A 是 B 的一部分",
    "creates": "A 创建 B",
    "influences": "A 影响 B",
    "belongs_to": "A 属于 B",
    "related_to": "A 与 B 相关",
    "located_in": "A 位于 B",
    "works_at": "A 在 B 工作",
    "authored_by": "A 由 B 创作",
    "contains": "A 包含 B",
    "collaborates_with": "A 与 B 协作",
    "leads": "A 领导 B",
    "developed_by": "A 由 B 开发",
    "other": "其他关系（请在 predicate 字段中自行描述具体关系）",
}
