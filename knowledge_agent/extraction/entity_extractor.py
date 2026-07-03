"""LLM 驱动的命名实体识别 (NER) 模块."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from knowledge_agent.config import settings


class EntityExtractor:
    """使用 LLM 进行命名实体识别 (NER).

    支持人物、组织、技术、概念、地点、日期等多种实体类型的识别，
    并具备正则回退方案以应对 LLM 输出解析失败的情况。

    Attributes:
        client: OpenAI 客户端实例.
        model: LLM 模型名称.
    """

    def __init__(self) -> None:
        """初始化 EntityExtractor，从配置创建 OpenAI 客户端。"""
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.llm_model

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_extraction_prompt(self, text: str) -> list[dict[str, str]]:
        """构建 NER 系统提示与用户提示。"""
        system_prompt = (
            "你是一个专业的命名实体识别 (NER) 系统。"
            "请从给定的文本中提取所有实体。\n\n"
            "实体类型包括：\n"
            '- "person": 人物\n'
            '- "organization": 组织 / 公司 / 机构\n'
            '- "technology": 技术 / 产品 / 框架 / 工具\n'
            '- "concept": 概念 / 主题 / 方法论\n'
            '- "location": 地点 / 地域\n'
            '- "date": 日期 / 时间\n'
            '- "other": 其他重要实体\n\n'
            "请以 JSON 数组格式返回，每个元素包含以下字段：\n"
            '- "name": 实体名称\n'
            '- "type": 实体类型\n'
            '- "description": 实体简要描述（10-30字）\n'
            '- "mentions": 实体在原文中的提及片段列表（至少包含一个）\n\n'
            "只返回 JSON 数组，不要包含其他说明文字。"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请提取以下文本中的实体：\n\n{text}"},
        ]

    # ------------------------------------------------------------------
    # JSON 解析
    # ------------------------------------------------------------------

    def _parse_json_response(self, response_text: str) -> list[dict[str, Any]] | None:
        """鲁棒地解析 LLM 返回的 JSON。

        自动处理：
        - Markdown 代码块包裹 (```json ... ```)
        - 尾随逗号
        - 嵌套在一段文字中间的 JSON 数组
        """
        text = response_text.strip()

        # 移除 markdown 代码块标记
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 处理尾随逗号后再试一次
        cleaned = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 尝试从响应中提取第一个 JSON 数组
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            candidate = array_match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            # 也尝试清理后的版本
            try:
                return json.loads(re.sub(r",\s*([\]}])", r"\1", candidate))
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # 正则回退
    # ------------------------------------------------------------------

    def _regex_fallback(self, text: str) -> list[dict[str, Any]]:
        """正则表达式回退方案：提取大写短语作为候选实体。

        当 LLM 输出解析完全失败时使用此回退。
        """
        entities: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 匹配连续大写开头的词组（如 "Knowledge Graph", "OpenAI"）
        for match in re.finditer(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b", text):
            name = match.group(0).strip()
            if not name or len(name) < 2:
                continue
            key = name.lower()
            if key not in seen:
                seen.add(key)
                entities.append({
                    "name": name,
                    "type": "concept",
                    "description": f"从文本中提取的候选实体: {name}",
                    "mentions": [name],
                })

        # 匹配全大写缩写（如 "API", "NER", "LLM"）
        for match in re.finditer(r"\b[A-Z]{2,}\b", text):
            name = match.group(0).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                entities.append({
                    "name": name,
                    "type": "concept",
                    "description": f"从文本中提取的候选实体: {name}",
                    "mentions": [name],
                })

        return entities

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[dict[str, Any]]:
        """从文本中提取实体。

        Args:
            text: 输入文本。

        Returns:
            实体列表，每个实体包含 ``name``, ``type``, ``description``, ``mentions`` 字段。

        Raises:
            RuntimeError: LLM API 调用失败时抛出，附带上下文信息。
        """
        if not text or not text.strip():
            return []

        messages = self._build_extraction_prompt(text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            raise RuntimeError(f"NER API 调用失败: {e}") from e

        content = response.choices[0].message.content or ""

        parsed = self._parse_json_response(content)
        if parsed is not None and isinstance(parsed, list):
            validated: list[dict[str, Any]] = []
            for ent in parsed:
                if isinstance(ent, dict) and ent.get("name"):
                    validated.append({
                        "name": ent["name"],
                        "type": ent.get("type", "other"),
                        "description": ent.get("description", ""),
                        "mentions": ent.get("mentions", [ent["name"]]),
                    })
            if validated:
                return validated

        # LLM 输出解析失败，回退到正则方案
        return self._regex_fallback(text)

    def extract_batch(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """批量从多个文本中提取实体。

        文本被分组（每组 5 个）处理，以平衡吞吐与稳定性。

        Args:
            texts: 输入文本列表。

        Returns:
            与输入 ``texts`` 顺序对应的实体列表的列表。
        """
        results: list[list[dict[str, Any]]] = []
        batch_size = 5

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                try:
                    entities = self.extract(text)
                except RuntimeError:
                    entities = []
                results.append(entities)

        return results
