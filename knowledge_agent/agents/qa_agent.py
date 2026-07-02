"""问答 Agent — 基于 RAG 的检索增强问答."""

from __future__ import annotations

from typing import Any, Iterator

from openai import OpenAI

from knowledge_agent.config import settings
from knowledge_agent.retrieval.hybrid_retriever import HybridRetriever

_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided context.

Guidelines:
1. Answer the question using ONLY the information provided in the context below.
2. If the context does not contain enough information to answer the question, say so clearly.
3. Cite specific sources when possible by referring to the source document or metadata.
4. Be concise and accurate. Do not make up information.
5. If the question is a greeting or general chat, you may respond naturally without relying on context.

Context:
{context}"""


class QAAgent:
    """基于检索增强生成 (RAG) 的问答 Agent.

    接收用户问题，通过混合检索获取相关上下文，
    调用 LLM 生成基于上下文的回答。
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        openai_client: OpenAI | None = None,
    ) -> None:
        """初始化 QAAgent.

        Args:
            hybrid_retriever: 已初始化的混合检索器实例.
            openai_client: OpenAI 客户端实例；若未提供，则从 settings 自动创建.
        """
        self._retriever = hybrid_retriever
        self._client = openai_client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def query(
        self,
        question: str,
        top_k: int | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """执行一次检索增强问答.

        1. 使用混合检索获取相关上下文.
        2. 构建包含上下文和历史消息的提示.
        3. 调用 LLM 生成回答.

        Args:
            question: 用户问题.
            top_k: 检索结果数量上限；默认使用 settings.retrieval_top_k.
            chat_history: 可选的历史对话记录，每个元素为 {"role": ..., "content": ...}.

        Returns:
            包含以下键的字典：
                - answer: LLM 生成的回答文本.
                - sources: 检索结果中的来源摘要列表（每项含 text 和 metadata）.
                - context_used: 检索到的原始文档块列表.

        Raises:
            RuntimeError: LLM API 调用失败时抛出.
        """
        # 1. 检索相关上下文
        retrieved = self._retriever.retrieve(question, top_k=top_k)

        # 2. 构建上下文文本
        context_text = self._format_context(retrieved)

        # 3. 构建消息
        messages = self._build_messages(
            system_prompt=_SYSTEM_PROMPT.format(context=context_text),
            question=question,
            chat_history=chat_history or [],
        )

        # 4. 调用 LLM
        try:
            response = self._client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM API call failed (model={settings.llm_model}): {exc}"
            ) from exc

        answer = response.choices[0].message.content or ""

        # 5. 整理来源信息
        sources = [
            {
                "text": item["text"],
                "metadata": item.get("metadata", {}),
            }
            for item in retrieved
        ]

        return {
            "answer": answer,
            "sources": sources,
            "context_used": retrieved,
        }

    def stream_query(
        self,
        question: str,
        top_k: int | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """流式版本的检索增强问答.

        用法与 query() 相同，但以生成器形式逐 token 返回回答文本，
        适用于实时流式输出场景。

        Args:
            question: 用户问题.
            top_k: 检索结果数量上限；默认使用 settings.retrieval_top_k.
            chat_history: 可选的历史对话记录.

        Yields:
            LLM 生成的文本片段.

        Raises:
            RuntimeError: LLM API 调用失败时抛出.
        """
        retrieved = self._retriever.retrieve(question, top_k=top_k)
        context_text = self._format_context(retrieved)

        messages = self._build_messages(
            system_prompt=_SYSTEM_PROMPT.format(context=context_text),
            question=question,
            chat_history=chat_history or [],
        )

        try:
            stream = self._client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM API call failed (model={settings.llm_model}): {exc}"
            ) from exc

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(retrieved: list[dict[str, Any]]) -> str:
        """将检索结果格式化为上下文文本.

        Args:
            retrieved: 检索结果列表.

        Returns:
            格式化的上下文字符串.
        """
        if not retrieved:
            return "(No relevant context found.)"

        parts: list[str] = []
        for i, item in enumerate(retrieved, start=1):
            source_info = ""
            meta = item.get("metadata", {}) or {}
            if meta.get("source"):
                source_info = f" [Source: {meta['source']}]"
            parts.append(f"[{i}]{source_info}\n{item.get('text', '')}")

        return "\n\n".join(parts)

    @staticmethod
    def _build_messages(
        system_prompt: str,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """构建发送给 LLM 的消息列表.

        Args:
            system_prompt: 系统提示词（含上下文）.
            question: 当前用户问题.
            chat_history: 历史对话记录.

        Returns:
            消息字典列表.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # 插入历史消息
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        # 追加当前问题
        messages.append({"role": "user", "content": question})

        return messages
