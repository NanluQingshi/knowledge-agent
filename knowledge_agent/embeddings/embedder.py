"""OpenAI 兼容的文本向量化封装."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

from knowledge_agent.config import settings

if TYPE_CHECKING:
    from openai.types.create_embedding_response import CreateEmbeddingResponse


class Embedder:
    """OpenAI 兼容 Embedding API 的包装器.

    默认从全局 settings 读取 api_key、base_url、model 和 dimension。
    所有参数均支持在构造时覆盖。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dim: int | None = None,
    ) -> None:
        """初始化 Embedder.

        Args:
            model: 嵌入模型名称，默认 settings.embedding_model.
            api_key: OpenAI API key，默认 settings.openai_api_key.
            base_url: OpenAI API base URL，默认 settings.openai_base_url.
            dim: 嵌入向量维度，默认 settings.embedding_dim.
        """
        self._model = model or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
        )

    @property
    def model(self) -> str:
        """当前使用的嵌入模型名称."""
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量将文本列表转换为向量.

        Args:
            texts: 待向量化的文本列表.

        Returns:
            与输入顺序对应的向量列表.

        Raises:
            RuntimeError: API 调用失败时抛出.
        """
        if not texts:
            return []

        try:
            response: CreateEmbeddingResponse = self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Embedding API call failed (model={self._model}): {exc}"
            ) from exc

        # response.data 默认按输入顺序返回，此处保留排序逻辑以确保健壮性
        sorted_data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in sorted_data]

    def embed_single(self, text: str) -> list[float]:
        """将单条文本转换为向量.

        Args:
            text: 待向量化的文本.

        Returns:
            向量列表.

        Raises:
            RuntimeError: API 调用失败时抛出.
        """
        results = self.embed([text])
        if not results:
            return []
        return results[0]

    def dimension(self) -> int:
        """返回当前嵌入模型的输出向量维度.

        Returns:
            向量维度.
        """
        return self._dim
