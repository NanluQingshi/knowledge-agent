"""OpenAI 兼容的文本向量化封装（含 sentence-transformers 本地回退）."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from openai import OpenAI

from knowledge_agent.config import settings

if TYPE_CHECKING:
    from openai.types.create_embedding_response import CreateEmbeddingResponse

logger = logging.getLogger(__name__)

_LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """文本向量化封装.

    优先使用 OpenAI 兼容 API；当 API key 未配置或调用失败时，
    自动回退到 ``sentence-transformers`` 本地模型
    (``all-MiniLM-L6-v2``, 384 维)。

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
        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.openai_base_url
        self._local_model: Any = None  # 延迟加载
        self._client: OpenAI | None = None

        if self._api_key:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )

    @property
    def model(self) -> str:
        """当前使用的嵌入模型名称."""
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量将文本列表转换为向量.

        优先调用 OpenAI API；API 不可用时自动回退到本地模型。

        Args:
            texts: 待向量化的文本列表.

        Returns:
            与输入顺序对应的向量列表.

        Raises:
            RuntimeError: API 和本地模型均失败时抛出.
        """
        if not texts:
            return []

        # 尝试 OpenAI API
        if self._client is not None:
            try:
                return self._embed_via_api(texts)
            except Exception as exc:
                logger.warning("OpenAI Embedding API failed, falling back to local model: %s", exc)

        # 回退到本地 sentence-transformers 模型
        return self._embed_via_local(texts)

    def _embed_via_api(self, texts: list[str]) -> list[list[float]]:
        """通过 OpenAI API 进行向量化."""
        response: CreateEmbeddingResponse = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        sorted_data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in sorted_data]

    def _embed_via_local(self, texts: list[str]) -> list[list[float]]:
        """通过本地 sentence-transformers 模型进行向量化."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

        if self._local_model is None:
            logger.info("Loading local embedding model: %s", _LOCAL_MODEL_NAME)
            self._local_model = SentenceTransformer(_LOCAL_MODEL_NAME)
            self._dim = self._local_model.get_sentence_embedding_dimension()

        embeddings = self._local_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """将单条文本转换为向量.

        Args:
            text: 待向量化的文本.

        Returns:
            向量列表.
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
