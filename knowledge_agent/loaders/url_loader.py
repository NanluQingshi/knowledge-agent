"""URL 加载器 — 支持从网页 URL 抓取并提取文本内容."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from knowledge_agent.loaders.base import BaseLoader, Document

logger = logging.getLogger(__name__)


class UrlLoader(BaseLoader):
    """URL 网页加载器，从指定 URL 抓取 HTML 并提取正文文本.

    依赖 httpx 和 beautifulsoup4 包，未安装时会给出清晰的安装提示。
    """

    def can_handle(self, file_path: Path) -> bool:
        # UrlLoader 不通过文件扩展名匹配，通过 ingest_url 方法调用
        return False

    def load(self, file_path: Path) -> list[Document]:
        raise NotImplementedError("UrlLoader does not support file loading. Use ingest_url() instead.")

    def ingest_url(
        self,
        url: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """从 URL 抓取网页并提取文本.

        Args:
            url: 网页 URL.
            metadata: 可选的附加元数据.

        Returns:
            提取的 Document 列表（通常为 1 个）.

        Raises:
            ImportError: httpx 或 beautifulsoup4 未安装时抛出.
            RuntimeError: 网络请求失败时抛出.
        """
        html = self._fetch_url(url)

        # 使用 HTMLLoader 的逻辑提取文本
        content = self._extract_text(html)
        if not content.strip():
            logger.warning("No extractable content found at %s", url)
            return []

        base_meta = {
            "filename": url.rstrip("/").split("/")[-1] or "webpage",
            "source_url": url,
            "type": "url",
            "size": len(html),
            **(metadata or {}),
        }

        return [
            Document(
                content=content,
                metadata=base_meta,
                source=url,
            )
        ]

    @staticmethod
    def _fetch_url(url: str) -> str:
        """发送 HTTP GET 请求获取网页内容."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required to fetch URLs. "
                "Install it with: pip install httpx"
            )

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url, headers={"User-Agent": "KnowledgeAgent/1.0"})
                response.raise_for_status()
                return response.text
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch URL {url}: {exc}") from exc

    @staticmethod
    def _extract_text(html: str) -> str:
        """从 HTML 中提取纯文本."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
        except ImportError:
            import re

            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text