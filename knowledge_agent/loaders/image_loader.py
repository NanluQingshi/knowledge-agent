"""图片加载器 — 支持 OCR 文本提取和 LLM 视觉描述."""

from __future__ import annotations

import logging
from pathlib import Path

from knowledge_agent.loaders.base import BaseLoader, Document

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


class ImageLoader(BaseLoader):
    """图片文件加载器.

    两种模式：
    - **OCR 模式**（默认）: 使用 pytesseract 提取图片中的文字
    - **LLM 视觉模式**: 使用 GPT-4o 等多模态模型描述图片内容

    当 pytesseract 不可用时自动回退到 LLM 视觉模式。
    """

    def __init__(self, use_llm_vision: bool = False) -> None:
        """初始化 ImageLoader.

        Args:
            use_llm_vision: 是否强制使用 LLM 视觉模式而非 OCR.
        """
        self._use_llm_vision = use_llm_vision

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in _SUPPORTED_EXTENSIONS

    def load(self, file_path: Path) -> list[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = self._extract_content(file_path)
        if not content.strip():
            return []

        stat = file_path.stat()
        return [
            Document(
                content=content,
                metadata={
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "type": file_path.suffix.lower().lstrip("."),
                    "image_mode": "ocr" if not self._use_llm_vision else "vision",
                },
                source=str(file_path.resolve()),
            )
        ]

    def _extract_content(self, path: Path) -> str:
        """提取图片内容."""
        if self._use_llm_vision:
            return self._describe_via_llm(path)

        # OCR 模式
        try:
            return self._ocr_extract(path)
        except (ImportError, OSError) as exc:
            logger.warning("OCR failed: %s, falling back to LLM vision", exc)
            return self._describe_via_llm(path)

    @staticmethod
    def _ocr_extract(path: Path) -> str:
        """使用 pytesseract 进行 OCR 文字提取."""
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract and Pillow are required for OCR. "
                "Install with: pip install pytesseract Pillow"
            )

        image = Image.open(str(path))
        text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        return text.strip() or "（图片中未识别到文字）"

    @staticmethod
    def _describe_via_llm(path: Path) -> str:
        """使用多模态 LLM 描述图片内容."""
        try:
            from openai import OpenAI
            from knowledge_agent.config import settings

            import base64

            with open(path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")

            ext = path.suffix.lower().lstrip(".")
            if ext == "jpg":
                ext = "jpeg"
            media_type = f"image/{ext}"

            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )

            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请详细描述这张图片的内容，包括文字、物体、场景等。",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{img_data}",
                                    "detail": "auto",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            return response.choices[0].message.content or "（图片描述生成失败）"
        except Exception as exc:
            logger.warning("LLM vision description failed: %s", exc)
            return f"（图片文件: {path.name}，无法提取内容）"
