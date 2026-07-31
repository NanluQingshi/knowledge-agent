"""知识库导出模块 — 导出为 Markdown / JSON 格式."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_agent.storage.doc_store import DocStore
from knowledge_agent.storage.vector_store import VectorStore


class Exporter:
    """知识库导出器.

    支持导出格式：
    - Markdown: 适合阅读和分享
    - JSON: 适合程序处理和迁移
    """

    def __init__(
        self,
        doc_store: DocStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._doc_store = doc_store or DocStore()
        self._vector_store = vector_store or VectorStore()

    def export_markdown(self, output_dir: str | Path) -> Path:
        """导出知识库为 Markdown 文件.

        每个文档源生成一个 .md 文件，包含其所有 chunk 的内容和元数据。

        Args:
            output_dir: 输出目录路径.

        Returns:
            输出目录路径.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        docs = self._doc_store.list_documents()
        if not docs:
            # 写一个空说明文件
            (out / "_index.md").write_text("# 知识库为空\n\n暂无文档。", encoding="utf-8")
            return out

        # 收集按 source 分组的文档
        by_source: dict[str, list[dict[str, Any]]] = {}
        for doc in docs:
            src = doc.get("source", "unknown")
            by_source.setdefault(src, []).append(doc)

        # 读取所有 chunk
        all_data = self._vector_store.collection.get(
            include=["documents", "metadatas"],
        )
        chunk_map: dict[str, list[str]] = {}
        if all_data.get("ids"):
            for i, cid in enumerate(all_data["ids"]):
                meta = (all_data["metadatas"] or [{}])[i] or {}
                doc_id = meta.get("doc_id", "")
                text = (all_data["documents"] or [""])[i] or ""
                chunk_map.setdefault(doc_id, []).append(text)

        index_lines = ["# 知识库导出\n", f"导出时间: {datetime.now(timezone.utc).isoformat()}\n"]
        index_lines.append(f"文档总数: {len(docs)}\n")

        for source, doc_list in by_source.items():
            filename = doc_list[0].get("filename", "unknown")
            safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename)
            md_lines = [
                f"# {filename}\n",
                f"- **来源**: {source}",
                f"- **版本**: {doc_list[0].get('version', 1)}",
                f"- **摄入时间**: {doc_list[0].get('ingested_at', '')}",
                f"- **文件类型**: {doc_list[0].get('file_type', '')}",
                "",
            ]

            for doc in doc_list:
                doc_id = doc["id"]
                chunks = chunk_map.get(doc_id, [])
                for i, chunk_text in enumerate(chunks):
                    md_lines.append(f"## Chunk {i + 1}\n")
                    md_lines.append(chunk_text)
                    md_lines.append("")

            filepath = out / safe_name
            filepath.write_text("\n".join(md_lines), encoding="utf-8")
            index_lines.append(f"- [{filename}]({safe_name})")

        (out / "_index.md").write_text("\n".join(index_lines), encoding="utf-8")
        return out

    def export_json(self, output_path: str | Path) -> Path:
        """导出知识库为 JSON 文件.

        Args:
            output_path: 输出 JSON 文件路径.

        Returns:
            输出文件路径.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        docs = self._doc_store.list_documents()
        all_data = self._vector_store.collection.get(
            include=["documents", "metadatas"],
        )

        chunks_export: list[dict[str, Any]] = []
        if all_data.get("ids"):
            for i, cid in enumerate(all_data["ids"]):
                chunks_export.append(
                    {
                        "id": cid,
                        "text": (all_data["documents"] or [""])[i] or "",
                        "metadata": (all_data["metadatas"] or [{}])[i] or {},
                    }
                )

        export = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "documents": docs,
            "chunks": chunks_export,
            "stats": {
                "documents": len(docs),
                "chunks": len(chunks_export),
            },
        }

        out.write_text(
            json.dumps(export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out
