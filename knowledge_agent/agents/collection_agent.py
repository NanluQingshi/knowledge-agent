"""采集 Agent — 负责监控数据源、解析文档与数据清洗."""

from __future__ import annotations

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from knowledge_agent.chunkers.recursive_chunker import RecursiveChunker
from knowledge_agent.config import settings
from knowledge_agent.embeddings.embedder import Embedder
from knowledge_agent.loaders import BaseLoader, MarkdownLoader, PDFLoader, TextLoader
from knowledge_agent.storage.doc_store import DocStore
from knowledge_agent.storage.vector_store import VectorStore

_DEFAULT_LOADERS: list[BaseLoader] = [TextLoader(), MarkdownLoader(), PDFLoader()]


class CollectionAgent:
    """文档采集与导入 Agent.

    负责：
    - 从文件或目录加载文档（通过 Loader 链自动识别格式）
    - 使用 Chunker 将文档分块
    - 使用 Embedder 生成向量并写入 VectorStore
    - 将文档元数据注册到 DocStore

    所有核心依赖均可在构造时注入，否则使用默认实现。
    """

    def __init__(
        self,
        loaders: list[BaseLoader] | None = None,
        chunker: Any | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        doc_store: DocStore | None = None,
    ) -> None:
        """初始化 CollectionAgent.

        Args:
            loaders: 文档加载器列表，按顺序尝试 can_handle 匹配文件.
                    默认包含 TextLoader、MarkdownLoader、PDFLoader.
            chunker: 文档分块器，默认 RecursiveChunker(settings.chunk_size, settings.chunk_overlap).
            embedder: 文本向量化器，默认 Embedder().
            vector_store: 向量存储，默认 VectorStore().
            doc_store: 文档元数据存储，默认 DocStore().
        """
        self._loaders = loaders or _DEFAULT_LOADERS
        self._chunker = chunker or RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._embedder = embedder or Embedder()
        self._vector_store = vector_store or VectorStore()
        self._doc_store = doc_store or DocStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_path(self, path: str | Path) -> dict[str, Any]:
        """从文件或目录加载文档并导入知识库.

        对目录进行递归遍历，逐个文件尝试可用的 Loader。
        每个文件的每个文档被分块、向量化后写入 VectorStore，
        同时向 DocStore 注册文档元数据。

        Args:
            path: 文件路径或目录路径.

        Returns:
            摘要字典，包含:
            - ``documents_loaded``: 加载的 Document 对象总数
            - ``chunks_created``: 切分出的 Chunk 总数
            - ``files_processed``: 成功处理的文件数
            - ``errors``: 错误信息列表，每项含 ``file`` 和 ``error`` 字段
        """
        target = Path(path)
        if not target.exists():
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "files_processed": 0,
                "errors": [{"file": str(target), "error": "Path does not exist"}],
            }

        files = self._collect_files(target)

        total_docs = 0
        total_chunks = 0
        files_processed = 0
        errors: list[dict[str, str]] = []

        for file_path in files:
            try:
                loader = self._match_loader(file_path)
                if loader is None:
                    # 没有匹配的 Loader，记录但不视为错误
                    continue

                documents = loader.load(file_path)
                if not documents:
                    continue

                for doc in documents:
                    content_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()
                    existing = self._doc_store.find_by_hash(content_hash)
                    if existing:
                        continue

                    # 检查同一来源是否有旧版本
                    previous_version_id = ""
                    prev_docs = self._doc_store.find_by_source(doc.source)
                    if prev_docs:
                        previous_version_id = prev_docs[0]["id"]

                    chunks = self._chunker.chunk(doc.content, doc.metadata)
                    if not chunks:
                        continue

                    doc_id = uuid.uuid4().hex
                    chunk_ids = [f"{doc_id}_chunk_{c.chunk_index}" for c in chunks]
                    chunk_texts = [c.text for c in chunks]
                    embeddings = self._embedder.embed(chunk_texts)

                    metadatas: list[dict[str, Any]] = []
                    for c in chunks:
                        meta = dict(c.metadata)
                        meta["doc_id"] = doc_id
                        meta["source"] = doc.source
                        metadatas.append(meta)

                    self._vector_store.add(chunks, embeddings, metadatas, chunk_ids)

                    self._doc_store.add_document(
                        doc_id=doc_id,
                        source=doc.source,
                        filename=Path(doc.source).name if doc.source else file_path.name,
                        file_type=file_path.suffix.lower(),
                        chunk_count=len(chunks),
                        content_hash=content_hash,
                        previous_version_id=previous_version_id,
                        metadata={
                            "original_metadata": doc.metadata,
                        },
                    )

                    total_docs += 1
                    total_chunks += len(chunks)

                files_processed += 1

            except Exception as exc:
                errors.append(
                    {
                        "file": str(file_path),
                        "error": str(exc),
                    }
                )

        return {
            "documents_loaded": total_docs,
            "chunks_created": total_chunks,
            "files_processed": files_processed,
            "errors": errors,
        }

    def ingest_path_parallel(
        self,
        path: str | Path,
        max_workers: int = 4,
    ) -> dict[str, Any]:
        """并行批量摄入（使用线程池加速大批量文件处理）.

        对目录中的每个文件使用独立的线程处理，互不阻塞。
        每个文件内部的加载、分块、向量化、存储是串行的。

        Args:
            path: 文件路径或目录路径.
            max_workers: 最大并发线程数，默认 4.

        Returns:
            与 ingest_path() 相同格式的摘要字典.
        """
        target = Path(path)
        if not target.exists():
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "files_processed": 0,
                "errors": [{"file": str(target), "error": "Path does not exist"}],
            }

        files = self._collect_files(target)
        if not files:
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "files_processed": 0,
                "errors": [],
            }

        total_docs = 0
        total_chunks = 0
        all_errors: list[dict[str, str]] = []

        # 每个线程使用独立的 Embedder 和 VectorStore（避免连接冲突）
        def _process_one(fp: Path) -> dict[str, Any]:
            """处理单个文件."""
            try:
                loader = self._match_loader(fp)
                if loader is None:
                    return {"docs": 0, "chunks": 0}

                documents = loader.load(fp)
                if not documents:
                    return {"docs": 0, "chunks": 0}

                local_embedder = Embedder()
                local_vector_store = VectorStore()
                local_doc_store = DocStore()
                local_chunker = RecursiveChunker(
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )

                doc_count = 0
                chunk_count = 0
                for doc in documents:
                    content_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()
                    existing = local_doc_store.find_by_hash(content_hash)
                    if existing:
                        continue

                    chunks = local_chunker.chunk(doc.content, doc.metadata)
                    if not chunks:
                        continue

                    doc_id = uuid.uuid4().hex
                    chunk_ids = [f"{doc_id}_chunk_{c.chunk_index}" for c in chunks]
                    chunk_texts = [c.text for c in chunks]
                    embeddings = local_embedder.embed(chunk_texts)

                    metadatas = []
                    for c in chunks:
                        meta = dict(c.metadata)
                        meta["doc_id"] = doc_id
                        meta["source"] = doc.source
                        metadatas.append(meta)

                    local_vector_store.add(chunks, embeddings, metadatas, chunk_ids)
                    local_doc_store.add_document(
                        doc_id=doc_id,
                        source=doc.source,
                        filename=Path(doc.source).name if doc.source else fp.name,
                        file_type=fp.suffix.lower(),
                        chunk_count=len(chunks),
                        content_hash=content_hash,
                        metadata={"original_metadata": doc.metadata},
                    )

                    doc_count += 1
                    chunk_count += len(chunks)

                return {"docs": doc_count, "chunks": chunk_count}

            except Exception as exc:
                return {"error": str(exc)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_one, fp): fp for fp in files}
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    all_errors.append(
                        {
                            "file": str(futures[future]),
                            "error": result["error"],
                        }
                    )
                else:
                    total_docs += result["docs"]
                    total_chunks += result["chunks"]

        return {
            "documents_loaded": total_docs,
            "chunks_created": total_chunks,
            "files_processed": len(files),
            "errors": all_errors,
        }

    def ingest_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """直接摄入原始文本（无需文件加载）.

        文本被分块、向量化后写入 VectorStore，并在 DocStore 中注册。

        Args:
            text: 原始文本内容.
            metadata: 可选的附加元数据.

        Returns:
            摘要字典，包含 ``documents_loaded``、``chunks_created``、
            ``files_processed``（固定为 0）和 ``errors``.
        """
        if not text or not text.strip():
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "files_processed": 0,
                "errors": [],
            }

        meta = metadata or {}
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self._doc_store.find_by_hash(content_hash)
        if existing:
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "files_processed": 0,
                "errors": [],
                "skipped": True,
                "existing_doc_id": existing[0]["id"],
            }

        doc_id = uuid.uuid4().hex

        chunks = self._chunker.chunk(text, meta)
        if not chunks:
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "files_processed": 0,
                "errors": [],
            }

        chunk_texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(chunk_texts)

        chunk_ids = [f"{doc_id}_chunk_{c.chunk_index}" for c in chunks]
        metadatas: list[dict[str, Any]] = []
        for c in chunks:
            m = dict(c.metadata)
            m["doc_id"] = doc_id
            m["source"] = "raw_text"
            metadatas.append(m)

        self._vector_store.add(chunks, embeddings, metadatas, chunk_ids)

        self._doc_store.add_document(
            doc_id=doc_id,
            source="raw_text",
            filename="raw_text",
            file_type="text",
            chunk_count=len(chunks),
            content_hash=content_hash,
            metadata=meta,
        )

        return {
            "documents_loaded": 1,
            "chunks_created": len(chunks),
            "files_processed": 0,
            "errors": [],
        }

    def get_stats(self) -> dict[str, Any]:
        """返回知识库当前统计信息.

        Returns:
            包含 ``total_documents``、``total_chunks``、``vector_store_size`` 的字典.
        """
        try:
            docs = self._doc_store.list_documents()
        except Exception:
            docs = []

        try:
            total_chunks = self._doc_store.get_total_chunks()
        except Exception:
            total_chunks = 0

        try:
            vec_size = self._vector_store.count()
        except Exception:
            vec_size = 0

        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "vector_store_size": vec_size,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_files(target: Path) -> list[Path]:
        """递归收集路径下所有文件.

        Args:
            target: 文件或目录路径.

        Returns:
            文件路径列表.
        """
        if target.is_file():
            return [target]
        return sorted(p for p in target.rglob("*") if p.is_file())

    def _match_loader(self, file_path: Path) -> BaseLoader | None:
        """尝试匹配一个能处理指定文件的 Loader.

        Args:
            file_path: 文件路径.

        Returns:
            匹配的 Loader 实例；若无匹配则返回 None.
        """
        for loader in self._loaders:
            try:
                if loader.can_handle(file_path):
                    return loader
            except Exception:
                continue
        return None
