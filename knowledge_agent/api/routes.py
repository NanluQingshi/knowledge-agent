"""FastAPI 路由 — 知识沉淀 Agent REST API."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from knowledge_agent.chunkers.recursive_chunker import RecursiveChunker
from knowledge_agent.config import settings
from knowledge_agent.embeddings.embedder import Embedder
from knowledge_agent.loaders import all_loaders
from knowledge_agent.loaders.base import Document
from knowledge_agent.storage.doc_store import DocStore
from knowledge_agent.storage.vector_store import VectorStore


class QueryRequest(BaseModel):
    question: str
    top_k: int = settings.retrieval_top_k


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    total_chunks_in_store: int


class DocumentItem(BaseModel):
    id: str
    filename: str
    source: str
    file_type: str
    chunk_count: int
    ingested_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    total_documents: int
    total_chunks: int


def _load_documents_from_path(path: Path) -> list[Document]:
    """从路径加载文档."""
    docs: list[Document] = []
    loaders = all_loaders()
    files: list[Path] = []

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())

    for fp in files:
        for loader in loaders:
            if loader.can_handle(fp):
                try:
                    loaded = loader.load(fp)
                    docs.extend(loaded)
                except Exception:
                    continue

    return docs


def _process_documents(docs: list[Document]) -> int:
    """分块、向量化并存储文档。返回创建的 chunk 数量."""
    if not docs:
        return 0

    chunker = RecursiveChunker()
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc.content, metadata={
            "source": doc.source,
            "filename": doc.metadata.get("filename", ""),
        })
        all_chunks.extend(chunks)

    embedder = Embedder()
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed(texts)

    vector_store = VectorStore()
    chunk_ids = [str(uuid.uuid4()) for _ in all_chunks]
    metadatas = [{"chunk_index": i, **chunk.metadata} for i, chunk in enumerate(all_chunks)]
    vector_store.add(chunks=all_chunks, embeddings=embeddings, metadatas=metadatas, ids=chunk_ids)

    doc_store = DocStore()
    by_source: dict[str, list[str]] = {}
    for chunk, cid in zip(all_chunks, chunk_ids):
        src = chunk.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(cid)

    for src, cids in by_source.items():
        fp = Path(src)
        doc_id = str(uuid.uuid4())
        doc_store.add_document(
            doc_id=doc_id,
            source=src,
            filename=fp.name,
            file_type=fp.suffix.lower().lstrip("."),
            chunk_count=len(cids),
            metadata={"chunk_ids": cids},
        )

    return len(all_chunks)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用."""

    app = FastAPI(
        title="Knowledge Agent API",
        description="知识沉淀 Agent — 文档摄入、向量检索、智能问答",
        version="0.1.0",
    )

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_file(file: UploadFile = File(None)):
        """摄入文档文件."""
        docs: list[Document] = []
        loaders = all_loaders()

        if file is None:
            raise HTTPException(status_code=400, detail="No file provided")

        # 保存临时文件
        tmp_path = Path(settings.data_dir) / "tmp"
        tmp_path.mkdir(parents=True, exist_ok=True)
        file_path = tmp_path / (file.filename or "uploaded_file")
        content = await file.read()
        file_path.write_bytes(content)

        for loader in loaders:
            if loader.can_handle(file_path):
                try:
                    docs = loader.load(file_path)
                except Exception as exc:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail=f"Failed to load file: {exc}")
                break

        file_path.unlink(missing_ok=True)

        if not docs:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

        chunks_count = _process_documents(docs)
        vector_store = VectorStore()

        return IngestResponse(
            status="ok",
            documents_loaded=len(docs),
            chunks_created=chunks_count,
            total_chunks_in_store=vector_store.count(),
        )

    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest):
        """问答查询."""
        from knowledge_agent.agents.qa_agent import QAAgent
        from knowledge_agent.retrieval.bm25_retriever import BM25Retriever
        from knowledge_agent.retrieval.hybrid_retriever import HybridRetriever
        from knowledge_agent.retrieval.vector_retriever import VectorRetriever

        vector_store = VectorStore()
        embedder = Embedder()

        if vector_store.count() == 0:
            raise HTTPException(status_code=404, detail="No documents ingested yet")

        vector_retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
        bm25_retriever = BM25Retriever()

        # Build BM25 index — 使用 get_all_documents 避免全量向量扫描
        all_results = vector_store.get_all_documents()
        if all_results:
            bm25_retriever.index(all_results)

        hybrid = HybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)
        agent = QAAgent(hybrid_retriever=hybrid)

        result = agent.query(req.question, top_k=req.top_k)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    @app.get("/documents", response_model=DocumentListResponse)
    async def list_documents():
        """列出所有已摄入的文档."""
        doc_store = DocStore()
        docs = doc_store.list_documents()

        return DocumentListResponse(
            documents=[
                DocumentItem(
                    id=doc["id"],
                    filename=doc["filename"],
                    source=doc["source"],
                    file_type=doc["file_type"],
                    chunk_count=doc["chunk_count"],
                    ingested_at=doc["ingested_at"],
                )
                for doc in docs
            ],
            total_chunks=doc_store.get_total_chunks(),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """健康检查."""
        doc_store = DocStore()
        vector_store = VectorStore()
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            total_documents=len(doc_store.list_documents()),
            total_chunks=vector_store.count(),
        )

    return app
