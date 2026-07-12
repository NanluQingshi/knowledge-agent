"""FastAPI 路由 — 知识沉淀 Agent REST API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from knowledge_agent.config import settings
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


class DeleteResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_orchestrator():
    """延迟导入并返回 Orchestrator 实例."""
    from knowledge_agent.agents.orchestrator import Orchestrator

    return Orchestrator()


# ---------------------------------------------------------------------------
# 评估端点
# ---------------------------------------------------------------------------


class EvalRetrievalRequest(BaseModel):
    top_k: int = settings.retrieval_top_k
    category: str | None = None


class EvalResponse(BaseModel):
    status: str
    summary: str
    num_queries: int
    details: list[dict[str, Any]] = []


@app.post("/evaluate/retrieval", response_model=EvalResponse)
async def evaluate_retrieval(req: EvalRetrievalRequest):
    """评估检索质量."""
    from knowledge_agent.evaluation import EvaluationRunner

    runner = EvaluationRunner()
    result = runner.evaluate_retrieval(top_k=req.top_k, category=req.category)
    return EvalResponse(
        status=result.get("status", "ok"),
        summary=result.get("summary", ""),
        num_queries=result.get("num_queries", 0),
        details=result.get("details", []),
    )


class EvalAnswerRequest(BaseModel):
    top_k: int = settings.retrieval_top_k
    category: str | None = None


@app.post("/evaluate/answer", response_model=EvalResponse)
async def evaluate_answer(req: EvalAnswerRequest):
    """评估答案质量."""
    from knowledge_agent.evaluation import EvaluationRunner

    runner = EvaluationRunner()
    result = runner.evaluate_answer_quality(top_k=req.top_k, category=req.category)
    return EvalResponse(
        status=result.get("status", "ok"),
        summary=result.get("summary", ""),
        num_queries=result.get("num_queries", 0),
        details=result.get("details", []),
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用."""

    app = FastAPI(
        title="Knowledge Agent API",
        description="知识沉淀 Agent — 文档摄入、向量检索、智能问答",
        version="0.1.0",
    )

    # CORS 配置 — 允许所有来源（开发环境）；生产环境请限制 origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_file(file: UploadFile = File(None)):
        """摄入文档文件."""
        if file is None:
            raise HTTPException(status_code=400, detail="No file provided")

        # 保存临时文件
        tmp_path = Path(settings.data_dir) / "tmp"
        tmp_path.mkdir(parents=True, exist_ok=True)
        file_path = tmp_path / (file.filename or "uploaded_file")
        content = await file.read()
        file_path.write_bytes(content)

        orchestrator = _get_orchestrator()
        try:
            result = orchestrator.run_full_pipeline(
                file_path,
                enable_extraction=True,
                enable_quality_check=True,
            )
            ingest_result = result.results.get("ingest", {})

            return IngestResponse(
                status="ok",
                documents_loaded=ingest_result.get("documents_loaded", 0),
                chunks_created=ingest_result.get("chunks_created", 0),
                total_chunks_in_store=VectorStore().count(),
            )
        finally:
            # 清理临时文件
            file_path.unlink(missing_ok=True)

    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest):
        """问答查询."""
        from knowledge_agent.agents.orchestrator import Orchestrator

        orchestrator = _get_orchestrator()
        result = orchestrator.run_query(req.question, top_k=req.top_k)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    @app.post("/query/stream")
    async def query_stream_endpoint(req: QueryRequest):
        """流式问答查询（Server-Sent Events）."""
        from fastapi.responses import StreamingResponse

        from knowledge_agent.agents.orchestrator import Orchestrator

        orchestrator = _get_orchestrator()

        async def event_stream():
            # 先发送来源信息
            # (为简化，先执行一次非流式查询获取 sources)
            result = orchestrator.run_query(req.question, top_k=req.top_k)
            import json
            sources_data = [
                {"text": s.get("text", "")[:200], "source": s.get("metadata", {}).get("source", "")}
                for s in result.get("sources", [])
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data}, ensure_ascii=False)}\n\n"

            # 流式输出回答
            for chunk in orchestrator.run_query_stream(req.question, top_k=req.top_k):
                yield f"data: {json.dumps({'type': 'token', 'data': chunk}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/documents", response_model=DocumentListResponse)
    async def list_documents(offset: int = 0, limit: int = 100):
        """列出已摄入的文档（支持分页）.

        Args:
            offset: 偏移量，默认 0.
            limit: 返回条数上限，默认 100，最大 1000.
        """
        from knowledge_agent.storage.doc_store import DocStore

        limit = min(limit, 1000)
        doc_store = DocStore()
        all_docs = doc_store.list_documents()
        docs_page = all_docs[offset : offset + limit]

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
                for doc in docs_page
            ],
            total_chunks=doc_store.get_total_chunks(),
        )

    @app.delete("/documents/{doc_id}", response_model=DeleteResponse)
    async def delete_document(doc_id: str):
        """删除指定文档."""
        orchestrator = _get_orchestrator()
        success = orchestrator.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return DeleteResponse(status="ok", message=f"Deleted document: {doc_id}")

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """健康检查."""
        from knowledge_agent.storage.doc_store import DocStore

        doc_store = DocStore()
        vector_store = VectorStore()
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            total_documents=len(doc_store.list_documents()),
            total_chunks=vector_store.count(),
        )

    return app
