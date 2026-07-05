"""CLI 入口 — 知识沉淀 Agent 命令行工具.

用法:
    ka ingest <path>      摄入文档（文件或目录）
    ka query <question>   提问
    ka serve              启动 REST API 服务
"""

from __future__ import annotations

import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from knowledge_agent.chunkers.recursive_chunker import RecursiveChunker
from knowledge_agent.config import settings
from knowledge_agent.embeddings.embedder import Embedder
from knowledge_agent.loaders import all_loaders
from knowledge_agent.loaders.base import Document
from knowledge_agent.storage.doc_store import DocStore
from knowledge_agent.storage.vector_store import VectorStore

console = Console()


def _load_documents(path: Path) -> list[Document]:
    """从路径加载文档（支持单文件和目录递归遍历）."""
    docs: list[Document] = []
    loaders = all_loaders()
    files: list[Path] = []

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        raise click.BadParameter(f"Path does not exist: {path}")

    for fp in files:
        for loader in loaders:
            if loader.can_handle(fp):
                try:
                    loaded = loader.load(fp)
                    docs.extend(loaded)
                except Exception as exc:
                    console.print(f"[yellow]Warning: failed to load {fp}: {exc}[/yellow]")

    return docs


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """知识沉淀 Agent — 文档摄入、向量检索、智能问答."""


@cli.command("ingest")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--chunk-size", default=settings.chunk_size, help="分块大小 (tokens)")
@click.option("--chunk-overlap", default=settings.chunk_overlap, help="分块重叠 (tokens)")
def ingest(path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """摄入文档 — 加载、分块、向量化、存储."""
    # 1. 加载文档
    console.print(f"[bold]Loading documents from: {path}[/bold]")
    docs = _load_documents(path)

    if not docs:
        console.print("[red]No supported documents found.[/red]")
        return

    console.print(f"  Loaded [green]{len(docs)}[/green] document sections")

    # 2. 分块
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc.content, metadata={
            "source": doc.source,
            "filename": doc.metadata.get("filename", ""),
        })
        all_chunks.extend(chunks)

    console.print(f"  Created [green]{len(all_chunks)}[/green] chunks")

    # 3. 向量化
    console.print("  Embedding chunks...")
    embedder = Embedder()
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed(texts)
    console.print(f"  Embeddings: [green]{len(embeddings)}[/green] vectors")

    # 4. 存储到向量数据库
    vector_store = VectorStore()
    chunk_ids = [str(uuid.uuid4()) for _ in all_chunks]
    metadatas = [{"chunk_index": i, **chunk.metadata} for i, chunk in enumerate(all_chunks)]
    vector_store.add(
        chunks=all_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=chunk_ids,
    )

    # 5. 存储文档元数据
    doc_store = DocStore()
    # Group by source file
    by_source: dict[str, list] = {}
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

    console.print(f"\n[bold green]Ingested {len(docs)} documents → {len(all_chunks)} chunks[/bold green]")
    console.print(f"  Vector store total: {vector_store.count()}")
    console.print(f"  Doc store total chunks: {doc_store.get_total_chunks()}")


@cli.command("query")
@click.argument("question", type=str)
@click.option("--top-k", default=settings.retrieval_top_k, help="检索结果数量")
def query(question: str, top_k: int) -> None:
    """提问 — RAG 检索增强问答."""
    from knowledge_agent.agents.qa_agent import QAAgent
    from knowledge_agent.retrieval.bm25_retriever import BM25Retriever
    from knowledge_agent.retrieval.hybrid_retriever import HybridRetriever
    from knowledge_agent.retrieval.vector_retriever import VectorRetriever

    # 1. 设置检索器
    vector_store = VectorStore()
    embedder = Embedder()

    vector_retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
    bm25_retriever = BM25Retriever()

    # 从 VectorStore 构建 BM25 索引
    if vector_store.count() > 0:
        # 获取所有文档用于 BM25 — 使用 get_all_documents 避免全量向量扫描
        corpus = vector_store.get_all_documents()
        if corpus:
            bm25_retriever.index(corpus)

    hybrid = HybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)
    agent = QAAgent(hybrid_retriever=hybrid)

    # 2. 查询
    console.print(f"[bold]Question:[/bold] {question}\n")
    console.print("[bold]Retrieving context...[/bold]")

    result = agent.query(question, top_k=top_k)

    # 3. 输出
    console.print(f"\n[bold green]Answer:[/bold green]\n{result['answer']}\n")

    if result.get("sources"):
        console.print("[bold]Sources:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Content", max_width=80)
        table.add_column("Source", max_width=40)
        for i, src in enumerate(result["sources"], start=1):
            preview = src["text"][:200].replace("\n", " ") + ("..." if len(src["text"]) > 200 else "")
            source_name = src.get("metadata", {}).get("source", src.get("metadata", {}).get("filename", "unknown"))
            table.add_row(str(i), preview, source_name)
        console.print(table)


@cli.command("serve")
@click.option("--host", default=settings.api_host, help="监听地址")
@click.option("--port", default=settings.api_port, help="监听端口")
@click.option("--reload", is_flag=True, default=False, help="开发模式热重载")
def serve_cmd(host: str, port: int, reload: bool) -> None:
    """启动 REST API 服务."""
    import uvicorn
    from knowledge_agent.api.routes import create_app

    app = create_app()
    console.print(f"[bold]Starting API server at http://{host}:{port}[/bold]")
    console.print("Endpoints:")
    console.print("  POST /ingest  — 摄入文档")
    console.print("  POST /query   — 问答查询")
    console.print("  GET  /documents — 列出已摄入文档")
    console.print("  GET  /health  — 健康检查")
    uvicorn.run(app, host=host, port=port, reload=reload)


def main() -> None:
    """入口函数."""
    cli()


if __name__ == "__main__":
    main()
