"""CLI 入口 — 知识沉淀 Agent 命令行工具.

用法:
    ka ingest <path>      摄入文档（文件或目录）
    ka query <question>   提问
    ka delete <doc-id>    删除文档
    ka serve              启动 REST API 服务
    ka webui              启动 Gradio Web UI
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from knowledge_agent.config import settings

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """知识沉淀 Agent — 文档摄入、向量检索、智能问答."""


@cli.command("ingest")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--extract/--no-extract", default=True, help="是否执行知识抽取（实体/关系提取）")
@click.option("--quality/--no-quality", default=True, help="是否执行质检（过期检测/缺口分析）")
def ingest(path: Path, extract: bool, quality: bool) -> None:
    """摄入文档 — 加载、分块、向量化、抽取、存储."""
    from knowledge_agent.agents.orchestrator import Orchestrator

    console.print(f"[bold]Ingesting from: {path}[/bold]")
    orchestrator = Orchestrator()
    result = orchestrator.run_full_pipeline(
        path,
        enable_extraction=extract,
        enable_quality_check=quality,
    )

    ingest_result = result.results.get("ingest", {})
    console.print(f"  Loaded [green]{ingest_result.get('documents_loaded', 0)}[/green] document sections")
    console.print(f"  Created [green]{ingest_result.get('chunks_created', 0)}[/green] chunks")

    if extract and "extraction" in result.results:
        ext = result.results["extraction"]
        console.print(
            f"  Extracted [green]{ext.get('entities_found', 0)}[/green] entities, "
            f"[green]{ext.get('relations_found', 0)}[/green] relations"
        )

    if quality and "quality" in result.results:
        q = result.results["quality"]
        console.print(
            f"  Quality: [yellow]{q.get('expired_documents', 0)}[/yellow] expired, "
            f"[yellow]{q.get('knowledge_gaps', 0)}[/yellow] knowledge gaps"
        )

    if result.errors:
        console.print(f"\n[yellow]{len(result.errors)} warning(s):[/yellow]")
        for err in result.errors:
            console.print(f"  {err.get('file', err.get('step', '?'))}: {err['error']}")

    console.print(f"\n[bold green]{result.summary}[/bold green]")


@cli.command("query")
@click.argument("question", type=str)
@click.option("--top-k", default=settings.retrieval_top_k, help="检索结果数量")
@click.option("--graphrag", is_flag=True, default=False, help="启用 GraphRAG 增强检索")
def query(question: str, top_k: int, graphrag: bool) -> None:
    """提问 — RAG 检索增强问答."""
    import time
    from knowledge_agent.agents.orchestrator import Orchestrator

    console.print(f"[bold]Question:[/bold] {question}\n")
    console.print("[bold]Retrieving context...[/bold]")

    start = time.perf_counter()
    orchestrator = Orchestrator()
    result = orchestrator.run_query(question, top_k=top_k, use_graphrag=graphrag)
    elapsed = (time.perf_counter() - start) * 1000

    console.print(f"\n[bold green]Answer:[/bold green]\n{result['answer']}\n")
    console.print(f"[dim]Query took {elapsed:.0f}ms[/dim]")

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


@cli.command("eval")
@click.argument("mode", type=click.Choice(["retrieval", "answer", "all"]), default="all")
@click.option("--top-k", default=settings.retrieval_top_k, help="检索深度")
@click.option("--category", default=None, help="按类别过滤评估条目")
@click.option("--dataset", default=None, help="外部评估数据集 JSON 文件路径")
def eval_cmd(mode: str, top_k: int, category: str | None, dataset: str | None) -> None:
    """评估 Agent 性能 — 检索质量 / 答案质量."""
    from knowledge_agent.evaluation import EvaluationDataset, EvaluationRunner

    if dataset:
        ds = EvaluationDataset.from_file(dataset)
    else:
        ds = EvaluationDataset()

    runner = EvaluationRunner(dataset=ds)

    if mode in ("retrieval", "all"):
        console.print("[bold]🔍 Retrieval Quality Evaluation[/bold]\n")
        ret_result = runner.evaluate_retrieval(top_k=top_k, category=category)
        console.print(ret_result.get("summary", ret_result.get("message", "")))
        console.print()

    if mode in ("answer", "all"):
        console.print("[bold]💬 Answer Quality Evaluation[/bold]\n")
        ans_result = runner.evaluate_answer_quality(top_k=top_k, category=category)
        console.print(ans_result.get("summary", ans_result.get("message", "")))
        console.print()


@cli.group("eval-dataset")
def eval_dataset() -> None:
    """管理评估数据集."""


@eval_dataset.command("add")
@click.option("--query", required=True, help="问题文本")
@click.option("--expected-doc-ids", default="", help="期望文档 ID（逗号分隔）")
@click.option("--expected-answer", default="", help="期望回答")
@click.option("--category", default="general", help="类别")
@click.option("--difficulty", default="medium", help="难度 (easy/medium/hard)")
def eval_dataset_add(query: str, expected_doc_ids: str, expected_answer: str, category: str, difficulty: str) -> None:
    """添加一条评估条目."""
    ds = EvaluationDataset()
    doc_ids = [x.strip() for x in expected_doc_ids.split(",") if x.strip()]
    item_id = ds.add_item(
        query=query,
        expected_doc_ids=doc_ids,
        expected_answer=expected_answer,
        category=category,
        difficulty=difficulty,
    )
    console.print(f"[bold green]Added evaluation item: {item_id}[/bold green]")


@eval_dataset.command("list")
@click.option("--category", default=None, help="按类别过滤")
def eval_dataset_list(category: str | None) -> None:
    """列出评估数据集条目."""
    ds = EvaluationDataset()
    items = ds.list_items(category=category)
    if not items:
        console.print("[yellow]No evaluation items found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", width=12)
    table.add_column("Query", max_width=50)
    table.add_column("Category", width=12)
    table.add_column("Difficulty", width=10)
    table.add_column("Expected Docs", width=12)
    for item in items:
        doc_count = len(item.get("expected_doc_ids", []))
        table.add_row(
            item["id"][:8] + "...",
            item["query"][:50],
            item.get("category", ""),
            item.get("difficulty", ""),
            str(doc_count),
        )
    console.print(table)
    console.print(f"\nTotal: {len(items)} items")


@eval_dataset.command("clear")
def eval_dataset_clear() -> None:
    """清空评估数据集."""
    ds = EvaluationDataset()
    ds.clear()
    console.print("[green]Evaluation dataset cleared.[/green]")


@eval_dataset.command("export")
@click.argument("output_path", type=str)
def eval_dataset_export(output_path: str) -> None:
    """导出评估数据集到 JSON 文件."""
    ds = EvaluationDataset()
    ds.export_to(output_path)
    console.print(f"[green]Exported {ds.size} items to {output_path}[/green]")
@click.argument("doc_id", type=str)
def delete(doc_id: str) -> None:
    """删除指定文档（从向量库和元数据存储中移除）."""
    from knowledge_agent.agents.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    result = orchestrator.delete_document(doc_id)
    if result:
        console.print(f"[bold green]Deleted document: {doc_id}[/bold green]")
    else:
        console.print(f"[red]Document not found: {doc_id}[/red]")


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
    console.print("  POST /ingest           — 摄入文档")
    console.print("  POST /query            — 问答查询")
    console.print("  GET  /documents        — 列出已摄入文档")
    console.print("  DELETE /documents/{id} — 删除文档")
    console.print("  GET  /health           — 健康检查")
    uvicorn.run(app, host=host, port=port, reload=reload)


@cli.command("webui")
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", default=7860, help="监听端口")
@click.option("--share", is_flag=True, default=False, help="生成公共链接")
def webui_cmd(host: str, port: int, share: bool) -> None:
    """启动 Gradio Web UI."""
    from knowledge_agent.webui import create_ui

    demo = create_ui()
    console.print(f"[bold]Starting Web UI at http://{host}:{port}[/bold]")
    console.print(f"  Share: {'enabled' if share else 'disabled'}")
    demo.launch(server_name=host, server_port=port, share=share)


def main() -> None:
    """入口函数."""
    from knowledge_agent.monitoring.logger import setup_logging
    setup_logging()
    cli()


if __name__ == "__main__":
    main()
