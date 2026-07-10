"""Gradio Web UI — 知识沉淀 Agent 可视化界面."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from knowledge_agent.agents.orchestrator import Orchestrator

_ORCHESTRATOR: Orchestrator | None = None


def _get_orchestrator() -> Orchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = Orchestrator()
    return _ORCHESTRATOR


# ---------------------------------------------------------------------------
# 底层逻辑
# ---------------------------------------------------------------------------


def _ingest_files(files: list[str] | None) -> str:
    """上传并摄入文档."""
    if not files:
        return "请先上传文件。"

    orchestrator = _get_orchestrator()
    results: list[str] = []

    for f in files:
        path = Path(f)
        if not path.exists():
            results.append(f"❌ {path.name}: 文件不存在")
            continue
        try:
            result = orchestrator.run_full_pipeline(
                path,
                enable_extraction=True,
                enable_quality_check=True,
            )
            if result.success:
                ingest = result.results.get("ingest", {})
                extract = result.results.get("extraction", {})
                parts = [
                    f"✅ **{path.name}**",
                    f"  - 文档: {ingest.get('documents_loaded', 0)}",
                    f"  - 分块: {ingest.get('chunks_created', 0)}",
                ]
                if extract:
                    parts.append(
                        f"  - 实体: {extract.get('entities_found', 0)}, "
                        f"关系: {extract.get('relations_found', 0)}"
                    )
                results.append("\n".join(parts))
            else:
                results.append(f"❌ {path.name}: {result.summary}")
        except Exception as exc:
            results.append(f"❌ {path.name}: {exc}")

    return "\n\n".join(results) if results else "未处理任何文件。"


def _answer_question(message: str, history: list[dict[str, str]]) -> str:
    """RAG 问答（流式）."""
    if not message or not message.strip():
        return "请输入问题。"

    orchestrator = _get_orchestrator()

    full_answer = ""
    for chunk in orchestrator.run_query_stream(message):
        full_answer += chunk
        yield full_answer


def _list_documents() -> str:
    """列出已摄入的文档."""
    orchestrator = _get_orchestrator()
    stats = orchestrator.get_system_report()
    storage = stats.get("storage", {})

    total_docs = storage.get("total_documents", 0)
    total_chunks = storage.get("total_chunks", 0)

    lines = [
        f"📚 **文档总数**: {total_docs}",
        f"🧩 **分块总数**: {total_chunks}",
    ]
    return "\n".join(lines)


def _system_health() -> str:
    """系统健康状态."""
    orchestrator = _get_orchestrator()
    report = orchestrator.get_system_report()
    health = orchestrator.get_knowledge_health()

    storage = report.get("storage", {})
    graph = report.get("graph", {})
    quality = report.get("quality", {})

    freshness_dist = health.get("freshness_distribution", {})

    lines = [
        "## 🖥️ 系统状态\n",
        f"**向量库**: {storage.get('vector_store_size', 0)} 条",
        f"**文档库**: {storage.get('total_documents', 0)} 篇",
        f"**知识图谱**: {graph.get('nodes', 0)} 节点 / {graph.get('edges', 0)} 边",
        "",
        "## 📊 健康状态\n",
        f"**过期文档**: {quality.get('expired_documents', 0)}",
        f"**知识缺口**: {quality.get('knowledge_gaps', 0)}",
        f"**新鲜度分布**:",
        f"  - 高 (>0.7): {freshness_dist.get('high (>0.7)', 0)}",
        f"  - 中 (0.3-0.7): {freshness_dist.get('medium (0.3-0.7)', 0)}",
        f"  - 低 (<0.3): {freshness_dist.get('low (<0.3)', 0)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------


def create_ui() -> gr.Blocks:
    """创建 Gradio Web UI."""
    with gr.Blocks(
        title="知识沉淀 Agent",
        theme=gr.themes.Soft(),
        css="""
        footer { display: none !important; }
        .gradio-container { max-width: 1200px !important; }
        """,
    ) as demo:
        gr.Markdown(
            "# 🧠 知识沉淀 Agent\n\n"
            "多源采集 + LLM 结构化抽取 + 混合存储 + 多 Agent 协作 + 持久化记忆"
        )

        with gr.Tab("📥 摄入文档"):
            gr.Markdown("### 上传文档文件\n支持 .txt / .md / .pdf / .log / .csv / .json 格式")
            file_input = gr.File(
                label="选择文件",
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf", ".log", ".csv", ".json"],
            )
            ingest_btn = gr.Button("🚀 开始摄入", variant="primary")
            ingest_output = gr.Markdown(label="摄入结果")

            ingest_btn.click(
                fn=_ingest_files,
                inputs=file_input,
                outputs=ingest_output,
            )

        with gr.Tab("💬 智能问答"):
            gr.Markdown("### 基于 RAG 的知识问答\n基于已摄入的文档进行检索增强问答。")
            chatbot = gr.ChatInterface(
                fn=_answer_question,
                title="",
                description="输入问题开始对话",
                type="messages",
            )

        with gr.Tab("📚 文档列表"):
            refresh_btn = gr.Button("🔄 刷新", variant="secondary")
            doc_output = gr.Markdown(label="文档信息")

            refresh_btn.click(
                fn=_list_documents,
                outputs=doc_output,
            )
            # 页面加载时自动显示
            demo.load(_list_documents, outputs=doc_output)

        with gr.Tab("📊 系统状态"):
            health_btn = gr.Button("🔄 刷新状态", variant="secondary")
            health_output = gr.Markdown(label="系统健康报告")

            health_btn.click(
                fn=_system_health,
                outputs=health_output,
            )
            demo.load(_system_health, outputs=health_output)

    return demo


def main() -> None:
    """启动 Gradio Web UI."""
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
