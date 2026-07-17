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


def _recall_relevant_memories(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """检索与当前查询语义相关的情景记忆（跨会话历史）."""
    try:
        orchestrator = _get_orchestrator()
        return orchestrator._episodic_memory.recall(query, top_k=top_k, memory_type="conversation")
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 底层逻辑
# ---------------------------------------------------------------------------


def _ingest_files(files: list[str] | None, progress: gr.Progress = gr.Progress()) -> str:
    """上传并摄入文档."""
    if not files:
        return "请先上传文件。"

    orchestrator = _get_orchestrator()
    progress(0, desc="准备摄入...")
    results: list[str] = []

    for i, f in enumerate(files):
        progress((i + 1) / len(files), desc=f"正在摄入: {Path(f).name}")
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


def _ingest_url(url: str, progress: gr.Progress = gr.Progress()) -> str:
    """从 URL 抓取网页并摄入."""
    if not url or not url.strip():
        return "请输入 URL。"

    url = url.strip()
    from knowledge_agent.loaders.url_loader import UrlLoader

    orchestrator = _get_orchestrator()
    progress(0, desc="正在抓取网页...")
    try:
        loader = UrlLoader()
        docs = loader.ingest_url(url)
        if not docs:
            return f"❌ 无法从 {url} 提取内容。"

        # 逐文档摄入
        total_chunks = 0
        for doc in docs:
            progress(0.5, desc="正在分块和向量化...")
            chunks = orchestrator._collection._chunker.chunk(doc.content, doc.metadata)
            if not chunks:
                continue

            import uuid
            chunk_ids = [uuid.uuid4().hex for _ in chunks]
            chunk_texts = [c.text for c in chunks]
            embeddings = orchestrator._collection._embedder.embed(chunk_texts)

            metadatas = []
            for c in chunks:
                meta = dict(c.metadata)
                meta["source"] = doc.source
                metadatas.append(meta)

            orchestrator._collection._vector_store.add(chunks, embeddings, metadatas, chunk_ids)
            total_chunks += len(chunks)

        # 注册到 DocStore
        orchestrator._collection._doc_store.add_document(
            doc_id=uuid.uuid4().hex,
            source=url,
            filename=url.rstrip("/").split("/")[-1] or "webpage",
            file_type="url",
            chunk_count=total_chunks,
            metadata={"source_url": url},
        )

        return f"✅ **{url}**\n  - 文档: {len(docs)}\n  - 分块: {total_chunks}"
    except (ImportError, RuntimeError) as exc:
        return f"❌ {exc}"


def _answer_question(message: str, history: list[dict[str, str]], use_enhanced: bool = False) -> str:
    """RAG 问答（流式），支持多轮对话历史和跨会话记忆检索."""
    if not message or not message.strip():
        return "请输入问题。"

    orchestrator = _get_orchestrator()

    # 检索相关历史记忆，注入到对话上下文中
    relevant_memories = _recall_relevant_memories(message, top_k=3)
    enriched_history = list(history)
    if relevant_memories:
        memory_context = "\n".join(
            f"[Past Q&A: {m.get('text', '')[:200]}]"
            for m in relevant_memories
        )
        # 添加一条系统级的记忆提示
        enriched_history.insert(
            0,
            {
                "role": "system",
                "content": (
                    "以下是从历史对话中检索到的相关记忆，可能对回答有帮助：\n"
                    f"{memory_context}"
                ),
            },
        )

    full_answer = ""
    for chunk in orchestrator.run_query_stream(
        message,
        chat_history=enriched_history,
        use_enhanced_search=use_enhanced,
    ):
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

    # 获取具体文档列表
    from knowledge_agent.storage.doc_store import DocStore
    docs = DocStore().list_documents()
    if docs:
        lines.append("")
        lines.append("### 文档列表")
        lines.append("| ID | 文件名 | 类型 | 版本 | Chunks | 时间 |")
        lines.append("|----|--------|------|------|--------|------|")
        for doc in docs[:50]:
            doc_id = doc["id"][:8] + "..."
            version = doc.get("version", 1)
            lines.append(
                f"| {doc_id} | {doc['filename']} | {doc['file_type']} | "
                f"v{version} | {doc['chunk_count']} | {doc['ingested_at'][:10]} |"
            )
        if len(docs) > 50:
            lines.append(f"\n*... 还有 {len(docs) - 50} 篇文档*")

    return "\n".join(lines)


def _get_document_versions(doc_id: str) -> str:
    """获取文档版本历史."""
    if not doc_id or not doc_id.strip():
        return "请输入文档 ID。"
    orchestrator = _get_orchestrator()
    versions = orchestrator.get_document_versions(doc_id.strip())
    if not versions:
        return f"未找到文档: {doc_id}"

    lines = [f"### 文档版本历史: {versions[0].get('filename', 'unknown')}"]
    for v in versions:
        v_id = v["id"][:8] + "..."
        v_num = v.get("version", "?")
        rolled_back = v.get("metadata", {}).get("rolled_back", False)
        tag = " ⬅️ 当前" if v == versions[0] else ""
        tag += " 🔙 已回滚" if rolled_back else ""
        lines.append(
            f"- v{v_num} | ID: {v_id} | {v['ingested_at'][:10]} | "
            f"{v['chunk_count']} chunks{tag}"
        )
    return "\n".join(lines)


def _delete_document(doc_id: str) -> str:
    """删除指定文档."""
    if not doc_id or not doc_id.strip():
        return "请输入文档 ID。"
    orchestrator = _get_orchestrator()
    success = orchestrator.delete_document(doc_id.strip())
    if success:
        return f"✅ 已删除文档: {doc_id}"
    return f"❌ 未找到文档: {doc_id}"


def _system_health() -> str:
    """系统健康状态."""
    orchestrator = _get_orchestrator()
    report = orchestrator.get_system_report()
    health = orchestrator.get_knowledge_health()
    memory_stats = orchestrator.get_memory_stats()

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
        "## 🧠 记忆系统\n",
        f"**情景记忆**: {memory_stats.get('episodic_count', 0)} 条",
        f"**语义记忆**: {memory_stats.get('semantic_facts', 0)} 条事实",
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


def _run_evaluation(mode: str) -> str:
    """运行 Agent 评估."""
    from knowledge_agent.evaluation import EvaluationRunner

    runner = EvaluationRunner()
    if mode == "retrieval":
        result = runner.evaluate_retrieval(top_k=5)
    else:
        result = runner.evaluate_answer_quality(top_k=5)
    return result.get("summary", result.get("message", "评估完成。"))



def _update_api_key(provider: str, key: str, base_url: str) -> str:
    """更新 API Key 配置（运行时生效，不持久化到 .env）. """
    if not key or not key.strip():
        return f"❌ {provider} API Key 不能为空。"
    try:
        import os
        if provider == "OpenAI":
            os.environ["KA_OPENAI_API_KEY"] = key.strip()
            if base_url:
                os.environ["KA_OPENAI_BASE_URL"] = base_url.strip()
        elif provider == "Anthropic":
            os.environ["KA_ANTHROPIC_API_KEY"] = key.strip()
        return f"✅ {provider} API Key 已更新（当前会话有效）。"
    except Exception as exc:
        return f"❌ 更新失败: {exc}"


def _render_graph() -> str:
    """生成知识图谱的 HTML 可视化."""
    try:
        from pyvis.network import Network
    except ImportError:
        return "pyvis 未安装，请执行: pip install pyvis"

    import tempfile

    orchestrator = _get_orchestrator()
    graph = orchestrator._extraction._graph_store.graph

    if graph.number_of_nodes() == 0:
        return "知识图谱为空，请先摄入文档并执行知识抽取。"

    net = Network(height="600px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")

    type_colors = {
        "person": "#ff6b6b",
        "organization": "#4ecdc4",
        "technology": "#45b7d1",
        "concept": "#f9ca24",
        "location": "#a29bfe",
        "date": "#fd79a8",
        "other": "#dfe6e9",
        "unknown": "#636e72",
    }

    for node, data in graph.nodes(data=True):
        label = data.get("name", node)
        etype = data.get("type", "unknown")
        color = type_colors.get(etype, "#636e72")
        net.add_node(node, label=label, title=label, color=color, size=15)

    for u, v, data in graph.edges(data=True):
        label = data.get("predicate", "related_to")
        net.add_edge(u, v, title=label, label=label, arrows="to", font={"size": 10, "color": "#aaa"})

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8")
    net.save_graph(tmp.name)
    return tmp.name


def _cache_stats() -> str:
    """查询缓存统计."""
    try:
        from knowledge_agent.cache import QueryCache
        cache = QueryCache()
        return f"**查询缓存**: {cache.size} 条 (TTL: 300s, 上限: 100 条)"
    except Exception:
        return "缓存未启用。"


def _clear_memories() -> str:
    """清空情景记忆."""
    try:
        orchestrator = _get_orchestrator()
        orchestrator._episodic_memory.clear()
        return "✅ 已清空所有情景记忆（对话历史）。"
    except Exception as exc:
        return f"❌ 清空失败: {exc}"


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

            gr.Markdown("---\n### 🌐 从 URL 抓取")
            with gr.Row():
                url_input = gr.Textbox(
                    label="网页 URL",
                    placeholder="https://example.com/article",
                    scale=3,
                )
                url_btn = gr.Button("🌐 抓取并摄入", variant="primary", scale=1)
            url_output = gr.Markdown()
            url_btn.click(
                fn=_ingest_url,
                inputs=url_input,
                outputs=url_output,
            )

        with gr.Tab("💬 智能问答"):
            gr.Markdown("### 基于 RAG 的知识问答\n基于已摄入的文档进行检索增强问答。")
            enhance_checkbox = gr.Checkbox(
                label="🔍 搜索增强（查询改写 + HyDE + 多查询融合）",
                value=False,
                info="启用后自动扩展查询，提升检索召回率，但响应会稍慢",
            )
            chatbot = gr.ChatInterface(
                fn=_answer_question,
                title="",
                description="输入问题开始对话",
                type="messages",
                additional_inputs=[enhance_checkbox],
            )
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话历史", variant="stop", size="sm")
                clear_output = gr.Markdown()
            clear_btn.click(
                fn=_clear_memories,
                outputs=clear_output,
            )

        with gr.Tab("⚙️ 设置"):
            gr.Markdown("### API Key 配置\n配置 LLM 和 Embedding 的 API Key（当前会话有效，重启后失效）。")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### OpenAI")
                    openai_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                    openai_url = gr.Textbox(label="Base URL", placeholder="https://api.openai.com/v1")
                    openai_btn = gr.Button("保存 OpenAI 配置", variant="primary")
                    openai_out = gr.Markdown()
                    openai_btn.click(
                        fn=_update_api_key,
                        inputs=[gr.State("OpenAI"), openai_key, openai_url],
                        outputs=openai_out,
                    )
                with gr.Column():
                    gr.Markdown("#### Anthropic")
                    anth_key = gr.Textbox(label="API Key", type="password", placeholder="sk-ant-...")
                    anth_btn = gr.Button("保存 Anthropic 配置", variant="primary")
                    anth_out = gr.Markdown()
                    anth_btn.click(
                        fn=_update_api_key,
                        inputs=[gr.State("Anthropic"), anth_key, gr.State("")],
                        outputs=anth_out,
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

            gr.Markdown("---\n### 📋 版本历史")
            with gr.Row():
                ver_input = gr.Textbox(
                    label="文档 ID",
                    placeholder="输入文档 ID 查看版本历史...",
                    scale=3,
                )
                ver_btn = gr.Button("查看版本", variant="secondary", scale=1)
            ver_output = gr.Markdown()
            ver_btn.click(
                fn=_get_document_versions,
                inputs=ver_input,
                outputs=ver_output,
            )

            gr.Markdown("---\n### 🗑️ 删除文档")
            with gr.Row():
                delete_input = gr.Textbox(
                    label="文档 ID",
                    placeholder="输入要删除的文档 ID...",
                    scale=3,
                )
                delete_btn = gr.Button("删除", variant="stop", scale=1)
            delete_output = gr.Markdown()
            delete_btn.click(
                fn=_delete_document,
                inputs=delete_input,
                outputs=delete_output,
            )

        with gr.Tab("🕸️ 知识图谱"):
            gr.Markdown("### 知识图谱可视化\n展示已提取的实体和关系网络。")
            with gr.Row():
                graph_btn = gr.Button("🔄 生成图谱", variant="primary", scale=1)
                graph_html = gr.HTML(label="知识图谱")
            graph_btn.click(
                fn=_render_graph,
                outputs=graph_html,
            )

        with gr.Tab("📊 系统状态"):
            health_btn = gr.Button("🔄 刷新状态", variant="secondary")
            health_output = gr.Markdown(label="系统健康报告")

            health_btn.click(
                fn=_system_health,
                outputs=health_output,
            )
            demo.load(_system_health, outputs=health_output)

        with gr.Tab("📋 评估"):
            gr.Markdown("### Agent 性能评估\n对已摄入的知识库进行检索质量和答案质量评估。")
            with gr.Row():
                eval_retrieval_btn = gr.Button("🔍 评估检索质量", variant="primary")
                eval_answer_btn = gr.Button("💬 评估答案质量", variant="primary")
            eval_output = gr.Markdown(label="评估结果")

            eval_retrieval_btn.click(
                fn=lambda: _run_evaluation("retrieval"),
                outputs=eval_output,
            )
            eval_answer_btn.click(
                fn=lambda: _run_evaluation("answer"),
                outputs=eval_output,
            )

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
