"""配置管理模块 — 支持环境变量和 .env 文件."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置，所有字段可从环境变量 / .env 文件读取."""

    # --- LLM ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Anthropic ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # --- Storage ---
    chroma_persist_dir: str = "./data/chroma"
    doc_db_path: str = "./data/docs.db"
    graph_db_path: str = "./data/graph.json"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Retrieval ---
    retrieval_top_k: int = 5
    hybrid_weight_vector: float = 0.6
    hybrid_weight_bm25: float = 0.4

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # --- Data ---
    data_dir: str = "./data/documents"

    model_config = {"env_prefix": "KA_", "env_file": ".env"}


settings = Settings()
