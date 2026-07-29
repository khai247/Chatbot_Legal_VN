"""
Cấu hình tập trung cho Phân hệ 2 (Advanced Retrieval & Agent).

Mọi tham số tuning (top-k, ngưỡng intersection, tên model...) đều đọc từ
biến môi trường (.env) để 3 thành viên trong nhóm không phải sửa code khi
đổi model / đổi threshold lúc thử nghiệm.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_query_model: str = "qwen2.5:3.5b"
    ollama_gen_model: str = "qwen3:8b"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "hspl_child_chunks"

    # --- Embedding / Reranker ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_tokens: int = 1024

    # --- Retrieval tuning (đúng theo sơ đồ kiến trúc đã chốt) ---
    top_k_dense: int = 50
    top_k_sparse: int = 50
    intersection_min: int = 15
    fallback_top_n: int = 5
    rerank_top_n: int = 5

    # --- Parent store (MongoDB) ---
    parent_store_uri: str = "mongodb://localhost:27017"
    parent_store_db: str = "hspl"
    parent_store_collection: str = "parent_articles"


settings = Settings()
