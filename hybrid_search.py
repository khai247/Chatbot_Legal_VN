"""
Giai đoạn 2 — Hybrid Search song song (Dense + Sparse)

ĐIỂM QUAN TRỌNG NHẤT của module này: filter `status` (còn hiệu lực) và
`metadata_filter` (số hiệu, loại văn bản, năm) phải được áp DỒNG THỜI cho
CẢ HAI luồng Dense và Sparse — đây là lỗ hổng đã phát hiện trong thiết kế
gốc (chỉ lọc ở Dense, Sparse quét toàn bộ) và đã được vá ở đây.

Dense (Qdrant): filter áp ở tầng pre-filtering ngay trong query Qdrant.
Sparse (BM25S): BM25S không có filter native theo payload, nên ta duy trì
một "allowed set" các parent_id hợp lệ (đã lọc theo status/metadata) trong
RAM, intersect với kết quả BM25 trước khi chốt top-k.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import bm25s
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import settings
from schemas import QueryPackage, RetrievalCandidate, VanBanStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding (dùng chung cho Dense query encode)
# ---------------------------------------------------------------------------

class Embedder:
    """Bọc BGE-M3 để encode câu hỏi thành dense vector.

    Lazy-load model để import module này không bắt buộc phải có GPU/torch
    sẵn sàng ngay (hữu ích khi chạy unit test chỉ cần mock).
    """

    _model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            logger.info("Đang load embedding model: %s", settings.embedding_model)
            self._model = BGEM3FlagModel(settings.embedding_model, use_fp16=True)
        return self._model

    def encode(self, text: str) -> list[float]:
        model = self._load()
        out = model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return out["dense_vecs"][0].tolist()


# ---------------------------------------------------------------------------
# Helper: build bộ lọc Qdrant DÙNG CHUNG logic với filter phía Sparse
# ---------------------------------------------------------------------------

def _build_qdrant_filter(query: QueryPackage) -> Filter | None:
    must: list[FieldCondition] = []

    if query.wants_status_filter():
        must.append(FieldCondition(key="status", match=MatchValue(value=VanBanStatus.CON_HIEU_LUC.value)))

    mf = query.metadata_filter
    if mf.so_hieu_van_ban:
        must.append(FieldCondition(key="so_hieu_van_ban", match=MatchValue(value=mf.so_hieu_van_ban)))
    if mf.loai_van_ban:
        must.append(FieldCondition(key="loai_van_ban", match=MatchValue(value=mf.loai_van_ban)))
    if mf.nam_ban_hanh:
        must.append(FieldCondition(key="nam_ban_hanh", match=MatchValue(value=mf.nam_ban_hanh)))

    return Filter(must=must) if must else None


def _parent_passes_filter(parent_meta: dict, query: QueryPackage) -> bool:
    """Áp CÙNG logic filter như `_build_qdrant_filter`, nhưng chạy trên
    metadata trong RAM — dùng cho luồng Sparse (BM25S không hỗ trợ filter
    native theo payload như Qdrant)."""

    if query.wants_status_filter() and parent_meta.get("status") != VanBanStatus.CON_HIEU_LUC.value:
        return False

    mf = query.metadata_filter
    if mf.so_hieu_van_ban and parent_meta.get("so_hieu_van_ban") != mf.so_hieu_van_ban:
        return False
    if mf.loai_van_ban and parent_meta.get("loai_van_ban") != mf.loai_van_ban:
        return False
    if mf.nam_ban_hanh and parent_meta.get("nam_ban_hanh") != mf.nam_ban_hanh:
        return False

    return True


# ---------------------------------------------------------------------------
# Luồng A — Dense retrieval (Qdrant)
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(self, client: QdrantClient | None = None, embedder: Embedder | None = None):
        self._client = client or QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._embedder = embedder or Embedder()

    def search(self, query: QueryPackage, top_k: int = settings.top_k_dense) -> list[RetrievalCandidate]:
        vector = self._embedder.encode(query.raw_question)
        qdrant_filter = _build_qdrant_filter(query)

        hits = self._client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            query_filter=qdrant_filter,  # pre-filtering: lọc ở tầng index, không phải sau khi search
            limit=top_k,
            with_payload=["parent_id"],
        )

        # Child-to-Parent Mapper: nhiều child có thể trỏ về cùng 1 parent_id,
        # chỉ giữ điểm cao nhất cho mỗi parent.
        best_by_parent: dict[str, float] = {}
        for hit in hits:
            parent_id = hit.payload["parent_id"]
            if parent_id not in best_by_parent or hit.score > best_by_parent[parent_id]:
                best_by_parent[parent_id] = hit.score

        candidates = [
            RetrievalCandidate(parent_id=pid, score=score, source="dense")
            for pid, score in best_by_parent.items()
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Luồng B — Sparse retrieval (BM25S)
# ---------------------------------------------------------------------------

@dataclass
class Bm25Corpus:
    """Corpus BM25 đã build sẵn (offline, ở bước ingestion/index-building),
    load lại khi khởi động service."""

    retriever: bm25s.BM25
    parent_ids: list[str]              # song song với thứ tự document trong index
    parent_metadata: dict[str, dict]   # parent_id -> {status, so_hieu_van_ban, ...}

    @classmethod
    def load(cls, index_dir: str, metadata: dict[str, dict]) -> "Bm25Corpus":
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        # corpus được lưu dưới dạng list các dict {"id": parent_id, "text": ...}
        parent_ids = [doc["id"] for doc in retriever.corpus]
        return cls(retriever=retriever, parent_ids=parent_ids, parent_metadata=metadata)


class SparseRetriever:
    def __init__(self, corpus: Bm25Corpus):
        self._corpus = corpus

    def search(self, query: QueryPackage, top_k: int = settings.top_k_sparse) -> list[RetrievalCandidate]:
        query_tokens = bm25s.tokenize(query.search_text(), stopwords="vi" if _has_vi_stopwords() else None)

        # Lấy dư (over-fetch) trước khi lọc, vì sau khi áp filter status/metadata
        # số lượng còn lại có thể ít hơn top_k mong muốn.
        over_fetch_k = min(top_k * 4, len(self._corpus.parent_ids))
        if over_fetch_k == 0:
            return []

        results, scores = self._corpus.retriever.retrieve(query_tokens, k=over_fetch_k)

        candidates: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for doc_idx, score in zip(results[0], scores[0]):
            parent_id = self._corpus.parent_ids[doc_idx]
            if parent_id in seen:
                continue
            seen.add(parent_id)

            meta = self._corpus.parent_metadata.get(parent_id, {})
            if not _parent_passes_filter(meta, query):  # <-- filter ĐỒNG BỘ với Dense
                continue

            candidates.append(RetrievalCandidate(parent_id=parent_id, score=float(score), source="sparse"))
            if len(candidates) >= top_k:
                break

        return candidates


def _has_vi_stopwords() -> bool:
    """bm25s không có stopword list tiếng Việt built-in; trả False để bỏ qua
    bước loại stopword thay vì lỗi ngầm. Nhóm có thể cắm bộ stopword tiếng
    Việt riêng ở đây nếu cần."""
    return False


# ---------------------------------------------------------------------------
# Orchestrator: chạy song song 2 luồng
# ---------------------------------------------------------------------------

class HybridSearchOrchestrator:
    """
    Chạy Dense và Sparse song song.

    Lưu ý hiệu năng: Qdrant call là I/O-bound (network) nên ThreadPoolExecutor
    phù hợp. BM25S là CPU-bound (numpy) — nếu benchmark thực tế cho thấy
    threading không đạt speedup như kỳ vọng (do tranh chấp GIL), đổi
    `executor_cls` sang `ProcessPoolExecutor` khi khởi tạo class này.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        executor_cls=ThreadPoolExecutor,
    ):
        self._dense = dense
        self._sparse = sparse
        self._executor_cls = executor_cls

    def search(self, query: QueryPackage) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
        with self._executor_cls(max_workers=2) as executor:
            dense_future = executor.submit(self._dense.search, query, settings.top_k_dense)
            sparse_future = executor.submit(self._sparse.search, query, settings.top_k_sparse)

            dense_results = dense_future.result()
            sparse_results = sparse_future.result()

        logger.info(
            "Hybrid search | dense=%d sparse=%d temporal_intent=%s",
            len(dense_results), len(sparse_results), query.temporal_intent,
        )
        return dense_results, sparse_results
