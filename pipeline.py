"""
Pipeline tổng của Phân hệ 2 — nối 5 giai đoạn lại thành một hàm duy nhất
mà Phân hệ 3 (Backend/UI) sẽ gọi vào.

    query = pipeline.run("Mua đất nông nghiệp cần giấy tờ gì?")

Khởi tạo các thành phần (model, kết nối DB) một lần khi service start,
KHÔNG khởi tạo lại trong mỗi request — các model embedding/reranker load
khá nặng.
"""

from __future__ import annotations

import logging
import time

from agent_prompt import AnswerGenerator
from hybrid_search import Bm25Corpus, DenseRetriever, Embedder, HybridSearchOrchestrator, SparseRetriever
from intersection_fallback import IntersectionFallbackPool
from parent_store import ParentStore
from query_analyzer import QueryAnalyzer
from reranker import Reranker
from schemas import AnswerResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class Phase2Pipeline:
    def __init__(self, bm25_index_dir: str):
        logger.info("Khởi tạo Phase2Pipeline...")

        self._parent_store = ParentStore()

        self._query_analyzer = QueryAnalyzer()

        embedder = Embedder()
        self._dense = DenseRetriever(embedder=embedder)

        bm25_metadata = self._parent_store.get_lightweight_metadata_index()
        bm25_corpus = Bm25Corpus.load(bm25_index_dir, metadata=bm25_metadata)
        self._sparse = SparseRetriever(corpus=bm25_corpus)

        self._hybrid = HybridSearchOrchestrator(dense=self._dense, sparse=self._sparse)
        self._pool_builder = IntersectionFallbackPool(parent_store=self._parent_store)
        self._reranker = Reranker()
        self._generator = AnswerGenerator()

        logger.info("Phase2Pipeline sẵn sàng.")

    def run(self, question: str) -> AnswerResult:
        t0 = time.perf_counter()

        # Giai đoạn 1
        query_package = self._query_analyzer.analyze(question)

        # Giai đoạn 2
        dense_results, sparse_results = self._hybrid.search(query_package)

        # Giai đoạn 3
        pool, parent_docs = self._pool_builder.build(query_package, dense_results, sparse_results)

        # Giai đoạn 4
        top_docs = self._reranker.rerank(query_package, pool, parent_docs)

        # Giai đoạn 5
        result = self._generator.generate(query_package, top_docs)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Pipeline hoàn tất trong %.2fs | dense=%d sparse=%d pool=%d top=%d",
            elapsed, len(dense_results), len(sparse_results), len(pool), len(top_docs),
        )
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chạy thử Phase 2 pipeline")
    parser.add_argument("question", type=str, help="Câu hỏi pháp luật cần tra cứu")
    parser.add_argument("--bm25-index", type=str, default="./bm25_index", help="Thư mục chứa BM25 index đã build")
    args = parser.parse_args()

    pipeline = Phase2Pipeline(bm25_index_dir=args.bm25_index)
    result = pipeline.run(args.question)

    print("\n=== TRẢ LỜI ===")
    print(result.answer)
    if result.warning:
        print(f"\n⚠ {result.warning}")
    print("\n=== TRÍCH DẪN ===")
    for c in result.citations:
        print(f"- {c.so_hieu_van_ban} | {c.ten_dieu} | {c.status.value}")
