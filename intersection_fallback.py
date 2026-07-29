"""
Giai đoạn 3 — Intersection Pool & Fallback

1. Giao (intersection) parent_id giữa Dense và Sparse -> giữ Điều luật vừa
   đúng ngữ nghĩa vừa chứa từ khoá chính xác.
2. Nếu pool < intersection_min (mặc định 15), bù thêm top-N (mặc định 5)
   của Dense thuần và Sparse thuần để không mất recall.
3. Defense-in-depth: dù Giai đoạn 2 đã lọc status/metadata, vẫn kiểm tra
   lại lần 2 với ParentStore trước khi đưa sang Reranker — phòng trường
   hợp BM25 index bị stale (văn bản mới đổi status nhưng chưa re-index).

Ghi chú: nhóm đã thử nghiệm Union+RRF so với Intersection+Fallback và xác
nhận Intersection+Fallback cho kết quả tốt hơn trên dataset thực tế, nên
giữ nguyên cách tiếp cận Intersection thay vì đổi sang RRF.
"""

from __future__ import annotations

import logging

from config import settings
from parent_store import ParentStore
from schemas import ParentDocument, QueryPackage, RetrievalCandidate, VanBanStatus

logger = logging.getLogger(__name__)


class IntersectionFallbackPool:
    def __init__(self, parent_store: ParentStore):
        self._parent_store = parent_store

    @staticmethod
    def _intersect(
        dense: list[RetrievalCandidate], sparse: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        sparse_ids = {c.parent_id for c in sparse}
        dense_by_id = {c.parent_id: c for c in dense}
        sparse_by_id = {c.parent_id: c for c in sparse}

        merged: list[RetrievalCandidate] = []
        for parent_id in dense_by_id.keys() & sparse_ids:
            # điểm hợp nhất: trung bình cộng đơn giản của 2 điểm (đã ở 2 thang
            # đo khác nhau — chỉ dùng để sắp xếp tương đối trong pool, không
            # phải điểm cuối cùng; Reranker ở Giai đoạn 4 mới là điểm quyết định)
            avg_score = (dense_by_id[parent_id].score + sparse_by_id[parent_id].score) / 2
            merged.append(RetrievalCandidate(parent_id=parent_id, score=avg_score, source="both"))

        merged.sort(key=lambda c: c.score, reverse=True)
        return merged

    def _apply_fallback(
        self,
        pool: list[RetrievalCandidate],
        dense: list[RetrievalCandidate],
        sparse: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        if len(pool) >= settings.intersection_min:
            return pool

        logger.info(
            "Intersection pool chỉ có %d (< %d) — kích hoạt fallback",
            len(pool), settings.intersection_min,
        )

        existing_ids = {c.parent_id for c in pool}
        extra: list[RetrievalCandidate] = []

        for candidate in dense[: settings.fallback_top_n]:
            if candidate.parent_id not in existing_ids:
                extra.append(candidate)
                existing_ids.add(candidate.parent_id)

        for candidate in sparse[: settings.fallback_top_n]:
            if candidate.parent_id not in existing_ids:
                extra.append(candidate)
                existing_ids.add(candidate.parent_id)

        return pool + extra

    def _defense_in_depth_filter(
        self, pool: list[RetrievalCandidate], query: QueryPackage
    ) -> tuple[list[RetrievalCandidate], dict[str, ParentDocument]]:
        """Lớp kiểm tra status LẦN 2, đọc trực tiếp từ ParentStore (nguồn sự
        thật), phòng trường hợp Qdrant/BM25 index bị stale so với Mongo."""

        parent_docs = self._parent_store.get_many([c.parent_id for c in pool])

        if not query.wants_status_filter():
            # temporal_intent = lich_su / khong_ro: không loại bỏ văn bản hết
            # hiệu lực, chỉ cần đảm bảo có đủ thông tin để cảnh báo ở Giai đoạn 5.
            return pool, parent_docs

        safe_pool = []
        dropped = 0
        for candidate in pool:
            doc = parent_docs.get(candidate.parent_id)
            if doc is None:
                continue  # không lấy được nội dung -> loại, tránh trích dẫn rỗng
            if doc.status != VanBanStatus.CON_HIEU_LUC:
                dropped += 1
                continue
            safe_pool.append(candidate)

        if dropped:
            logger.warning(
                "Defense-in-depth loại %d văn bản hết hiệu lực bị lọt qua Giai đoạn 2 "
                "(nghi ngờ index bị stale)", dropped,
            )

        return safe_pool, parent_docs

    def build(
        self,
        query: QueryPackage,
        dense: list[RetrievalCandidate],
        sparse: list[RetrievalCandidate],
    ) -> tuple[list[RetrievalCandidate], dict[str, ParentDocument]]:
        pool = self._intersect(dense, sparse)
        pool = self._apply_fallback(pool, dense, sparse)
        pool, parent_docs = self._defense_in_depth_filter(pool, query)

        logger.info("Pool cuối cùng sau Giai đoạn 3: %d ứng viên", len(pool))
        return pool, parent_docs
