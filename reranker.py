"""
Giai đoạn 4 — Reranker (BGE-Reranker-M3)

Chấm điểm lại pool ứng viên (~15-20 Điều luật) bằng full-text để chọn ra
Top 5 chính xác nhất. Có kiểm tra độ dài token trước khi đưa vào reranker
— BGE-Reranker-M3 có giới hạn context (mặc định coi là 1024 token qua
config `reranker_max_tokens`); Điều luật dài hơn sẽ bị cắt theo Khoản liên
quan nhất thay vì bị truncate cứng ở cuối (silent truncation).
"""

from __future__ import annotations

import logging
import re

from config import settings
from schemas import ParentDocument, QueryPackage, RerankedDocument, RetrievalCandidate

logger = logging.getLogger(__name__)

_KHOAN_SPLIT_RE = re.compile(r"(?=\n?\s*\d+\.\s)")  # tách theo "1. ", "2. " ... đầu dòng Khoản


class Reranker:
    _model = None
    _tokenizer = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info("Đang load reranker model: %s", settings.reranker_model)
            self._model = FlagReranker(settings.reranker_model, use_fp16=True)
        return self._model

    def _token_count(self, text: str) -> int:
        """Đếm token bằng chính tokenizer của reranker nếu có sẵn; nếu
        không truy cập được, ước lượng thô ~ 1 token / 3 ký tự tiếng Việt
        (an toàn hơn là giả định 1 token/từ vì tiếng Việt có dấu)."""
        try:
            model = self._load()
            return len(model.tokenizer.encode(text))
        except Exception:
            return max(1, len(text) // 3)

    def _truncate_to_relevant_khoan(self, doc: ParentDocument, query: QueryPackage) -> str:
        """Cắt nội dung Điều luật xuống các Khoản liên quan nhất, thay vì
        cắt cứng ở cuối văn bản (tránh mất đúng phần Khoản chứa câu trả lời)."""

        segments = [s.strip() for s in _KHOAN_SPLIT_RE.split(doc.noi_dung) if s.strip()]
        if len(segments) <= 1:
            # Không tách được theo Khoản -> đành cắt cứng theo ký tự, ưu tiên phần đầu
            budget_chars = settings.reranker_max_tokens * 3
            return doc.noi_dung[:budget_chars]

        keywords = {*(k.lower() for k in query.keywords), *(t.lower() for t in query.legal_terms)}

        def relevance(segment: str) -> int:
            seg_lower = segment.lower()
            return sum(1 for kw in keywords if kw and kw in seg_lower)

        # Luôn giữ đoạn mở đầu (thường là câu dẫn của Điều) + sắp các Khoản còn
        # lại theo mức độ khớp từ khoá, nhiều nhất trước.
        head, *rest = segments
        rest_sorted = sorted(rest, key=relevance, reverse=True)

        kept = [head]
        token_budget = settings.reranker_max_tokens
        used = self._token_count(head)

        for seg in rest_sorted:
            seg_tokens = self._token_count(seg)
            if used + seg_tokens > token_budget:
                continue
            kept.append(seg)
            used += seg_tokens

        return "\n".join(kept)

    def _prepare_text(self, doc: ParentDocument, query: QueryPackage) -> tuple[str, bool]:
        full_text = f"{doc.ten_dieu}\n{doc.noi_dung}"
        token_count = self._token_count(full_text)

        if token_count <= settings.reranker_max_tokens:
            return full_text, False

        logger.info(
            "parent_id=%s vượt giới hạn token (%d > %d) — cắt theo Khoản liên quan",
            doc.parent_id, token_count, settings.reranker_max_tokens,
        )
        truncated = f"{doc.ten_dieu}\n{self._truncate_to_relevant_khoan(doc, query)}"
        return truncated, True

    def rerank(
        self,
        query: QueryPackage,
        pool: list[RetrievalCandidate],
        parent_docs: dict[str, ParentDocument],
    ) -> list[RerankedDocument]:
        model = self._load()

        pairs: list[tuple[str, str]] = []
        docs_in_order: list[ParentDocument] = []
        truncated_flags: list[bool] = []

        for candidate in pool:
            doc = parent_docs.get(candidate.parent_id)
            if doc is None:
                continue
            text, was_truncated = self._prepare_text(doc, query)
            pairs.append((query.raw_question, text))
            docs_in_order.append(doc)
            truncated_flags.append(was_truncated)

        if not pairs:
            return []

        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]

        reranked = [
            RerankedDocument(parent_doc=doc, rerank_score=float(score), was_truncated=trunc)
            for doc, score, trunc in zip(docs_in_order, scores, truncated_flags)
        ]
        reranked.sort(key=lambda r: r.rerank_score, reverse=True)

        top = reranked[: settings.rerank_top_n]
        logger.info("Reranker giữ lại top %d / %d ứng viên", len(top), len(reranked))
        return top
