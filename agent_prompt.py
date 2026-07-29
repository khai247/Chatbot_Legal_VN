"""
Giai đoạn 5 — Agentic Prompting & Sinh câu trả lời

Đóng gói Top 5 Điều luật vào prompt System/User tách biệt, ép LLM:
  1. Chỉ trả lời dựa trên Context được cấp (chống ảo giác).
  2. Luôn nêu rõ trạng thái hiệu lực của từng Điều luật được trích dẫn —
     đặc biệt quan trọng khi temporal_intent != hien_tai, vì lúc đó pool
     có thể chứa văn bản đã hết hiệu lực một cách hợp lệ (xem schemas.py).
"""

from __future__ import annotations

import logging

import ollama

from config import settings
from schemas import AnswerResult, Citation, QueryPackage, RerankedDocument, VanBanStatus

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
Bạn là trợ lý tra cứu văn bản pháp luật Việt Nam. Chỉ được trả lời dựa trên
CONTEXT được cung cấp bên dưới. Nếu Context không đủ thông tin để trả lời,
hãy nói rõ là không tìm thấy quy định liên quan — TUYỆT ĐỐI không suy diễn
hay bịa thông tin ngoài Context.

Quy tắc bắt buộc khi trích dẫn:
1. Luôn nêu rõ Số hiệu văn bản và tên Điều/Khoản khi trích dẫn.
2. Luôn nêu rõ trạng thái hiệu lực của văn bản được trích dẫn. Nếu văn bản
   đã hết hiệu lực, PHẢI nói rõ điều này và nêu văn bản thay thế nếu có
   trong Context, ví dụ: "Điều 5, Nghị định 43/2014 — ĐÃ HẾT HIỆU LỰC, hiện
   được thay thế bởi Nghị định 148/2020".
3. Không được trình bày văn bản hết hiệu lực như thể nó đang còn hiệu lực.
"""


def _format_context(docs: list[RerankedDocument]) -> str:
    blocks = []
    for i, item in enumerate(docs, start=1):
        doc = item.parent_doc
        status_label = {
            VanBanStatus.CON_HIEU_LUC: "CÒN HIỆU LỰC",
            VanBanStatus.HET_HIEU_LUC: "ĐÃ HẾT HIỆU LỰC",
            VanBanStatus.HET_HIEU_LUC_MOT_PHAN: "HẾT HIỆU LỰC MỘT PHẦN",
        }[doc.status]

        thay_the = f" | Thay thế bởi: {doc.van_ban_thay_the}" if doc.van_ban_thay_the else ""

        blocks.append(
            f"[Nguồn {i}] {doc.so_hieu_van_ban} — {doc.ten_dieu}\n"
            f"Trạng thái: {status_label}{thay_the}\n"
            f"Nội dung: {doc.noi_dung}\n"
        )
    return "\n".join(blocks)


class AnswerGenerator:
    def __init__(self, client: ollama.Client | None = None):
        self._client = client or ollama.Client(host=settings.ollama_host)

    def generate(self, query: QueryPackage, top_docs: list[RerankedDocument]) -> AnswerResult:
        if not top_docs:
            return AnswerResult(
                answer="Không tìm thấy quy định pháp luật liên quan đến câu hỏi này trong dữ liệu hiện có.",
                citations=[],
                used_temporal_intent=query.temporal_intent,
            )

        context = _format_context(top_docs)
        user_prompt = f"CÂU HỎI: {query.raw_question}\n\nCONTEXT:\n{context}"

        response = self._client.chat(
            model=settings.ollama_gen_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.1},
        )
        answer_text = response["message"]["content"]

        citations = [
            Citation(
                so_hieu_van_ban=item.parent_doc.so_hieu_van_ban,
                ten_dieu=item.parent_doc.ten_dieu,
                status=item.parent_doc.status,
                van_ban_thay_the=item.parent_doc.van_ban_thay_the,
            )
            for item in top_docs
        ]

        warning = None
        het_hieu_luc_docs = [c for c in citations if c.status != VanBanStatus.CON_HIEU_LUC]
        if het_hieu_luc_docs:
            warning = (
                f"Câu trả lời có tham chiếu {len(het_hieu_luc_docs)} văn bản đã hết hiệu lực "
                f"(do câu hỏi liên quan đến giai đoạn lịch sử hoặc không xác định rõ thời điểm)."
            )

        return AnswerResult(
            answer=answer_text,
            citations=citations,
            used_temporal_intent=query.temporal_intent,
            warning=warning,
        )
