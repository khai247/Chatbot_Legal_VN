"""
Schema dùng chung xuyên suốt 5 giai đoạn của Phân hệ 2.

Dùng Pydantic để: (1) ép cấu trúc ổn định cho JSON output không xác định
của LLM ở Giai đoạn 1, và (2) làm "hợp đồng" rõ ràng giữa các module do
3 người trong nhóm cùng phát triển.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TemporalIntent(str, Enum):
    """Ý định thời gian của câu hỏi — quyết định có lọc 'còn hiệu lực' hay không."""

    HIEN_TAI = "hien_tai"   # mặc định: chỉ lấy văn bản còn hiệu lực
    LICH_SU = "lich_su"     # hỏi về văn bản/giai đoạn cũ: không lọc status
    KHONG_RO = "khong_ro"   # không đủ tín hiệu: lấy cả 2, ưu tiên còn hiệu lực


class VanBanStatus(str, Enum):
    CON_HIEU_LUC = "con_hieu_luc"
    HET_HIEU_LUC = "het_hieu_luc"
    HET_HIEU_LUC_MOT_PHAN = "het_hieu_luc_mot_phan"


class MetadataFilter(BaseModel):
    """Bộ lọc metadata bóc tách từ câu hỏi — dùng chung cho cả Dense và Sparse."""

    so_hieu_van_ban: Optional[str] = None     # vd: "45/2013/QH13"
    loai_van_ban: Optional[str] = None        # vd: "Luật", "Nghị định", "Thông tư"
    nam_ban_hanh: Optional[int] = None

    def is_empty(self) -> bool:
        return not any([self.so_hieu_van_ban, self.loai_van_ban, self.nam_ban_hanh])


class QueryPackage(BaseModel):
    """
    Đầu ra chuẩn hoá của Giai đoạn 1 (Query Analyzer).
    Đây là "hợp đồng" duy nhất mà Giai đoạn 2 (Hybrid Search) đọc vào —
    cả Dense lẫn Sparse PHẢI cùng đọc từ đối tượng này để đảm bảo đồng bộ
    filter (đây chính là lỗ hổng đã phát hiện và vá trong thiết kế).
    """

    raw_question: str
    keywords: list[str] = Field(default_factory=list)
    legal_terms: list[str] = Field(default_factory=list)
    metadata_filter: MetadataFilter = Field(default_factory=MetadataFilter)
    temporal_intent: TemporalIntent = TemporalIntent.HIEN_TAI
    is_fallback_parse: bool = False  # True nếu Qwen3.5 trả JSON lỗi và phải fallback

    @field_validator("keywords", "legal_terms", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    def search_text(self) -> str:
        """Chuỗi văn bản dùng cho BM25 / fallback embedding: keyword + thuật ngữ pháp lý."""
        terms = list(dict.fromkeys([*self.keywords, *self.legal_terms]))  # unique, giữ thứ tự
        return " ".join(terms) if terms else self.raw_question

    def wants_status_filter(self) -> bool:
        """Chỉ áp filter 'còn hiệu lực' khi ý định là hiện tại."""
        return self.temporal_intent == TemporalIntent.HIEN_TAI


class RetrievalCandidate(BaseModel):
    """Một Parent (Điều luật) ứng viên, kèm nguồn gốc để debug/audit."""

    parent_id: str
    score: float
    source: str  # "dense" | "sparse"


class ParentDocument(BaseModel):
    """Nội dung đầy đủ của một Điều luật, lấy từ Document Store (MongoDB)."""

    parent_id: str
    so_hieu_van_ban: str
    loai_van_ban: str
    ten_dieu: str
    noi_dung: str
    ngay_ban_hanh: Optional[str] = None
    ngay_hieu_luc: Optional[str] = None
    status: VanBanStatus = VanBanStatus.CON_HIEU_LUC
    van_ban_thay_the: Optional[str] = None  # số hiệu văn bản thay thế, nếu có


class RerankedDocument(BaseModel):
    parent_doc: ParentDocument
    rerank_score: float
    was_truncated: bool = False


class Citation(BaseModel):
    so_hieu_van_ban: str
    ten_dieu: str
    status: VanBanStatus
    van_ban_thay_the: Optional[str] = None


class AnswerResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_temporal_intent: TemporalIntent
    warning: Optional[str] = None  # vd: cảnh báo khi trích dẫn văn bản hết hiệu lực
