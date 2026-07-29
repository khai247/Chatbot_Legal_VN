"""
Test logic lõi của Giai đoạn 1 và Giai đoạn 3 mà KHÔNG cần Qdrant/Ollama/
Mongo thật — dùng mock/fake object. Chạy: `pytest tests/ -v`

Đây chỉ là smoke test để 3 thành viên yên tâm là logic filter/fallback
đúng như thiết kế, không thay thế cho test tích hợp với dữ liệu thật.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intersection_fallback import IntersectionFallbackPool
from schemas import MetadataFilter, ParentDocument, QueryPackage, RetrievalCandidate, TemporalIntent, VanBanStatus


class FakeParentStore:
    """Giả lập ParentStore, không kết nối Mongo thật."""

    def __init__(self, docs: dict[str, ParentDocument]):
        self._docs = docs

    def get_many(self, parent_ids):
        return {pid: self._docs[pid] for pid in parent_ids if pid in self._docs}


def make_doc(pid: str, status: VanBanStatus) -> ParentDocument:
    return ParentDocument(
        parent_id=pid,
        so_hieu_van_ban=f"VB-{pid}",
        loai_van_ban="Luật",
        ten_dieu=f"Điều {pid}",
        noi_dung="Nội dung mẫu.",
        status=status,
    )


def test_intersection_giu_dung_parent_id_chung():
    dense = [
        RetrievalCandidate(parent_id="A", score=0.9, source="dense"),
        RetrievalCandidate(parent_id="B", score=0.8, source="dense"),
    ]
    sparse = [
        RetrievalCandidate(parent_id="B", score=5.0, source="sparse"),
        RetrievalCandidate(parent_id="C", score=4.0, source="sparse"),
    ]
    pool = IntersectionFallbackPool._intersect(dense, sparse)
    assert [c.parent_id for c in pool] == ["B"]


def test_fallback_kich_hoat_khi_pool_qua_nho(monkeypatch):
    import config

    monkeypatch.setattr(config.settings, "intersection_min", 3)
    monkeypatch.setattr(config.settings, "fallback_top_n", 2)

    docs = {pid: make_doc(pid, VanBanStatus.CON_HIEU_LUC) for pid in ["A", "B", "C", "D"]}
    store = FakeParentStore(docs)
    builder = IntersectionFallbackPool(parent_store=store)

    dense = [
        RetrievalCandidate(parent_id="A", score=0.9, source="dense"),
        RetrievalCandidate(parent_id="B", score=0.8, source="dense"),
    ]
    sparse = [
        RetrievalCandidate(parent_id="C", score=5.0, source="sparse"),
        RetrievalCandidate(parent_id="D", score=4.0, source="sparse"),
    ]

    query = QueryPackage(raw_question="test", temporal_intent=TemporalIntent.HIEN_TAI)
    pool, _ = builder.build(query, dense, sparse)

    # intersection rỗng (không có id chung) -> fallback lấy top-2 mỗi luồng
    assert len(pool) == 4


def test_defense_in_depth_loai_van_ban_het_hieu_luc_khi_hien_tai():
    docs = {
        "A": make_doc("A", VanBanStatus.CON_HIEU_LUC),
        "B": make_doc("B", VanBanStatus.HET_HIEU_LUC),  # lọt qua Giai đoạn 2 do index stale
    }
    store = FakeParentStore(docs)
    builder = IntersectionFallbackPool(parent_store=store)

    dense = [
        RetrievalCandidate(parent_id="A", score=0.9, source="dense"),
        RetrievalCandidate(parent_id="B", score=0.85, source="dense"),
    ]
    sparse = [
        RetrievalCandidate(parent_id="A", score=5.0, source="sparse"),
        RetrievalCandidate(parent_id="B", score=4.9, source="sparse"),
    ]

    query = QueryPackage(raw_question="test", temporal_intent=TemporalIntent.HIEN_TAI)
    pool, parent_docs = builder.build(query, dense, sparse)

    assert [c.parent_id for c in pool] == ["A"]


def test_lich_su_khong_loai_van_ban_het_hieu_luc():
    docs = {
        "A": make_doc("A", VanBanStatus.CON_HIEU_LUC),
        "B": make_doc("B", VanBanStatus.HET_HIEU_LUC),
    }
    store = FakeParentStore(docs)
    builder = IntersectionFallbackPool(parent_store=store)

    dense = [
        RetrievalCandidate(parent_id="A", score=0.9, source="dense"),
        RetrievalCandidate(parent_id="B", score=0.85, source="dense"),
    ]
    sparse = [
        RetrievalCandidate(parent_id="A", score=5.0, source="sparse"),
        RetrievalCandidate(parent_id="B", score=4.9, source="sparse"),
    ]

    query = QueryPackage(raw_question="Luật đất đai 2013 quy định gì?", temporal_intent=TemporalIntent.LICH_SU)
    pool, parent_docs = builder.build(query, dense, sparse)

    assert {c.parent_id for c in pool} == {"A", "B"}


def test_query_package_wants_status_filter():
    q1 = QueryPackage(raw_question="x", temporal_intent=TemporalIntent.HIEN_TAI)
    q2 = QueryPackage(raw_question="x", temporal_intent=TemporalIntent.LICH_SU)
    assert q1.wants_status_filter() is True
    assert q2.wants_status_filter() is False


def test_metadata_filter_is_empty():
    assert MetadataFilter().is_empty() is True
    assert MetadataFilter(nam_ban_hanh=2020).is_empty() is False
