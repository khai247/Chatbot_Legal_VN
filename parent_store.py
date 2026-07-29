"""
Wrapper truy xuất Parent Chunk (toàn văn Điều luật) từ Document Store.

Phân hệ 1 (Data Ingestion) ghi dữ liệu vào MongoDB; Phân hệ 2 chỉ đọc.
Tách riêng module này để 2 người code Phân hệ 1 / Phân hệ 2 không đụng
code của nhau — chỉ cần thống nhất schema field.
"""

from __future__ import annotations

import logging

from pymongo import MongoClient

from config import settings
from schemas import ParentDocument, VanBanStatus

logger = logging.getLogger(__name__)


class ParentStore:
    def __init__(self, client: MongoClient | None = None):
        self._client = client or MongoClient(settings.parent_store_uri)
        self._collection = self._client[settings.parent_store_db][settings.parent_store_collection]

    def get_many(self, parent_ids: list[str]) -> dict[str, ParentDocument]:
        if not parent_ids:
            return {}

        cursor = self._collection.find({"parent_id": {"$in": parent_ids}})
        result: dict[str, ParentDocument] = {}
        for doc in cursor:
            doc.pop("_id", None)
            try:
                result[doc["parent_id"]] = ParentDocument(**doc)
            except Exception as exc:  # dữ liệu Mongo không khớp schema
                logger.warning("Bỏ qua parent_id=%s do lỗi schema: %s", doc.get("parent_id"), exc)

        missing = set(parent_ids) - result.keys()
        if missing:
            logger.warning("Không tìm thấy %d parent_id trong ParentStore: %s", len(missing), missing)

        return result

    def get_lightweight_metadata_index(self) -> dict[str, dict]:
        """
        Trả về {parent_id: {status, so_hieu_van_ban, loai_van_ban, nam_ban_hanh}}
        cho TOÀN BỘ tập dữ liệu — dùng để nạp vào SparseRetriever (Giai đoạn 2)
        làm "allowed set" filter trong RAM mà không cần query Mongo mỗi lần.
        Nên cache/refresh định kỳ (vd: mỗi khi re-index BM25) thay vì gọi mỗi request.
        """
        projection = {
            "_id": 0, "parent_id": 1, "status": 1,
            "so_hieu_van_ban": 1, "loai_van_ban": 1, "nam_ban_hanh": 1,
        }
        return {
            doc["parent_id"]: {
                "status": doc.get("status", VanBanStatus.CON_HIEU_LUC.value),
                "so_hieu_van_ban": doc.get("so_hieu_van_ban"),
                "loai_van_ban": doc.get("loai_van_ban"),
                "nam_ban_hanh": doc.get("nam_ban_hanh"),
            }
            for doc in self._collection.find({}, projection)
        }
