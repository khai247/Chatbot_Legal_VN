"""
Giai đoạn 1 — Query Analyzer (Single-Pass)

Đẩy câu hỏi tự nhiên của người dùng qua một prompt duy nhất trên Ollama
(Qwen3.5) để bóc tách thành QueryPackage chuẩn hoá.

Có JSON Schema Validator + fallback: nếu LLM trả JSON lỗi/thiếu field,
KHÔNG để pipeline gãy — tự chuyển sang dùng câu hỏi gốc làm keywords thô.
Đây là điểm đã xác định là "single point of failure" nên bắt buộc phải
có lớp phòng thủ này.
"""

from __future__ import annotations

import json
import logging
import re

import ollama
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed

from config import settings
from schemas import MetadataFilter, QueryPackage, TemporalIntent

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
Bạn là bộ phân tích câu hỏi cho hệ thống tra cứu văn bản pháp luật Việt Nam.
Nhiệm vụ: đọc câu hỏi của người dùng (có thể dùng ngôn ngữ đời thường, dân
gian) và trả về DUY NHẤT một JSON object hợp lệ, không kèm giải thích, không
markdown, không code fence.

Schema JSON bắt buộc:
{
  "keywords": [string],        // từ khoá thô để tìm kiếm từ khoá
  "legal_terms": [string],     // thuật ngữ pháp lý tương ứng, vd "mua đất" -> "chuyển nhượng quyền sử dụng đất"
  "metadata_filter": {
      "so_hieu_van_ban": string | null,   // vd "45/2013/QH13", chỉ điền nếu người dùng nêu rõ
      "loai_van_ban": string | null,      // "Luật" | "Nghị định" | "Thông tư" | null
      "nam_ban_hanh": number | null
  },
  "temporal_intent": "hien_tai" | "lich_su" | "khong_ro"
}

Quy tắc xác định temporal_intent:
- "hien_tai": câu hỏi không có mốc thời gian quá khứ, hỏi kiểu "hiện nay",
  "bây giờ", hoặc không nhắc gì tới thời gian (đây là mặc định).
- "lich_su": câu hỏi có mốc thời gian quá khứ rõ ràng ("năm 2019", "trước
  đây", "luật cũ", "hồi đó"), hoặc hỏi trực tiếp về một văn bản đã biết là
  cũ/đã hết hiệu lực.
- "khong_ro": không đủ tín hiệu để xác định.

Chỉ trả JSON. Không thêm bất kỳ văn bản nào khác.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class QueryAnalyzer:
    def __init__(self, client: ollama.Client | None = None):
        self._client = client or ollama.Client(host=settings.ollama_host)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    def _call_llm(self, question: str) -> str:
        response = self._client.chat(
            model=settings.ollama_query_model,
            format="json",  # ép Ollama ràng buộc JSON ở tầng decode khi model hỗ trợ
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            options={"temperature": 0.0},
        )
        return response["message"]["content"]

    def _parse(self, raw_text: str, raw_question: str) -> QueryPackage:
        match = _JSON_BLOCK_RE.search(raw_text)
        if not match:
            raise ValueError("Không tìm thấy JSON object trong output của LLM")

        data = json.loads(match.group(0))

        metadata = data.get("metadata_filter") or {}
        return QueryPackage(
            raw_question=raw_question,
            keywords=data.get("keywords") or [],
            legal_terms=data.get("legal_terms") or [],
            metadata_filter=MetadataFilter(
                so_hieu_van_ban=metadata.get("so_hieu_van_ban"),
                loai_van_ban=metadata.get("loai_van_ban"),
                nam_ban_hanh=metadata.get("nam_ban_hanh"),
            ),
            temporal_intent=TemporalIntent(data.get("temporal_intent") or "hien_tai"),
        )

    def _fallback_package(self, raw_question: str) -> QueryPackage:
        """
        Fallback an toàn khi LLM lỗi: dùng chính câu hỏi làm keywords thô,
        không áp metadata_filter, mặc định temporal_intent = hien_tai (an
        toàn hơn là mặc định lich_su, vì tránh vô tình trả về văn bản hết
        hiệu lực khi không chắc ý định người dùng).
        """
        tokens = [t for t in re.split(r"\s+", raw_question.strip()) if len(t) > 1]
        return QueryPackage(
            raw_question=raw_question,
            keywords=tokens,
            legal_terms=[],
            metadata_filter=MetadataFilter(),
            temporal_intent=TemporalIntent.HIEN_TAI,
            is_fallback_parse=True,
        )

    def analyze(self, question: str) -> QueryPackage:
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi rỗng")

        try:
            raw_text = self._call_llm(question)
            package = self._parse(raw_text, question)
            logger.info("Query analyzer OK | temporal_intent=%s", package.temporal_intent)
            return package
        except (ValueError, KeyError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Query analyzer parse thất bại (%s) — dùng fallback", exc)
            return self._fallback_package(question)
        except Exception as exc:  # lỗi kết nối Ollama, timeout, v.v.
            logger.error("Query analyzer lỗi hạ tầng (%s) — dùng fallback", exc)
            return self._fallback_package(question)
