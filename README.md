# Phân hệ 2 — Advanced Retrieval & Agent

Code triển khai 5 giai đoạn theo đúng kiến trúc đã chốt:

```
Query Analyzer → Hybrid Search (Dense+Sparse) → Intersection & Fallback
→ Reranker → Agentic Prompting & Sinh câu trả lời
```

## Cấu trúc file

| File | Giai đoạn | Vai trò |
|---|---|---|
| `schemas.py` | — | Pydantic models dùng chung (QueryPackage, ParentDocument...) |
| `config.py` | — | Cấu hình đọc từ `.env` |
| `query_analyzer.py` | GĐ1 | Gọi Ollama, bóc tách JSON, validate + fallback |
| `hybrid_search.py` | GĐ2 | Dense (Qdrant) + Sparse (BM25S) chạy song song, filter đồng bộ |
| `parent_store.py` | — | Đọc Parent Chunk từ MongoDB (do Phân hệ 1 ghi) |
| `intersection_fallback.py` | GĐ3 | Giao 2 luồng + fallback + kiểm tra hiệu lực lần 2 |
| `reranker.py` | GĐ4 | BGE-Reranker-M3, kiểm soát độ dài token |
| `agent_prompt.py` | GĐ5 | Prompt System/User, sinh câu trả lời + cảnh báo hiệu lực |
| `pipeline.py` | — | Ghép 5 giai đoạn, entrypoint cho Phân hệ 3 gọi vào |
| `tests/test_pipeline_smoke.py` | — | Test logic GĐ1/GĐ3 bằng mock, không cần service ngoài |

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # rồi chỉnh lại theo môi trường thật
```

Yêu cầu hạ tầng chạy sẵn:
- **Ollama** local đã pull model trong `.env` (`OLLAMA_QUERY_MODEL`, `OLLAMA_GEN_MODEL`)
- **Qdrant** đã có collection chứa Child Chunks (do Phân hệ 1 nạp), payload cần có tối thiểu các field: `parent_id`, `status`, `so_hieu_van_ban`, `loai_van_ban`, `nam_ban_hanh`
- **MongoDB** đã có collection Parent Chunks với schema khớp `ParentDocument` trong `schemas.py`
- **BM25S index** đã build sẵn (xem phần dưới)

## Build BM25S index (chạy 1 lần, hoặc mỗi khi Phân hệ 1 cập nhật dữ liệu)

```python
import bm25s

# corpus: list các dict {"id": parent_id, "text": toàn văn Điều luật}
corpus = [...]
corpus_texts = [doc["text"] for doc in corpus]

retriever = bm25s.BM25(corpus=corpus)
corpus_tokens = bm25s.tokenize(corpus_texts)
retriever.index(corpus_tokens)
retriever.save("./bm25_index")
```

## Chạy thử pipeline (CLI)

```bash
python pipeline.py "Mua đất nông nghiệp cần giấy tờ gì?" --bm25-index ./bm25_index
```

## Chạy test logic (không cần service ngoài)

```bash
pip install pytest
pytest tests/ -v
```

6 test hiện có kiểm tra đúng phần dễ sai nhất của thiết kế:
- Intersection giữ đúng `parent_id` chung giữa 2 luồng.
- Fallback kích hoạt đúng khi pool nhỏ hơn `intersection_min`.
- Defense-in-depth loại văn bản hết hiệu lực khi `temporal_intent = hien_tai`.
- Không loại văn bản hết hiệu lực khi `temporal_intent = lich_su`.

## Những điểm PHẢI giữ nguyên khi sửa code (đã thống nhất trong nhóm)

1. **Filter `status` + `metadata_filter` phải áp cho CẢ Dense lẫn Sparse**, kể cả nhánh fallback. Đây là lỗ hổng đã phát hiện và vá — đừng vô tình bỏ lại filter ở một luồng khi refactor.
2. **`temporal_intent` quyết định có lọc "còn hiệu lực" hay không** — mặc định `hien_tai`. Đừng hard-code filter status cứng cho mọi câu hỏi.
3. **Intersection + Fallback** được giữ (không đổi sang Union+RRF) vì nhóm đã thử nghiệm và Intersection cho kết quả tốt hơn trên dataset thực tế.
4. Reranker luôn kiểm tra token length trước khi đưa full-text vào — tránh silent truncation.
5. Prompt Giai đoạn 5 luôn phải ép LLM nêu trạng thái hiệu lực của văn bản trích dẫn.

## TODO cần 3 người thống nhất thêm

- [ ] Schema chính xác của `payload` trong Qdrant (tên field có khớp `parent_id`, `status`... như trong code không) — cần khớp với Phân hệ 1.
- [ ] Schema chính xác của document MongoDB (`ParentDocument`) — cần khớp với Phân hệ 1.
- [ ] Endpoint để Phân hệ 3 gọi `Phase2Pipeline.run()` — có thể wrap bằng FastAPI route riêng, xem `pipeline.py`.
- [ ] Benchmark `ThreadPoolExecutor` vs `ProcessPoolExecutor` cho `HybridSearchOrchestrator` bằng dữ liệu thật.
- [ ] Kiểm tra `reranker_max_tokens` thực tế theo checkpoint model đang dùng.
