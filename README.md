# Hệ thống RAG Hỏi-Đáp Văn Bản Pháp Luật Việt Nam

> Hệ thống Retrieval-Augmented Generation (RAG) chạy local, hỗ trợ tra cứu và hỏi-đáp văn bản pháp luật Việt Nam bằng ngôn ngữ tự nhiên, có trích dẫn chính xác đến Điều/Khoản và cảnh báo tình trạng hiệu lực văn bản.

---

## Mục lục

1. [Bài toán & Động lực](#1-bài-toán--động-lực)
2. [Kiến trúc tổng quan](#2-kiến-trúc-tổng-quan)
3. [Nguồn dữ liệu](#3-nguồn-dữ-liệu)
4. [Phân hệ 1 — Data Ingestion & Pipeline](#4-phân-hệ-1--data-ingestion--pipeline)
5. [Phân hệ 2 — Advanced Retrieval & Agent](#5-phân-hệ-2--advanced-retrieval--agent)
6. [Phân hệ 3 — Backend API & UI/UX](#6-phân-hệ-3--backend-api--uiux)
7. [Các quyết định thiết kế quan trọng](#7-các-quyết-định-thiết-kế-quan-trọng)
8. [Công nghệ sử dụng](#8-công-nghệ-sử-dụng)
9. [Cài đặt & Chạy dự án](#9-cài-đặt--chạy-dự-án)
10. [Cấu trúc thư mục](#10-cấu-trúc-thư-mục)
11. [Phân công nhóm](#11-phân-công-nhóm)
12. [Hạn chế đã biết & Hướng phát triển](#12-hạn-chế-đã-biết--hướng-phát-triển)

---

## 1. Bài toán & Động lực

Tra cứu văn bản pháp luật Việt Nam theo cách truyền thống (tìm từ khóa trên các cổng thông tin) đòi hỏi người dùng phải tự biết đúng thuật ngữ pháp lý, tự đọc và đối chiếu nhiều văn bản để biết quy định nào còn hiệu lực. Hệ thống này giải quyết bài toán đó bằng cách cho phép người dùng hỏi bằng ngôn ngữ đời thường và nhận câu trả lời có trích dẫn chính xác, kèm cảnh báo nếu quy định được trích dẫn đã hết hiệu lực.

**Ràng buộc thiết kế xuyên suốt dự án:**

- **Chạy local-first**: toàn bộ mô hình suy luận (embedding, reranker, LLM sinh câu trả lời) chạy trên Ollama/CPU cục bộ, không phụ thuộc API trả phí — phù hợp bối cảnh cuộc thi và khả năng triển khai on-premise.
- **Chống ảo giác**: mọi câu trả lời phải bắt nguồn từ Context được truy hồi, có trích dẫn Điều/Khoản cụ thể; nếu không tìm thấy quy định liên quan, hệ thống phải nói rõ thay vì suy diễn.
- **Nhạy cảm với hiệu lực pháp lý**: đây là bài toán đặc thù của domain pháp luật — một câu trả lời đúng nội dung nhưng dựa trên Điều luật đã bị bãi bỏ là một câu trả lời sai và nguy hiểm hơn nhiều so với việc không trả lời được.

---

## 2. Kiến trúc tổng quan

Hệ thống chia thành 3 phân hệ độc lập, mỗi thành viên trong nhóm phụ trách một phân hệ, ghép nối theo pipeline tuần tự:

```
[Dataset văn bản pháp luật]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHÂN HỆ 1 — Data Ingestion & Pipeline                       │
│  Parser (Nghị định → Điều → Khoản) → Parent-Child Chunking   │
│  → Embedding (BGE-M3) → Nạp vào Qdrant                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHÂN HỆ 2 — Advanced Retrieval & Agent                       │
│  Query Analyzer → Hybrid Search (Dense+Sparse, song song)     │
│  → Intersection & Fallback → Reranker (BGE) → Agentic         │
│  Prompting (Ollama Qwen3) → Câu trả lời + trích dẫn           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHÂN HỆ 3 — Backend API & UI/UX                              │
│  FastAPI (streaming) → Giao diện Split-Screen (Chainlit)      │
│  Trái: hội thoại kèm Source Tag | Phải: toàn văn Điều luật     │
└─────────────────────────────────────────────────────────────┘
```

**Kho dữ liệu duy nhất: Qdrant** — không dùng thêm hệ quản trị CSDL riêng (đã cân nhắc và loại bỏ phương án dùng MongoDB cho Parent Chunks, xem [mục 7](#7-các-quyết-định-thiết-kế-quan-trọng)), giúp hạ tầng gọn, chỉ một hệ thống cần vận hành và backup.

---

## 3. Nguồn dữ liệu

Dự án sử dụng bộ dữ liệu **[`tmquan/vbpl-vn`](https://huggingface.co/datasets/tmquan/vbpl-vn)** trên HuggingFace — kho văn bản pháp luật Việt Nam quy mô lớn (~158K văn bản), mỗi văn bản gồm:

- Metadata có cấu trúc sẵn: `doc_number` (số hiệu), `doc_type`/`legal_type` (loại văn bản), `year`, `issue_date`, `title`, `legal_area`, `scope`...
- Nội dung dạng `markdown` (raw text đã làm sạch).
- `structure_json`: phân đoạn document → section → paragraph → sentence, kèm id ổn định và char-span trỏ ngược về `markdown`.
- `extracted_json`: kết quả NER/trích dẫn thực thể (statute_refs, relations) — chưa được kiểm chứng độ tin cậy đầy đủ, dùng ở mức tham khảo.

**Giới hạn quan trọng đã xác định qua khảo sát thực tế dữ liệu** (ảnh hưởng trực tiếp đến thiết kế Phân hệ 1 và 2):

- `structure_json` **không** tách sẵn theo đơn vị Điều/Khoản ở mọi văn bản — nhiều văn bản gộp toàn bộ nội dung vào một `section`/`paragraph` duy nhất. Việc tách Điều/Khoản vẫn phải tự thực hiện bằng Regex trên text.
- Dataset **không có field trạng thái hiệu lực** (`status`) hay văn bản thay thế — phải tự suy luận (xem mục 4).
- ~11.5K văn bản có `markdown = null` (chỉ có metadata, không có nội dung) — bị loại khỏi pipeline ingestion.

---

## 4. Phân hệ 1 — Data Ingestion & Pipeline

**Mục tiêu:** biến văn bản pháp luật thô thành các đơn vị Parent (Điều) và Child (Khoản) có cấu trúc, gắn đầy đủ metadata, sẵn sàng cho việc tìm kiếm ngữ nghĩa.

### 4.1 Phân cấp dữ liệu

Mỗi văn bản (Nghị định/Luật/Thông tư...) có 3 tầng, nhưng Parent-Child Chunking chỉ dùng 2 tầng dưới cùng:

```
Văn bản (nguồn metadata dùng chung: số hiệu, ngày ban hành, loại VB...)
   │
   ├─ Điều 1  ──►  PARENT (đơn vị cấp ngữ cảnh đầy đủ cho LLM)
   │    ├─ Khoản 1 ──►  CHILD (đơn vị để embedding & tìm kiếm)
   │    └─ Khoản 2 ──►  CHILD
   └─ Điều 2  ──►  PARENT
        └─ ...
```

**Lý do chọn Điều = Parent, Khoản = Child** (không chọn cả văn bản làm Parent): embedding hoạt động chính xác nhất trên đoạn văn ngắn tập trung một ý (Khoản); nhưng LLM cần đọc trọn câu dẫn/điều kiện áp dụng của cả Điều để không hiểu sai ngữ cảnh khi sinh câu trả lời — đưa cả văn bản vào Prompt sẽ tốn context một cách không cần thiết.

### 4.2 Xử lý các trường hợp thiếu tầng

| Trường hợp | Cách xử lý |
|---|---|
| Điều không có Khoản (đoạn văn liền, không đánh số) | Chính Điều trở thành Child của chính nó; nếu quá dài thì tách thêm theo câu, vẫn giữ chung `parent_id` |
| Văn bản không có Điều (một số Quyết định/Công văn ngắn) | Toàn văn bản được coi là "Điều giả" duy nhất (`dieu_so = 0`), sau đó áp lại đúng logic ở trên |

### 4.3 Thuật toán tách (tổng quát, không rẽ nhánh phức tạp theo từng loại văn bản)

```
1. Regex tách theo "Điều N."
   → không tìm thấy: coi cả văn bản là 1 Điều giả
2. Với mỗi Điều, regex tách theo số thứ tự đầu dòng (Khoản)
   → không tìm thấy: Điều tự làm Child duy nhất của chính nó
3. Nếu 1 Child vượt ngưỡng token embedding → tách thêm theo câu
```

### 4.4 Suy luận trạng thái hiệu lực (`status`, `van_ban_thay_the`, `ngay_hieu_luc`)

Vì dataset không có sẵn field này, hệ thống khai thác một mẫu câu chuẩn gần như cố định trong "Điều khoản thi hành" (thường là Điều cuối) của văn bản pháp luật Việt Nam, ví dụ: *"...có hiệu lực thi hành kể từ ngày .../.../... . Văn bản này thay thế .../.../..."*.

- `ngay_hieu_luc`, `van_ban_thay_the`: regex-extract trực tiếp từ Điều khoản thi hành của chính văn bản đó.
- `status`: **không tự văn bản nào biết nó hết hiệu lực** — chỉ suy ra được khi có văn bản KHÁC tuyên bố thay thế/bãi bỏ nó. Hệ thống xây một **bảng quan hệ (reverse index)** dạng edge-list tối giản:

  ```
  van_ban_relations: { so_hieu_van_ban_nguon, so_hieu_van_ban_dich, loai_quan_he }
  ```

  quét một lần trong bước ingestion trên toàn corpus. `status` của một văn bản = `het_hieu_luc` nếu số hiệu của nó xuất hiện là "đích" của quan hệ `thay_the`/`bai_bo` trong bảng này, mặc định `con_hieu_luc` nếu không tìm thấy quan hệ nào.

  > Đây thực chất là một dạng knowledge-graph tối giản (edge-list 2 cột + loại quan hệ), được chọn thay vì triển khai Graph DB đầy đủ (Neo4j/GraphRAG) do đánh đổi effort/lợi ích không phù hợp với quy mô và timeline cuộc thi — xem thêm mục 7.

### 4.5 Embedding & nạp dữ liệu

Mỗi Child Chunk được encode bằng **BGE-M3** (dense vector 1024 chiều), nạp vào Qdrant kèm đầy đủ payload metadata phục vụ pre-filtering ở Phân hệ 2.

---

## 5. Phân hệ 2 — Advanced Retrieval & Agent

**Mục tiêu:** nhận câu hỏi ngôn ngữ tự nhiên, truy hồi đúng Điều luật liên quan, sinh câu trả lời có trích dẫn và cảnh báo hiệu lực chính xác.

### 5.1 Sơ đồ 5 giai đoạn

```
Câu hỏi
   │
   ▼
[GĐ1] Query Analyzer (Ollama, single-pass)
   → keywords, legal_terms, metadata_filter, temporal_intent
   │
   ▼
[GĐ2] Hybrid Search song song
   ├─ Dense (Qdrant, BGE-M3)      ─┐  filter status + metadata
   └─ Sparse (BM25S, CPU)         ─┘  ĐỒNG BỘ trên cả 2 luồng
   │
   ▼
[GĐ3] Intersection Pool & Fallback
   → giao 2 luồng, bù thêm nếu pool < ngưỡng, kiểm tra hiệu lực lần 2
   │
   ▼
[GĐ4] Reranker (BGE-Reranker-M3, cross-encoder)
   → kiểm soát độ dài token, cắt theo Khoản liên quan nếu vượt giới hạn
   │
   ▼
[GĐ5] Agentic Prompting (Ollama Qwen3)
   → câu trả lời + trích dẫn Điều/Khoản + cảnh báo hiệu lực
```

### 5.2 Điểm thiết kế quan trọng nhất: `temporal_intent`

Không phải mọi câu hỏi đều nên bị lọc "chỉ lấy văn bản còn hiệu lực" — câu hỏi kiểu *"Luật Đất đai 2013 quy định gì về..."* cần chính văn bản đã hết hiệu lực. Giai đoạn 1 phân loại ý định thời gian của câu hỏi thành 3 nhóm và áp filter tương ứng:

| `temporal_intent` | Khi nào | Filter áp dụng |
|---|---|---|
| `hien_tai` (mặc định) | Không có tín hiệu thời gian | Chỉ lấy văn bản `con_hieu_luc` |
| `lich_su` | Có mốc thời gian quá khứ rõ ràng | Không lọc status; ưu tiên văn bản khớp mốc thời gian |
| `khong_ro` | Không đủ tín hiệu | Lấy cả 2, ưu tiên còn hiệu lực, cảnh báo rõ trong câu trả lời |

Nguyên tắc: **không lọc bỏ âm thầm** văn bản hết hiệu lực — luôn hiển thị kèm cảnh báo rõ ràng, vì im lặng bỏ qua nguy hiểm hơn hiển thị có cảnh báo.

### 5.3 Vì sao Intersection thay vì Union + RRF

Union + Reciprocal Rank Fusion (RRF) là lựa chọn phổ biến hơn để tránh mất recall khi kết hợp Dense/Sparse. Nhóm đã thử nghiệm cả hai và xác nhận **Intersection + Fallback cho kết quả tốt hơn trên dataset thực tế** — Intersection triệt tiêu ~70% nhiễu ngay từ đầu (chỉ giữ văn bản vừa đúng ngữ nghĩa vừa đúng từ khóa), cơ chế Fallback (bù thêm Top-N mỗi luồng khi giao quá nhỏ) đảm bảo không mất recall ở các câu hỏi khó/thuật ngữ hiếm.

### 5.4 Vá lỗ hổng đồng bộ filter

Thiết kế ban đầu chỉ áp filter `status`/`metadata_filter` ở luồng Dense (Qdrant hỗ trợ pre-filtering native); luồng Sparse (BM25S không hỗ trợ filter theo payload) quét toàn bộ corpus không lọc — dẫn đến rủi ro văn bản hết hiệu lực lọt vào qua nhánh Fallback. Đã khắc phục bằng cách duy trì một **allowed-set** các `parent_id` hợp lệ (đã lọc status/metadata) trong bộ nhớ, intersect với kết quả BM25 trước khi chốt danh sách ứng viên — đảm bảo 2 luồng luôn làm việc trên cùng một không gian dữ liệu đã lọc.

### 5.5 Kiểm soát giới hạn token của Reranker

BGE-Reranker-M3 có giới hạn context (~512–1024 token tùy checkpoint). Thay vì để bị cắt cứng ở cuối văn bản (silent truncation, có thể mất đúng phần chứa câu trả lời), hệ thống tách Điều luật dài theo từng Khoản và ưu tiên giữ lại các Khoản khớp từ khóa/thuật ngữ pháp lý đã trích ở Giai đoạn 1.

### 5.6 Chống ảo giác & minh bạch hiệu lực ở tầng Prompt

System Prompt của Giai đoạn 5 ép buộc LLM: (1) chỉ trả lời dựa trên Context được cấp, nói rõ "không tìm thấy" nếu thiếu thông tin; (2) luôn nêu trạng thái hiệu lực của mọi Điều luật được trích dẫn, và nêu văn bản thay thế nếu có.

---

## 6. Phân hệ 3 — Backend API & UI/UX

**Mục tiêu:** truyền tải kết quả từ Phân hệ 2 tới người dùng một cách trực quan, có khả năng tự kiểm chứng.

```
Phân hệ 2 (câu trả lời + trích dẫn)
         │
         ▼ Streaming (token-by-token)
   FastAPI Backend
         │
         ▼
   Giao diện Split-Screen (Chainlit)
   ┌───────────────────┬────────────────────┐
   │   MÀN HÌNH TRÁI    │    MÀN HÌNH PHẢI    │
   │  Chatbot + Source  │  Toàn văn Điều luật │
   │  Tag (Điều, Luật)  │  gốc khi click Tag  │
   └───────────────────┴────────────────────┘
```

- **Streaming**: trả lời hiển thị theo thời gian thực, giảm cảm giác chờ khi context dài.
- **Split-screen + Source Tag click-to-view**: người dùng có thể tự đối chiếu câu trả lời với nguyên văn Điều luật — quan trọng với domain pháp luật, nơi độ tin cậy của trích dẫn quyết định giá trị sử dụng thực tế của hệ thống.
- Source Tag hiển thị kèm trạng thái hiệu lực, đồng bộ với nguyên tắc minh bạch ở Phân hệ 2.

---

## 7. Các quyết định thiết kế quan trọng

Bảng dưới tóm tắt các lựa chọn kiến trúc đã cân nhắc nhiều phương án trước khi chốt — thể hiện quá trình đánh giá đánh đổi (trade-off) có chủ đích, không phải mặc định theo thói quen:

| Quyết định | Phương án đã cân nhắc | Lựa chọn cuối & lý do |
|---|---|---|
| Kho lưu Parent Chunk | MongoDB riêng / Qdrant collection riêng (không vector) / nhồi chung payload Child | **Qdrant collection riêng** — gọn hạ tầng (1 hệ thống duy nhất), tách bạch khỏi payload Child để không phình index |
| Kết hợp Dense + Sparse | Union + RRF / Intersection + Fallback | **Intersection + Fallback** — đã thử nghiệm thực tế, cho kết quả tốt hơn trên dataset của cuộc thi |
| Lọc văn bản hết hiệu lực | Lọc cứng mọi câu hỏi / không lọc / lọc có điều kiện | **Lọc có điều kiện theo `temporal_intent`** — tránh sai với câu hỏi về văn bản lịch sử |
| Suy luận `status` hiệu lực | Bỏ qua / Knowledge Graph đầy đủ (Neo4j, NER+RE tự động) / bảng quan hệ tối giản | **Bảng quan hệ tối giản (edge-list) suy từ Regex trên "Điều khoản thi hành"** — đạt phần lớn lợi ích của KG với chi phí thấp hơn nhiều, phù hợp timeline cuộc thi |
| Reranker | Bi-encoder (nhanh) / Cross-encoder (chính xác hơn) | **Cross-encoder (BGE-Reranker-M3)** — vì đầu vào Giai đoạn 4 đã tinh gọn (~15-20 ứng viên) nên chi phí tính toán chấp nhận được, đổi lại độ chính xác cao hơn |
| Reranker chạy trên model pretrained hay fine-tuned | Pretrained BGE-M3/BGE-Reranker / fine-tune trên corpus pháp luật VN | **Pretrained cho MVP**; fine-tune bằng hard-negative mining (nếu có tập câu hỏi mẫu từ ban tổ chức) được ghi nhận là hướng cải thiện có tiềm năng cao nhất, để ở mục Hướng phát triển |

---

## 8. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Vector DB / Document Store | Qdrant |
| Sparse retrieval | BM25S (CPU) |
| Embedding | BGE-M3 |
| Reranker | BGE-Reranker-M3 |
| LLM (Query Analyzer & sinh câu trả lời) | Qwen3 / Qwen3.5 qua Ollama (local) |
| Backend | FastAPI (streaming) |
| Giao diện | Chainlit (split-screen UI) |
| Ngôn ngữ | Python |
| Dữ liệu nguồn | [`tmquan/vbpl-vn`](https://huggingface.co/datasets/tmquan/vbpl-vn) (HuggingFace) |

---

## 9. Cài đặt & Chạy dự án

```bash
# 1. Clone & cài đặt
git clone <repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Cấu hình môi trường
cp .env.example .env   # chỉnh Qdrant URL, tên model Ollama...

# 3. Chạy Ollama local & pull model
ollama pull qwen2.5:3.5b
ollama pull qwen3:8b

# 4. Chạy pipeline ingestion (Phân hệ 1) — nạp dữ liệu vào Qdrant

# 5. Chạy thử truy vấn (Phân hệ 2)
python pipeline.py "Mua đất nông nghiệp cần giấy tờ gì?"

# 6. Khởi động Backend + UI (Phân hệ 3)
uvicorn main:app --reload
```

> Chi tiết cấu hình từng phân hệ xem README riêng trong thư mục tương ứng.

---

## 10. Cấu trúc thư mục

```
.
├── phan_he_1_ingestion/       # Parser, chunking, ingestion vào Qdrant
├── phan_he_2_retrieval/       # Query Analyzer, Hybrid Search, Reranker, Agent
│   ├── schemas.py
│   ├── config.py
│   ├── query_analyzer.py
│   ├── hybrid_search.py
│   ├── intersection_fallback.py
│   ├── reranker.py
│   ├── agent_prompt.py
│   ├── pipeline.py
│   └── tests/
├── phan_he_3_backend_ui/      # FastAPI + Chainlit
├── docs/                      # Tài liệu kiến trúc chi tiết
├── requirements.txt
├── .env.example
└── README.md                  # (file này)
```

---

## 11. Phân công nhóm

| Thành viên | Phân hệ phụ trách | Phạm vi chính |
|---|---|---|
| Thành viên 1 | Phân hệ 1 — Data Ingestion & Pipeline | Parser, Parent-Child Chunking, suy luận hiệu lực, embedding & nạp Qdrant |
| Thành viên 2 | Phân hệ 2 — Advanced Retrieval & Agent | Query Analyzer, Hybrid Search, Intersection & Fallback, Reranker, Agentic Prompting |
| Thành viên 3 | Phân hệ 3 — Backend API & UI/UX | FastAPI streaming, giao diện split-screen Chainlit |

---

## 12. Hạn chế đã biết & Hướng phát triển

**Hạn chế đã biết (chủ động ghi nhận, không che giấu khi đánh giá):**

- Suy luận `status` hiệu lực dựa trên Regex trên "Điều khoản thi hành" — có thể bỏ sót các trường hợp diễn đạt không theo mẫu câu chuẩn, hoặc quan hệ hiệu lực phức tạp (hết hiệu lực một phần theo từng Điều/Khoản cụ thể chưa được mô hình hóa).
- Embedding & Reranker dùng model pretrained, chưa fine-tune riêng cho văn phong/thuật ngữ pháp luật Việt Nam.
- Chưa có eval pipeline định lượng (retrieval hit-rate, answer accuracy) trên tập câu hỏi mẫu.

**Hướng phát triển tiếp theo:**

- Fine-tune BGE-M3/BGE-Reranker bằng hard-negative mining (dùng chính BM25 hiện có để mine negative khó) nếu có tập (câu hỏi, Điều luật đáp án) từ ban tổ chức.
- Mở rộng bảng quan hệ văn bản (`van_ban_relations`) thành knowledge-graph đầy đủ hơn nếu cần trả lời tốt các câu hỏi multi-hop về quan hệ giữa văn bản (sửa đổi, căn cứ, hướng dẫn).
- Xây dựng bộ eval tự động (retrieval hit-rate@k, answer accuracy) để đo lường định lượng trước khi tinh chỉnh tiếp.
