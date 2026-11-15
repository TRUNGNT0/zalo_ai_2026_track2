# Ingestion

Ingestion có một lõi đọc và chuyển kiểu dùng chung cho cả SQL và pandas:

```text
CSV / Excel / SQLite / Parquet
              ↓
  format adapter → SourceReader → RawTable + lineage
              ↓
  ValueNormalizer → Typed + conversion issues
              ↓
  BusinessIngestion → IngestedBusiness
       ┌──────┴────────┐
       ↓               ↓
store.py + catalog   datasets.loader
DuckDB + Semantic    pandas fallback
       ↓
SQLContextBuilder → SQLSolver
```

`IngestionPipeline` trong `processing.pipeline` là façade cho ingestion một
business: gọi `BusinessIngestion.ingest_business`, xuất database Raw/Typed/Semantic, xây catalog
tương thích SQL agent và sinh `DATABASE_GUIDE.md`. SQL agent dùng catalog có cấu
trúc từ chính database; guide là tài liệu hỗ trợ, không thay catalog trong prompt.

`DatasetLoader` gọi cùng `BusinessIngestion.ingest_business` khi nạp Typed DataFrames cho
baseline pandas. Đây là hai đích thực thi trên cùng reader/type conversion,
không phải hai bộ đọc/chuyển kiểu độc lập. `app.ingestion.run` điều phối nhiều
business và calendar, còn `processing.pipeline` giữ API/CLI tương thích cho một
business. Cả hai dùng chung ingestion, catalog, store và fingerprint.

`SourceReader.read_tables` dispatch theo phần mở rộng file qua `CsvAdapter`, `ExcelAdapter`,
`SqliteAdapter` hoặc `ParquetAdapter`. Các adapter dùng parser loss-aware trong
`app/ingestion/raw.py` để chuẩn hóa dữ liệu thô, nhờ đó ingestion và kiểm kê
manifest giữ chung provenance, lỗi đọc, metadata ô Excel và quan hệ SQLite.

Nếu thiếu manifest, ingestion tự tạo một BusinessManifest v2 nháp ở
`<split>/manifests/<business_id>.json` trước khi xây catalog (với API một
business ngoài cấu trúc split, file được đặt cạnh output database). Việc tạo
nháp chạy local, không gọi LLM: chỉ ánh xạ định nghĩa README rõ ràng và bằng
chứng trực tiếp; vai trò bảng, grain và các ý nghĩa chưa có bằng chứng được để
`unknown`/`unresolved`. Có thể chỉnh sửa/duyệt manifest rồi chạy lại ingestion.

Các lớp chính:

| Module | Trách nhiệm |
|---|---|
| `sources/adapters/` | Đọc định dạng CSV, Excel, SQLite, Parquet |
| `sources/dispatch.py` — `SourceReader` | Chọn adapter theo phần mở rộng file |
| `raw.py` — `RawData` | Chuẩn hóa `RawTable`, vị trí nguồn và metadata để typing/persistence dùng chung |
| `processing/normalization.py` — `IdentifierNormalizer`, `ValueNormalizer` | Chuẩn hóa định danh và chuyển kiểu bảo thủ; ghi nhận lỗi parse, giữ giá trị nguồn |
| `layered.py` — `BusinessIngestion` | Ghép Raw/Typed/semantic metadata cho một business |
| `store.py` — `IngestionStore` | Xuất DuckDB và báo cáo |
| `catalog.py` — `CatalogService` | Catalog vật lý/semantic, manifest, quan hệ, quality và view |
| `organization/` | Phân tích và tạo các view đã được kiểm tra |
| `datasets/loader.py` — `DatasetLoader` | Giao diện Split/Business cho inference và evaluation |
| `datasets/profiling.py` — `DatasetProfiler` | Tạo profile và gộp sheet cho prompt |
| `datasets/splitting.py` — `BusinessSplitter` | Chia nhóm business có tính xác định |
| `agent/sql_context.py` | Đưa catalog, completeness, views và rulebook vào SQL prompt |

Ví dụ import:

```python
from app.ingestion.datasets.loader import Business, DatasetLoader, Split
from app.ingestion.datasets.profiling import DatasetProfiler
from app.ingestion.datasets.splitting import BusinessSplitter
from app.ingestion.processing.pipeline import IngestionPipeline

split = DatasetLoader.load_split("dataset/train")
business = split.business("<business_id>")
profile = DatasetProfiler.profile_text(business.tables, limit=5000)
held_out = BusinessSplitter.holdout_businesses(["biz_a", "biz_b"], n_holdout=1)
```

Các lệnh CLI:

```bash
python -m app.ingestion.cli.ingest --help
python -m app.ingestion.cli.build_manifests --help
python -m app.ingestion.cli.audit_manifests --help
python -m app.ingestion.cli.upgrade_manifests --help
```

Các file `build_manifests.py`, `audit_manifests.py` và
`upgrade_manifests.py` ở gốc package chỉ chuyển tiếp sang CLI mới để giữ
tương thích với các lệnh `python -m app.ingestion.<tên cũ>`.
Các import Python khác dùng đường dẫn mới như ví dụ trên.

## SQL context và parse completeness

Catalog giữ contract hiện có của SQL agent: physical/source names, semantic
mapping, relationships, facts, unresolved issues và quality. Pipeline thêm
lineage Raw/Typed, parse failures, grain và các semantic views; không dùng
`DATABASE_GUIDE.md` làm thay thế cho metadata có cấu trúc.

SQL context đưa `parse_failed_count`, số NULL nguồn và tỷ lệ parse thành công
theo cột. Vì SQL `SUM` bỏ qua NULL, lỗi parse ở cột dùng để tính tổng là dấu
hiệu kết quả có thể thiếu; model không được xem NULL đó là 0 hoặc khẳng định
tổng đầy đủ khi không có chính sách xử lý đã xác minh.
