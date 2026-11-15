"""Cấu hình chạy: một dataclass thay cho các biến toàn cục của notebook."""
from __future__ import annotations

from dataclasses import dataclass, field

BACKENDS = ("local", "gemini")
LOCAL_ENGINES = ("auto", "vllm", "hf")
HF_QUANT_MODES = ("auto", "4bit", "none")


@dataclass
class Config:
    # --- dữ liệu ---
    train_dir: str = "/kaggle/input/datasets/trungnguyen0510/train-track2-zaloai/train"
    test_dir: str = "/kaggle/input/datasets/trungnguyen0510/testpublic-track2-zaloai/public_test"
    out_dir: str = "/kaggle/working"

    # --- phạm vi chạy ---
    max_biz: int | None = None        # số business train để chạy thử; None = tất cả (muốn thử nhanh: --max-biz 1)
    per_biz: int | None = None        # tối đa ngần này câu mỗi business; None = hết
    limit_test: int | None = None     # số câu public test tối đa; None = tất cả
    time_budget_s: float | None = None  # hết giờ thì dừng gọi model, câu còn lại ghi abstain

    # --- luật tất định (mặc định tắt) ---
    use_rules: bool = False
    rule_on: dict = field(default_factory=lambda: {"top": True, "low": True, "revenue": True})

    # --- sinh nhiều lần + đồng thuận ---
    n_samples: int = 3                # số lần sinh độc lập tối đa cho một câu
    agree_min: int = 2                # cần ít nhất ngần này lần ra CÙNG kết quả mới trả lời
    max_retry: int = 2                # số lần tự sửa lỗi cho mỗi lời giải
    code_timeout_s: int = 60          # giới hạn thời gian chạy một đoạn code
    prompt_version: str = "v7"        # đổi khi sửa prompt để vô hiệu cache cũ
    use_helpers: bool = True          # nêu các hàm trợ giúp (to_number, period_range, order_state...) trong prompt
    collect_all: bool = False         # sinh ĐỦ n_samples (không dừng sớm khi đã đồng thuận): để thu dữ liệu SFT/few-shot
    rulebook_select: bool = False     # chỉ đưa các mục rulebook liên quan câu hỏi (prompt ngắn hơn; phải đo PA có/không dùng)
    run_name: str = ""                # nhãn lần chạy: tách cache và file kết quả giữa các model/cấu hình
    min_agree_by_type: dict = field(default_factory=dict)   # vd {"table": 3, "unanswerable": 3}: loại đáp án cần đồng thuận cao hơn

    # --- gợi ý tất định đưa vào prompt ---
    use_hints: bool = True            # khoảng thời gian đã quy đổi + giá trị khớp trong dữ liệu (do mã tính, không do LLM)
    exemplar_path: str | None = None  # kho (câu hỏi, lời giải đúng) từ tập train; None = không dùng few-shot truy xuất
    exemplar_k: int = 2               # số ví dụ tương tự đưa vào prompt
    fewshot_bank: str | None = "assets/fewshot_bank.jsonl"
    fewshot_k: int = 4

    # --- view chuẩn & công nợ (DataMate semantic layer) ---
    views: str = "cached"             # "cached" (dùng view đã kiểm thử) | "llm" (draft bằng LLM nếu thiếu) | "none"
    views_from: str | None = "configs/public_v9"  # thư mục chứa _views_<shop>.json, _debt_<shop>.json
    debt_views: bool = True           # nạp view công nợ v_debt_orders, v_debt_pay và macro debt_at
    hide_raw: bool = False            # ẩn bảng thô khi đã có canonical views
    answer_unanswerable: bool = True  # cho phép trả lời unanswerable khi đồng thuận
    canon_names: bool = True          # chuẩn hóa tên shop/bảng khi nạp
    dataset_dir: str = "dataset"      # thư mục chứa train/ và public_test/

    # --- backend LLM ---
    backend: str = "local"            # "local" | "gemini"
    query_engine: str = "sql"         # sql (DuckDB) | pandas (baseline)
    database_dir: str = "data/cache"
    manifest_dir: str | None = None
    ingestion_mode: str = "layered"  # "direct" reads sources into DuckDB without manifests/catalog preparation
    require_prepared_database: bool = False
    ingestion_overrides: dict = field(default_factory=dict)
    rebuild_database: bool = False
    rewrite_questions: bool = True
    limit_train: int | None = None
    gemini_model: str = "gemini-3.5-flash"
    thinking: str = "low"             # thinking_level của Gemini
    local_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"   # repo HF hoặc đường dẫn thư mục
    local_lora: str | None = None     # thư mục LoRA adapter sau khi fine-tune
    lora_rank: int = 64
    local_engine: str = "auto"        # "auto" | "vllm" | "hf"
    hf_base_model: str | None = None  # model gốc fp16 khi engine hf gặp bản AWQ/GPTQ
    hf_quant: str = "auto"            # engine hf: "auto" (nén 4-bit chỉ khi VRAM không đủ cho bf16) | "4bit" | "none" (bf16 đầy đủ, nhanh hơn)
    tensor_parallel: int = 1
    vllm_extra: dict = field(default_factory=lambda: {"enforce_eager": True})
    vllm_env: dict = field(default_factory=lambda: {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    max_model_len: int = 26384        # tổng token tối đa (prompt + trả lời)
    max_new_tokens: int = 6096
    prefill_chunk: int = 512          # engine hf: nạp prompt từng đoạn để khỏi OOM (0 = tắt)
    gpu_memory_utilization: float = 0.88
    temperature: float = 0.7          # phải > 0 để các lần sinh độc lập khác nhau
    chat_kwargs: dict = field(default_factory=dict)
    allow_cpu: bool = False           # False = backend local không có GPU thì dừng ngay
    generation_timeout: float | None = 600.0  # Giới hạn thời gian sinh một lần (giây)

    def __post_init__(self):
        if self.query_engine not in ("sql", "pandas"):
            raise ValueError("query_engine phải là sql hoặc pandas")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend phải thuộc {BACKENDS}, nhận {self.backend!r}")
        if self.local_engine not in LOCAL_ENGINES:
            raise ValueError(f"local_engine phải thuộc {LOCAL_ENGINES}, nhận {self.local_engine!r}")
        if self.hf_quant not in HF_QUANT_MODES:
            raise ValueError(f"hf_quant phải thuộc {HF_QUANT_MODES}, nhận {self.hf_quant!r}")
        if self.ingestion_mode not in ("direct", "layered"):
            raise ValueError("ingestion_mode phải là direct hoặc layered")
        if not 1 <= self.agree_min <= self.n_samples:
            raise ValueError("cần 1 <= agree_min <= n_samples")
        if any(v > self.n_samples for v in self.min_agree_by_type.values()):
            raise ValueError("min_agree_by_type không được lớn hơn n_samples")
        if self.n_samples > 1 and self.temperature <= 0:
            raise ValueError("n_samples > 1 cần temperature > 0, nếu không các lần sinh giống hệt nhau")

    @property
    def is_local(self) -> bool:
        return self.backend == "local"

    @property
    def context_limits(self) -> tuple[int, int, int]:
        """(rulebook, readme, profile) tính bằng ký tự. Model local có ngữ cảnh nhỏ hơn Gemini nhiều."""
        return (16000, 4000, 16000) if self.is_local else (30000, 15000, 40000)

    @property
    def token_budget(self) -> int | None:
        """Ngân sách token cho phần ngữ cảnh (chỉ local); 800 token dự phòng cho system + lần retry."""
        return self.max_model_len - self.max_new_tokens - 800 if self.is_local else None
