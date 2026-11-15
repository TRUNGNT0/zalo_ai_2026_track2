"""Nạp một split (questions/answers/rulebook/calendar) và dữ liệu từng business."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Business:
    tables: dict
    readme: str
    cache: dict = field(default_factory=dict)   # kết quả dựng sẵn dùng lại giữa các câu (bảng đã gộp, profile...)
    ingestion: object | None = None  # Raw/Typed/Semantic audit when layered ingestion is requested


@dataclass
class Split:
    root: str
    questions: pd.DataFrame
    answers: pd.DataFrame | None
    rulebook: str
    calendar: pd.DataFrame | None
    _businesses: dict = field(default_factory=dict)
    layered: bool = False
    ingestion_overrides: dict = field(default_factory=dict)

    def business(self, biz_id) -> Business:
        if biz_id not in self._businesses:
            directory = f"{self.root}/data/{biz_id}"
            readme_path = f"{directory}/README.md"
            readme = open(readme_path, encoding="utf-8").read() if os.path.exists(readme_path) else ""
            if self.layered:
                from ..layered import BusinessIngestion
                from ..catalog import CatalogService
                from pathlib import Path
                manifest = Path(self.root) / "manifests" / f"{biz_id}.json"
                ingested = BusinessIngestion.ingest_business(directory, rulebook=self.rulebook,
                                           overrides=self.ingestion_overrides.get(biz_id),
                                           manifest=CatalogService.load_manifest(manifest if manifest.exists() else None, biz_id))
                self._businesses[biz_id] = Business(tables=ingested.frames, readme=readme, ingestion=ingested)
            else:
                self._businesses[biz_id] = Business(tables=DatasetLoader.load_tables(directory), readme=readme)
        return self._businesses[biz_id]

class DatasetLoader:
    @staticmethod
    def read_jsonl(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def find_root_dir(split_dir: str) -> str:
        """Tìm thư mục chứa questions.jsonl (phòng khi cấu trúc lồng thêm một cấp)."""
        for dirpath, _dirs, files in os.walk(split_dir):
            if "questions.jsonl" in files:
                return dirpath
        raise FileNotFoundError(split_dir)

    @staticmethod
    def fix_header(df: pd.DataFrame, min_fill: float = 0.6) -> pd.DataFrame:
        """Sheet kiểu sổ tay: tiêu đề nằm ở dòng 1-3 chứ không phải dòng 0 (cột thành "Unnamed: n") -> dời tiêu đề lên."""
        cols = [str(c) for c in df.columns]
        if len(cols) < 2 or sum(c.startswith("Unnamed") for c in cols) / len(cols) < 0.5:
            return df
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            if row.notna().sum() >= min_fill * len(cols) and all(isinstance(v, str) for v in row.dropna()):
                new = df.iloc[i + 1:].copy()
                names, seen = [], {}
                for j, v in enumerate(row):
                    name = str(v).strip() if pd.notna(v) else f"col_{j}"
                    seen[name] = seen.get(name, 0) + 1
                    names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
                new.columns = names
                return new.reset_index(drop=True).infer_objects()
        return df

    @staticmethod
    def load_tables(biz_dir: str) -> dict[str, pd.DataFrame]:
        """Compatibility API: use the same readers and typing as SQL ingestion."""
        try:
            from ..layered import BusinessIngestion
            return BusinessIngestion.ingest_business(biz_dir).frames
        except Exception:
            from datamate.ingest import load_business
            p = Path(biz_dir)
            db, _ = load_business(p.parent.parent, p.name)
            frames = {}
            for (t,) in db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall():
                frames[t] = db.execute(f'SELECT * FROM "{t}"').df()
            db.close()
            return frames

    @staticmethod
    def load_split(split_dir: str, *, layered: bool = False, ingestion_overrides: dict | None = None) -> Split:
        root = DatasetLoader.find_root_dir(split_dir)
        questions = pd.DataFrame(DatasetLoader.read_jsonl(f"{root}/questions.jsonl"))
        answers_path = f"{root}/answers.jsonl"
        answers = pd.DataFrame(DatasetLoader.read_jsonl(answers_path)) if os.path.exists(answers_path) else None
        rulebook_path, calendar_path = f"{root}/rulebook.md", f"{root}/calendar.csv"
        rulebook = open(rulebook_path, encoding="utf-8").read() if os.path.exists(rulebook_path) else ""
        if os.path.exists(calendar_path):
            try:
                from ..layered import BusinessIngestion
                calendar = BusinessIngestion.ingest_calendar(calendar_path).frames["calendar"]
            except Exception:
                calendar = pd.read_csv(calendar_path)
        else:
            calendar = None
        return Split(root=root, questions=questions, answers=answers, rulebook=rulebook, calendar=calendar,
                     layered=layered, ingestion_overrides=ingestion_overrides or {})
