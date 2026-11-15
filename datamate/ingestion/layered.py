"""Business ingestion API: raw rows, typed frames and reviewable semantic metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from datamate.text import colnorm
from .raw import RawTable, RawData
from .sources.dispatch import ADAPTERS
from .processing.normalization import ValueNormalizer


@dataclass
class IngestedTable:
    raw: RawTable
    typed: pd.DataFrame
    columns: list[dict]
    flags: list[list[str]]
    issues: list[dict]


@dataclass
class IngestedBusiness:
    business_id: str
    readme: str
    rulebook: str
    tables: dict[str, IngestedTable] = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)
    semantic: dict = field(default_factory=dict)
    manifest: dict | None = None
    input_fingerprint: str | None = None

    @property
    def frames(self):
        """Typed frames with original names. Provenance is separate from data columns."""
        return {name: table.typed for name, table in self.tables.items()}

class BusinessIngestion:
    @staticmethod
    def ingest_calendar(path, *, rulebook=""):
        """Shared challenge calendar, with an explicit schema for its known columns."""
        path = Path(path)
        raw = RawData.read_tables(path, path.parent, {})[0]
        columns = {name: {"type": kind} for name, kind in {
            "date": "date", "weekday": "integer", "lunar_day": "integer", "lunar_month": "integer",
            "lunar_year": "integer", "lunar_leap": "boolean", "holiday": "string",
        }.items() if name in raw.headers}
        frame, metadata, flags, issues = ValueNormalizer.convert_table(raw, "", {"columns": columns})
        result = IngestedBusiness("_shared", "Lịch dùng chung của split", rulebook)
        result.tables[raw.name] = IngestedTable(raw, frame, metadata, flags, issues)
        result.semantic = {"views": {}, "policy": "Shared calendar with explicit column types"}
        return result

    @staticmethod
    def ingest_business(directory, *, rulebook="", overrides=None, strict=True, manifest=None):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        overrides = overrides or {}
        readme_path = directory / "README.md"
        readme = readme_path.read_text(encoding="utf-8-sig") if readme_path.exists() else ""
        result = IngestedBusiness(directory.name, readme, rulebook, manifest=manifest)
        paths = sorted(p for p in directory.rglob("*") if p.is_file()
                       and not any(part.startswith(".") for part in p.relative_to(directory).parts))
        unknown_files = set(overrides.get("files", {})) - {p.relative_to(directory).as_posix() for p in paths}
        if unknown_files:
            raise ValueError(f"Unknown override files: {sorted(unknown_files)}")
        for path in paths:
            if not path.is_file() or path.suffix.lower() in (".md", ".txt"):
                continue
            if path.suffix.casefold() not in ADAPTERS:
                problem = RawData.issue("unsupported_file", file=str(path), review_required=True)
                result.issues.append(problem)
                if strict:
                    raise ValueError(f"Unsupported input: {path}")
                continue
            try:
                relative = path.relative_to(directory).as_posix()
                file_options = dict(overrides.get("files", {}).get(relative, {}))
                if manifest:
                    from datamate.catalog.builder import CatalogBuilder
                    if manifest.get("schema_version") == "2.0":
                        file_options["expected_headers"] = [c["source_name"] for t in manifest["tables"]
                                                             if t["source_file"] == relative for c in t["columns"]]
                    else:
                        file_options["expected_headers"] = [c.rsplit(".", 1)[-1] for t in manifest["tables"]
                                                             if relative in t["source_files"] for c in t["columns"]]
                for raw in RawData.read_tables(path, directory, file_options):
                    if raw.name in result.tables:
                        raise ValueError(f"Logical table collision: {raw.name}; rename source files")
                    options = dict(overrides.get("tables", {}).get(raw.name, {}))
                    if manifest:
                        _, descriptions, _ = CatalogBuilder.choose_table(
                            manifest, relative, raw.source["source_object"], tuple(raw.headers))
                        columns = {c: {"description": d} for c, d in descriptions.items() if c in raw.headers}
                        for c, spec in options.get("columns", {}).items():
                            columns[c] = columns.get(c, {}) | spec
                        options["columns"] = columns
                    frame, columns, flags, issues = ValueNormalizer.convert_table(raw, readme, options)
                    frame.attrs["source"] = raw.source
                    frame.attrs["positions"] = raw.positions
                    result.tables[raw.name] = IngestedTable(raw, frame, columns, flags, issues)
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"Ingestion failed for {path}: {exc}") from exc
                result.issues.append(RawData.issue("file_read_failed", file=str(path), error=str(exc), review_required=True))
        unknown_tables = set(overrides.get("tables", {})) - set(result.tables)
        if unknown_tables:
            raise ValueError(f"Unknown override tables: {sorted(unknown_tables)}")
        if not result.tables and strict:
            raise ValueError(f"No readable source tables found in {directory}")
        return BusinessIngestion.finish_semantic(result, overrides)

    @staticmethod
    def finish_semantic(result, overrides):
        keys = []
        aliases = {}
        for name, table in result.tables.items():
            for column in table.columns:
                aliases.setdefault(column["search_alias"], []).append({"table": name, "column": column["name"]})
                if column["identifier"]:
                    series = table.typed[column["name"]].dropna()
                    keys.append({"table": name, "column": column["name"], "alias": colnorm(column["name"]),
                                 "non_null": len(series), "unique": int(series.nunique()),
                                 "unique_non_null": bool(len(series) and series.is_unique)})
        # Names alone are insufficient evidence to establish a foreign key.
        result.semantic = {"aliases": aliases, "key_candidates": keys,
                           "confirmed_links": overrides.get("links", []),
                           "views": overrides.get("views", {}),
                           "policy": "Identity views by default; no inferred revenue, joins or unit conversions."}
        return result
