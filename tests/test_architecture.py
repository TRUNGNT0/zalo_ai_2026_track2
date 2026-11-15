"""Protect offline ingestion boundaries and supported command-line entry points."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class TestArchitecture:
    def test_ingestion_core_does_not_import_llm(self):
        # Functions and classes are both public APIs in the merged implementation.
        # The boundary we need to protect is deterministic ingestion, not syntax style.
        script = """
import sys
import datamate.ingestion.processing.pipeline
import datamate.ingestion.datasets.loader
from pathlib import Path
from tempfile import TemporaryDirectory
from datamate.ingestion.processing.pipeline import IngestionPipeline
with TemporaryDirectory() as tmp:
    root = Path(tmp)
    business = root / 'split' / 'data' / 'b1'
    business.mkdir(parents=True)
    (business / 'items.csv').write_text('ID,Amount\\n001,10\\n')
    IngestionPipeline.ensure_manifest(business, root / 'out' / 'b1')
assert not any(name.startswith(('app.llm', 'google.genai')) for name in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_all_application_modules_import(self):
        for path in (ROOT / "app").rglob("*.py"):
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            importlib.import_module(module)

    @pytest.mark.parametrize(
        "module",
        ["build_manifests", "audit_manifests", "upgrade_manifests", "pipeline",
         "cli.build_manifests", "cli.audit_manifests", "cli.upgrade_manifests", "cli.ingest"],
    )
    def test_existing_cli_entry_points(self, module):
        result = subprocess.run(
            [sys.executable, "-m", f"app.ingestion.{module}", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--help" in result.stdout

    @pytest.mark.parametrize("cli_prefix", ["", "cli."])
    def test_offline_cli_migration_audit_and_ingestion(self, tmp_path, cli_prefix):
        dataset = tmp_path / "data"
        business = dataset / "biz_cli"
        business.mkdir(parents=True)
        (business / "README.md").write_text(
            "# Dữ liệu\n## Quy ước\n**Amount** là thành tiền dòng.\n",
            encoding="utf-8",
        )
        (business / "sales.csv").write_text("ID,Amount\n001,10\n002,20\n")
        (tmp_path / "rulebook.md").write_text("# Rules\n## 1. Sales\nRules\n")
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "biz_cli.json").write_text(json.dumps({
            "schema_version": "1.0",
            "business_id": "biz_cli",
            "tables": [{
                "table_role": "sales_records",
                "source_files": ["sales.csv"],
                "columns": {"Amount": "thành tiền dòng"},
                "source_refs": ["readme.conventions"],
            }],
            "facts": [],
            "unresolved": [],
        }))
        generated = tmp_path / "generated"
        output = tmp_path / "catalog"
        commands = [
            ["upgrade_manifests", "--dataset-dir", str(dataset),
             "--manifest-dir", str(legacy), "--output-dir", str(generated)],
            ["audit_manifests", "--dataset-dir", str(dataset),
             "--manifest-dir", str(generated), "--output-dir", str(tmp_path / "audit")],
            ["pipeline", "--business-dir", str(business),
             "--manifest", str(generated / "biz_cli.json"), "--output-dir", str(output)],
        ]
        for module, *arguments in commands:
            if cli_prefix:
                module = cli_prefix + ("ingest" if module == "pipeline" else module)
            result = subprocess.run(
                [sys.executable, "-m", f"app.ingestion.{module}", *arguments],
                cwd=ROOT, text=True, capture_output=True, timeout=30,
            )
            assert result.returncode == 0, result.stderr
        catalog = json.loads((output / "catalog.json").read_text())
        assert catalog["tables"][0]["row_count"] == 2
        assert catalog["quality"]["coverage"]["documented_columns"] == 1
