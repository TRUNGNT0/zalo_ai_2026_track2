"""Manifest-independent source inspection shared by generation and validation."""

from pathlib import Path
from typing import Any, Iterator
import hashlib
import json
import random

from datamate.ingestion.sources.adapters.common import SourceTable
from datamate.ingestion.sources.dispatch import SourceReader

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".parquet",
}


class SourceInventory:
    """Discover source objects, profile their data and emit evidence."""

    @staticmethod
    def read(path: Path, root: Path, expected_headers=()) -> Iterator[SourceTable]:
        relative = path.relative_to(root).as_posix()
        yield from SourceReader.read_tables(
            path, relative, {"expected_headers": list(expected_headers)}
        )

    @staticmethod
    def inspect(root: Path) -> dict[str, Any]:
        """Scan every object, with exact counts and bounded, deterministic samples.

        Distinct counts are exact only up to the cap. Samples never prove uniqueness.
        A heuristic header remains explicitly unconfirmed.
        """
        objects = []
        fingerprints = {}
        for path in SourceInventory.discover(root):
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            fingerprints[relative] = digest.hexdigest()
            for source in SourceInventory.read(path, root):
                rng = random.Random(0)
                n = 0
                profiles = [dict(null_count=0, negative_count=0, values=set(), distinct_capped=False,
                                 sample=[], observed_types=set()) for _ in source.source_headers]
                for row in source.rows:
                    n += 1
                    if len(row) != len(profiles):
                        raise ValueError(f'Invalid row width: {relative}/{source.source_object}')
                    for profile, value in zip(profiles, row):
                        if value is None:
                            profile['null_count'] += 1
                            continue
                        rendered = str(value)
                        profile['observed_types'].add(type(value).__name__)
                        if isinstance(value, (int, float)) and value < 0:
                            profile['negative_count'] += 1
                        if len(profile['values']) < 1000:
                            profile['values'].add(rendered)
                        elif rendered not in profile['values']:
                            profile['distinct_capped'] = True
                        sample = profile['sample']
                        if len(sample) < 5:
                            sample.append(rendered[:160])
                        else:
                            index = rng.randrange(n)
                            if index < 5:
                                sample[index] = rendered[:160]
                columns = []
                for i, (name, profile) in enumerate(zip(source.source_headers, profiles)):
                    values = profile.pop('values')
                    profile['observed_types'] = sorted(profile['observed_types'])
                    # Avoid forwarding free-text customer/contact values to the LLM.
                    if name and any(term in name.casefold() for term in ('khach', 'khách', 'phone', 'email', 'dien_thoai', 'điện thoại', 'diachi', 'địa chỉ')):
                        profile['sample'] = []
                    profile['distinct_count'] = None if profile['distinct_capped'] else len(values)
                    profile['distinct_lower_bound'] = len(values)
                    columns.append({'source_name': name, 'position': i + 1,
                                    'source_type': source.source_types[i] if source.source_types else None,
                                    **profile})
                objects.append({'source_file': relative, 'source_object': source.source_object,
                                'header_row_number': source.header_row_number,
                                'header_status': source.header_status, 'row_count': n,
                                'columns': columns, 'relationships': list(source.relationships)})
        return {'files': fingerprints, 'objects': objects}

    @staticmethod
    def evidence(inventory: dict) -> list[dict[str, str]]:
        return [{'evidence_id': f"profile:{t['source_file']}::{t['source_object']}",
                 'text': json.dumps(t, ensure_ascii=False)} for t in inventory['objects']]

    @staticmethod
    def discover(business_dir: Path) -> list[Path]:
        if not business_dir.is_dir():
            raise NotADirectoryError(f"Business input directory does not exist: {business_dir}")

        return sorted(
            path
            for path in business_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        )
