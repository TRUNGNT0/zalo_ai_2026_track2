"""Offline v1 -> v2 upgrade. Preserve evidence; never invent undocumented meanings.

Run with --output-dir to stage, or --output-dir dataset/train/manifests to replace.
Optional per-business reviewed overlays can add mappings and relationships.
"""

from pathlib import Path
import argparse
import json
import re

from jsonschema import Draft202012Validator

from datamate.ingestion.processing.normalization import IdentifierNormalizer
from datamate.manifests.service import BusinessManifestService, SCHEMA_FILE
from datamate.manifests.validation import ManifestValidator
from datamate.storage.json_store import JsonStore

# These aliases classify existing descriptions, never source names alone.
SEMANTICS = {
    'mã đơn': 'order_id', 'mã hóa đơn': 'order_id', 'mã hàng': 'product_id',
    'mã sản phẩm': 'product_id', 'mã khách': 'customer_id', 'trạng thái đơn': 'order_status',
    'ngày đặt': 'order_date', 'ngày bán': 'order_date', 'số lượng': 'quantity',
    'thành tiền sau giảm': 'line_amount', 'thành tiền dòng': 'line_amount',
    'đơn giá': 'unit_price', 'hệ số quy đổi': 'unit_conversion_factor',
    'số tiền trả lần đó': 'payment_amount', 'số khách phải trả cho cả đơn': 'order_payable',
    'thành tiền của dòng (đã trừ giảm giá dòng)': 'line_amount',
    'số lượng đã đổi ra đơn vị cơ bản': 'base_quantity',
    'giá vốn một đơn vị bán': 'unit_cost',
    'tồn sau lần nhập/bán/trả đó, theo đơn vị cơ bản': 'stock_balance',
}


class ManifestMigrator:
    """Upgrade existing mappings and apply reviewed source-specific corrections."""

    @staticmethod
    def mapped_column(name, description, refs):
        semantic = SEMANTICS.get(description.casefold(), IdentifierNormalizer.normalize(description))
        unit = ('base_unit' if 'đơn vị cơ bản' in description else
                'sale_unit' if 'đơn vị bán' in description else
                'purchase_unit' if 'đơn vị nhập' in description else
                'currency' if semantic in {'line_amount', 'payment_amount', 'order_payable'} else 'unknown')
        aggregation = ('last' if semantic == 'stock_balance' else
                       'sum' if semantic in {'line_amount', 'payment_amount', 'quantity', 'base_quantity'} else 'none')
        return {'source_name': name, 'semantic_name': semantic, 'description': description,
                'status': 'documented', 'source_refs': refs, 'unit_basis': unit,
                'time_role': 'order_date' if semantic == 'order_date' else None,
                'aggregation': aggregation}

    @staticmethod
    def upgrade(legacy, business_input):
        tables = []
        for source in business_input['source_inventory']['objects']:
            columns, roles, refs = [], set(), set()
            for col in source['columns']:
                name = col['source_name']
                if name is None:
                    continue
                matches = []
                for table in legacy['tables']:
                    if source['source_file'] not in table['source_files']:
                        continue
                    for key in (name, f"{source['source_object']}.{name}"):
                        if key in table['columns']:
                            matches.append((table, table['columns'][key]))
                descriptions = {desc for _, desc in matches}
                if len(descriptions) == 1:
                    description = next(iter(descriptions))
                    col_refs = sorted({ref for t, _ in matches for ref in t['source_refs']})
                    columns.append(ManifestMigrator.mapped_column(name, description, col_refs))
                    roles.update(t['table_role'] for t, _ in matches)
                    refs.update(col_refs)
                else:
                    columns.append({'source_name': name, 'semantic_name': None, 'description': None,
                                    'status': 'unresolved', 'source_refs': [], 'unit_basis': 'unknown',
                                    'time_role': None, 'aggregation': 'unknown'})
            tables.append({'source_file': source['source_file'], 'source_object': source['source_object'],
                           'header_row_number': source['header_row_number'],
                           'table_role': next(iter(roles)) if len(roles) == 1 else 'unknown',
                           'grain': None, 'columns': columns, 'source_refs': sorted(refs)})
        facts = list(legacy['facts'])
        # Retain the full documented conventions instead of losing unstructured rules.
        for section in business_input['readme_sections']:
            facts.append({'fact_id': 'source_' + section['evidence_id'], 'value': section['text'],
                          'source_refs': [section['evidence_id']]})
        relationships = []
        for source in business_input['source_inventory']['objects']:
            for rel in source['relationships']:
                relationships.append({'relationship_id': f"sqlite:{source['source_file']}:{rel['from_table']}:{rel['constraint_id']}",
                                      'from': {'source_file': source['source_file'], 'source_object': rel['from_table'], 'columns': rel['from_columns']},
                                      'to': {'source_file': source['source_file'], 'source_object': rel['to_table'], 'columns': rel['to_columns']},
                                      'cardinality': 'many_to_one', 'source': 'sqlite_foreign_key',
                                      'source_refs': [f"profile:{source['source_file']}::{source['source_object']}"]})
        return {'schema_version': '2.0', 'business_id': legacy['business_id'], 'tables': tables,
                'facts': facts, 'relationships': relationships,
                'rule_refs': [r['evidence_id'] for r in business_input['rulebook_sections']],
                'unresolved': list(legacy['unresolved'])}

    @staticmethod
    def enrich_explicit_readme_definitions(manifest, sections):
        """Use only literal 'column is meaning' statements with a unique binding.

        Repeated headers across objects need explicit table qualification; no fuzzy
        matching or source-role guessing is used to resolve that ambiguity.
        """
        patterns = [
            (re.compile(r"`([^`]+)\.([^`]+)`\s+là\s+([^;\n]+)"), True),
            (re.compile(r"\*\*([^*]+)\*\*\s+là\s+([^;\n]+)"), False),
        ]
        proposals = {}
        for section in sections:
            file_section = re.search(r"^##\s+Cột trong `([^`]+)`", section['text'], re.M)
            if file_section:
                source_file = file_section.group(1)
                for name, description in re.findall(
                    r"^- \*\*(.*?)\*\*:\s*(.+)$", section['text'], re.M
                ):
                    candidates = [
                        (table, column)
                        for table in manifest['tables']
                        for column in table['columns']
                        if Path(table['source_file']).name == source_file
                        and column['source_name'] == name
                    ]
                    if len(candidates) == 1:
                        table, column = candidates[0]
                        key = (table['source_file'], table['source_object'], name)
                        proposals.setdefault(key, []).append(
                            (column, description.strip(), section['evidence_id'])
                        )
            for pattern, qualified in patterns:
                for match in pattern.finditer(section['text']):
                    if qualified:
                        object_name, name, description = match.groups()
                    else:
                        name, description = match.groups()
                        object_name = None
                    # Stop at sentence boundary; skip compound statements involving
                    # more marked-up columns, whose scope cannot be established here.
                    description = description.split('. ')[0].rstrip('.').strip()
                    if '**' in description or '`' in description:
                        continue
                    candidates = [(t, c) for t in manifest['tables'] for c in t['columns']
                                  if c['source_name'] == name and
                                  (object_name is None or t['source_object'] == object_name)]
                    if len(candidates) != 1:
                        continue
                    table, col = candidates[0]
                    if col['status'] == 'documented':
                        continue
                    key = (table['source_file'], table['source_object'], name)
                    proposals.setdefault(key, []).append((col, description, section['evidence_id']))
        for proposed in proposals.values():
            descriptions = {description for _, description, _ in proposed}
            if len(descriptions) != 1:
                continue
            col = proposed[0][0]
            col.update(ManifestMigrator.mapped_column(col['source_name'], proposed[0][1], sorted({p[2] for p in proposed})))

    @staticmethod
    def apply_overlay(manifest, overlay):
        by_source = {(t['source_file'], t['source_object']): t for t in manifest['tables']}
        for patch in overlay.get('tables', []):
            table = by_source[(patch['source_file'], patch['source_object'])]
            for key in ('table_role', 'grain'):
                if key in patch:
                    table[key] = patch[key]
            by_name = {c['source_name']: c for c in table['columns']}
            for column in patch.get('columns', []):
                by_name[column['source_name']].update(column)
        manifest['relationships'].extend(overlay.get('relationships', []))

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--dataset-dir', type=Path, default=Path('dataset/train/data'))
        parser.add_argument('--manifest-dir', type=Path, default=Path('dataset/train/manifests'))
        parser.add_argument('--output-dir', type=Path, required=True)
        parser.add_argument('--overlays-dir', type=Path, default=Path('dataset/train/manifest_overrides'))
        parser.add_argument('--business-id')
        args = parser.parse_args()
        schema = json.loads(SCHEMA_FILE.with_name('business_manifest.v2.schema.json').read_text())
        rulebook = (args.dataset_dir.parent / 'rulebook.md').read_text()
        for root in sorted(args.dataset_dir.iterdir()):
            if not root.is_dir() or (args.business_id and root.name != args.business_id):
                continue
            old = json.loads((args.manifest_dir / f'{root.name}.json').read_text())
            data = BusinessManifestService.build_input(root, rulebook, schema)
            previous_evidence = args.manifest_dir / 'evidence' / f'{root.name}.json'
            if previous_evidence.exists():
                previous = json.loads(previous_evidence.read_text())
                for section in previous.get('survey_results', []):
                    if section['evidence_id'] not in data['_evidence_ids']:
                        data['survey_results'].append({**section, 'text': 'Archived survey from v1 (not re-surveyed by the generic profiler):\n' + section['text']})
                        data['_evidence_ids'].append(section['evidence_id'])
            manifest = ManifestMigrator.upgrade(old, data) if old['schema_version'] == '1.0' else old
            ManifestMigrator.enrich_explicit_readme_definitions(manifest, data['readme_sections'])
            overlay_path = args.overlays_dir / f'{root.name}.json'
            if overlay_path.exists():
                # Apply overlays only once; relationship IDs identify existing overrides.
                overlay = json.loads(overlay_path.read_text())
                ids = {r['relationship_id'] for r in overlay.get('relationships', [])}
                manifest['relationships'] = [r for r in manifest['relationships'] if r['relationship_id'] not in ids]
                ManifestMigrator.apply_overlay(manifest, overlay)
            report = ManifestValidator.validate(manifest, data['source_inventory'], data['_evidence_ids'])
            if report['errors']:
                raise ValueError(f'{root.name}: {report["errors"]}')
            manifest['unresolved'] = ManifestValidator.merge_issues(
                [i for i in manifest['unresolved'] if not i['topic'].startswith(('column_mapping:', 'header:', 'table:'))],
                report['unresolved'])
            Draft202012Validator(schema).validate(manifest)
            JsonStore.write(manifest, args.output_dir / f'{root.name}.json')
            JsonStore.write(BusinessManifestService.build_evidence(data), args.output_dir / 'evidence' / f'{root.name}.json')
            JsonStore.write(report, args.output_dir / 'reports' / f'{root.name}.json')
            print(root.name, report['coverage'], flush=True)
