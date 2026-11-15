"""Structural validation and explicit coverage, independent of LLM assertions."""


class ManifestValidator:
    """Validate source bindings, semantic constraints and coverage."""

    COLUMN_CONTRACTS = {
        'line_amount': ('currency', 'sum'),
        'payment_amount': ('currency', 'sum'),
        'base_quantity': ('base_unit', 'sum'),
        'stock_balance': ('base_unit', 'last'),
        'unit_cost': ('sale_unit', 'none'),
    }
    TIME_ROLES = frozenset({'order_date', 'payment_date', 'movement_time', 'effective_date'})

    @staticmethod
    def issue(topic, reason, needed='Provide source-specific evidence for this mapping.'):
        return {'topic': topic, 'reason': reason, 'needed_evidence': needed}

    @staticmethod
    def validate(manifest: dict, inventory: dict, evidence_ids=None) -> dict:
        objects = {(t['source_file'], t['source_object']): t for t in inventory['objects']}
        errors = []
        unresolved = []
        mapped = set()
        seen = set()
        if manifest['schema_version'] != '2.0':
            # Legacy groups may span files/sheets. Each mapping must exist in at least
            # one selected source object; object names are never column mappings.
            for table in manifest['tables']:
                candidates = [t for key, t in objects.items() if key[0] in table['source_files']]
                for name in table['columns']:
                    matches = [(t, c) for t in candidates for c in t['columns']
                               if name in (c['source_name'], f"{t['source_object']}.{c['source_name']}")]
                    if not matches:
                        errors.append(f'Unknown source column: {table["source_files"]}/{name}')
            return {'errors': errors, 'unresolved': [], 'coverage': {}}
        for table in manifest['tables']:
            key = (table['source_file'], table['source_object'])
            if key in seen:
                errors.append(f'Duplicate source object: {key}')
            seen.add(key)
            source = objects.get(key)
            if source is None:
                errors.append(f'Unknown source object: {key}')
                continue
            if table['header_row_number'] != source['header_row_number']:
                errors.append(f'Header disagrees with observed source: {key}')
            available = {c['source_name'] for c in source['columns']}
            column_seen = set()
            for col in table['columns']:
                name = col['source_name']
                if name not in available:
                    errors.append(f'Unknown source column: {key}/{name}')
                if name in column_seen:
                    errors.append(f'Duplicate source column: {key}/{name}')
                column_seen.add(name)
                if col['status'] == 'documented':
                    if not col['semantic_name'] or not col['description'] or not col['source_refs']:
                        errors.append(f'Documented column lacks semantics/evidence: {key}/{name}')
                    else:
                        mapped.add((*key, name))
                if evidence_ids is not None:
                    unknown = set(col['source_refs']) - set(evidence_ids)
                    if unknown:
                        errors.append(f'Unknown column evidence: {sorted(unknown)}')
            if evidence_ids is not None and set(table['source_refs']) - set(evidence_ids):
                errors.append(f'Unknown table evidence: {key}')
        if evidence_ids is not None:
            for fact in manifest['facts']:
                if set(fact['source_refs']) - set(evidence_ids):
                    errors.append(f"Unknown fact evidence: {fact['fact_id']}")
            if set(manifest['rule_refs']) - set(evidence_ids):
                errors.append('Unknown rule references')
        relationship_ids = set()
        for rel in manifest['relationships']:
            if rel['relationship_id'] in relationship_ids:
                errors.append(f'Duplicate relationship ID: {rel["relationship_id"]}')
            relationship_ids.add(rel['relationship_id'])
            if len(rel['from']['columns']) != len(rel['to']['columns']):
                errors.append(f'Relationship key lengths differ: {rel["relationship_id"]}')
            for endpoint in (rel['from'], rel['to']):
                key = (endpoint['source_file'], endpoint['source_object'])
                source = objects.get(key)
                available = {c['source_name'] for c in source['columns']} if source else set()
                if not source or not set(endpoint['columns']) <= available:
                    errors.append(f'Invalid relationship endpoint: {key}/{endpoint["columns"]}')
            if rel['source'] != 'inferred_candidate' and not rel['source_refs']:
                errors.append(f'Relationship lacks evidence: {rel["relationship_id"]}')
            if evidence_ids is not None and set(rel['source_refs']) - set(evidence_ids):
                errors.append(f'Unknown relationship evidence: {rel["relationship_id"]}')
        total = 0
        for key, source in objects.items():
            if key not in seen:
                unresolved.append(ManifestValidator.issue(f'table:{key[0]}:{key[1]}', 'Source object is absent from manifest.'))
            if source.get('header_status') not in ('sqlite_schema', 'parquet_schema', 'manifest_header_match'):
                unresolved.append(ManifestValidator.issue(f'header:{key[0]}:{key[1]}', 'Header was selected heuristically.', 'Confirm the header using source documentation.'))
            for col in source['columns']:
                total += 1
                if (*key, col['source_name']) not in mapped:
                    unresolved.append(ManifestValidator.issue(f"column_mapping:{key[0]}:{key[1]}:{col['source_name']}", 'Observed column has no documented semantic mapping.'))
        return {'errors': errors, 'unresolved': unresolved,
                'coverage': {'source_objects': len(objects), 'manifest_objects': len(seen),
                             'source_columns': total, 'documented_columns': len(mapped),
                             'unresolved_columns': total - len(mapped)}}

    @staticmethod
    def merge_issues(*groups):
        result = {}
        for group in groups:
            for item in group:
                result.setdefault(item['topic'], item)
        return list(result.values())

    @staticmethod
    def validate_column_semantics(column):
        if column['status'] != 'documented':
            return []
        semantic = column['semantic_name']
        errors = []
        expected = ManifestValidator.COLUMN_CONTRACTS
        if semantic in expected:
            unit, aggregation = expected[semantic]
            if column['unit_basis'] not in (unit, 'unknown'):
                errors.append(f'{semantic} requires unit_basis={unit}')
            if column['aggregation'] not in (aggregation, 'unknown'):
                errors.append(f'{semantic} requires aggregation={aggregation}')
        if semantic in {'unit_price', 'unit_conversion_factor'} and column['aggregation'] == 'sum':
            errors.append(f'{semantic} cannot be summed as a line amount')
        if semantic in ManifestValidator.TIME_ROLES:
            if column['time_role'] not in (None, semantic):
                errors.append(f'{semantic} has conflicting time_role={column["time_role"]}')
        return errors
