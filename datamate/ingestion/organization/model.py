"""Stage 3: lossless order/line projections and explicitly scoped aggregates.

Offline deterministic rules. No revenue policy, as-of filter, unit conversion,
rounding, source deletion, or arbitrary functional-dependency normalization.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from datamate.text import colnorm, norm
from ..raw import RawData
from .analyze import OrganizationAnalyzer
from .build import OrganizationBuilder

MODEL_POLICY = 'organization-model-v1'

class OrganizationModelBuilder:
    @staticmethod
    def role(column):
        """Specific source-field meanings; never infer additivity from numeric type."""
        full_description = norm(column.get('description') or '')
        description = re.split(r'[;:(,]', full_description)[0].strip()
        name = colnorm(re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', column['name']))
        domain = OrganizationAnalyzer.concept(column)
        if domain:
            return domain
        names = {
            'order_date': {'ordered_at', 'order_date', 'ngay_ban', 'ngayban', 'ngay_dat_hang'},
            'order_status': {'order_status', 'trang_thai_don', 'trang_thai_don_hang'},
            'customer_name': {'customer_name', 'ten_khach', 'ten_khach_hang'},
            'order_channel': {'order_channel', 'sales_channel'},
            'order_total': {'order_total', 'total_bill', 'tong_bill'},
            'item_name': {'product_name', 'item_name', 'ten_san_pham', 'ten_hang', 'ten_hang_hoa'},
            'line_amount': {'line_amount', 'line_total', 'item_amount'},
            'quantity': {'quantity', 'qty', 'so_luong', 'soluong', 'sl'},
            'unit': {'unit', 'uom', 'dvt', 'don_vi_tinh', 'don_vi'},
            'currency': {'currency', 'currency_code', 'tien_te', 'loai_tien'},
            'payment_amount': {'payment_amount', 'paid_amount'},
            'event_time': {'paid_at', 'payment_time', 'thoi_gian_thu'},
        }
        meanings = {
            'order_date': ('ngay ban', 'ngay dat', 'ngay dat hang'),
            'order_status': ('trang thai don',),
            'customer_name': ('ten khach',),
            'order_channel': ('kenh ban',),
            'order_total': ('tong tien don',),
            'item_name': ('ten hang', 'ten san pham'),
            'line_amount': ('thanh tien sau giam', 'thanh tien dong hang'),
            'quantity': ('so luong',),
            'unit': ('don vi tinh',),
            'currency': ('ma tien te', 'loai tien'),
            'event_time': ('thoi diem thu',),
        }
        found = {r for r, aliases in names.items() if name in aliases}
        for r, prefixes in meanings.items():
            if any(description == p or description.startswith(p + ' ') for p in prefixes):
                # 'số lượng đã đổi ra đơn vị cơ bản' is not literal source quantity.
                if r == 'quantity' and description != 'so luong':
                    continue
                found.add(r)
        if 'quantity' in found and (any(token in full_description for token in ('quy doi', 'don vi co ban', 'da doi')) or any(token in name for token in ('quy_doi', 'base_quantity', 'so_luong_le'))):
            found.discard('quantity')
        return next(iter(found)) if len(found) == 1 else None

    @staticmethod
    def cleanup_models(con):
        exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='model_transformations'").fetchone()[0]
        if not exists:
            return
        current = OrganizationBuilder.catalog(con)
        for (text,) in con.execute("SELECT details_json FROM meta.model_transformations WHERE status='applied' ORDER BY transformation_id").fetchall():
            op = json.loads(text)
            for output in op.get('outputs', {}).values():
                name = output['name']
                fingerprint = output.get('definition_fingerprint')
                if name in current and fingerprint is not None and fingerprint == OrganizationBuilder.definition_fingerprint(con, current[name]):
                    con.execute('DROP VIEW semantic.' + RawData.quote_identifier(name))

    @staticmethod
    def build_models(con, analysis):
        """Caller cleans stage-3 views before rebuilding stage 2; owns transaction."""
        raw_tables = {}
        for name, db, lineage, source, columns in con.execute('SELECT logical_name, db_name, lineage_column, source_json, columns_json FROM meta.tables ORDER BY logical_name').fetchall():
            raw_tables[name] = {'name': name, 'db': db, 'lineage': lineage, 'source': json.loads(source),
                                'columns': json.loads(columns)}
        profile = {(p['table'], p['column']): p for p in analysis['profiles']}
        sources, covered = [], set()
        for op in analysis['stage2']['transformations']:
            if op['kind'] != 'merged' or op['status'] != 'applied':
                continue
            first = op['sources'][0]
            trace = op['lineage_columns']
            sources.append({'name': op['output_name'], 'ref': 'semantic.' + RawData.quote_identifier(op['output_name']),
                            'members': op['sources'], 'columns': raw_tables[first]['columns'],
                            'trace_columns': list(trace.values()), 'trace': trace,
                            'column_mapping': op['column_mapping'], 'logical_role': OrganizationAnalyzer.table_concept(first)})
            covered.update(op['sources'])
        for name, t in raw_tables.items():
            if name not in covered:
                sources.append({'name': name, 'ref': 'typed.' + RawData.quote_identifier(t['db']), 'members': [name],
                                'columns': t['columns'], 'trace_columns': [t['lineage']],
                                'trace': None, 'logical_role': OrganizationAnalyzer.table_concept(name)})
        sources.sort(key=lambda s: s['name'])
        operations, validations, source_cache = [], [], {}
        occupied = {name.casefold() for (name,) in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='semantic'").fetchall()}

        def check(op, name, passed, **details):
            validations.append({'transformation_id': op['id'], 'check': name, 'passed': bool(passed), 'details': json.loads(json.dumps(details, default=str))})
            return bool(passed)

        def operation(kind, source):
            identifier = OrganizationAnalyzer.stable_id(kind, [MODEL_POLICY, source['ref']])
            label = colnorm(source['name'].split('::')[0])[:28] or 'data'
            op = {'id': identifier, 'kind': kind, 'source': source['name'], 'source_ref': source['ref'], 'members': source['members'],
                  'status': 'candidate', 'reason': 'insufficient_evidence', 'outputs': {}, 'policy_version': MODEL_POLICY,
                  'scope': 'All source records and statuses; no as_of_date, cancellation/return filter or business rounding.',
                  'label': label}
            operations.append(op)
            return op

        def parse_count(s, column):
            return sum(profile[n, s.get('column_mapping', {}).get(n, {}).get(column, column)]['parse_failed_count']
                       for n in s['members'])

        def source_stats(s):
            if s['ref'] in source_cache:
                return source_cache[s['ref']]
            n = con.execute('SELECT count(*) FROM ' + s['ref']).fetchone()[0]
            structural, raw_ok = 0, True
            for member in s['members']:
                t = raw_tables[member]
                structural += con.execute(f"SELECT count(*) FROM raw.{RawData.quote_identifier(t['db'])} WHERE flags_json LIKE '%repeated_header%' OR flags_json LIKE '%possible_total%' OR flags_json LIKE '%blank_row%'").fetchone()[0]
                total, unique = con.execute(f'SELECT count(*),count(DISTINCT {RawData.quote_identifier(t["lineage"])}) FROM typed.{RawData.quote_identifier(t["db"])}').fetchone()
                raw_total, raw_unique = con.execute(f'SELECT count(*),count(DISTINCT row_id) FROM raw.{RawData.quote_identifier(t["db"])}').fetchone()
                orphans = con.execute(f'SELECT count(*) FROM typed.{RawData.quote_identifier(t["db"])} t WHERE NOT EXISTS (SELECT 1 FROM raw.{RawData.quote_identifier(t["db"])} r WHERE r.row_id=t.{RawData.quote_identifier(t["lineage"])})').fetchone()[0]
                raw_ok &= total == unique == raw_total == raw_unique and orphans == 0
            trace = ','.join(map(RawData.quote_identifier, s['trace_columns']))
            unique_trace = con.execute(f'SELECT count(*) FROM (SELECT DISTINCT {trace} FROM {s["ref"]})').fetchone()[0]
            raw_ok &= n == unique_trace
            source_cache[s['ref']] = {'rows': n, 'structural_rows': structural, 'raw_lineage_valid': bool(raw_ok)}
            return source_cache[s['ref']]

        def key_stats(s, keys):
            complete = ' AND '.join(OrganizationAnalyzer.present(k) for k in keys)
            n, full = con.execute(f'SELECT count(*),count(*) FILTER (WHERE {complete}) FROM {s["ref"]}').fetchone()
            unique = con.execute('SELECT count(*) FROM (SELECT DISTINCT ' + ','.join(map(RawData.quote_identifier, keys)) + ' FROM ' + s['ref'] + ')').fetchone()[0]
            return n, full, unique

        def install(op, queries, validate):
            for tag, sql in queries.items():
                name = f'{tag}_{op["label"]}_{op["id"].rsplit("_",1)[1][:10]}'
                op['outputs'][tag] = {'name': name, 'sql': sql}
            if any(out['name'].casefold() in occupied for out in op['outputs'].values()):
                op['reason'] = 'output_name_in_use'
                return
            for out in op['outputs'].values():
                con.execute('CREATE VIEW semantic.' + RawData.quote_identifier(out['name']) + ' AS ' + out['sql'])
            if validate():
                current = OrganizationBuilder.catalog(con)
                for out in op['outputs'].values():
                    out['definition_fingerprint'] = OrganizationBuilder.definition_fingerprint(con, current[out['name']])
                    occupied.add(out['name'].casefold())
                op.update(status='applied', reason='checks_passed')
            else:
                for out in op['outputs'].values():
                    con.execute('DROP VIEW semantic.' + RawData.quote_identifier(out['name']))
                op.update(status='rejected', reason='post_model_validation_failed')

        for s in sources:
            columns = {c['name']: c for c in s['columns']}
            roles = {c: OrganizationModelBuilder.role(spec) for c, spec in columns.items() if 'derived_from' not in spec}
            order_keys = [c for c, r in roles.items() if r == 'order']
            line_evidence = (s['logical_role'] == 'line' or any(r == 'line' for r in roles.values()) or
                             (any(r in ('product', 'item_name') for r in roles.values()) and
                              any(r in ('quantity', 'line_amount') for r in roles.values())))
            payment = s['logical_role'] == 'payment'
            if not order_keys or not (line_evidence or payment):
                continue
            stats = source_stats(s)
            branch = [c for c, r in roles.items() if r == 'branch']
            unambiguous = len(order_keys) == 1 and len(branch) <= 1
            keys = branch + order_keys if unambiguous else []
            n, full, distinct = key_stats(s, keys) if keys else (stats['rows'], 0, 0)
            source_clean = stats['raw_lineage_valid'] and not stats['structural_rows'] and n > 0
            key_clean = bool(keys) and n == full and all(parse_count(s, k) == 0 for k in keys)

            if line_evidence:
                op = operation('split_order_lines', s)
                op.update(keys=keys, evidence={'source_grain': 'line_candidate', 'roles': roles, **stats})
                gate = check(op, 'source_and_key_complete', source_clean and key_clean,
                             complete_key_rows=full, **stats)
                repeated = check(op, 'repeated_order_key', n > distinct > 0, rows=n, distinct_orders=distinct)
                attrs = [c for c, r in roles.items() if r in ('order_date', 'order_status', 'customer', 'customer_name', 'order_channel', 'order_total') and c not in keys]
                op['order_attributes'] = attrs
                if not gate or not repeated or not attrs:
                    op['reason'] = 'source_key_or_order_attributes_unresolved'
                elif not check(op, 'header_types_resolved', all(columns[c]['type']=='date' for c in attrs if roles[c]=='order_date')):
                    op['reason'] = 'order_attribute_type_unresolved'
                elif not check(op, 'header_parse_clean', all(parse_count(s, c) == 0 for c in attrs), parse_failed_cells=sum(parse_count(s, c) for c in attrs)):
                    op['reason'] = 'order_attributes_parse_failed'
                else:
                    header_cols = keys + attrs
                    h = ','.join(map(RawData.quote_identifier, header_cols))
                    k = ','.join(map(RawData.quote_identifier, keys))
                    conflicts = con.execute(f'SELECT count(*) FROM (SELECT {k} FROM (SELECT DISTINCT {h} FROM {s["ref"]}) GROUP BY {k} HAVING count(*)>1)').fetchone()[0]
                    if not check(op, 'attributes_functionally_depend_on_order_key', conflicts == 0, conflicting_keys=conflicts):
                        op['reason'] = 'conflicting_order_attributes'
                    else:
                        all_cols = list(columns) + s['trace_columns']
                        detail_cols = [c for c in all_cols if c not in attrs]
                        op.update(detail_columns=detail_cols, lineage={'detail_trace_columns': s['trace_columns'],
                                  'parent_path': 'order keys -> detail view -> source trace -> meta.tables/raw', 'source_trace': s['trace']})
                        queries = {'orders': f'SELECT DISTINCT {h} FROM {s["ref"]}',
                                   'lines': 'SELECT ' + ','.join(map(RawData.quote_identifier, detail_cols)) + ' FROM ' + s['ref']}
                        def validate_split():
                            parent = 'semantic.' + RawData.quote_identifier(op['outputs']['orders']['name'])
                            detail = 'semantic.' + RawData.quote_identifier(op['outputs']['lines']['name'])
                            pn = con.execute('SELECT count(*) FROM ' + parent).fetchone()[0]
                            dn = con.execute('SELECT count(*) FROM ' + detail).fetchone()[0]
                            cardinality = check(op, 'split_cardinality', pn == distinct and dn == n, parent_rows=pn, expected_parents=distinct, detail_rows=dn, source_rows=n)
                            condition = ' AND '.join('d.'+RawData.quote_identifier(c)+'=p.'+RawData.quote_identifier(c) for c in keys)
                            projection = ','.join(('p.' if c in attrs else 'd.') + RawData.quote_identifier(c) for c in all_cols)
                            reconstructed = f'SELECT {projection} FROM {detail} d LEFT JOIN {parent} p ON {condition}'
                            original = 'SELECT ' + ','.join(map(RawData.quote_identifier, all_cols)) + ' FROM ' + s['ref']
                            diffs = con.execute(f'SELECT count(*) FROM (({original} EXCEPT ALL {reconstructed}) UNION ALL ({reconstructed} EXCEPT ALL {original}))').fetchone()[0]
                            op['reconstruction_sql'] = reconstructed
                            return check(op, 'lossless_reconstruction', diffs == 0, differing_rows=diffs) and cardinality
                        install(op, queries, validate_split)

            op = operation('aggregate_source_events', s)
            op.update(keys=keys, source_kind='payments' if payment else 'lines',
                      evidence={'roles': roles, **stats}, measure_decisions=[])
            if not check(op, 'source_and_key_complete', source_clean and key_clean, complete_key_rows=full, **stats):
                op['reason'] = 'source_or_key_unresolved'
                continue
            identity = [c for c, r in roles.items() if r == ('payment' if payment else 'line')]
            if payment or identity:
                ikeys = branch + identity if payment else keys + identity
                valid_identity = len(identity) == 1
                identity_stats = key_stats(s, ikeys) if valid_identity else (n, 0, 0)
                if not check(op, 'event_identity_complete_unique', valid_identity and identity_stats[0] == identity_stats[1] == identity_stats[2] and all(parse_count(s, c) == 0 for c in identity), counts=identity_stats):
                    op['reason'] = 'event_identity_unresolved'
                    continue
            # Status is retained as a group dimension, never mapped to eligible sales.
            status = [c for c, r in roles.items() if r == 'order_status']
            if not check(op, 'status_parse_clean', all(parse_count(s, c) == 0 for c in status), parse_failed_cells=sum(parse_count(s, c) for c in status)):
                op['reason'] = 'status_parse_failed'
                continue
            groups = keys + status
            numeric = []
            for c, spec in columns.items():
                r = roles.get(c)
                if payment and (r == 'payment_amount' or norm(spec.get('description') or '') == 'so tien'):
                    numeric.append((c, 'money'))
                elif not payment and r in ('line_amount', 'quantity'):
                    numeric.append((c, 'money' if r == 'line_amount' else 'quantity'))
            prepared = []
            unit_columns = [c for c, r in roles.items() if r == 'unit']
            currency_columns = [c for c, r in roles.items() if r == 'currency']
            for c, unit_kind in numeric:
                spec = columns[c]
                decision = {'column': c, 'role': roles.get(c), 'status': 'deferred', 'reason': 'unit_unresolved'}
                op['measure_decisions'].append(decision)
                nulls = con.execute(f'SELECT count(*) FROM {s["ref"]} WHERE {RawData.quote_identifier(c)} IS NULL').fetchone()[0]
                if spec['type'] not in ('number', 'integer'):
                    decision['reason'] = 'measure_not_numeric'
                    continue
                if nulls or parse_count(s, c):
                    decision.update(reason='measure_missing_or_parse_failed', null_rows=nulls, parse_failed_cells=parse_count(s, c))
                    continue
                unit_field_candidates = currency_columns if unit_kind == 'money' else unit_columns
                declared = spec.get('unit')
                # Currency output conventions in rulebook do not establish source units.
                if len(unit_field_candidates) > 1:
                    decision['reason'] = 'multiple_unit_columns'
                    continue
                unit_field = unit_field_candidates[0] if unit_field_candidates else None
                if unit_field:
                    invalid = con.execute(f'SELECT count(*) FROM {s["ref"]} WHERE NOT {OrganizationAnalyzer.present(unit_field)}').fetchone()[0]
                    if invalid or parse_count(s, unit_field):
                        decision['reason'] = 'unit_column_incomplete'
                        continue
                    if declared:
                        mismatches = con.execute(f'SELECT count(*) FROM {s["ref"]} WHERE {RawData.quote_identifier(unit_field)}<>?', [str(declared)]).fetchone()[0]
                        if mismatches:
                            decision['reason'] = 'unit_metadata_conflict'
                            continue
                    if unit_field not in groups:
                        groups.append(unit_field)
                elif not declared:
                    continue
                decision.update(status='included', reason='explicit_unit_and_complete_measure',
                                unit=declared, unit_column=unit_field)
                prepared.append((c, decision))
            used = {g.casefold() for g in groups}
            count_alias = OrganizationBuilder.reserve('_org_source_rows', used)
            projections = [RawData.quote_identifier(g) for g in groups] + ['count(*) AS ' + RawData.quote_identifier(count_alias)]
            for c, decision in prepared:
                alias = OrganizationBuilder.reserve('source_sum__' + c, used)
                decision['output_column'] = alias
                projections.append('sum(' + RawData.quote_identifier(c) + ') AS ' + RawData.quote_identifier(alias))
            times = [c for c, r in roles.items() if r in ('event_time', 'order_date')]
            time_outputs = {}
            op["timestamp_decisions"] = []
            for c in times:
                if columns[c]['type'] != 'date':
                    op['timestamp_decisions'].append({'column': c, 'status': 'deferred', 'reason': 'timestamp_not_typed'})
                    continue
                nulls = con.execute(f'SELECT count(*) FROM {s["ref"]} WHERE {RawData.quote_identifier(c)} IS NULL').fetchone()[0]
                if parse_count(s, c) or nulls:
                    op['timestamp_decisions'].append({'column': c, 'status': 'deferred', 'reason': 'timestamp_missing_or_parse_failed', 'null_rows': nulls, 'parse_failed_cells': parse_count(s, c)})
                    continue
                op['timestamp_decisions'].append({'column': c, 'status': 'included', 'reason': 'complete_typed_timestamp'})
                lower, upper = OrganizationBuilder.reserve('source_first__' + c, used), OrganizationBuilder.reserve('source_last__' + c, used)
                time_outputs[c] = [lower, upper]
                projections += ['min('+RawData.quote_identifier(c)+') AS '+RawData.quote_identifier(lower), 'max('+RawData.quote_identifier(c)+') AS '+RawData.quote_identifier(upper)]
            sql = 'SELECT ' + ','.join(projections) + ' FROM ' + s['ref'] + ' GROUP BY ' + ','.join(map(RawData.quote_identifier, groups))
            op.update(group_columns=groups, count_column=count_alias, timestamp_columns=time_outputs,
                      lineage={'group_columns': groups, 'source_trace_columns': s['trace_columns'], 'source_trace': s['trace']},
                      warning='Source-event summaries only. Counts are not eligible-order counts; sums are not revenue or debt. Distinct units/statuses remain separate groups.')
            def validate_aggregate():
                target = 'semantic.' + RawData.quote_identifier(op['outputs']['aggregate']['name'])
                rows, covered_rows = con.execute(f'SELECT count(*),coalesce(sum({RawData.quote_identifier(count_alias)}),0) FROM {target}').fetchone()
                expected_groups = con.execute('SELECT count(*) FROM (SELECT DISTINCT ' + ','.join(map(RawData.quote_identifier, groups)) + ' FROM ' + s['ref'] + ')').fetchone()[0]
                valid = check(op, 'aggregate_group_counts', rows == expected_groups and covered_rows == n,
                              groups=rows, expected_groups=expected_groups, source_rows=n, covered_rows=covered_rows)
                for c, decision in prepared:
                    before = con.execute('SELECT sum(' + RawData.quote_identifier(c) + ') FROM ' + s['ref']).fetchone()[0]
                    after = con.execute('SELECT sum(' + RawData.quote_identifier(decision['output_column']) + ') FROM ' + target).fetchone()[0]
                    valid &= check(op, 'sum_conserved:' + c, before == after, before=before, after=after)
                # Full per-group comparison also verifies timestamps and NULL status grouping.
                diffs = con.execute(f'SELECT count(*) FROM (({sql} EXCEPT ALL SELECT * FROM {target}) UNION ALL (SELECT * FROM {target} EXCEPT ALL {sql}))').fetchone()[0]
                valid &= check(op, 'per_group_values_verified', diffs == 0, differing_rows=diffs)
                return valid
            install(op, {'aggregate': sql}, validate_aggregate)

        con.execute('CREATE OR REPLACE TABLE meta.model_transformations (transformation_id VARCHAR, kind VARCHAR, status VARCHAR, source_name VARCHAR, details_json VARCHAR)')
        con.execute('CREATE OR REPLACE TABLE meta.model_validations (transformation_id VARCHAR, check_name VARCHAR, passed BOOLEAN, details_json VARCHAR)')
        relationships = []
        for op in operations:
            if op['kind'] == 'split_order_lines' and op['status'] == 'applied':
                relationships.append({'transformation_id': op['id'], 'source_view': op['outputs']['lines']['name'],
                                      'target_view': op['outputs']['orders']['name'], 'columns': op['keys'],
                                      'relationship_type': 'many_to_one', 'evidence': 'validated_lossless_decomposition'})
            con.execute('INSERT INTO meta.model_transformations VALUES (?,?,?,?,?)', [op['id'], op['kind'], op['status'], op['source'], json.dumps(op, ensure_ascii=False, default=str)])
        for v in validations:
            con.execute('INSERT INTO meta.model_validations VALUES (?,?,?,?)', [v['transformation_id'], v['check'], v['passed'], json.dumps(v['details'], ensure_ascii=False, default=str)])
        con.execute('CREATE OR REPLACE TABLE meta.model_relationships (transformation_id VARCHAR, source_view VARCHAR, target_view VARCHAR, columns_json VARCHAR, details_json VARCHAR)')
        for rel in relationships:
            con.execute('INSERT INTO meta.model_relationships VALUES (?,?,?,?,?)', [rel['transformation_id'], rel['source_view'], rel['target_view'], json.dumps(rel['columns'], ensure_ascii=False), json.dumps(rel, ensure_ascii=False)])
        con.execute('CREATE OR REPLACE TABLE meta.model_policy (policy_version VARCHAR, details_json VARCHAR)')
        con.execute('INSERT INTO meta.model_policy VALUES (?,?)', [MODEL_POLICY, json.dumps({'split_entities': ['orders/lines'], 'source_selection': 'prefer verified merged views; exclude their Typed members', 'business_filters': 'none', 'units': 'explicit column metadata or complete source unit/currency field'}, ensure_ascii=False)])
        return {'policy_version': MODEL_POLICY, 'transformations': operations, 'validations': validations, 'relationships': relationships,
                'policy': 'Lossless order projections and all-source-event summaries. No business eligibility, inferred unit conversion or missing-value imputation.'}

    @staticmethod
    def write_model_report(result, directory):
        directory = Path(directory)
        (directory / 'organization-model.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        safe = lambda v: str(v).replace('|', '/').replace('\n', ' ')
        lines = ['# Tổ chức database — đợt 3', '', f'Policy: `{MODEL_POLICY}`.', '',
                 'Tách order/line kiểm tra dựng lại không mất dữ liệu; tổng hợp bản ghi nguồn theo khóa/status/đơn vị.',
                 'Không tự tính doanh thu, công nợ hoặc áp as_of_date/rulebook. Không gọi LLM/API.', '',
                 '| Loại | Nguồn | Trạng thái | Lý do | Views |', '|---|---|---|---|---|']
        for op in result['transformations']:
            names = ', '.join(out['name'] for out in op['outputs'].values()) if op['status']=='applied' else ''
            lines.append(f"| {op['kind']} | {safe(op['source'])} | {op['status']} | {op['reason']} | {safe(names)} |")
        lines += ['', '## Chỉ tiêu chưa cộng', '', '| Nguồn | Cột | Lý do |', '|---|---|---|']
        for op in result['transformations']:
            for measure in op.get('measure_decisions', []):
                if measure['status'] != 'included':
                    lines.append(f"| {safe(op['source'])} | {safe(measure['column'])} | {measure['reason']} |")
        lines += ['', 'Tra meta.model_transformations / meta.model_validations hoặc organization-model.json để lấy SQL, mapping và lineage.',
                  'View aggregate có thể chỉ có số dòng nguồn khi không đủ căn cứ cộng chỉ tiêu.',
                  'Counts không phải số đơn hợp lệ; source_sum không phải doanh thu/công nợ. Giữ các trạng thái và đơn vị thành nhóm riêng.',
                  'Tách chỉ là projections/views; Raw/Typed và view nền vẫn giữ nguyên.']
        (directory / 'ORGANIZATION_MODEL.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
