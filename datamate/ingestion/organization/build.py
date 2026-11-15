"""Stage 2: verified UNION ALL and single-edge LEFT JOIN views.

Only generated views are managed. Raw/Typed, identity views, and configured views
are never replaced. The caller owns the transaction; unexpected SQL failures abort it.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from datamate.text import colnorm, norm
from ..raw import RawData
from .analyze import OrganizationAnalyzer

BUILD_POLICY = 'organization-build-v1'

class OrganizationBuilder:
    @staticmethod
    def literal(value):
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def reserve(name, used):
        candidate, suffix = name, 2
        while candidate.casefold() in used:
            candidate = f'{name}_{suffix}'
            suffix += 1
        used.add(candidate.casefold())
        return candidate

    @staticmethod
    def period_name(value):
        """Only explicit year+month labels; no guessed year or date semantics."""
        value = colnorm(value)
        for expression, reverse in [(r'(?:t|thang_?)?(\d{1,2})_(\d{4})', True),
                                    (r'(\d{4})_(?:t|thang_?)?(\d{1,2})', False)]:
            match = re.fullmatch(expression, value)
            if match:
                a, b = map(int, match.groups())
                year, month = (b, a) if reverse else (a, b)
                if 1900 <= year <= 2200 and 1 <= month <= 12:
                    return f'{year:04}-{month:02}'
        return None

    @staticmethod
    def family(table):
        source = table['source']
        sheet = source.get('sheet') or (source.get('source_object') if source.get('format') in ('xlsx', 'xls') else None)
        if sheet:
            return ('workbook', source['file']), OrganizationBuilder.period_name(sheet)
        stem = Path(source['file']).stem
        match = re.fullmatch(r'(.+?)[_. -]((?:\d{4}[_ .-](?:[Tt])?\d{1,2})|(?:(?:[Tt])?\d{1,2}[_ .-]\d{4}))', stem)
        if match:
            return ('files', str(Path(source['file']).parent / match[1])), OrganizationBuilder.period_name(match[2])
        return None, None

    @staticmethod
    def catalog(con):
        return {name: sql for name, sql in con.execute(
            "SELECT view_name, sql FROM duckdb_views() WHERE schema_name='semantic'").fetchall()}

    @staticmethod
    def definition_fingerprint(con, definition):
        """Compare parsed SELECT definitions, allowing UNION ALL reassociation on reload.

        Do not strip parentheses with regex: that could hide meaningful user edits.
        Only flatten unmodified UNION ALL nodes, preserving branch order, projections,
        predicates, CTEs, types, limits and all other AST fields.
        """
        if definition.startswith('CREATE VIEW '):
            definition = definition.split(' AS ', 1)[1]
        parsed = json.loads(con.execute('SELECT json_serialize_sql(?)', [definition]).fetchone()[0])
        if parsed.get('error'):
            # Unsupported SQL is not treated as an unchanged owned definition.
            return None
        def clean(value):
            if isinstance(value, list):
                return [clean(v) for v in value]
            if not isinstance(value, dict):
                return value
            node = {k: clean(v) for k, v in value.items() if k not in ('query_location', 'query_location_length')}
            if (node.get('type') == 'SET_OPERATION_NODE' and node.get('setop_type') == 'UNION'
                    and node.get('setop_all') is True and not node.get('modifiers')
                    and node.get('cte_map') == {'map': []}):
                attributes = {k: v for k, v in node.items() if k not in ('left', 'right', 'children')}
                branches = []
                for child in (node.get('children') or [node['left'], node['right']]):
                    if child.get('_union_all_attributes') == attributes:
                        branches.extend(child['_branches'])
                    else:
                        branches.append(child)
                return {'_union_all_attributes': attributes, '_branches': branches}
            return node
        return hashlib.sha256(json.dumps(clean(parsed), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    @staticmethod
    def build_connection(con, analysis, *, validate_catalog=True):
        """Build and validate extra semantic views using current transaction data."""
        tables = {}
        for logical, db, lineage, source, columns in con.execute(
                'SELECT * EXCLUDE (cell_metadata_json) FROM meta.tables ORDER BY logical_name').fetchall():
            tables[logical] = {'db': db, 'lineage': lineage, 'source': json.loads(source),
                               'columns': json.loads(columns)}
        validations, operations = [], []

        def check(op, name, passed, **details):
            validations.append({'transformation_id': op['id'], 'check': name,
                                'passed': bool(passed), 'details': details})
            return bool(passed)

        # Remove only views installed by us whose catalog definition has not been edited.
        exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='transformations'").fetchone()[0]
        current = OrganizationBuilder.catalog(con)
        if exists:
            for (text,) in con.execute("SELECT details_json FROM meta.transformations WHERE status='applied' ORDER BY transformation_id").fetchall():
                old = json.loads(text)
                name = old.get('output_name')
                if name in current:
                    previous = old.get('definition_fingerprint')
                    if previous is None and old.get('installed_sql'):
                        previous = OrganizationBuilder.definition_fingerprint(con, old['installed_sql'])
                    if previous is not None and previous == OrganizationBuilder.definition_fingerprint(con, current[name]):
                        con.execute(f'DROP VIEW semantic.{RawData.quote_identifier(name)}')
        occupied = {name.casefold() for (name,) in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='semantic'").fetchall()}
        profiles = {(p['table'], p['column']): p for p in analysis['profiles']}

        def operation(kind, sources, identity):
            identifier = OrganizationAnalyzer.stable_id(kind, identity)
            label = colnorm(sources[0].split('::')[0])[:32] or 'data'
            return {'id': identifier, 'kind': kind, 'sources': sources,
                    'output_name': f'{kind}_{label}_{identifier.rsplit("_", 1)[1][:10]}',
                    'status': 'candidate', 'reason': 'insufficient_evidence'}

        def lineage_check(op, logical):
            t = tables[logical]
            total, unique = con.execute('SELECT count(*), count(DISTINCT ' + RawData.quote_identifier(t['lineage']) +
                                        ') FROM typed.' + RawData.quote_identifier(t['db'])).fetchone()
            raw_total, raw_unique = con.execute('SELECT count(*), count(DISTINCT row_id) FROM raw.' + RawData.quote_identifier(t['db'])).fetchone()
            orphan = con.execute('SELECT count(*) FROM typed.' + RawData.quote_identifier(t['db']) + ' t WHERE NOT EXISTS '
                                 '(SELECT 1 FROM raw.' + RawData.quote_identifier(t['db']) + ' r WHERE r.row_id=t.' + RawData.quote_identifier(t['lineage']) + ')').fetchone()[0]
            return check(op, 'raw_lineage_valid:' + logical,
                         total == unique == raw_total == raw_unique and orphan == 0,
                         typed_rows=total, typed_unique_lineage=unique, raw_rows=raw_total,
                         raw_unique_lineage=raw_unique, orphan_lineage=orphan)

        def install(op, sql, validate):
            name = op['output_name']
            op['sql'] = sql
            if name.casefold() in occupied:
                op['reason'] = 'output_name_in_use'
                return
            con.execute(f'CREATE VIEW semantic.{RawData.quote_identifier(name)} AS {sql}')
            if validate():
                op.update(status='applied', reason='checks_passed', sql=sql,
                          installed_sql=OrganizationBuilder.catalog(con)[name],
                          definition_fingerprint=OrganizationBuilder.definition_fingerprint(con, OrganizationBuilder.catalog(con)[name]))
                occupied.add(name.casefold())
            else:
                con.execute(f'DROP VIEW semantic.{RawData.quote_identifier(name)}')
                op.update(status='rejected', reason='post_build_validation_failed')

        groups = defaultdict(list)
        for name, table in tables.items():
            group, period = OrganizationBuilder.family(table)
            if group:
                groups[group].append((name, period))
        for group, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            members.sort()
            names = [name for name, _ in members]
            op = operation('merged', names, [group, names])
            op['evidence'] = {'family': list(group), 'periods': dict(members),
                              'assumption': 'Same source family and compatible schema represent the same row grain; heuristic, not a business guarantee.'}
            operations.append(op)
            periods = [period for _, period in members]
            if not check(op, 'explicit_distinct_month_partitions', all(periods) and len(set(periods)) == len(periods), periods=periods):
                op['reason'] = 'unknown_or_overlapping_period_labels'
                continue
            mappings = {}
            for name in names:
                columns = tables[name]['columns']
                mappings[name] = {colnorm(c['name']): c for c in columns}
            first = mappings[names[0]]
            schema_ok = bool(first) and all(len(mappings[n]) == len(tables[n]['columns']) and set(mappings[n]) == set(first) for n in names)
            def signature(c):
                return (c['type'], c.get('unit'), norm(c.get('description') or ''),
                        colnorm(c.get('derived_from', '')))
            if schema_ok:
                schema_ok = all(signature(mappings[n][alias]) == signature(c)
                                for n in names for alias, c in first.items())
            # Require identical physical SQL types too: DECIMAL(38,0) and (38,2)
            # must not be implicitly promoted at the expense of precision.
            sql_types = {}
            for name in names:
                sql_types[name] = {c: t for c, t in con.execute(
                    'SELECT column_name, data_type FROM information_schema.columns WHERE table_schema=? AND table_name=?',
                    ['typed', tables[name]['db']]).fetchall()}
            if schema_ok:
                schema_ok = all(sql_types[n][mappings[n][alias]['name']] == sql_types[names[0]][c['name']]
                                for n in names for alias, c in first.items())
            if not check(op, 'compatible_column_contract', schema_ok):
                op['reason'] = 'column_type_unit_or_description_mismatch'
                continue
            # Require exactly one common date column with every row in its labelled month.
            # This is partition evidence, not guessing from filename alone.
            candidates = [alias for alias, c in first.items() if c['type'] == 'date']
            matching_dates = []
            counts = {}
            for alias in candidates:
                valid = True
                for name, period in members:
                    column = mappings[name][alias]['name']
                    n, in_month = con.execute(f"SELECT count(*), count(*) FILTER (WHERE strftime({RawData.quote_identifier(column)}, '%Y-%m')=?) FROM typed.{RawData.quote_identifier(tables[name]['db'])}", [period]).fetchone()
                    counts[name] = n
                    valid &= n > 0 and n == in_month
                if valid:
                    matching_dates.append(alias)
            if not check(op, 'fully_verified_date_partition', len(matching_dates) == 1,
                         matching_date_columns=matching_dates):
                op['reason'] = 'date_partition_unverified_or_ambiguous'
                continue
            date_alias = matching_dates[0]
            # All accepted row-identity keys common to the partitions must remain unique
            # globally. Never silently reinterpret a repeated entity ID as a new key.
            common_keys = None
            for name in names:
                sets = {tuple(sorted(colnorm(c) for c in k['columns'])) for k in analysis['keys']
                        if k['table'] == name and k['status'] == 'accepted'}
                common_keys = sets if common_keys is None else common_keys & sets
            union_keys_ok = True
            for key in sorted(common_keys or []):
                parts = ['SELECT ' + ','.join(RawData.quote_identifier(mappings[n][a]['name']) + ' AS ' + RawData.quote_identifier(a) for a in key) +
                         ' FROM typed.' + RawData.quote_identifier(tables[n]['db']) for n in names]
                all_rows, unique_rows = con.execute('SELECT count(*), count(DISTINCT (' +
                    ','.join(map(RawData.quote_identifier, key)) + ')) FROM (' + ' UNION ALL '.join(parts) + ')').fetchone()
                union_keys_ok &= check(op, 'global_key_unique:' + ','.join(key), all_rows == unique_rows,
                                      rows=all_rows, distinct_keys=unique_rows)
            if not union_keys_ok:
                op['reason'] = 'cross_partition_key_collision'
                continue
            output_columns = [c['name'] for c in first.values()]
            used = {c.casefold() for c in output_columns}
            trace = {role: OrganizationBuilder.reserve('_org_' + role, used) for role in ('source_table', 'source_row_id', 'period')}
            selects = []
            for name, period in members:
                t = tables[name]
                projections = [RawData.quote_identifier(mappings[name][alias]['name']) + ' AS ' + RawData.quote_identifier(c['name']) for alias, c in first.items()]
                projections += [OrganizationBuilder.literal(name) + ' AS ' + RawData.quote_identifier(trace['source_table']),
                                RawData.quote_identifier(t['lineage']) + ' AS ' + RawData.quote_identifier(trace['source_row_id']),
                                OrganizationBuilder.literal(period) + ' AS ' + RawData.quote_identifier(trace['period'])]
                selects.append('SELECT ' + ','.join(projections) + ' FROM typed.' + RawData.quote_identifier(t['db']))
            sql = ' UNION ALL '.join(selects)
            op.update(column_mapping={n: {c['name']: mappings[n][a]['name'] for a, c in first.items()} for n in names},
                      lineage_columns=trace, partition_date_column=first[date_alias]['name'],
                      quality={'parse_failed_cells': sum(p['parse_failed_count'] for p in analysis['profiles'] if p['table'] in names),
                               'warning': 'All rows and pre-existing Typed issues are preserved; this view is not a data-cleaning operation.'})
            def validate_merge():
                target = 'semantic.' + RawData.quote_identifier(op['output_name'])
                actual = dict(con.execute(f'SELECT {RawData.quote_identifier(trace["source_table"])}, count(*) FROM {target} GROUP BY 1').fetchall())
                row_ok = check(op, 'source_row_counts_preserved', actual == counts, expected=counts, actual=actual)
                total, unique = con.execute(f'SELECT count(*), count(DISTINCT ({RawData.quote_identifier(trace["source_table"])},{RawData.quote_identifier(trace["source_row_id"])})) FROM {target}').fetchone()
                lineage_ok = check(op, 'lineage_unique', total == unique, rows=total, distinct_lineage=unique)
                # Compare complete business projections + source lineage in both directions.
                diffs = 0
                for n in names:
                    t = tables[n]
                    expected = 'SELECT ' + ','.join(RawData.quote_identifier(mappings[n][a]['name']) for a in first) + ',' + RawData.quote_identifier(t['lineage']) + ' FROM typed.' + RawData.quote_identifier(t['db'])
                    observed = 'SELECT ' + ','.join(map(RawData.quote_identifier, output_columns)) + ',' + RawData.quote_identifier(trace['source_row_id']) + ' FROM ' + target + ' WHERE ' + RawData.quote_identifier(trace['source_table']) + '=' + OrganizationBuilder.literal(n)
                    diffs += con.execute(f'SELECT count(*) FROM (({expected} EXCEPT ALL {observed}) UNION ALL ({observed} EXCEPT ALL {expected}))').fetchone()[0]
                values_ok = check(op, 'all_source_values_preserved', diffs == 0, differing_rows=diffs)
                raw_lineage_ok = all([lineage_check(op, n) for n in names])
                return row_ok and lineage_ok and values_ok and raw_lineage_ok
            install(op, sql, validate_merge)

        # Single edge only: never combine sibling fact tables or chain multiple joins.
        for relation in analysis['relationships']:
            if relation['status'] != 'accepted':
                continue
            source, parent = relation['source_table'], relation['target_table']
            s, p = tables[source], tables[parent]
            op = operation('joined', [source, parent], relation['id'])
            op.update(relationship_id=relation['id'], grain=source,
                      warning='Parent values may repeat across source rows. Do not sum parent measures on this view without grouping by the parent key.')
            operations.append(op)
            # Recheck actual target uniqueness and reference coverage, not cached counts.
            keys = relation['target_columns']
            n, distinct, complete = con.execute('SELECT count(*), count(DISTINCT (' + ','.join(map(RawData.quote_identifier, keys)) +
                ')), count(*) FILTER (WHERE ' + ' AND '.join(f'{RawData.quote_identifier(c)} IS NOT NULL' for c in keys) +
                ') FROM typed.' + RawData.quote_identifier(p['db'])).fetchone()
            if not check(op, 'target_key_still_unique_complete', n > 0 and n == distinct == complete,
                         rows=n, distinct_keys=distinct, complete_keys=complete):
                op['reason'] = 'target_key_changed'
                continue
            condition = ' AND '.join('c.' + RawData.quote_identifier(a) + '=p.' + RawData.quote_identifier(b) for a, b in zip(relation['source_columns'], keys))
            full = ' AND '.join(f'c.{RawData.quote_identifier(c)} IS NOT NULL' for c in relation['source_columns'])
            missing = con.execute(f'SELECT count(*) FROM typed.{RawData.quote_identifier(s["db"])} c LEFT JOIN typed.{RawData.quote_identifier(p["db"])} p ON {condition} WHERE ({full}) AND p.{RawData.quote_identifier(keys[0])} IS NULL').fetchone()[0]
            if not check(op, 'reference_coverage_still_complete', missing == 0, missing_rows=missing):
                op['reason'] = 'reference_coverage_changed'
                continue
            source_columns = [c['name'] for c in s['columns']] + [s['lineage']]
            used = {c.casefold() for c in source_columns}
            projection = ['c.' + RawData.quote_identifier(c) for c in source_columns]
            parent_mapping = {}
            for column in [c['name'] for c in p['columns']] + [p['lineage']]:
                alias = OrganizationBuilder.reserve((colnorm(parent)[:24] or 'parent') + '__' + column, used)
                parent_mapping[column] = alias
                projection.append('p.' + RawData.quote_identifier(column) + ' AS ' + RawData.quote_identifier(alias))
            sql = 'SELECT ' + ','.join(projection) + f' FROM typed.{RawData.quote_identifier(s["db"])} c LEFT JOIN typed.{RawData.quote_identifier(p["db"])} p ON {condition}'
            op.update(source_lineage_column=s['lineage'], parent_columns=parent_mapping,
                      parent_lineage_column=parent_mapping[p['lineage']])
            def validate_join():
                target = 'semantic.' + RawData.quote_identifier(op['output_name'])
                expected_count = con.execute('SELECT count(*) FROM typed.' + RawData.quote_identifier(s['db'])).fetchone()[0]
                actual_count, distinct_lineage = con.execute(f'SELECT count(*), count(DISTINCT {RawData.quote_identifier(s["lineage"])} ) FROM {target}').fetchone()
                count_ok = check(op, 'source_grain_preserved', actual_count == expected_count == distinct_lineage,
                                 before=expected_count, after=actual_count, distinct_lineage=distinct_lineage)
                observed = 'SELECT ' + ','.join(map(RawData.quote_identifier, source_columns)) + ' FROM ' + target
                expected = 'SELECT ' + ','.join(map(RawData.quote_identifier, source_columns)) + ' FROM typed.' + RawData.quote_identifier(s['db'])
                diffs = con.execute(f'SELECT count(*) FROM (({expected} EXCEPT ALL {observed}) UNION ALL ({observed} EXCEPT ALL {expected}))').fetchone()[0]
                values_ok = check(op, 'all_source_values_preserved', diffs == 0, differing_rows=diffs)
                raw_lineage_ok = all([lineage_check(op, n) for n in (source, parent)])
                return values_ok and count_ok and raw_lineage_ok
            install(op, sql, validate_join)

        # Also ensure removing an obsolete owned view did not break another view.
        # Failure aborts the transaction and restores the previous catalog.
        if validate_catalog:
            for view in OrganizationBuilder.catalog(con):
                con.execute('SELECT * FROM semantic.' + RawData.quote_identifier(view) + ' LIMIT 0')

        con.execute('CREATE OR REPLACE TABLE meta.transformations (transformation_id VARCHAR, kind VARCHAR, status VARCHAR, output_name VARCHAR, details_json VARCHAR)')
        con.execute('CREATE OR REPLACE TABLE meta.transformation_validations (transformation_id VARCHAR, check_name VARCHAR, passed BOOLEAN, details_json VARCHAR)')
        for op in operations:
            con.execute('INSERT INTO meta.transformations VALUES (?,?,?,?,?)',
                        [op['id'], op['kind'], op['status'], op['output_name'], json.dumps(op, ensure_ascii=False)])
        for v in validations:
            con.execute('INSERT INTO meta.transformation_validations VALUES (?,?,?,?)',
                        [v['transformation_id'], v['check'], v['passed'], json.dumps(v['details'], ensure_ascii=False)])
        return {'policy_version': BUILD_POLICY, 'transformations': operations, 'validations': validations,
                'policy': 'Verified views only; no source deletion, deduplication, unit conversion, splitting or automatic business aggregates.'}

    @staticmethod
    def write_build_report(result, directory):
        directory = Path(directory)
        (directory / 'organization-build.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        safe = lambda value: str(value).replace('|', '/').replace('\n', ' ')
        lines = ['# Tổ chức database — đợt 2', '', f"Policy: `{BUILD_POLICY}`.", '',
                 'View gộp UNION ALL và view LEFT JOIN một quan hệ; giữ nguyên Raw, Typed và view nền.',
                 'Các quyết định là heuristic theo policy, không bảo đảm đúng nghiệp vụ với mọi nguồn.', '',
                 '| Thao tác | Nguồn | Trạng thái | View Semantic | Lý do |', '|---|---|---|---|---|']
        for op in result['transformations']:
            lines.append(f"| {op['kind']} | {safe(', '.join(op['sources']))} | {op['status']} | {safe(op['output_name'])} | {op['reason']} |")
        lines += ['', 'Chỉ view có status=applied mới được tạo bởi lần chạy này.', '',
                  '## Cách dùng', '',
                  'Tra `meta.transformations` để lấy tên view và `details_json` để lấy mapping/lineage.',
                  'Với merged: dùng cặp source_table/source_row_id để về bảng Raw tương ứng.',
                  'Với joined: giữ mức chi tiết nguồn; giá trị từ bảng cha có thể lặp, không cộng trực tiếp tiền bảng cha.',
                  'Các lỗi Typed có sẵn được giữ; UNION/JOIN không tự sửa NULL hoặc đổi đơn vị.', '',
                  'Toàn bộ SQL, bằng chứng và kiểm tra ở organization-build.json và meta.transformation_validations.']
        (directory / 'ORGANIZATION_BUILD.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
