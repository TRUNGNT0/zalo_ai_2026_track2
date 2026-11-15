"""Offline relationship diagrams from exported metadata, without inferring new links."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb


class OrganizationDiagram:
    @staticmethod
    def read_diagram(database):
        """Only metadata/catalog reads; never modify or sample business data."""
        database = Path(database)
        if not database.is_file():
            raise FileNotFoundError(database)
        with duckdb.connect(str(database), read_only=True) as con:
            available = {name for (name,) in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='meta'").fetchall()}
            if not {'business', 'tables'} <= available:
                raise ValueError('Not an ingestion export: meta.business / meta.tables missing')
            business = con.execute('SELECT business_id FROM meta.business').fetchone()[0]
            nodes, edges, diagnostics = [], [], []
            if not ({'relationships', 'model_relationships'} & available):
                diagnostics.append('Chưa có metadata quan hệ; chạy lại organization trước khi kết luận về liên kết.')
            typed, semantic = {}, {}
            column_catalog = {}
            for schema, table, name, kind in con.execute("""
                SELECT table_schema,table_name,column_name,data_type
                FROM information_schema.columns WHERE table_schema IN ('typed','semantic')
                ORDER BY table_schema,table_name,ordinal_position
            """).fetchall():
                column_catalog.setdefault((schema, table), []).append({'name': name, 'type': kind})
            row_counts = {}
            if 'column_profiles' in available:
                for logical, text in con.execute('SELECT logical_name,details_json FROM meta.column_profiles').fetchall():
                    row_counts.setdefault(logical, json.loads(text).get('rows'))

            def node(schema, sql_name, label, category, logical=None, technical=()):
                columns = column_catalog.get((schema, sql_name))
                if columns is None:
                    diagnostics.append(f'Missing catalog object: {schema}.{sql_name}')
                    return None
                identifier = f'n{len(nodes)}'
                nodes.append({'id': identifier, 'label': label, 'sql': schema+'.'+sql_name,
                              'category': category, 'logical': logical,
                              'rows': row_counts.get(logical) if schema == 'typed' else None,
                              'columns': [dict(c, badges=[], technical=c['name'] in technical) for c in columns],
                              'keys': []})
                return identifier

            for logical, sql_name, lineage in con.execute('SELECT logical_name,db_name,lineage_column FROM meta.tables ORDER BY logical_name').fetchall():
                typed[logical] = node('typed', sql_name, logical, 'typed', logical, [lineage])
            lookup = {n['id']: n for n in nodes}

            def mark(identifier, columns, badge):
                if identifier is None:
                    return
                for c in lookup[identifier]['columns']:
                    if c['name'] in columns and badge not in c['badges']:
                        c['badges'].append(badge)

            def link(child, parent, source_columns, target_columns, category, status, details):
                if child is None or parent is None:
                    return
                source_names = {c['name'] for c in lookup[child]['columns']}
                target_names = {c['name'] for c in lookup[parent]['columns']}
                if len(source_columns) != len(target_columns) or not source_columns or not set(source_columns) <= source_names or not set(target_columns) <= target_names:
                    diagnostics.append('Invalid relationship endpoints in metadata; omitted')
                    return
                edges.append({'id': f'e{len(edges)}', 'child': child, 'parent': parent,
                              'source_columns': source_columns, 'target_columns': target_columns,
                              'category': category, 'status': status, 'details': details})
                if status == 'accepted' and category != 'lineage':
                    mark(child, source_columns, 'R')

            if 'key_candidates' in available:
                for logical, text in con.execute("SELECT logical_name,details_json FROM meta.key_candidates WHERE status='accepted' ORDER BY key_id").fetchall():
                    key = json.loads(text)
                    identifier = typed.get(logical)
                    if identifier is not None:
                        lookup[identifier]['keys'].append({'columns': key['columns'], 'kind': 'U', 'description': 'Snapshot-local uniqueness candidate; not a declared primary key'})
                        mark(identifier, key['columns'], 'U')
            if 'relationships' in available:
                for child, parent, status, text in con.execute("SELECT source_table,target_table,status,details_json FROM meta.relationships WHERE status IN ('accepted','candidate') ORDER BY relationship_id").fetchall():
                    detail = json.loads(text)
                    link(typed.get(child), typed.get(parent), detail['source_columns'], detail['target_columns'], 'relationship', status, detail)

            stage2 = []
            if 'transformations' in available:
                for (text,) in con.execute("SELECT details_json FROM meta.transformations WHERE status='applied' ORDER BY transformation_id").fetchall():
                    op = json.loads(text)
                    stage2.append(op)
                    technical = list(op.get('lineage_columns', {}).values()) if op['kind'] == 'merged' else [op.get('source_lineage_column'), op.get('parent_lineage_column')]
                    name = op['output_name']
                    semantic[name] = node('semantic', name, ('Gộp · ' if op['kind']=='merged' else 'Nối · ')+', '.join(op['sources']), 'derived', technical=technical)
            stage3 = []
            if 'model_transformations' in available:
                for (text,) in con.execute("SELECT details_json FROM meta.model_transformations WHERE status='applied' ORDER BY transformation_id").fetchall():
                    op = json.loads(text)
                    stage3.append(op)
                    for tag, out in op['outputs'].items():
                        label = {'orders': 'Đơn hàng', 'lines': 'Dòng hàng', 'aggregate': 'Tổng hợp nguồn'}[tag] + ' · ' + op['source']
                        technical = op.get('lineage', {}).get('detail_trace_columns', []) if tag=='lines' else []
                        semantic[out['name']] = node('semantic', out['name'], label, 'split' if tag in ('orders','lines') else 'derived', technical=technical)
            lookup.update({n['id']: n for n in nodes})
            if 'model_relationships' in available:
                for child, parent, columns, text in con.execute('SELECT source_view,target_view,columns_json,details_json FROM meta.model_relationships ORDER BY transformation_id').fetchall():
                    keys = json.loads(columns)
                    parent_id = semantic.get(parent)
                    if parent_id is not None:
                        lookup[parent_id]['keys'].append({'columns': keys, 'kind': 'K', 'description': 'Logical key from validated lossless decomposition; not a physical constraint'})
                        mark(parent_id, keys, 'K')
                    link(semantic.get(child), parent_id, keys, keys, 'model', 'accepted', json.loads(text))
            # Provenance arrows are deliberately a different category, never shown as FKs.
            for op in stage2:
                child = semantic.get(op['output_name'])
                for source in op['sources']:
                    parent = typed.get(source)
                    if child is not None and parent is not None:
                        edges.append({'id': f'e{len(edges)}', 'child': child, 'parent': parent,
                                      'source_columns': [], 'target_columns': [], 'category': 'lineage', 'status': 'accepted',
                                      'details': {'kind': op['kind'], 'meaning': 'Data derivation, not a foreign key'}})
            for op in stage3:
                parent = semantic.get(op['source']) if op.get('source_ref', '').startswith('semantic.') else typed.get(op['source'])
                # Older exports may lack source_ref; member names disambiguate fallback.
                if parent is None:
                    parent = typed.get(op['source']) or semantic.get(op['source'])
                for out in op['outputs'].values():
                    child = semantic.get(out['name'])
                    if child is not None and parent is not None:
                        edges.append({'id': f'e{len(edges)}', 'child': child, 'parent': parent,
                                      'source_columns': [], 'target_columns': [], 'category': 'lineage', 'status': 'accepted',
                                      'details': {'kind': op['kind'], 'meaning': 'Data derivation, not a foreign key'}})
            return {'business_id': business, 'nodes': nodes, 'edges': edges, 'diagnostics': diagnostics}

    @staticmethod
    def render_diagrams(businesses, selected):
        payload = json.dumps({'businesses': businesses, 'selected': selected}, ensure_ascii=False)
        # Safe in a script data block even for adversarial table/column names.
        payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
        return Path(__file__).with_name('diagram.html').read_text(encoding='utf-8').replace('@@GRAPH_DATA@@', payload)

    @staticmethod
    def main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('business', nargs='?', default='biz_111', help='Business ID, e.g. biz_111 / dev_010')
        parser.add_argument('--split', choices=['train', 'public_test'])
        parser.add_argument('--out-dir', type=Path, help='Root containing train/ and public_test/')
        parser.add_argument('--port', type=int, default=4214)
        parser.add_argument('--no-serve', action='store_true', help='Generate an offline HTML file only')
        args = parser.parse_args(argv)
        if not 1 <= args.port <= 65535:
            parser.error('Invalid port')
        root = args.out_dir or Path(__file__).resolve().parents[3] / 'outputs' / 'ingestion'
        graphs, failures = [], []
        for split in ['train', 'public_test']:
            for database in sorted((root / split).glob('*/data.duckdb')):
                if database.parent.name == '_shared':
                    continue
                try:
                    graph = OrganizationDiagram.read_diagram(database)
                    graph.update(split=split, id=split+'/'+database.parent.name)
                    graphs.append(graph)
                except Exception as exc:
                    failures.append(f'{split}/{database.parent.name}: {exc}')
        selected = [g for g in graphs if g['business_id'] == args.business and (not args.split or g['split']==args.split)]
        if len(selected) != 1:
            for error in failures:
                print('Skipped:', error, flush=True)
            parser.error('Business not found or ambiguous; use --split. Available: '+', '.join(g['id'] for g in graphs))
        content = OrganizationDiagram.render_diagrams(graphs, selected[0]['id'])
        root.mkdir(parents=True, exist_ok=True)
        output = root / 'relationships.html'
        output.write_text(content, encoding='utf-8')
        print(f'Generated {output}: {len(graphs)} businesses; selected {selected[0]["id"]}.', flush=True)
        for error in failures:
            print('Skipped:', error, flush=True)
        if args.no_serve:
            return 0
        data = content.encode('utf-8')
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split('?', 1)[0] not in ('/', '/relationships.html'):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *_):
                pass
        try:
            server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
        except OSError as exc:
            parser.error(f'Cannot start local server: {exc}. HTML is available at {output}; use --port or --no-serve.')
        print(f'Open http://localhost:{args.port} — keep this terminal running; Ctrl+C to stop.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0

if __name__ == '__main__':
    raise SystemExit(OrganizationDiagram.main())
