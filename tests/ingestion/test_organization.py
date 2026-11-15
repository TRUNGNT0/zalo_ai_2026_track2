"""Stage-1 safety and SQL evidence tests, with real DuckDB exports."""
import csv
import json
import sys

import duckdb
import pytest

from datamate.ingestion.layered import BusinessIngestion
from datamate.ingestion.store import IngestionStore
from datamate.ingestion.organization.analyze import OrganizationAnalyzer
from datamate.ingestion.organization.run import OrganizationCommand


@pytest.fixture
def export(tmp_path):
    def build(tables, overrides=None):
        source = tmp_path / 'business'
        source.mkdir(exist_ok=True)
        for name, (header, rows) in tables.items():
            with (source / (name + '.csv')).open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        options = {'files': {name+'.csv': {'header_row': 1} for name in tables}}
        options.update(overrides or {})
        business = BusinessIngestion.ingest_business(source, overrides=options)
        destination = tmp_path / 'export' / 'business' / 'data.duckdb'
        IngestionStore.write_database(business, destination)
        result = json.loads((destination.parent / 'organization.json').read_text())
        return destination, result
    return build


def find_key(result, table, columns):
    return next(k for k in result['keys'] if k['table'] == table and set(k['columns']) == set(columns))


def find_relation(result, source, target):
    return next(r for r in result['relationships'] if r['source_table'] == source and r['target_table'] == target)


def test_profiles_and_nullable_relationship_and_idempotence(export):
    database, result = export({
        'customers': (['customer_id'], [['001'], ['002']]),
        'orders': (['order_id', 'customer_id'], [['O1', '001'], ['O2', '001'], ['O3', '']]),
    })
    r = find_relation(result, 'orders', 'customers')
    assert r['status'] == 'accepted'
    assert r['relationship_type'] == 'many_to_one'
    assert r['matched_rows'] == 2 and r['all_null_reference_rows'] == 1
    assert r['join_rows'] == 3 and r['join_extra_rows'] == 0
    p = next(p for p in result['profiles'] if p['table'] == 'orders' and p['column'] == 'customer_id')
    assert p['null_count'] == 1 and p['distinct_non_null'] == 1
    with duckdb.connect(str(database), read_only=True) as con:
        before = con.execute('SELECT * FROM meta.tables ORDER BY logical_name').fetchall()
        assert con.execute('SELECT count(*) FROM meta.relationships').fetchone()[0] > 0
    assert OrganizationAnalyzer.analyze_database(database) == result
    with duckdb.connect(str(database), read_only=True) as con:
        assert before == con.execute('SELECT * FROM meta.tables ORDER BY logical_name').fetchall()
        assert con.execute("SELECT count(*) FROM information_schema.views WHERE table_schema='semantic' AND table_name LIKE 't_%'").fetchone()[0] == 2


def test_coincidental_values_do_not_create_relationship(export):
    _, result = export({'a': (['order_id'], [['001'], ['002']]),
                        'b': (['product_id'], [['001'], ['002']]),
                        'c': (['amount'], [['001'], ['002']])})
    assert not result['relationships']
    assert not any(k['table'] == 'c' for k in result['keys'])


def test_duplicate_parent_is_not_join_target(export):
    _, result = export({'customers': (['customer_id'], [['001'], ['001']]),
                        'orders': (['order_id', 'customer_id'], [['O1', '001'], ['O2', '001']])})
    assert find_key(result, 'customers', ['customer_id'])['status'] == 'rejected'
    assert not any(r['target_table'] == 'customers' for r in result['relationships'])


def test_orphans_are_preserved_as_candidate(export):
    _, result = export({'customers': (['customer_id'], [['001'], ['002']]),
                        'orders': (['order_id', 'customer_id'], [['O1', '001'], ['O2', '999']])})
    r = find_relation(result, 'orders', 'customers')
    assert r['status'] == 'candidate' and r['unmatched_rows'] == 1
    assert r['coverage'] == .5 and r['join_extra_rows'] == 0


def test_competing_targets_are_not_arbitrarily_selected(export):
    _, result = export({'customers_a': (['customer_id'], [['001'], ['002']]),
                        'customers_b': (['customer_id'], [['001'], ['002']]),
                        'orders': (['order_id', 'customer_id'], [['O1', '001'], ['O2', '001']])})
    rs = [r for r in result['relationships'] if r['source_table'] == 'orders']
    assert len(rs) == 2
    assert all(r['status'] == 'candidate' and r['reason'] == 'ambiguous_target' for r in rs)


def test_composite_keys_and_scoped_relationship(export):
    _, result = export({
        'orders': (['branch_id', 'order_id'], [['B1', 'O1'], ['B1', 'O2'], ['B2', 'O1']]),
        'items': (['branch_id', 'order_id', 'line_number'],
                  [['B1', 'O1', '1'], ['B1', 'O1', '2'], ['B1', 'O2', '1'], ['B2', 'O1', '1']]),
    })
    k = find_key(result, 'orders', ['branch_id', 'order_id'])
    assert k['status'] == 'accepted'
    assert not any(k['table'] == 'orders' and len(k['columns']) > 2 for k in result['keys'])
    r = find_relation(result, 'items', 'orders')
    assert r['status'] == 'accepted' and len(r['source_columns']) == 2
    assert r['join_rows'] == 4


def test_missing_key_and_empty_table_not_accepted(export):
    _, result = export({'customers': (['customer_id', 'name'], [['001', 'A'], ['', 'B']]),
                        'empty': (['order_id'], [])})
    assert find_key(result, 'customers', ['customer_id'])['status'] == 'rejected'
    assert find_key(result, 'empty', ['order_id'])['status'] == 'rejected'


def test_generic_identifier_is_candidate_not_semantically_accepted(export):
    _, result = export({'unknown': (['Mã'], [['001'], ['002']])})
    k = find_key(result, 'unknown', ['Mã'])
    assert k['status'] == 'candidate' and not result['relationships']


def test_primary_description_alias_and_conflict():
    assert OrganizationAnalyzer.concept({'name': 'Mã chứng từ', 'description': 'mã đơn'}) == 'order'
    assert OrganizationAnalyzer.concept({'name': 'SKU', 'description': 'mã khách'}) is None
    assert OrganizationAnalyzer.concept({'name': 'note', 'description': 'ghi chú; mã đơn liên quan'}) is None


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows filesystem does not support double quotes in filenames')
def test_quotes_in_names_and_case_sensitive_values(export):
    _, result = export({'parent"x': (['customer_id'], [['AbC']]),
                        'child': (['order_id', 'customer_id'], [['O1', 'abc']])})
    assert find_relation(result, 'child', 'parent"x')['unmatched_rows'] == 1


def test_cli_and_nonexistent_database(export, tmp_path):
    database, _ = export({'customers': (['customer_id'], [['001']])})
    assert OrganizationCommand.main(['--out-dir', str(database.parent.parent)]) == 0
    assert (database.parent.parent / 'organization-summary.json').is_file()
    assert OrganizationCommand.main(['--database', str(tmp_path / 'missing.duckdb')]) == 1
    assert not (tmp_path / 'missing.duckdb').exists()
    with pytest.raises(SystemExit):
        OrganizationCommand.main(['--out-dir', str(database.parent.parent), '--business', 'unknown'])


def test_unique_foreign_key_is_not_assumed_parent_identity(export):
    _, result = export({'customers': (['customer_id'], [['001'], ['002']]),
                        'orders': (['order_id', 'customer_id'], [['O1', '001'], ['O2', '002']])})
    assert find_relation(result, 'orders', 'customers')['status'] == 'accepted'
    reverse = find_relation(result, 'customers', 'orders')
    assert reverse['status'] == 'candidate' and reverse['target_role_compatible'] is False


def test_parse_failures_cannot_turn_missing_references_into_acceptance(export):
    _, result = export({'customers': (['customer_id'], [['1'], ['2']]),
                        'orders': (['order_id', 'customer_id'], [['O1', '1'], ['O2', 'bad']])},
                       {'tables': {name: {'columns': {'customer_id': {'type': 'integer'}}}
                                   for name in ['customers', 'orders']}})
    r = find_relation(result, 'orders', 'customers')
    assert r['coverage'] == 1 and r['status'] == 'candidate'
    p = next(p for p in result['profiles'] if p['table'] == 'orders' and p['column'] == 'customer_id')
    assert p['parse_failed_count'] == 1


def test_partial_composite_reference_not_treated_as_valid_null(export):
    _, result = export({
        'orders': (['branch_id', 'order_id'], [['B1', 'O1'], ['B1', 'O2'], ['B2', 'O1']]),
        'items': (['branch_id', 'order_id', 'line_number'],
                  [['B1', 'O1', '1'], ['B1', '', '2']]),
    })
    r = find_relation(result, 'items', 'orders')
    assert r['status'] == 'candidate' and r['reason'] == 'partial_or_blank_reference'
    assert r['partial_or_blank_reference_rows'] == 1


def test_structural_rows_block_automatic_key_acceptance(export):
    _, result = export({'customers': (['customer_id', 'note'], [['001', 'ok'], ['002', 'Tổng cộng']])})
    assert find_key(result, 'customers', ['customer_id'])['status'] == 'candidate'


def test_accidentally_unique_order_product_pair_not_auto_identity(export):
    _, result = export({'items': (['order_id', 'product_id'],
                                  [['O1', 'P1'], ['O1', 'P2'], ['O2', 'P1']])})
    assert find_key(result, 'items', ['order_id', 'product_id'])['status'] == 'candidate'


def test_export_analysis_failure_does_not_replace_existing_database(export, tmp_path, monkeypatch):
    database, _ = export({'customers': (['customer_id'], [['001']])})
    original = database.read_bytes()
    business = BusinessIngestion.ingest_business(tmp_path / 'business')
    def fail(_):
        raise RuntimeError('injected analysis failure')
    monkeypatch.setattr('app.ingestion.organization.analyze.OrganizationAnalyzer.analyze_connection', fail)
    with pytest.raises(RuntimeError, match='injected analysis'):
        IngestionStore.write_database(business, database)
    assert database.read_bytes() == original
    assert not list(database.parent.glob('.ingest-*'))


def test_search_limits_are_reported(export):
    columns = ['ma_key_' + str(i) for i in range(12)]
    _, result = export({'unknown': (columns, [['X']*12, ['X']*12])})
    check = next(v for v in result['validations'] if v['check'] == 'key_search_complete_within_limits')
    assert check['passed'] is False
    assert check['details']['eligible_columns'] == 12
    assert len(check['details']['searched_columns']) == 10
    assert max(len(k['columns']) for k in result['keys']) == 3
    assert len(result['keys']) == 175
