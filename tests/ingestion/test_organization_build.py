"""Stage 2: exact conservation, safe joins, catalog ownership and rollback."""
import csv
import json

import duckdb
import pytest
from openpyxl import Workbook

from datamate.ingestion.layered import BusinessIngestion
from datamate.ingestion.store import IngestionStore
from datamate.ingestion.organization.analyze import OrganizationAnalyzer
from datamate.ingestion.organization.build import OrganizationBuilder
from datamate.ingestion.raw import RawData


@pytest.fixture
def export(tmp_path):
    def build(tables=None, overrides=None, sheets=None):
        source = tmp_path / 'business'
        source.mkdir(exist_ok=True)
        for name, (header, rows) in (tables or {}).items():
            with (source / (name + '.csv')).open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        options = {'files': {name+'.csv': {'header_row': 1} for name in (tables or {})}}
        if sheets:
            book = Workbook()
            book.remove(book.active)
            for name, (header, rows) in sheets.items():
                sheet = book.create_sheet(name)
                sheet.append(header)
                for row in rows:
                    sheet.append(row)
            book.save(source / 'sales.xlsx')
            options['files']['sales.xlsx'] = {'header_row': 1}
        options.update(overrides or {})
        database = tmp_path / 'output' / 'business' / 'data.duckdb'
        IngestionStore.write_database(BusinessIngestion.ingest_business(source, overrides=options), database)
        result = json.loads((database.parent / 'organization.json').read_text())
        return database, result
    return build


def operations(result, kind):
    return [op for op in result['stage2']['transformations'] if op['kind'] == kind]


def monthly(rows_a=None, rows_b=None):
    header = ['order_id', 'date', 'amount']
    return {'sales_2025_01': (header, rows_a or [['O1', '2025-01-10', '10']]),
            'sales_2025_02': (header, rows_b or [['O2', '2025-02-10', '20']])}


def test_merge_months_preserves_exact_values_and_lineage(export):
    database, result = export(monthly())
    op, = operations(result, 'merged')
    assert op['status'] == 'applied'
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT order_id, date::DATE::VARCHAR, amount::INT FROM semantic.' + RawData.quote_identifier(op['output_name']) + ' ORDER BY order_id').fetchall() == [('O1', '2025-01-10', 10), ('O2', '2025-02-10', 20)]
        assert con.execute('SELECT count(*) FROM meta.transformations').fetchone()[0] == 1
        assert con.execute('SELECT count(*) FROM meta.transformation_validations WHERE NOT passed').fetchone()[0] == 0
    assert OrganizationAnalyzer.analyze_database(database) == result
    assert (database.parent / 'ORGANIZATION_BUILD.md').is_file()


def test_reordered_and_normalized_column_mapping(export):
    database, result = export({'sales_2025_01': (['Order ID', 'date', 'amount'], [['O1', '2025-01-01', '10']]),
                               'sales_2025_02': (['amount', 'order_id', 'date'], [['20', 'O2', '2025-02-01']])})
    op, = operations(result, 'merged')
    assert op['status'] == 'applied'
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT "Order ID", amount::INT FROM semantic.' + RawData.quote_identifier(op['output_name']) + ' ORDER BY "Order ID"').fetchall() == [('O1', 10), ('O2', 20)]


def test_workbook_month_sheets_are_merged(export):
    _, result = export(sheets={'T01_2025': (['order_id', 'date'], [['O1', '2025-01-01']]),
                              '02.2025': (['order_id', 'date'], [['O2', '2025-02-01']])})
    op, = operations(result, 'merged')
    assert op['status'] == 'applied'


def test_partition_labels_must_not_overlap(export):
    _, result = export(sheets={'T01_2025': (['order_id', 'date'], [['O1', '2025-01-01']]),
                              '01.2025': (['order_id', 'date'], [['O2', '2025-01-02']])})
    op, = operations(result, 'merged')
    assert op['status'] == 'candidate' and op['reason'] == 'unknown_or_overlapping_period_labels'


@pytest.mark.parametrize('value', ['2025-01-15', ''])
def test_missing_or_wrong_month_date_blocks_merge(export, value):
    _, result = export(monthly(rows_b=[['O2', value, '20']]))
    op, = operations(result, 'merged')
    assert op['status'] == 'candidate' and op['reason'] == 'date_partition_unverified_or_ambiguous'


def test_cross_partition_identity_collision_blocks_merge(export):
    _, result = export(monthly(rows_b=[['O1', '2025-02-01', '20']]))
    op, = operations(result, 'merged')
    assert op['status'] == 'candidate' and op['reason'] == 'cross_partition_key_collision'


@pytest.mark.parametrize('change', ['unit', 'description', 'precision'])
def test_mismatched_contracts_are_not_implicitly_cast(export, change):
    overrides = {'tables': {'sales_2025_01': {'columns': {'amount': {'type': 'number'}}},
                            'sales_2025_02': {'columns': {'amount': {'type': 'number'}}}}}
    tables = monthly()
    if change == 'precision':
        tables['sales_2025_02'][1][0][2] = '20.50'
        overrides['tables']['sales_2025_02']['columns']['amount']['decimal'] = '.'
    else:
        overrides['tables']['sales_2025_02']['columns']['amount'][change] = 'different'
    _, result = export(tables, overrides)
    op, = operations(result, 'merged')
    assert op['status'] == 'candidate' and op['reason'] == 'column_type_unit_or_description_mismatch'


def test_duplicate_source_rows_are_kept_and_null_values_preserved(export):
    rows_a = [['O1', '2025-01-01', 'bad']]*2
    rows_b = [['O2', '2025-02-01', 'bad']]*2
    options = {'tables': {name: {'columns': {'amount': {'type': 'number'}}}
                           for name in monthly()}}
    database, result = export(monthly(rows_a, rows_b), options)
    op, = operations(result, 'merged')
    assert op['status'] == 'applied' and op['quality']['parse_failed_cells'] == 4
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT count(*), count(amount) FROM semantic.' + RawData.quote_identifier(op['output_name'])).fetchone() == (4, 0)


def test_technical_column_names_are_collision_safe(export):
    header = ['order_id', 'date', '_org_source_table', '_org_source_row_id', '_org_period', '__ingest_row_id']
    database, result = export({'sales_2025_01': (header, [['O1', '2025-01-01', 'x', 'y', 'z', 'user1']]),
                               'sales_2025_02': (header, [['O2', '2025-02-01', 'x', 'y', 'z', 'user2']])})
    op, = operations(result, 'merged')
    assert op['status'] == 'applied'
    assert op['lineage_columns']['source_table'] == '_org_source_table_2'
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT __ingest_row_id FROM semantic.' + RawData.quote_identifier(op['output_name']) + ' ORDER BY order_id').fetchall() == [('user1',), ('user2',)]


def relational():
    return {'customers': (['customer_id', 'name'], [['001', 'A'], ['002', 'B']]),
            'orders': (['order_id', 'customer_id', 'amount', 'customers__name'],
                       [['O1', '001', '10', 'source'], ['O2', '001', '20', 'source'], ['O3', '', '30', 'source']])}


def test_join_preserves_source_grain_nulls_measures_and_column_names(export):
    database, result = export(relational())
    op, = operations(result, 'joined')
    assert op['status'] == 'applied'
    assert op['parent_columns']['name'] == 'customers__name_2'
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT order_id, amount::INT, customers__name, customers__name_2 FROM semantic.' + RawData.quote_identifier(op['output_name']) + ' ORDER BY order_id').fetchall() == [('O1', 10, 'source', 'A'), ('O2', 20, 'source', 'A'), ('O3', 30, 'source', None)]
        assert con.execute('SELECT sum(amount) FROM semantic.' + RawData.quote_identifier(op['output_name'])).fetchone()[0] == 60


def test_no_view_for_unaccepted_relationship(export):
    data = relational()
    data['orders'][1][0][1] = 'missing'
    _, result = export(data)
    assert not operations(result, 'joined')


def test_stale_analysis_cannot_join_duplicated_parent(export):
    database, result = export(relational())
    with duckdb.connect(str(database)) as con:
        db, = con.execute("SELECT db_name FROM meta.tables WHERE logical_name='customers'").fetchone()
        con.execute('INSERT INTO typed.' + RawData.quote_identifier(db) + ' SELECT * FROM typed.' + RawData.quote_identifier(db) + ' LIMIT 1')
        con.execute('BEGIN')
        try:
            stage2 = OrganizationBuilder.build_connection(con, result)
            op, = stage2['transformations']
            assert op['reason'] == 'target_key_changed' and op['status'] == 'candidate'
        finally:
            con.execute('ROLLBACK')


def test_modified_generated_view_is_not_overwritten(export):
    database, result = export(monthly())
    op, = operations(result, 'merged')
    with duckdb.connect(str(database)) as con:
        con.execute('CREATE OR REPLACE VIEW semantic.' + RawData.quote_identifier(op['output_name']) + ' AS SELECT 42 AS user_value')
    for _ in range(2):
        refreshed = OrganizationAnalyzer.analyze_database(database)
        changed, = operations(refreshed, 'merged')
        assert changed['status'] == 'candidate' and changed['reason'] == 'output_name_in_use'
        with duckdb.connect(str(database), read_only=True) as con:
            assert con.execute('SELECT * FROM semantic.' + RawData.quote_identifier(op['output_name'])).fetchall() == [(42,)]


def test_unexpected_builder_failure_rolls_back_database_and_catalog(export, monkeypatch):
    database, result = export(monthly())
    op, = operations(result, 'merged')
    def fail(con, analysis, **kwargs):
        con.execute('DROP VIEW semantic.' + RawData.quote_identifier(op['output_name']))
        raise RuntimeError('injected stage2 failure')
    monkeypatch.setattr('app.ingestion.organization.build.OrganizationBuilder.build_connection', fail)
    with pytest.raises(RuntimeError, match='injected stage2'):
        OrganizationAnalyzer.analyze_database(database)
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT count(*) FROM semantic.' + RawData.quote_identifier(op['output_name'])).fetchone()[0] == 2
        assert con.execute("SELECT status FROM meta.transformations").fetchone()[0] == 'applied'


def test_configured_views_are_preserved(export):
    database, result = export(monthly(), {'views': {'manual_view': 'SELECT 123 AS value'}})
    OrganizationAnalyzer.analyze_database(database)
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT * FROM semantic.manual_view').fetchall() == [(123,)]


def test_obsolete_owned_view_is_removed_after_refresh(export):
    database, result = export(relational())
    op, = operations(result, 'joined')
    with duckdb.connect(str(database)) as con:
        parent, = con.execute("SELECT db_name FROM meta.tables WHERE logical_name='customers'").fetchone()
        con.execute('DELETE FROM typed.' + RawData.quote_identifier(parent))
    refreshed = OrganizationAnalyzer.analyze_database(database)
    assert not operations(refreshed, 'joined')
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM information_schema.views WHERE table_schema='semantic' AND table_name=?", [op['output_name']]).fetchone()[0] == 0


def test_external_dependency_prevents_removing_required_view(export):
    database, result = export(relational())
    op, = operations(result, 'joined')
    with duckdb.connect(str(database)) as con:
        con.execute('CREATE VIEW semantic.external_view AS SELECT * FROM semantic.' + RawData.quote_identifier(op['output_name']))
        parent, = con.execute("SELECT db_name FROM meta.tables WHERE logical_name='customers'").fetchone()
        con.execute('DELETE FROM typed.' + RawData.quote_identifier(parent))
    with pytest.raises(duckdb.Error):
        OrganizationAnalyzer.analyze_database(database)
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT count(*) FROM semantic.external_view').fetchone()[0] == 3
        assert con.execute("SELECT status FROM meta.transformations").fetchone()[0] == 'applied'


def test_no_merge_for_unrelated_csv_with_identical_headers(export):
    _, result = export({'orders': (['id', 'date'], [['001', '2025-01-01']]),
                        'payments': (['id', 'date'], [['002', '2025-02-01']])})
    assert not operations(result, 'merged')


def test_broken_raw_lineage_rejects_generated_view(export):
    database, result = export(monthly())
    with duckdb.connect(str(database)) as con:
        db, lineage = con.execute("SELECT db_name, lineage_column FROM meta.tables WHERE logical_name='sales_2025_01'").fetchone()
        con.execute('UPDATE typed.' + RawData.quote_identifier(db) + ' SET ' + RawData.quote_identifier(lineage) + "='missing-id'")
    refreshed = OrganizationAnalyzer.analyze_database(database)
    op, = operations(refreshed, 'merged')
    assert op['status'] == 'rejected' and op['reason'] == 'post_build_validation_failed'
    assert any(not v['passed'] and v['check'].startswith('raw_lineage_valid:') for v in refreshed['stage2']['validations'])
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM information_schema.views WHERE table_schema='semantic' AND table_name=?", [op['output_name']]).fetchone()[0] == 0


def test_many_union_branches_survive_catalog_serialization_and_repeated_refresh(export):
    tables = {f'sales_2025_{i:02}': (['order_id', 'date'], [[f'O{i}', f'2025-{i:02}-01']])
              for i in range(1, 6)}
    database, result = export(tables)
    original, = operations(result, 'merged')
    assert original['status'] == 'applied'
    for _ in range(3):
        refreshed = OrganizationAnalyzer.analyze_database(database)
        op, = operations(refreshed, 'merged')
        assert op['status'] == 'applied'
        assert op['definition_fingerprint'] == original['definition_fingerprint']
        with duckdb.connect(str(database), read_only=True) as con:
            assert con.execute('SELECT count(*) FROM semantic.' + RawData.quote_identifier(op['output_name'])).fetchone()[0] == 5


def test_meaningful_union_edit_is_not_hidden_by_fingerprint(export):
    tables = {f'sales_2025_{i:02}': (['order_id', 'date'], [[f'O{i}', f'2025-{i:02}-01']])
              for i in range(1, 4)}
    database, result = export(tables)
    original, = operations(result, 'merged')
    with duckdb.connect(str(database)) as con:
        # A row filter is a meaningful edit, not serialization whitespace/parentheses.
        con.execute('CREATE OR REPLACE VIEW semantic.' + RawData.quote_identifier(original['output_name']) +
                    ' AS SELECT * FROM (' + original['sql'] + ') WHERE order_id<>\'O1\'')
    refreshed = OrganizationAnalyzer.analyze_database(database)
    op, = operations(refreshed, 'merged')
    assert op['status'] == 'candidate' and op['reason'] == 'output_name_in_use'
