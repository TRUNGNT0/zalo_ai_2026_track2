"""Production contracts and guide accuracy against the actual export."""
import hashlib

import duckdb
import pytest

from datamate.ingestion.layered import BusinessIngestion
from datamate.ingestion.store import IngestionStore
from datamate.ingestion.documentation import DatabaseGuide
from datamate.ingestion.documentation.run import DocumentationCommand
from datamate.ingestion.run import IngestionCommand
from datamate.ingestion.retype import RetypeService
from datamate.ingestion.organization.analyze import OrganizationAnalyzer


def source_at(directory):
    directory.mkdir(parents=True)
    (directory / 'items.csv').write_text('order_id,line_number,customer_id,quantity,unit\nO1,1,C1,2,kg\nO1,2,C1,3,kg\n', encoding='utf-8')
    (directory / 'README.md').write_text('Doanh nghiệp thử nghiệm.\n```\nNguồn tham khảo\n```', encoding='utf-8')
    return BusinessIngestion.ingest_business(directory, overrides={'files': {'items.csv': {'header_row': 1}}})


def test_guide_is_readonly_stable_and_standalone(tmp_path):
    business = source_at(tmp_path / 'source')
    database = tmp_path / 'out' / 'data.duckdb'
    IngestionStore.write_database(business, database, mode='production')
    assert {p.name for p in database.parent.iterdir()} == {'data.duckdb', 'catalog.json', 'DATABASE_GUIDE.md'}
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    guide = DatabaseGuide.write_guide(database)
    content = guide.read_text()
    assert DatabaseGuide.write_guide(database).read_text() == content
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert 'Doanh nghiệp thử nghiệm' in content
    assert 'không phải ràng buộc PK/FK vật lý' in content
    with duckdb.connect(str(database), read_only=True) as con:
        for schema, name, column in con.execute("SELECT table_schema,table_name,column_name FROM information_schema.columns WHERE table_schema IN ('typed','semantic')").fetchall():
            assert f'"{schema}"."{name}"' in content
            assert column in content
        assert con.execute('SELECT count(*) FROM meta.column_profiles').fetchone()[0] > 0


def test_mode_switch_and_organization_regenerate(tmp_path):
    business = source_at(tmp_path / 'source')
    database = tmp_path / 'out' / 'data.duckdb'
    IngestionStore.write_database(business, database)
    assert (database.parent / 'organization.json').exists()
    (database.parent / 'notes.md').write_text('User notes')
    OrganizationAnalyzer.analyze_database(database, mode='production')
    assert {p.name for p in database.parent.iterdir()} == {'data.duckdb', 'catalog.json', 'DATABASE_GUIDE.md', 'notes.md'}
    OrganizationAnalyzer.analyze_database(database, mode='dev')
    assert (database.parent / 'organization.json').exists()


def test_cli_production_and_retype_without_summary(tmp_path):
    root = tmp_path / 'split'
    source_at(root / 'data' / 'biz')
    (root / 'questions.jsonl').write_text('')
    out = tmp_path / 'out'
    assert IngestionCommand.main(['--split-dir', str(root), '--out-dir', str(out), '--mode', 'production']) == 0
    assert {p.name for p in out.iterdir()} == {'biz'}
    RetypeService.main(['--out-dir', str(out), '--mode', 'production'])
    assert {p.name for p in (out / 'biz').iterdir()} == {
        'data.duckdb', 'catalog.json', 'DATABASE_GUIDE.md', 'manifest.json'
    }
    assert DocumentationCommand.main(['--out-dir', str(out), '--mode', 'production']) == 0


def test_missing_database_never_created(tmp_path):
    database = tmp_path / 'missing.duckdb'
    with pytest.raises(FileNotFoundError):
        DatabaseGuide.write_guide(database)
    assert not database.exists()


def test_failed_guide_drops_stale_document(tmp_path):
    business = source_at(tmp_path / 'source')
    database = tmp_path / 'out' / 'data.duckdb'
    IngestionStore.write_database(business, database, mode='production')
    with duckdb.connect(str(database)) as con:
        con.execute('DROP TABLE meta.business')
    with pytest.raises(duckdb.Error):
        DatabaseGuide.write_guide(database)
    assert not (database.parent / 'DATABASE_GUIDE.md').exists()


def test_dev_after_production_retype_restores_summary(tmp_path):
    root = tmp_path / 'split'
    source_at(root / 'data' / 'biz')
    (root / 'questions.jsonl').write_text('')
    out = tmp_path / 'out'
    assert IngestionCommand.main(['--split-dir', str(root), '--out-dir', str(out), '--mode', 'production']) == 0
    RetypeService.main(['--out-dir', str(out), '--mode', 'dev'])
    import json
    result = json.loads((out / 'summary.json').read_text())['businesses'][0]
    assert result['tables'] == 1 and result['rows'] == 2
