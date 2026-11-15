from unittest.mock import Mock
import json
import sqlite3

from jsonschema import Draft202012Validator
from openpyxl import Workbook
import pytest

from datamate.ingestion.processing.pipeline import IngestionPipeline
from datamate.ingestion.sources.inventory import SourceInventory
from datamate.manifests.migration import ManifestMigrator
from datamate.manifests.service import BusinessManifestService, SCHEMA_FILE
from datamate.manifests.validation import ManifestValidator
from datamate.rules.rulebook import Rulebook

class TestManifestV2:
    @staticmethod
    def fixture_business(tmp_path):
        root = tmp_path / 'biz_test'
        root.mkdir()
        (root / 'README.md').write_text('# Dữ liệu\n## Bảng\nMã đơn nối orders và lines.\n## Quy ước\namount là thành tiền dòng.\n')
        db = sqlite3.connect(root / 'pos.sqlite')
        db.executescript('CREATE TABLE orders (shop TEXT, id TEXT, amount INTEGER); CREATE TABLE lines (shop TEXT, id TEXT, amount INTEGER);')
        db.executemany('INSERT INTO orders VALUES (?, ?, ?)', [('a','1',10),('b','1',20)])
        db.executemany('INSERT INTO lines VALUES (?, ?, ?)', [('a','1',4),('a','1',6),('b','1',20)])
        db.commit(); db.close()
        schema = json.loads(SCHEMA_FILE.with_name('business_manifest.v2.schema.json').read_text())
        data = BusinessManifestService.build_input(root, '# Rules\n## 1. Revenue\nRevenue rules\n## 6. Industry\n### Pharmacy\nExpiry rules\n', schema)
        old = {'schema_version':'1.0','business_id':'biz_test','tables':[{'table_role':'sales_records','source_files':['pos.sqlite'],'columns':{'lines.amount':'thành tiền dòng'},'source_refs':['readme.conventions']}], 'facts':[], 'unresolved':[]}
        manifest = ManifestMigrator.upgrade(old, data)
        relationship = {'relationship_id':'orders_lines','from':{'source_file':'pos.sqlite','source_object':'lines','columns':['shop','id']},'to':{'source_file':'pos.sqlite','source_object':'orders','columns':['shop','id']},'cardinality':'many_to_one','source':'readme','source_refs':['readme.bang']}
        manifest['relationships'] = [relationship]
        return root, data, manifest, schema

    def test_v2_rejects_invented_columns_and_unknown_relationship_endpoints(self, tmp_path):
        _, data, manifest, _ = TestManifestV2.fixture_business(tmp_path)
        manifest['tables'][0]['columns'][0]['source_name'] = 'orders'
        manifest['relationships'][0]['to']['columns'] = ['missing', 'id']
        report = ManifestValidator.validate(manifest, data['source_inventory'], data['_evidence_ids'])
        assert any('Unknown source column' in e for e in report['errors'])
        assert any('Invalid relationship endpoint' in e for e in report['errors'])

    def test_v2_exact_table_binding_and_verified_composite_relationship(self, tmp_path):
        root, _, manifest, schema = TestManifestV2.fixture_business(tmp_path)
        Draft202012Validator(schema).validate(manifest)
        path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
        catalog = IngestionPipeline.ingest(root, tmp_path/'output', path)
        tables = {t['source_objects'][0]:t for t in catalog['tables']}
        amounts = {k:next(c for c in v['columns'] if c['source_name']=='amount') for k,v in tables.items()}
        assert amounts['lines']['semantic_name'] == 'line_amount'
        assert amounts['orders']['semantic_name'] is None
        rel = catalog['relationships'][0]
        assert rel['status'] == 'verified'
        assert rel['validation']['source_duplicate_keys'] == 1
        assert rel['validation']['target_duplicate_keys'] == 0
        assert catalog['quality']['capabilities']['revenue']['status'] == 'blocked'

    @pytest.mark.parametrize('change,expected', [('single_key','failed_validation'), ('orphan','failed_validation'), ('candidate','candidate')])
    def test_bad_or_inferred_joins_never_become_verified(self, tmp_path, change, expected):
        root, _, manifest, _ = TestManifestV2.fixture_business(tmp_path)
        rel = manifest['relationships'][0]
        if change == 'single_key':
            rel['from']['columns'] = rel['to']['columns'] = ['id']
        elif change == 'candidate':
            rel['source'] = 'inferred_candidate'
        else:
            db=sqlite3.connect(root/'pos.sqlite'); db.execute("INSERT INTO lines VALUES ('c','2',99)"); db.commit(); db.close()
        path=tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
        catalog=IngestionPipeline.ingest(root,tmp_path/'output',path)
        assert catalog['relationships'][0]['status']==expected
        assert any(i['topic']=='relationship:orders_lines' for i in catalog['unresolved'])

    def test_generator_repairs_physical_error_and_computes_missing_columns(self, tmp_path):
        _,data,manifest,schema=TestManifestV2.fixture_business(tmp_path)
        bad=json.loads(json.dumps(manifest)); bad['tables'][0]['columns'][0]['source_name']='invented'
        llm=Mock();llm.generate.side_effect=[json.dumps(bad),json.dumps(manifest)]
        result=BusinessManifestService.generate(llm,data,schema)
        assert llm.generate.call_count==2
        assert 'Unknown source column' in llm.generate.call_args.args[0]
        assert any(i['topic'].startswith('column_mapping:') for i in result['unresolved'])

    def test_profiler_covers_multiple_sheets_and_records_uncertain_header(self, tmp_path):
        root=tmp_path/'biz_excel';root.mkdir()
        wb=Workbook();s=wb.active;s.title='2025';s.append(['Report']);s.append(['ID','Amount']);s.append(['001',-10]);s.append(['002',None])
        s=wb.create_sheet('2026');s.append(['ID','Amount']);s.append(['003',30]);wb.save(root/'sales.xlsx')
        inventory=SourceInventory.inspect(root)
        assert len(inventory['objects'])==2
        first=inventory['objects'][0]
        assert first['header_row_number']==2
        assert first['header_status']=='heuristic_header'
        assert first['columns'][1]['negative_count']==1
        assert first['columns'][1]['null_count']==1
        assert len(inventory['files']['sales.xlsx'])==64

    def test_rulebook_keeps_preamble_and_nested_industry_rules(self):
        sections=Rulebook.parse('# Rules\nas_of_date\n## 1. Sales\nSales\n## 6. Industry\n### Pharmacy\nExpiry\n## 10. Units\nUnits')
        assert sections[0]['evidence_id']=='rulebook.preamble'
        assert '### Pharmacy\nExpiry' in sections[2]['text']

    def test_legacy_table_name_is_rejected_as_column(self, tmp_path):
        _,data,_,_=TestManifestV2.fixture_business(tmp_path)
        legacy={'schema_version':'1.0','tables':[{'source_files':['pos.sqlite'],'columns':{'orders':'conversion'}}]}
        report=ManifestValidator.validate(legacy,data['source_inventory'])
        assert report['errors']

    def test_join_guard_blocks_reverse_fanout_and_unverified_edges(self):
        from datamate.catalog.relationships import RelationshipService
        from datamate.catalog.relationships import RelationshipService
        relation={'relationship_id':'r','from_table':'lines','to_table':'orders','cardinality':'many_to_one','status':'verified'}
        catalog={'relationships':[relation]}
        assert RelationshipService.validate_join(catalog,'r',aggregate_from='lines')['allowed']
        assert not RelationshipService.validate_join(catalog,'r',aggregate_from='orders')['allowed']
        relation['status']='candidate'
        assert not RelationshipService.validate_join(catalog,'r')['allowed']
        assert RelationshipService.verified(catalog)==[]

    def test_rejects_sum_of_stock_balance_and_mismatched_unit_basis(self):
        from datamate.manifests.validation import ManifestValidator
        col=ManifestMigrator.mapped_column('balance','tồn sau lần nhập/bán/trả đó, theo đơn vị cơ bản',['readme.conventions'])
        assert ManifestValidator.validate_column_semantics(col)==[]
        col['aggregation']='sum'
        assert any('aggregation=last' in e for e in ManifestValidator.validate_column_semantics(col))
        col=ManifestMigrator.mapped_column('cost','giá vốn một đơn vị bán',['readme.conventions'])
        col['unit_basis']='base_unit'
        assert any('unit_basis=sale_unit' in e for e in ManifestValidator.validate_column_semantics(col))

    def test_readme_enrichment_requires_literal_unique_binding(self, tmp_path):
        from datamate.manifests.migration import ManifestMigrator
        _,_,manifest,_=TestManifestV2.fixture_business(tmp_path)
        sections=[{'evidence_id':'readme.bang','text':'**id** là mã đơn; `orders.amount` là số khách phải trả cho cả đơn.'}]
        ManifestMigrator.enrich_explicit_readme_definitions(manifest,sections)
        for table in manifest['tables']:
            assert next(c for c in table['columns'] if c['source_name']=='id')['status']=='unresolved'
        orders=next(t for t in manifest['tables'] if t['source_object']=='orders')
        assert next(c for c in orders['columns'] if c['source_name']=='amount')['semantic_name']=='order_payable'
