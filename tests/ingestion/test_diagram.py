"""Read-only ER diagrams: real metadata, no guessed links or embedded cell data."""
import csv
import hashlib
import json
from html.parser import HTMLParser

import duckdb
import pytest

from datamate.ingestion.layered import BusinessIngestion
from datamate.ingestion.store import IngestionStore
from datamate.ingestion.organization.diagram import OrganizationDiagram


@pytest.fixture
def export(tmp_path):
    def create(tables, business='biz_example'):
        source=tmp_path/'source'/business;source.mkdir(parents=True)
        for name,(headers,rows) in tables.items():
            with (source/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
        overrides={'files':{name+'.csv':{'header_row':1} for name in tables}}
        database=tmp_path/'outputs'/'train'/business/'data.duckdb'
        IngestionStore.write_database(BusinessIngestion.ingest_business(source,overrides=overrides),database)
        return database
    return create


def basic():
    return {'customers':(['customer_id','email'],[['001','sensitive@example.net'],['002','private@example.net']]),
            'orders':(['order_id','customer_id'],[['O1','001'],['O2','001']])}


def test_verified_links_columns_and_read_only_snapshot(export):
    database=export(basic())
    before=hashlib.sha256(database.read_bytes()).hexdigest()
    graph=OrganizationDiagram.read_diagram(database)
    assert hashlib.sha256(database.read_bytes()).hexdigest()==before
    typed={n['logical']:n for n in graph['nodes'] if n['category']=='typed'}
    links=[e for e in graph['edges'] if e['category']=='relationship' and e['status']=='accepted']
    assert len(links)==1
    edge=links[0]
    assert edge['child']==typed['orders']['id'] and edge['parent']==typed['customers']['id']
    assert edge['source_columns']==edge['target_columns']==['customer_id']
    assert edge['details']['relationship_type']=='many_to_one'
    assert next(c for c in typed['orders']['columns'] if c['name']=='customer_id')['badges']==['R']
    assert 'U' in next(c for c in typed['customers']['columns'] if c['name']=='customer_id')['badges']
    assert not any('PK' in c['badges'] for n in graph['nodes'] for c in n['columns'])
    assert 'sensitive@example.net' not in json.dumps(graph)
    assert len([e for e in graph['edges'] if e['category']=='lineage'])==2


def test_candidate_relationship_keeps_evidence_and_unverified_status(export):
    data=basic();data['orders'][1][1][1]='missing'
    graph=OrganizationDiagram.read_diagram(export(data))
    candidates=[e for e in graph['edges'] if e['category']=='relationship']
    assert candidates and all(e['status']=='candidate' for e in candidates)
    assert any(e['details']['unmatched_rows']==1 for e in candidates)


def test_logical_split_relationship_is_separate_from_derivation(export):
    graph=OrganizationDiagram.read_diagram(export({'order_items':(['order_id','line_number','ordered_at','product_id','quantity','unit'],[
        ['O1','1','2025-01-01','P1','2','ly'],['O1','2','2025-01-01','P2','3','ly']])}))
    models=[e for e in graph['edges'] if e['category']=='model']
    assert len(models)==1 and models[0]['source_columns']==['order_id']
    parent=next(n for n in graph['nodes'] if n['id']==models[0]['parent'])
    assert parent['category']=='split' and parent['keys'][0]['kind']=='K'
    assert len([e for e in graph['edges'] if e['category']=='lineage'])==3


def test_missing_catalog_view_is_reported_not_faked(export):
    database=export(basic())
    with duckdb.connect(str(database)) as con:
        view=con.execute("SELECT output_name FROM meta.transformations WHERE status='applied'").fetchone()[0]
        con.execute('DROP VIEW semantic."'+view.replace('"','""')+'"')
    graph=OrganizationDiagram.read_diagram(database)
    assert graph['diagnostics'] and not any(n['category']=='derived' for n in graph['nodes'])
    assert len([e for e in graph['edges'] if e['category']=='relationship' and e['status']=='accepted'])==1


def test_metadata_without_relationship_analysis_does_not_invent_links(export):
    database=export(basic())
    with duckdb.connect(str(database)) as con:
        for table in ['relationships','transformations','model_relationships','model_transformations']:
            con.execute('DROP TABLE meta.'+table)
    graph=OrganizationDiagram.read_diagram(database)
    assert not graph['edges'] and len(graph['nodes'])==2
    assert graph['diagnostics']


def test_render_escapes_script_injection_and_preserves_unicode():
    evil='</script><script>alert("x")</script> & Mã khách'
    graph={'id':'train/x','business_id':'x','split':'train','nodes':[{'label':evil}],'edges':[],'diagnostics':[]}
    html=OrganizationDiagram.render_diagrams([graph],'train/x')
    assert evil not in html and '\\u003c/script' in html
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.scripts=[];self.current=None
        def handle_starttag(self,tag,attrs):
            if tag=='script':self.current=''
        def handle_data(self,data):
            if self.current is not None:self.current+=data
        def handle_endtag(self,tag):
            if tag=='script':self.scripts.append(self.current);self.current=None
    parser=Parser();parser.feed(html)
    assert len(parser.scripts)==2
    assert json.loads(parser.scripts[0])['businesses'][0]['nodes'][0]['label']==evil


def test_offline_cli_and_invalid_selection(export):
    database=export(basic())
    root=database.parents[2]
    assert OrganizationDiagram.main(['biz_example','--out-dir',str(root),'--no-serve'])==0
    html=(root/'relationships.html').read_text()
    assert 'train/biz_example' in html and '@@GRAPH_DATA@@' not in html
    with pytest.raises(SystemExit):OrganizationDiagram.main(['not-found','--out-dir',str(root),'--no-serve'])
    assert (root/'relationships.html').read_text()==html


def test_nonexistent_database_is_not_created(tmp_path):
    database=tmp_path/'missing.duckdb'
    with pytest.raises(FileNotFoundError):OrganizationDiagram.read_diagram(database)
    assert not database.exists()
