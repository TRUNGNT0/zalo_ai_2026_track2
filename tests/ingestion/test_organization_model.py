"""Stage-3 lossless decomposition, measure semantics, units and lifecycle."""
import csv
import json

import duckdb
import pytest

from datamate.ingestion.layered import BusinessIngestion
from datamate.ingestion.store import IngestionStore
from datamate.ingestion.organization.analyze import OrganizationAnalyzer
from datamate.ingestion.raw import RawData


@pytest.fixture
def export(tmp_path):
    def build(tables, overrides=None):
        source = tmp_path / 'business'
        source.mkdir(exist_ok=True)
        for name, (header, rows) in tables.items():
            with (source / (name+'.csv')).open('w', newline='', encoding='utf-8') as f:
                writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
        options={'files': {name+'.csv': {'header_row': 1} for name in tables}}
        options.update(overrides or {})
        database=tmp_path/'output'/'business'/'data.duckdb'
        IngestionStore.write_database(BusinessIngestion.ingest_business(source,overrides=options),database)
        return database,json.loads((database.parent/'organization.json').read_text())
    return build


def wide():
    header=['order_id','line_number','ordered_at','order_status','customer_id','product_id','quantity','unit','line_amount','currency']
    rows=[['O1','1','2025-01-01 06:03:59','completed','C1','P1','2','ly','20','VND'],
          ['O1','2','2025-01-01 06:03:59','completed','C1','P2','3','ly','30','VND'],
          ['O2','1','2025-01-02 21:15:30','cancelled','C2','P1','4','ly','40','VND']]
    return {'order_items':(header,rows)}


def get_op(result, kind):
    return next(op for op in result['stage3']['transformations'] if op['kind']==kind)


def view(op, tag):
    return 'semantic.'+RawData.quote_identifier(op['outputs'][tag]['name'])


def test_lossless_split_reconstructs_values_and_keeps_precise_timestamps(export):
    database,result=export(wide())
    split=get_op(result,'split_order_lines')
    assert split['status']=='applied'
    assert split['keys']==['order_id']
    with duckdb.connect(str(database),read_only=True) as con:
        assert con.execute('SELECT order_id,ordered_at::VARCHAR,customer_id FROM '+view(split,'orders')+' ORDER BY order_id').fetchall()==[
            ('O1','2025-01-01 06:03:59','C1'),('O2','2025-01-02 21:15:30','C2')]
        assert con.execute('SELECT count(*) FROM '+view(split,'lines')).fetchone()[0]==3
        assert len(con.execute(split['reconstruction_sql']).fetchall())==3
    checks=[v for v in result['stage3']['validations'] if v['transformation_id']==split['id']]
    assert all(v['passed'] for v in checks)
    assert json.loads(json.dumps(OrganizationAnalyzer.analyze_database(database),default=str))==result


def test_aggregate_preserves_status_and_is_not_revenue(export):
    database,result=export(wide())
    aggregate=get_op(result,'aggregate_source_events')
    assert aggregate['status']=='applied'
    measures={m['column']:m for m in aggregate['measure_decisions']}
    assert measures['line_amount']['status']=='included'
    assert measures['quantity']['status']=='included'
    with duckdb.connect(str(database),read_only=True) as con:
        rows=con.execute('SELECT order_id,order_status,'+RawData.quote_identifier(aggregate['count_column'])+',"source_sum__quantity","source_sum__line_amount" FROM '+view(aggregate,'aggregate')+' ORDER BY order_id').fetchall()
        assert rows==[('O1','completed',2,5,50),('O2','cancelled',1,4,40)]
    assert 'no as_of_date' in aggregate['scope']


@pytest.mark.parametrize('field,value', [('customer_id','DIFFERENT'),('ordered_at','2025-01-03 12:00:00'),('order_status','returned')])
def test_conflicting_order_attributes_block_entire_split(export,field,value):
    data=wide();header,rows=data['order_items'];rows[1][header.index(field)]=value
    _,result=export(data)
    op=get_op(result,'split_order_lines')
    assert op['status']=='candidate' and op['reason']=='conflicting_order_attributes'
    assert not op['outputs']


def test_null_and_nonnull_attribute_are_a_real_fd_conflict(export):
    data=wide();data['order_items'][1][0][4]=''
    _,result=export(data)
    assert get_op(result,'split_order_lines')['reason']=='conflicting_order_attributes'


def test_header_parse_failure_does_not_make_false_consistency(export):
    data=wide();data['order_items'][1][0][2]='bad';data['order_items'][1][1][2]='bad'
    _,result=export(data,{'tables':{'order_items':{'columns':{'ordered_at':{'type':'date'}}}}})
    assert get_op(result,'split_order_lines')['reason']=='order_attributes_parse_failed'
    agg=get_op(result,'aggregate_source_events')
    assert agg['status']=='applied' and not agg['timestamp_columns']


def test_branch_is_part_of_order_scope(export):
    data=wide();header,rows=data['order_items'];header.insert(0,'branch_id')
    for i,row in enumerate(rows):row.insert(0,'B1' if i<2 else 'B2')
    rows[2][1]='O1'
    _,result=export(data)
    split=get_op(result,'split_order_lines')
    assert split['status']=='applied' and split['keys']==['branch_id','order_id']


def test_unknown_unit_defers_sum_without_losing_source_count(export):
    data=wide();header,rows=data['order_items'];idx=header.index('currency');header.pop(idx)
    for row in rows:row.pop(idx)
    database,result=export(data)
    agg=get_op(result,'aggregate_source_events')
    decision=next(m for m in agg['measure_decisions'] if m['column']=='line_amount')
    assert decision['status']=='deferred' and decision['reason']=='unit_unresolved'
    with duckdb.connect(str(database),read_only=True) as con:
        assert con.execute('SELECT sum('+RawData.quote_identifier(agg['count_column'])+') FROM '+view(agg,'aggregate')).fetchone()[0]==3
        names=[row[0] for row in con.execute('DESCRIBE '+view(agg,'aggregate')).fetchall()]
        assert 'source_sum__line_amount' not in names


def test_mixed_quantity_units_and_currencies_stay_separate(export):
    data=wide();data['order_items'][1][1][7]='hop';data['order_items'][1][1][9]='USD'
    database,result=export(data)
    agg=get_op(result,'aggregate_source_events')
    with duckdb.connect(str(database),read_only=True) as con:
        rows=con.execute('SELECT order_id,unit,currency,"source_sum__quantity" FROM '+view(agg,'aggregate')+' ORDER BY order_id,unit').fetchall()
        assert rows==[('O1','hop','USD',3),('O1','ly','VND',2),('O2','ly','VND',4)]


def payments():
    return {'payments':(['payment_id','order_id','payment_amount','currency','paid_at'],[
        ['P1','O1','10','VND','2025-01-01 06:03:59'],
        ['P2','O1','15','VND','2025-01-02 14:30:45'],
        ['P3','O2','20','VND','2025-01-03 22:00:01']])}


def test_payment_summary_count_sum_and_first_last_time(export):
    database,result=export(payments())
    agg=get_op(result,'aggregate_source_events')
    assert agg['source_kind']=='payments' and agg['status']=='applied'
    assert not any(op['kind']=='split_order_lines' for op in result['stage3']['transformations'])
    with duckdb.connect(str(database),read_only=True) as con:
        rows=con.execute('SELECT order_id,"source_sum__payment_amount","source_first__paid_at"::VARCHAR,"source_last__paid_at"::VARCHAR FROM '+view(agg,'aggregate')+' ORDER BY order_id').fetchall()
        assert rows==[('O1',25,'2025-01-01 06:03:59','2025-01-02 14:30:45'),('O2',20,'2025-01-03 22:00:01','2025-01-03 22:00:01')]


def test_duplicate_payment_identity_blocks_summary(export):
    data=payments();data['payments'][1][1][0]='P1'
    _,result=export(data)
    agg=get_op(result,'aggregate_source_events')
    assert agg['status']=='candidate' and agg['reason']=='event_identity_unresolved'


def test_null_measure_does_not_silently_become_zero(export):
    data=payments();data['payments'][1][1][2]=''
    database,result=export(data)
    agg=get_op(result,'aggregate_source_events')
    assert agg['status']=='applied'
    decision=agg['measure_decisions'][0]
    assert decision['status']=='deferred' and decision['reason']=='measure_missing_or_parse_failed'
    with duckdb.connect(str(database),read_only=True) as con:
        assert con.execute('SELECT sum('+RawData.quote_identifier(agg['count_column'])+') FROM '+view(agg,'aggregate')).fetchone()[0]==3


def test_explicit_metadata_unit_can_enable_sum(export):
    data=payments();header,rows=data['payments'];header.pop(3)
    for row in rows:row.pop(3)
    _,result=export(data,{'tables':{'payments':{'columns':{'payment_amount':{'unit':'VND'}}}}})
    decision=get_op(result,'aggregate_source_events')['measure_decisions'][0]
    assert decision['status']=='included' and decision['unit']=='VND'


def test_unit_metadata_conflict_defers_sum(export):
    data=payments();data['payments'][1][1][3]='USD'
    _,result=export(data,{'tables':{'payments':{'columns':{'payment_amount':{'unit':'VND'}}}}})
    assert get_op(result,'aggregate_source_events')['measure_decisions'][0]['reason']=='unit_metadata_conflict'


def test_rulebook_answer_currency_is_not_source_unit_evidence(export):
    data=payments();header,rows=data['payments'];header.pop(3)
    for row in rows:row.pop(3)
    database,_=export(data)
    with duckdb.connect(str(database)) as con:
        con.execute("UPDATE meta.business SET rulebook='Tiền là số nguyên VND'")
    result=OrganizationAnalyzer.analyze_database(database)
    assert get_op(result,'aggregate_source_events')['measure_decisions'][0]['reason']=='unit_unresolved'


def test_structural_rows_and_missing_keys_are_not_aggregated(export):
    data=wide();data['order_items'][1][1][0]=''
    _,result=export(data)
    assert all(op['status']=='candidate' for op in result['stage3']['transformations'])


def test_corrupted_lineage_blocks_models(export):
    database,_=export(wide())
    with duckdb.connect(str(database)) as con:
        db,lineage=con.execute('SELECT db_name,lineage_column FROM meta.tables').fetchone()
        con.execute('UPDATE typed.'+RawData.quote_identifier(db)+' SET '+RawData.quote_identifier(lineage)+"='orphan'")
    result=OrganizationAnalyzer.analyze_database(database)
    assert all(op['status']=='candidate' for op in result['stage3']['transformations'])


def test_modified_model_view_is_preserved(export):
    database,result=export(wide())
    old=get_op(result,'aggregate_source_events')
    with duckdb.connect(str(database)) as con:
        con.execute('CREATE OR REPLACE VIEW '+view(old,'aggregate')+' AS SELECT 42 AS manual_value')
    for _ in range(2):
        refreshed=OrganizationAnalyzer.analyze_database(database)
        op=get_op(refreshed,'aggregate_source_events')
        assert op['status']=='candidate' and op['reason']=='output_name_in_use'
        with duckdb.connect(str(database),read_only=True) as con:
            assert con.execute('SELECT * FROM '+view(old,'aggregate')).fetchall()==[(42,)]


def test_model_failure_rolls_back_all_layers(export,monkeypatch):
    database,result=export(wide())
    old=get_op(result,'aggregate_source_events')
    def fail(con,analysis):
        raise RuntimeError('injected model error')
    monkeypatch.setattr('app.ingestion.organization.model.OrganizationModelBuilder.build_models', fail)
    with pytest.raises(RuntimeError,match='injected model'):
        OrganizationAnalyzer.analyze_database(database)
    with duckdb.connect(str(database),read_only=True) as con:
        assert con.execute('SELECT sum('+RawData.quote_identifier(old['count_column'])+') FROM '+view(old,'aggregate')).fetchone()[0]==3


def test_merged_source_is_used_once_and_rebuilds_dependency_order(export):
    data=wide();header,rows=data['order_items']
    second=[row.copy() for row in rows]
    for row in second:
        row[0]='N'+row[0];row[4]='N'+row[4];row[2]=row[2].replace('2025-01','2025-02')
    database,result=export({'sales_2025_01':(header,rows),'sales_2025_02':(header,second)})
    assert len(result['stage2']['transformations'])==1
    assert result['stage2']['transformations'][0]['status']=='applied'
    assert len(result['stage3']['transformations'])==2
    for _ in range(2):
        result=OrganizationAnalyzer.analyze_database(database)
        assert all(op['status']=='applied' for op in result['stage3']['transformations'])
        agg=get_op(result,'aggregate_source_events')
        with duckdb.connect(str(database),read_only=True) as con:
            assert con.execute('SELECT sum('+RawData.quote_identifier(agg['count_column'])+') FROM '+view(agg,'aggregate')).fetchone()[0]==6


def test_numeric_arbitrary_columns_are_not_summed(export):
    data=wide();header,rows=data['order_items'];header.append('random_score')
    for i,row in enumerate(rows):row.append(str(i))
    _,result=export(data)
    agg=get_op(result,'aggregate_source_events')
    assert not any(m['column']=='random_score' for m in agg['measure_decisions'])


def test_split_relationship_is_recorded_for_query_planning(export):
    database,result=export(wide())
    split=get_op(result,'split_order_lines')
    rel,=result['stage3']['relationships']
    assert rel['source_view']==split['outputs']['lines']['name']
    assert rel['target_view']==split['outputs']['orders']['name']
    assert rel['columns']==['order_id'] and rel['relationship_type']=='many_to_one'
    with duckdb.connect(str(database),read_only=True) as con:
        assert con.execute('SELECT count(*) FROM meta.model_relationships').fetchone()[0]==1


def test_typed_string_date_is_reported_as_unresolved(export):
    data=wide();data['order_items'][1][0][2]='bad';data['order_items'][1][1][2]='bad'
    _,result=export(data)
    assert get_op(result,'split_order_lines')['reason']=='order_attribute_type_unresolved'
    agg=get_op(result,'aggregate_source_events')
    assert agg['timestamp_decisions'][0]['reason']=='timestamp_not_typed'


def test_total_row_flag_blocks_modeling(export):
    data=wide();data['order_items'][1][0][5]='Tổng cộng'
    _,result=export(data)
    assert all(op['status']=='candidate' for op in result['stage3']['transformations'])
    assert all(op['evidence']['structural_rows']>0 for op in result['stage3']['transformations'])


def test_empty_unit_field_cannot_enable_quantity_sum(export):
    data=wide();data['order_items'][1][0][7]=''
    _,result=export(data)
    agg=get_op(result,'aggregate_source_events')
    quantity=next(m for m in agg['measure_decisions'] if m['column']=='quantity')
    assert quantity['status']=='deferred' and quantity['reason']=='unit_column_incomplete'


def test_converted_quantity_is_not_assumed_to_use_source_unit(export):
    data=wide()
    _,result=export(data,{'tables':{'order_items':{'columns':{'quantity':{'description':'số lượng (đã đổi ra đơn vị cơ bản)'}}}}})
    agg=get_op(result,'aggregate_source_events')
    assert not any(m['column']=='quantity' for m in agg['measure_decisions'])


def test_decimal_payment_sums_remain_exact(export):
    from decimal import Decimal
    data=payments()
    for row,value in zip(data['payments'][1],['0.10','0.20','1.25']):row[2]=value
    database,result=export(data,{'tables':{'payments':{'columns':{'payment_amount':{'type':'number','decimal':'.'}}}}})
    agg=get_op(result,'aggregate_source_events')
    with duckdb.connect(str(database),read_only=True) as con:
        assert con.execute('SELECT "source_sum__payment_amount" FROM '+view(agg,'aggregate')+" WHERE order_id='O1'").fetchone()[0]==Decimal('0.30')


def test_null_status_is_a_separate_group_not_dropped(export):
    data=wide();data['order_items'][1][1][3]=''
    database,result=export(data)
    agg=get_op(result,'aggregate_source_events')
    with duckdb.connect(str(database),read_only=True) as con:
        assert con.execute('SELECT '+RawData.quote_identifier(agg['count_column'])+' FROM '+view(agg,'aggregate')+' WHERE order_status IS NULL').fetchone()[0]==1
        assert con.execute('SELECT sum('+RawData.quote_identifier(agg['count_column'])+') FROM '+view(agg,'aggregate')).fetchone()[0]==3
