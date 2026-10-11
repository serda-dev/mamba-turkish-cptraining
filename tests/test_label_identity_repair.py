import hashlib,json
import pytest
from scripts.cpt_repair_label_identity import recover_row,signature
from src.data.classified import document_id

SOURCE={'id':'source','repo':'owner/source','revision':'a'*40,'config':'default','split':'train'}


def parent(text,ordinal=100000):
    docid=document_id(SOURCE,ordinal,text)
    return {'source_id':'source','row_ordinal':ordinal,'doc_id':docid,
            'document_hash':hashlib.sha256(text.encode()).hexdigest(),'input_bytes':len(text.encode()),
            'input_truncated':False,'predicted_route':'KEEP',
            'predicted_label_json':json.dumps({'id':docid,'type':'general','q':{'utility':3},'noise':[]})}


def test_exact_text_recovery_changes_only_identity_and_retains_parent_provenance():
    text='Aynı metnin tam UTF-8 içeriği'; original=parent(text)
    row=recover_row(SOURCE,2247,text,original)
    assert row['row_ordinal']==2247 and row['doc_id']==document_id(SOURCE,2247,text)
    assert row['parent_doc_id']==original['doc_id'] and row['parent_row_ordinal']==100000
    assert signature(row)==signature(original)
    assert original['row_ordinal']==100000


@pytest.mark.parametrize('field,value',[('document_hash','0'*64),('doc_id','doc_wrong'),('input_bytes',1),('source_id','other')])
def test_recovery_does_not_authenticate_modified_parent(field,value):
    original=parent('Metin');original[field]=value
    with pytest.raises(ValueError):recover_row(SOURCE,7,'Metin',original)


def test_inconsistent_predictions_are_detected_even_for_identical_content():
    one=parent('Metin',100000);two=parent('Metin',200000)
    assert signature(one)==signature(two)
    label=json.loads(two['predicted_label_json']);label['q']['utility']=1
    two['predicted_label_json']=json.dumps(label)
    assert signature(one)!=signature(two)
