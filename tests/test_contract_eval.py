from src.eval.contracts import score_response

CASE={'expected':{'sonuc':12}}
def test_content_correct_extra_prose_is_contract_failure():
    result=score_response('Cevap: {"sonuc":12}',CASE)
    assert result['task_correct'] is True
    assert result['json_parse_valid'] is False
    assert result['contract_and_task_correct'] is False

def test_valid_format_wrong_answer_separate():
    result=score_response('{"sonuc":11}',CASE)
    assert result['schema_valid'] is True
    assert result['task_correct'] is False

def test_malformed_response_is_not_called_semantic_failure():
    result=score_response('sonuc=12',CASE)
    assert result['task_correct'] is None
    assert not result['schema_valid']

def test_type_and_extra_key_contract():
    assert not score_response('{"sonuc":true}',{'expected':{'sonuc':1}})['task_correct']
    result=score_response('{"sonuc":12,"extra":"x"}',CASE)
    assert result['task_correct'] is True and not result['schema_valid']
