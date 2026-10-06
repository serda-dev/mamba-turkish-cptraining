import json
from pathlib import Path
import pytest
from src.cli import build_phase_text_stream, cache_data_identity, phase_token_cache_dir, validate_runtime_config
from src.data.curriculum import iter_english_texts
from src.data.mixing import token_balanced_mix

class Tok:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

def test_english_sources_are_not_silently_dropped(tmp_path):
    entries = {}
    for key in ('a', 'b'):
        root = tmp_path / key
        root.mkdir()
        (root/'x.jsonl').write_text(json.dumps({'text': key*10})+'\n')
        entries[key] = {'local_path': str(root)}
    cfg = {'datasets': {'english': entries}}
    assert list(iter_english_texts({'english_sources': [{'dataset':'a'}, {'dataset':'b'}]}, cfg)) == ['a'*10,'b'*10]

def test_token_mix_uses_token_lengths_and_stops_at_exhaustion():
    result = list(token_balanced_mix([['t'*99]*100, ['e'*9]*1000], [.8,.2], Tok()))
    counts = [sum(len(text)+1 for text, i in result if i == source) for source in (0,1)]
    assert abs(counts[0]/sum(counts)-.8) < .01
    assert list(token_balanced_mix([[], ['e']], [.8,.2], Tok())) == []

def test_validation_cache_has_distinct_path_and_identity(tmp_path):
    cfg={'paths':{'cache_dir':str(tmp_path)}, 'token_cache':{'split':'validation'}}
    assert phase_token_cache_dir(cfg,1).name == 'phase_1_validation'
    assert cache_data_identity(cfg,{'id':1})['split'] == 'validation'

def test_continuous_config_rejects_unfrozen_labels_and_wrong_storage(tmp_path):
    cfg = {'training': {'mode': 'continuous_cpt', 'max_tokens':10},
           'phases':[{'id':1}], 'model':{'revision':'a'*40},
           'datasets':{'turkish':{'classified':{'labels_revision':None}}},
           'paths':{'storage_root':str(tmp_path), **{key:str(tmp_path/key) for key in ('cache_dir','checkpoint_dir','log_dir','manifest_dir')}}}
    with pytest.raises(ValueError,match='Freeze'):
        validate_runtime_config(cfg)
    cfg['datasets']['turkish']['classified']['labels_revision']='b'*40
    validate_runtime_config(cfg)
    cfg['paths']['cache_dir']='/tmp'
    with pytest.raises(ValueError,match='storage_root'):
        validate_runtime_config(cfg)
