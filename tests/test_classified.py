import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from src.data.classified import content_split, document_id, iter_classified_texts


def fixture_config(tmp_path, records):
    sources, labels = [], []
    for sid, rows in records.items():
        path = tmp_path / f'{sid}.jsonl'
        source = {'id': sid, 'repo': f'owner/{sid}', 'revision': 'a' * 40,
                  'config': 'default', 'split': 'train', 'text_field': 'text',
                  'license': 'cc-by-4.0', 'local_path': str(path)}
        sources.append(source)
        path.write_text(''.join(json.dumps({'text': text}) + '\n' for text, route in rows))
        for ordinal, (text, route) in enumerate(rows):
            labels.append({'source_id': sid, 'row_ordinal': ordinal,
                           'doc_id': document_id(source, ordinal, text),
                           'document_hash': hashlib.sha256(text.encode()).hexdigest(),
                           'input_bytes': len(text.encode()), 'input_truncated': False,
                           'predicted_label_json': '{"q":{"utility":3}}', 'predicted_route': route})
    label_path = tmp_path / 'labels.jsonl'
    label_path.write_text(''.join(json.dumps(row) + '\n' for row in reversed(labels)))
    settings = {'source_plan': {'sources': sources}, 'labels_local_path': str(label_path),
                'audit_db': str(tmp_path / 'audit.sqlite'), 'validation_fraction': .3}
    return {'datasets': {'turkish': {'classified': settings}}}, settings, labels


def rewrite_labels(settings, labels):
    from pathlib import Path
    Path(settings['labels_local_path']).write_text(''.join(json.dumps(row) + '\n' for row in labels))


def test_join_out_of_order_labels_global_dedup_and_all_routes_audited(tmp_path):
    config, settings, labels = fixture_config(tmp_path, {
        'one': [('merhaba', 'PREMIUM'), ('onar', 'MODEL_REPAIR'), ('tekrar', 'KEEP')],
        'two': [('merhaba', 'KEEP'), ('incele', 'REVIEW'), ('sil', 'DROP'),
                ('düzelt', 'DETERMINISTIC_REPAIR')]})
    assert list(iter_classified_texts(config, 'all')) == ['merhaba', 'tekrar']
    with sqlite3.connect(settings['audit_db']) as db:
        assert db.execute('SELECT count(*) FROM labels').fetchone()[0] == len(labels)
        assert db.execute('SELECT count(*) FROM decisions').fetchone()[0] == len(labels)
        assert db.execute("SELECT count(*) FROM decisions WHERE decision='duplicate'").fetchone()[0] == 1
        assert db.execute('SELECT count(DISTINCT route) FROM decisions').fetchone()[0] == 6
        assert db.execute("SELECT payload FROM labels WHERE source_id='one' AND ordinal=1").fetchone()[0]
    manifest = json.loads((tmp_path / 'audit.manifest.json').read_text())
    assert manifest['complete'] and manifest['verified_rows'] == len(labels)
    # A second reconstruction pass cannot lose accepted rows to the previous pass's dedup index.
    assert list(iter_classified_texts(config, 'all')) == ['merhaba', 'tekrar']


def test_content_holdout_is_disjoint_and_reproducible(tmp_path):
    texts = [f'Türkçe belge {i}' for i in range(100)]
    config, _, _ = fixture_config(tmp_path, {'one': [(t, 'KEEP') for t in texts]})
    train = set(iter_classified_texts(config, 'train'))
    validation = set(iter_classified_texts(config, 'validation'))
    assert train and validation and not train & validation
    assert train | validation == set(texts)
    digest = hashlib.sha256(texts[0].encode()).hexdigest()
    assert (texts[0] in validation) == (content_split(digest, .3, 42) == 'validation')


@pytest.mark.parametrize('field,value', [('document_hash', '0'*64), ('doc_id', 'wrong'), ('input_bytes', 1)])
def test_identity_mismatch_fails_and_manifest_is_incomplete(tmp_path, field, value):
    config, settings, labels = fixture_config(tmp_path, {'one': [('Türkçe', 'KEEP')]})
    labels[0][field] = value
    rewrite_labels(settings, labels)
    with pytest.raises(ValueError, match='mismatch'):
        list(iter_classified_texts(config))
    manifest = json.loads((tmp_path / 'audit.manifest.json').read_text())
    assert not manifest['complete'] and manifest['error']


def test_missing_raw_row_fails(tmp_path):
    config, settings, labels = fixture_config(tmp_path, {'one': [('a', 'KEEP'), ('b', 'DROP')]})
    (tmp_path / 'one.jsonl').write_text('{"text":"a"}\n')
    with pytest.raises(ValueError, match='Source ended'):
        list(iter_classified_texts(config, 'all'))


def test_cosmos_and_truncated_inputs_are_audited_but_excluded(tmp_path):
    config, settings, labels = fixture_config(tmp_path, {'cosmos': [('cosmos', 'KEEP')], 'one': [('long', 'PREMIUM')]})
    labels[1]['input_truncated'] = True
    rewrite_labels(settings, labels)
    assert list(iter_classified_texts(config, 'all')) == []
    with sqlite3.connect(settings['audit_db']) as db:
        assert {r[0] for r in db.execute('SELECT decision FROM decisions')} == {'cosmos_review_gate', 'truncated_input_review'}
    settings['allow_cosmos'] = True
    with pytest.raises(ValueError, match='cosmos_review_evidence'):
        list(iter_classified_texts(config))


def test_early_close_cannot_claim_complete_manifest(tmp_path):
    config, _, _ = fixture_config(tmp_path, {'one': [('a', 'KEEP'), ('b', 'KEEP')]})
    stream = iter_classified_texts(config, 'all')
    assert next(stream) == 'a'
    stream.close()
    manifest = json.loads((tmp_path / 'audit.manifest.json').read_text())
    assert not manifest['complete'] and manifest['verified_rows'] == 1


def test_repair_cannot_be_enabled_as_clean_route(tmp_path):
    config, settings, _ = fixture_config(tmp_path, {'one': [('a', 'KEEP')]})
    settings['accepted_routes'] = ['MODEL_REPAIR']
    with pytest.raises(ValueError, match='Only PREMIUM/KEEP'):
        list(iter_classified_texts(config))


def test_duplicate_labels_fail_instead_of_being_lost(tmp_path):
    config, settings, labels = fixture_config(tmp_path, {'one': [('a', 'KEEP')]})
    rewrite_labels(settings, labels * 2)
    with pytest.raises(ValueError, match='Duplicate label'):
        list(iter_classified_texts(config))


def test_changed_policy_requires_new_database(tmp_path):
    config, settings, _ = fixture_config(tmp_path, {'one': [('a', 'KEEP')]})
    list(iter_classified_texts(config, 'all'))
    settings['split_seed'] = 43
    with pytest.raises(ValueError, match='different labels/source plan/policy'):
        list(iter_classified_texts(config))


def test_sources_are_interleaved_before_first_source_exhausts(tmp_path):
    config, settings, _ = fixture_config(tmp_path, {
        'one': [(f'a{i}', 'KEEP') for i in range(20)],
        'two': [(f'b{i}', 'KEEP') for i in range(20)]})
    stream = iter_classified_texts(config, 'all')
    assert [next(stream) for _ in range(4)] == ['a0', 'b0', 'a1', 'b1']
    stream.close()


def test_cosmos_explicit_review_evidence_unlocks_only_clean_inputs(tmp_path):
    config, settings, _ = fixture_config(tmp_path, {'cosmos': [('x', 'KEEP'), ('y', 'MODEL_REPAIR')]})
    settings.update(allow_cosmos=True, cosmos_review_evidence='review-report.json')
    assert list(iter_classified_texts(config, 'all')) == ['x']


@pytest.mark.parametrize('vector', json.loads(
    (Path(__file__).parent / 'fixtures' / 'upstream_identity_v1.json').read_text(encoding='utf-8')
)['vectors'])
def test_identity_matches_independent_pinned_upstream_vectors(vector):
    # Expected values came from upstream contracts.py, not this module's helper.
    from src.data.classified import document_hash
    assert document_hash(vector['text']) == vector['document_hash']
    assert document_id(vector['source'], vector['ordinal'], vector['text']) == vector['doc_id']
