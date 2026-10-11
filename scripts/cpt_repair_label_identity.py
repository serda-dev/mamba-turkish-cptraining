"""Recover canonical source coordinates by exact full-text hash, with provenance.

Parent annotations are immutable. Conflicting predictions for the same content
are excluded. Every recovered annotation must authenticate its old document ID
against the complete pinned raw text before receiving a canonical ID.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from src.data.classified import document_hash, document_id, _iter_source
from src.data.streaming import closing_iterator


def signature(row):
    label = json.loads(row['predicted_label_json'])
    if label.pop('id') != row['doc_id']:
        raise ValueError('Parent label id mismatch')
    return json.dumps({'label':label,'route':row['predicted_route'],
        'input_bytes':row['input_bytes'],'truncated':row['input_truncated']},sort_keys=True)


def recover_row(source, ordinal, text, parent):
    digest = document_hash(text)
    if parent['source_id'] != source['id'] or digest != parent['document_hash']:
        raise ValueError('Parent content hash mismatch')
    if parent['input_bytes'] != len(text.encode('utf-8')):
        raise ValueError('Parent full-text byte count mismatch')
    if document_id(source,parent['row_ordinal'],text) != parent['doc_id']:
        raise ValueError('Parent document ID does not authenticate this full text')
    signature(parent)
    row = dict(parent)
    row['parent_row_ordinal'],row['parent_doc_id'] = parent['row_ordinal'],parent['doc_id']
    row['identity_repair_version'] = 'canonical-source-sha256-v1'
    row['row_ordinal'] = ordinal
    row['doc_id'] = document_id(source,ordinal,text)
    label = json.loads(row['predicted_label_json'])
    label['id'] = row['doc_id']
    row['predicted_label_json'] = json.dumps(label,ensure_ascii=False,sort_keys=True)
    return row


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-audit',required=True)
    parser.add_argument('--source-plan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        raise ValueError('Repair already completed; do not overwrite an audited artifact')
    parent=sqlite3.connect('file:'+args.parent_audit+'?mode=ro',uri=True)
    db=sqlite3.connect(out/'hash-index.sqlite')
    db.execute('PRAGMA cache_size=-32768')
    db.executescript('CREATE TABLE IF NOT EXISTS hashes(hash TEXT PRIMARY KEY,payload TEXT,signature TEXT,conflict INTEGER DEFAULT 0,matched INTEGER DEFAULT 0) WITHOUT ROWID;')
    report={'version':'canonical-source-sha256-v1','parent_labels_repo':'serda-dev/filtered-turkish-dataset',
            'parent_labels_revision':'388300940d3675640553e817e2b6eae1b3b6947b',
            'dedup_policy':'first canonical occurrence per source/full UTF8 SHA256',
            'conflict_policy':'exclude any differing score/type/noise/route/truncation/byte count', 'sources':{}}
    for source in json.loads(Path(args.source_plan).read_text())['sources']:
        sid=source['id']
        if sid not in {'bella-akademik','bella-ozenli'}:continue
        db.execute('DELETE FROM hashes'); db.commit()
        count=0
        for payload, in parent.execute('SELECT payload FROM labels WHERE source_id=? ORDER BY ordinal',(sid,)):
            row=json.loads(payload); sig=signature(row)
            db.execute('INSERT INTO hashes(hash,payload,signature) VALUES(?,?,?) ON CONFLICT(hash) DO UPDATE SET conflict=CASE WHEN hashes.signature!=excluded.signature THEN 1 ELSE hashes.conflict END',
                       (row['document_hash'],payload,sig))
            count+=1
            if count%10000==0:db.commit()
            if count%100000==0:print(f'REPAIR_INDEX {sid} {count}',flush=True)
        db.commit()
        folder=out/'labels'/sid; folder.mkdir(parents=True,exist_ok=True)
        # A failed prior pass is retried in a fresh output directory.
        if list(folder.glob('*.parquet')):raise ValueError('Partial artifacts exist; use a fresh output directory')
        pending=[]; accepted=scanned=unlabeled=conflicted=duplicates=0; part=0
        def flush():
            nonlocal part
            if not pending:return
            path=folder/f'part-{part:06d}.parquet'
            pq.write_table(pa.Table.from_pylist(pending),path.with_suffix('.tmp'),compression='zstd')
            path.with_suffix('.tmp').replace(path)
            part+=1;pending.clear()
        with closing_iterator(_iter_source(source)) as stream:
            for ordinal,raw in enumerate(stream):
                scanned+=1
                text=raw.get(source['text_field'])
                if not isinstance(text,str) or not text.strip():continue
                digest=document_hash(text)
                found=db.execute('SELECT payload,conflict,matched FROM hashes WHERE hash=?',(digest,)).fetchone()
                if found is None:unlabeled+=1
                elif found[1]:conflicted+=1
                elif found[2]:duplicates+=1
                else:
                    recovered=recover_row(source,ordinal,text,json.loads(found[0]))
                    pending.append(recovered);accepted+=1
                    db.execute('UPDATE hashes SET matched=1 WHERE hash=?',(digest,))
                    if len(pending)>=10000:flush();db.commit()
                if scanned%10000==0:
                    print(f'REPAIR_JOIN {sid} scanned={scanned} recovered={accepted}',flush=True)
        flush();db.commit()
        report['sources'][sid]={'source':source,'parent_annotations':count,'scanned_raw_rows':scanned,
            'recovered_unique_annotations':accepted,'unlabeled_raw_rows':unlabeled,
            'conflicting_raw_rows':conflicted,'duplicate_raw_rows':duplicates,
            'hashes_without_canonical_match':db.execute('SELECT count(*) FROM hashes WHERE matched=0 AND conflict=0').fetchone()[0],
            'ambiguous_parent_hashes':db.execute('SELECT count(*) FROM hashes WHERE conflict=1').fetchone()[0]}
        (out/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(f'REPAIR_SOURCE_DONE {sid} {accepted}',flush=True)
    report['files']={p.relative_to(out/'labels').as_posix():{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in sorted((out/'labels').rglob('*.parquet'))}
    (out/'COMPLETE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    db.close();parent.close()


if __name__=='__main__':main()
