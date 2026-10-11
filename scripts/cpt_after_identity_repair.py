"""Release authenticated repairs, prepare the frozen corpus, then enable rental.

This CPU-only coordinator never rents directly. The supervisor validates the
entire 1B training / 2M validation cache before it may request an instance.
"""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

PROJECT = Path('/home/serda/cpt-production')
REPAIR = Path('/home/serda/cpt-identity-repair-20261011')
ROOT = Path('/home/serda/cpt-run-20261011')
PREFIX = 'repairs/identity-20261011'
COLUMNS = ['source_id','row_ordinal','doc_id','document_hash','input_bytes',
           'input_truncated','predicted_label_json','predicted_route']


def release():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    report = json.loads((REPAIR/'COMPLETE.json').read_text())
    if set(report['sources']) != {'bella-akademik','bella-ozenli'}:
        raise ValueError('Repair does not cover both affected sources')
    destination = REPAIR/'release'
    paths = []
    for relative, expected in report['files'].items():
        original = REPAIR/'labels'/relative
        if original.stat().st_size != expected['size'] or hashlib.sha256(original.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('Repair artifact checksum mismatch')
        table = pq.read_table(original)
        target = destination/'labels'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(table.select(COLUMNS),target,compression='zstd')
        provenance = destination/'provenance'/relative
        provenance.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(table.select(['source_id','row_ordinal','doc_id','document_hash',
                       'parent_row_ordinal','parent_doc_id','identity_repair_version']),
                       provenance,compression='zstd')
        paths.append(PREFIX+'/labels/'+relative)
    for sid, stats in report['sources'].items():
        rows = sum(pq.read_metadata(p).num_rows for p in (destination/'labels'/sid).glob('*.parquet'))
        if not rows or rows != stats['recovered_unique_annotations']:
            raise ValueError('Release row counts do not match repair report')
    report['release_files'] = {}
    for p in sorted(destination.rglob('*.parquet')):
        data = p.read_bytes()
        report['release_files'][p.relative_to(destination).as_posix()] = {
            'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'git_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}
    (destination/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    api = HfApi()
    receipt = ROOT/'label_release.json'
    if receipt.exists():
        revision = json.loads(receipt.read_text())['revision']
    else:
        commit = api.upload_folder(repo_id=report['parent_labels_repo'],repo_type='dataset',
            folder_path=destination,path_in_repo=PREFIX,
            commit_message='Add canonical full-text identity repairs with parent provenance')
        revision = commit.oid
    info = api.dataset_info(report['parent_labels_repo'],revision=revision,files_metadata=True)
    files = {p.rfilename:p for p in info.siblings}
    for name, expected in report['release_files'].items():
        remote = files[PREFIX+'/'+name]
        lfs = remote.lfs
        actual = (lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']) if lfs else remote.blob_id
        if remote.size != expected['size'] or actual != expected['sha256' if lfs else 'git_sha1']:
            raise ValueError('Independent remote label verification failed')
    receipt.write_text(json.dumps({'repo':info.id,'revision':revision,'prefix':PREFIX,
        'verified_files':len(report['release_files']),'sources':report['sources']},indent=2))
    return revision,paths


def main():
    import yaml
    ROOT.mkdir(exist_ok=True)
    while not (REPAIR/'COMPLETE.json').exists():
        active = subprocess.run(['systemctl','--user','is-active','linguai-cpt-identity-repair'],
                                capture_output=True,text=True).stdout.strip()
        if active not in {'active','activating'}:
            raise RuntimeError('Identity repair ended without COMPLETE; rental remains disabled')
        print('WAITING_FOR_AUTHENTICATED_REPAIR',flush=True)
        time.sleep(30)
    revision,paths = release()
    config = yaml.safe_load(Path('/home/serda/cpt-run-20261007/run.yaml').read_text())
    classified = config['datasets']['turkish']['classified']
    classified['labels_revision'] = revision
    classified['labels_files'] = [p for p in classified['labels_files']
        if not p.startswith(('data/bella-akademik/','data/bella-ozenli/'))] + paths
    (ROOT/'run.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    if shutil.disk_usage(ROOT).free < 15*1024**3:
        raise RuntimeError('Insufficient free disk; rental remains disabled')
    template = Path('/home/serda/cpt-run-20261007/prepare.sh').read_text()
    (ROOT/'prepare.sh').write_text(template.replace('/home/serda/cpt-run-20261007',str(ROOT)))
    (ROOT/'prepare.sh').chmod(0o700)
    # No early-rental flag; the full cache is a mandatory gate.
    state = ROOT/'supervisor.json'
    if not state.exists():state.write_text(json.dumps({'stage':'WAITING_FOR_CACHE'}))
    subprocess.run(['systemctl','--user','reset-failed','linguai-cpt-prepare'],check=False)
    subprocess.run(['systemd-run','--user','--unit=linguai-cpt-prepare',
        '--property=StandardOutput=append:'+str(ROOT/'prepare.log'),
        '--property=StandardError=append:'+str(ROOT/'prepare.log'),str(ROOT/'prepare.sh')],check=True)
    subprocess.run(['systemctl','--user','start','linguai-cpt-supervisor'],check=True)
    print('PREPARATION_STARTED_RENTAL_GATED_BY_COMPLETE_CACHE',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        ROOT.mkdir(exist_ok=True)
        (ROOT/'coordinator_failure.json').write_text(json.dumps({'error':str(exc),'time':time.time()}))
        raise
