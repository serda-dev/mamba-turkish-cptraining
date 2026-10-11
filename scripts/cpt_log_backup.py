"""Capture CPT logs on VPS and verify a versioned archive on the private Hub."""
import base64
import hashlib
import json
import shlex
import tarfile
import time
from pathlib import Path


def backup_logs(state,root):
    import fcntl
    root=Path(root)
    (root/'log-backup').mkdir(parents=True,exist_ok=True)
    with (root/'log-backup/backup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _backup_logs(state,root)


def _backup_logs(state,root):
    from scripts.cpt_vps_supervisor import ssh
    from huggingface_hub import HfApi
    import yaml
    root=Path(root)
    destination=root/'log-backup/files'
    destination.mkdir(parents=True,exist_ok=True)
    code='''import base64,hashlib,json
from pathlib import Path
root=Path('/mnt/home_extra')
patterns=['training.log','training.before-batch-opt.log','prefetch.log',
 'run_status.json','run.yaml','run.before-batch-opt.yaml','eval_*.json',
 'log/*.log','log/*.jsonl','log/config_*.json','checkpoint/latest.json',
 'batch-benchmark.log','batch-benchmark/results.json',
 'batch-benchmark/*/run.log','batch-benchmark/*/memory.json']
paths=sorted({p for pattern in patterns for p in root.glob(pattern)})
assert (root/'training.log') in paths and (root/'run_status.json') in paths
rows=[]
total=0
for p in paths:
 if not p.is_file() or p.is_symlink():continue
 data=p.read_bytes();total+=len(data)
 assert len(data)<=64*1024**2 and total<=128*1024**2,'Log snapshot size limit exceeded'
 rows.append({'path':p.relative_to(root).as_posix(),'size':len(data),
 'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()})
print(json.dumps(rows))
'''
    rows=json.loads(ssh(state,'python -c '+shlex.quote(code),timeout=120))
    for row in rows:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe log path')
        data=base64.b64decode(row['data'],validate=True)
        if len(data)!=row['size'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Log transfer checksum mismatch')
        path=destination/'gpu'/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.tmp');temp.write_bytes(data);temp.replace(path)
    local=destination/'vps';local.mkdir(exist_ok=True)
    for name in ['supervisor.log','prepare.log','coordinator.log','run.yaml',
                 'run.before-batch-opt.yaml','label_release.json','batch-benchmark.json',
                 'recovered_eval_baseline.json','current_eval_100m.json','remote_training.log']:
        source=root/name
        if source.is_file():
            (local/name).write_bytes(source.read_bytes())
    repair=Path('/home/serda/cpt-identity-repair-20261011')
    for name in ['run.log','COMPLETE.json']:
        if (repair/name).is_file():(local/('identity-repair-'+name)).write_bytes((repair/name).read_bytes())
    manifest={'version':1,'instance_id':state['instance_id'],'captured_at':time.time(),
        'files':{p.relative_to(destination).as_posix():{'size':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=root/'log-backup'/f'{root.name}.tar.gz'
    temporary=archive.with_suffix('.tmp')
    with tarfile.open(temporary,'w:gz') as tar:
        for p in sorted(destination.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(destination).as_posix(),recursive=False)
    temporary.replace(archive)
    repo=yaml.safe_load((root/'run.yaml').read_text())['checkpointing']['hub_repo']
    remote_path='run/log-archives/'+archive.name
    api=HfApi()
    commit=api.upload_file(repo_id=repo,path_or_fileobj=str(archive),path_in_repo=remote_path,
        commit_message=f'Archive CPT logs from instance {state["instance_id"]}')
    info=api.model_info(repo,revision=commit.oid,files_metadata=True)
    remote=next(f for f in info.siblings if f.rfilename==remote_path)
    data=archive.read_bytes();sha=hashlib.sha256(data).hexdigest()
    git_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    lfs=remote.lfs
    actual=(lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']) if lfs else remote.blob_id
    if remote.size!=len(data) or actual!=(sha if lfs else git_sha):
        raise ValueError('Independent Hub log archive verification failed')
    receipt={'verified':True,'repo':repo,'revision':commit.oid,'path':remote_path,
        'size':len(data),'sha256':sha,'file_count':len(manifest['files']),
        'captured_at':manifest['captured_at'],'verified_at':time.time()}
    temporary=root/'log-backup/receipt.tmp';temporary.write_text(json.dumps(receipt,indent=2))
    temporary.replace(root/'log-backup/receipt.json')
    return receipt


def main():
    from scripts.cpt_vps_supervisor import ROOT,STATE
    state=json.loads(STATE.read_text())
    if not state.get('instance_id') or state.get('stage') in {'COMPLETE','STOPPED_EXPORTED','FAILED_NO_CHECKPOINT'}:
        print('No active instance; existing archives retained',flush=True)
        return
    print(json.dumps(backup_logs(state,ROOT)),flush=True)


if __name__=='__main__':main()
