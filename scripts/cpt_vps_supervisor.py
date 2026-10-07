"""Prepare first, then rent one A100 and supervise costs and durable export.

Run as a persistent user systemd service. The state file prevents duplicate rents.
Only this run's labelled instance can be stopped/destroyed. No API key leaves VPS.
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path('/home/serda/cpt-run-20261007')
PROJECT = Path('/home/serda/cpt-production')
STATE = ROOT/'supervisor.json'
LABEL = 'linguai-cpt-1b-20261007'
IMAGE = 'serdadev/jamba2-cpt@sha256:5762cb5fffab046aca0c6c3a7a32fbc1733a8f51352b54244f738f0b6f4eb8e4'
KEY = Path('/home/serda/.config/vastai/linguai_team_api_key')
SSH_KEY = '/home/serda/.ssh/id_ed25519'


def save(state, stage=None, **fields):
    if stage:
        state['stage'] = stage
    state.update(fields, updated_at=time.time())
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATE)
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), state['stage'], flush=True)


def vast(*args):
    # Capture stderr: never echo the CLI command or its private key.
    process = subprocess.run(['vastai','--api-key',KEY.read_text().strip(),*map(str,args),'--raw'],
                              capture_output=True, text=True, timeout=45)
    if process.returncode:
        raise RuntimeError('Vast CLI failed: ' + ' '.join(map(str,args[:2])))
    return json.loads(process.stdout)


def connection(state):
    return ['-i',SSH_KEY,'-p',str(state['ssh_port']),'-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','StrictHostKeyChecking=accept-new','-o',f'UserKnownHostsFile={ROOT}/known_hosts']


def ssh(state, command, data=None, timeout=60):
    return subprocess.run(['ssh',*connection(state),'root@'+state['ssh_host'],command],
                          input=data, capture_output=True, timeout=timeout, check=True).stdout.decode()


def copy_to(state, path, target, recursive=False):
    options = connection(state)
    options[2] = '-P'
    subprocess.run(['scp',*(['-r'] if recursive else []),*options,str(path),
                    f'root@{state["ssh_host"]}:{target}'], capture_output=True, check=True, timeout=1800)


def validate_cache():
    for split, tokens in [('phase_1',1000000000),('phase_1_validation',2000000)]:
        directory = ROOT/'cache/token_cache'/split
        manifest = json.loads((directory/'manifest.json').read_text())
        if not manifest['complete'] or manifest['total_tokens'] != tokens:
            raise ValueError(f'{split} cache is incomplete or below target')
        if abs(manifest['source_token_counts']['turkish']/tokens - .8) > .005:
            raise ValueError('Cache Turkish/English ratio differs from plan')
        for shard in manifest['shards']:
            for field, expected in shard['sha256'].items():
                digest = hashlib.sha256()
                with (directory/shard[field]).open('rb') as handle:
                    for block in iter(lambda: handle.read(8*1024**2), b''):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ValueError('Cache checksum mismatch')
    audit = json.loads((ROOT/'manifests/corpus.json').read_text())
    counts = audit['yielded_source_tokens']
    weights = audit['source_weights']
    if audit['error'] or any(abs(counts[k]/sum(counts.values())-w) > .005 for k,w in weights.items()):
        raise ValueError('Turkish source token quotas differ from plan')
    # Audit incomplete is expected: the requested token cap closes raw streams early.
    return {'train_tokens':1000000000,'validation_tokens':2000000,'source_tokens':counts}


def offer():
    query = 'gpu_name=A100_SXM4 gpu_ram>=79 num_gpus=1 rentable=true rented=false reliability>0.98 verified=true'
    offers = vast('search','offers',query,'--storage',180,'--limit',20,'--order','dph_total')
    candidates = [o for o in offers if o['dph_total'] <= 1.25 and o['end_date'] > time.time()+72*3600
                  and o.get('inet_up_cost',1) <= .01 and o.get('inet_down_cost',1) <= .01]
    return min(candidates,key=lambda o:o['dph_total']) if candidates else None


def make_archive():
    tracked = subprocess.check_output(['git','-C',str(PROJECT),'ls-files','-z']).decode().split('\0')
    additions = ['src/train/hub.py','scripts/cpt_evaluate.py','scripts/cpt_remote_run.py',
                 'scripts/cpt_vps_supervisor.py','configs/cpt_1b_20261007.yaml','configs/cpt_1b_sources.json']
    with tarfile.open(ROOT/'code.tar.gz','w:gz') as archive:
        for relative in sorted(set(tracked+additions)-{''}):
            if (PROJECT/relative).is_file():
                archive.add(PROJECT/relative,arcname=relative,recursive=False)


def recover_and_verify(state):
    # Check every completed local checkpoint; incomplete temporary directories are ignored.
    code = '''import json
from pathlib import Path
from src.train.hub import upload_checkpoint
root=Path('/mnt/home_extra')
completed=[]
for path in (root/'checkpoint').rglob('checkpoint_metadata.json'):
 meta=json.loads(path.read_text())
 completed.append((meta['tokens_seen'],path.parent))
if completed:
 _,path=max(completed,key=lambda item:item[0])
 receipt_path=path/'hub_receipt.json'
 if not receipt_path.exists():
  upload_checkpoint(path,'serda-dev/Jamba2-3B-Turkish-CPT-1B-20261007')
 receipt=json.loads(receipt_path.read_text())
 receipt['checkpoint_metadata']=json.loads((path/'checkpoint_metadata.json').read_text())
 print(json.dumps(receipt))
else:
 print(json.dumps({'no_checkpoint':True}))
'''
    command = 'cd /workspace/mamba-cpt-tr && HF_TOKEN="$(cat /mnt/home_extra/hf_token)" python -c '+shlex.quote(code)
    receipt = json.loads(ssh(state,command,timeout=3600))
    if receipt.get('no_checkpoint'):
        return receipt
    from huggingface_hub import HfApi
    info = HfApi().model_info(receipt['repo_id'],revision=receipt['revision'],files_metadata=True)
    files = {f.rfilename:f for f in info.siblings}
    for name, expected in receipt['files'].items():
        actual = files[receipt['path']+'/'+name]
        lfs = actual.lfs
        remote_hash = (lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256']) if lfs else actual.blob_id
        if actual.size != expected['size'] or remote_hash != expected['sha256' if lfs else 'git_sha1']:
            raise ValueError('Independent Hub checkpoint verification failed')
    (ROOT/'final_hub_receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def main():
    ROOT.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {'stage':'WAITING_FOR_CACHE'}
    save(state)
    while True:
        try:
            if state['stage'] in {'COMPLETE','STOPPED_EXPORTED','PREPARATION_FAILED','PAUSED_RECOVERY_REQUIRED','RENT_AMBIGUOUS'}:
                return
            if not state.get('instance_id'):
                if shutil.disk_usage(ROOT).free < 15*1024**3:
                    subprocess.run(['docker','stop','-t','30','linguai-cpt-1b-prepare'],capture_output=True)
                    save(state,'PREPARATION_FAILED',error='VPS free disk below 15 GiB')
                    return
                prep = (ROOT/'preparation.status').read_text().strip() if (ROOT/'preparation.status').exists() else ''
                if prep != 'CACHE_READY':
                    active = subprocess.run(['systemctl','--user','is-active','linguai-cpt-prepare'],capture_output=True,text=True)
                    if active.stdout.strip() not in {'active','activating'}:
                        save(state,'PREPARATION_FAILED',error='Preparation service stopped before CACHE_READY')
                        return
                    time.sleep(60)
                    continue
                if not state.get('cache_validation'):
                    save(state,'CHECKING_FROZEN_CACHE',cache_validation=validate_cache())
                existing = [i for i in vast('show','instances') if i.get('label') == LABEL]
                if existing:
                    if len(existing) != 1:
                        save(state,'RENT_AMBIGUOUS',error='Multiple instances with run label')
                        return
                    save(state,'RENTED',instance_id=existing[0]['id'])
                elif state.get('rent_requested'):
                    save(state,'RENT_AMBIGUOUS',error='Prior rent request has no recoverable instance; no duplicate rental attempted')
                    return
                else:
                    selected = offer()
                    if not selected:
                        save(state,'WAITING_FOR_A100_PRICE')
                        time.sleep(600)
                        continue
                    user = vast('show','user')
                    if user.get('id') != 733063 or not user.get('is_team') or user.get('credit',0) < 90:
                        save(state,'WAITING_FOR_TEAM_CREDIT',credit=user.get('credit'))
                        time.sleep(600)
                        continue
                    save(state,'RENT_REQUESTED',rent_requested=True,offer=selected,
                         started_at=time.time(),initial_credit=user['credit'],hourly_rate=selected['dph_total'])
                    rented = vast('create','instance',selected['id'],'--disk',180,'--image',IMAGE,'--ssh','--direct',
                                  '--label',LABEL,'--env','-e AUTO_INSTALL=0 -e AUTO_TRAIN=0 -e HF_HOME=/mnt/home_extra/hf-cache',
                                  '--onstart-cmd','bash /workspace/mamba-cpt-tr/scripts/vast_onstart.sh','--cancel-unavail')
                    if not rented.get('success') or not rented.get('new_contract'):
                        raise RuntimeError('Rental request did not return a contract')
                    save(state,'RENTED',instance_id=rented['new_contract'])
                    vast('attach','ssh',state['instance_id'],Path(SSH_KEY+'.pub').read_text().strip())
                    save(state,ssh_key_attached=True)
            if not state.get('ssh_key_attached'):
                vast('attach','ssh',state['instance_id'],Path(SSH_KEY+'.pub').read_text().strip())
                save(state,ssh_key_attached=True)
            instances = vast('show','instances')
            instance = next((i for i in instances if i['id'] == state['instance_id']),None)
            if instance is None:
                save(state,'PAUSED_RECOVERY_REQUIRED',error='Run instance disappeared')
                return
            user = vast('show','user')
            # Balance catches bandwidth and any parallel team spending conservatively.
            spent = max(state['initial_credit']-user['credit'],
                        (time.time()-state['started_at'])/3600*state['hourly_rate'])
            if spent >= 90 or user['credit'] <= 7:
                vast('stop','instance',state['instance_id'])
                save(state,'PAUSED_RECOVERY_REQUIRED',spent_estimate=spent,
                     error='Budget boundary: GPU stopped; disk retained for checkpoint recovery')
                return
            if instance.get('actual_status') != 'running':
                if time.time()-state['started_at'] > 1800:
                    vast('stop','instance',state['instance_id'])
                    save(state,'PAUSED_RECOVERY_REQUIRED',error='Instance failed to start within 30 minutes')
                    return
                time.sleep(60)
                continue
            state['ssh_host'],state['ssh_port'] = instance['ssh_host'],instance['ssh_port']
            if not state.get('remote_started'):
                ssh(state,'mkdir -p /mnt/home_extra /workspace/mamba-cpt-tr')
                make_archive()
                save(state,code_archive_sha256=hashlib.sha256((ROOT/'code.tar.gz').read_bytes()).hexdigest())
                copy_to(state,ROOT/'code.tar.gz','/mnt/home_extra/code.tar.gz')
                ssh(state,'tar -xzf /mnt/home_extra/code.tar.gz -C /workspace/mamba-cpt-tr')
                copy_to(state,ROOT/'cache','/mnt/home_extra/',recursive=True)
                copy_to(state,ROOT/'manifest','/mnt/home_extra/',recursive=True)
                copy_to(state,ROOT/'run.yaml','/mnt/home_extra/run.yaml')
                ssh(state,'umask 077; cat > /mnt/home_extra/hf_token',
                    data=Path('/home/serda/.cache/huggingface/token').read_bytes())
                save(state,'LAUNCHING_REMOTE',remote_launch_requested=True)
                # A restart during launch checks for a live process before creating another.
                launch = 'cd /workspace/mamba-cpt-tr; if ! pgrep -f "^python -u -m scripts.cpt_remote_run$" >/dev/null; then nohup env HF_HOME=/mnt/home_extra/hf-cache HUGGINGFACE_HUB_CACHE=/mnt/home_extra/hf-cache/hub TRANSFORMERS_CACHE=/mnt/home_extra/hf-cache/transformers python -u -m scripts.cpt_remote_run > /mnt/home_extra/training.log 2>&1 < /dev/null & echo $! > /mnt/home_extra/job.pid; fi'
                ssh(state,launch)
                save(state,'RUNNING',remote_started=True)
            if spent >= 86 or user['credit'] <= 11:
                ssh(state,'touch /mnt/home_extra/STOP')
                save(state,'STOP_REQUESTED',spent_estimate=spent)
            probe = "python -c \"import json,os,time; from pathlib import Path; r=Path('/mnt/home_extra'); s=json.loads((r/'run_status.json').read_text()) if (r/'run_status.json').exists() else {}; s['alive']=Path('/proc/'+(r/'job.pid').read_text().strip()).exists(); s['log_age']=time.time()-(r/'training.log').stat().st_mtime; print(json.dumps(s))\""
            remote = json.loads(ssh(state,probe))
            state['remote_status'] = remote
            save(state,spent_estimate=spent,credit=user['credit'])
            if remote['log_age'] > 3600 and remote['alive']:
                # First recheck occurred in earlier 10-minute polling passes; request safe save.
                ssh(state,'touch /mnt/home_extra/STOP')
            if not remote['alive']:
                # Export and verify before destroying; exceptions retain the instance until recovery/budget stop.
                receipt = recover_and_verify(state)
                if remote.get('state') == 'DONE' and (receipt.get('no_checkpoint') or
                        not receipt['checkpoint_metadata']['phase_completed'] or
                        receipt['checkpoint_metadata']['tokens_seen'] != 1000000000):
                    raise ValueError('DONE state does not match final verified checkpoint')
                for path in ['run_status.json','training.log','eval_baseline.json','eval_100m.json','eval_final.json']:
                    try:
                        content = ssh(state,'cat '+shlex.quote('/mnt/home_extra/'+path),timeout=90)
                        (ROOT/('remote_'+path)).write_text(content)
                    except subprocess.CalledProcessError:
                        pass
                vast('destroy','instance',state['instance_id'],'-y')
                save(state,'COMPLETE' if remote.get('state') == 'DONE' else 'STOPPED_EXPORTED',hub_receipt=receipt)
                return
            time.sleep(600)
        except Exception as exc:
            # Do not log command arguments which might contain credentials.
            save(state,error=type(exc).__name__)
            time.sleep(60)


if __name__ == '__main__':
    main()
