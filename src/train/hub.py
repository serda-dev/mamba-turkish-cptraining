"""Upload complete checkpoints and verify every remote file before cleanup."""
import hashlib
import json
import os
from pathlib import Path


def upload_checkpoint(path, repo_id, api=None):
    from huggingface_hub import HfApi
    api = api or HfApi(token=os.environ.get('HF_TOKEN'))
    path = Path(path)
    api.create_repo(repo_id, private=True, exist_ok=True)
    files = {}
    for file in sorted(path.rglob('*')):
        if not file.is_file() or file.name == 'hub_receipt.json':
            continue
        sha = hashlib.sha256()
        git = hashlib.sha1()
        git.update(f'blob {file.stat().st_size}\0'.encode())
        with file.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b''):
                sha.update(block)
                git.update(block)
        files[file.relative_to(path).as_posix()] = {'size': file.stat().st_size,
                                                  'sha256': sha.hexdigest(), 'git_sha1': git.hexdigest()}
    prefix = 'checkpoints/' + path.name
    commit = api.upload_folder(repo_id=repo_id, folder_path=str(path), path_in_repo=prefix,
                               ignore_patterns=['hub_receipt.json'], commit_message=f'Checkpoint {path.name}')
    remote = {s.rfilename: s for s in api.model_info(repo_id, revision=commit.oid, files_metadata=True).siblings}
    for name, expected in files.items():
        entry = remote.get(prefix + '/' + name)
        if entry is None or entry.size != expected['size']:
            raise ValueError(f'Checkpoint upload size mismatch: {name}')
        lfs = entry.lfs
        actual = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else entry.blob_id
        wanted = expected['sha256'] if lfs else expected['git_sha1']
        if actual != wanted:
            raise ValueError(f'Checkpoint upload hash mismatch: {name}')
    receipt = {'repo_id': repo_id, 'revision': commit.oid, 'path': prefix, 'files': files, 'verified': True}
    (path / 'hub_receipt.json').write_text(json.dumps(receipt, indent=2))
    api.upload_file(repo_id=repo_id, path_or_fileobj=json.dumps(receipt).encode(), path_in_repo='latest_verified.json',
                    commit_message=f'Verified {path.name}')
    return receipt


def prune_remote_checkpoints(repo_id, keep=2):
    """Keep two periodic snapshots plus milestone/stopped/final snapshots."""
    import re
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    directories = sorted({name.split('/')[1] for name in api.list_repo_files(repo_id)
                          if name.startswith('checkpoints/') and len(name.split('/')) > 2
                          and re.fullmatch(r'step_\d{6}', name.split('/')[1])})
    for name in directories[:-keep]:
        api.delete_folder(repo_id=repo_id, path_in_repo='checkpoints/'+name,
                          commit_message=f'Prune verified periodic checkpoint {name}')
