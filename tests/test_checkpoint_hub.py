import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.train.hub import upload_checkpoint


class FakeHub:
    def __init__(self, bad=False):
        self.bad = bad
        self.files = []
        self.pointer_written = False
    def create_repo(self, *args, **kwargs):
        assert kwargs['private'] is True
    def upload_folder(self, **kwargs):
        for path in Path(kwargs['folder_path']).rglob('*'):
            if path.is_file():
                data = path.read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                self.files.append(SimpleNamespace(rfilename=kwargs['path_in_repo']+'/'+path.name,
                       size=len(data), lfs=SimpleNamespace(sha256='bad' if self.bad else sha)))
        return SimpleNamespace(oid='a'*40)
    def model_info(self, *args, **kwargs):
        return SimpleNamespace(siblings=self.files)
    def upload_file(self, **kwargs):
        self.pointer_written = True


def test_verified_checkpoint_only_updates_pointer_after_all_hashes_match(tmp_path):
    (tmp_path/'model.safetensors').write_bytes(b'weights')
    (tmp_path/'training_state.pt').write_bytes(b'optimizer and exact cursor')
    api = FakeHub()
    receipt = upload_checkpoint(tmp_path, 'owner/run', api=api)
    assert api.pointer_written and receipt['verified']
    assert len(receipt['files']) == 2


def test_bad_remote_hash_keeps_local_checkpoint_without_verified_pointer(tmp_path):
    (tmp_path/'training_state.pt').write_bytes(b'cursor')
    api = FakeHub(bad=True)
    with pytest.raises(ValueError, match='hash mismatch'):
        upload_checkpoint(tmp_path, 'owner/run', api=api)
    assert not api.pointer_written and not (tmp_path/'hub_receipt.json').exists()
    assert (tmp_path/'training_state.pt').is_file()
