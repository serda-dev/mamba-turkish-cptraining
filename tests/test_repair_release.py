import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import cpt_after_identity_repair as coordinator


def test_release_retains_loader_schema_and_separate_parent_provenance(tmp_path,monkeypatch):
    import hashlib
    import huggingface_hub
    monkeypatch.setattr(coordinator,'REPAIR',tmp_path/'repair')
    monkeypatch.setattr(coordinator,'ROOT',tmp_path/'run')
    coordinator.ROOT.mkdir()
    report={'sources':{},'files':{},'parent_labels_repo':'owner/labels'}
    row={key:'value' for key in coordinator.COLUMNS}
    row.update(row_ordinal=7,input_bytes=30,input_truncated=False,
               parent_row_ordinal=100000,parent_doc_id='old',identity_repair_version='v1')
    for sid in ['bella-akademik','bella-ozenli']:
        row['source_id']=sid
        path=coordinator.REPAIR/'labels'/sid/'part-000000.parquet'
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row]),path)
        report['sources'][sid]={'recovered_unique_annotations':1}
        report['files'][sid+'/'+path.name]={'size':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (coordinator.REPAIR/'COMPLETE.json').write_text(json.dumps(report))

    class Api:
        def upload_folder(self,**kwargs):
            assert kwargs['repo_type']=='dataset'
            self.folder=kwargs['folder_path']
            return SimpleNamespace(oid='pinned-revision')
        def dataset_info(self,repo,**kwargs):
            assert kwargs['revision']=='pinned-revision'
            files=[]
            for path in self.folder.rglob('*.parquet'):
                data=path.read_bytes()
                files.append(SimpleNamespace(rfilename=coordinator.PREFIX+'/'+path.relative_to(self.folder).as_posix(),
                    size=len(data),lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())))
            return SimpleNamespace(id=repo,siblings=files)
    monkeypatch.setattr(huggingface_hub,'HfApi',Api)
    revision,paths=coordinator.release()
    assert revision=='pinned-revision' and len(paths)==2
    for path in (coordinator.REPAIR/'release'/'labels').rglob('*.parquet'):
        assert pq.read_schema(path).names==coordinator.COLUMNS
    for path in (coordinator.REPAIR/'release'/'provenance').rglob('*.parquet'):
        assert pq.read_table(path).to_pylist()[0]['parent_doc_id']=='old'
    assert json.loads((coordinator.ROOT/'label_release.json').read_text())['verified_files']==4


def test_repair_failure_cannot_start_cache_or_rental(tmp_path,monkeypatch):
    monkeypatch.setattr(coordinator,'REPAIR',tmp_path/'missing')
    monkeypatch.setattr(coordinator,'ROOT',tmp_path/'run')
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='failed\n')
    monkeypatch.setattr(coordinator.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='rental remains disabled'):
        coordinator.main()
    assert len(calls)==1 and calls[0][2]=='is-active'
