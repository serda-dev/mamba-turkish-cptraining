import json
import time
import pytest
from scripts import cpt_vps_supervisor as supervisor


def setup_run(tmp_path, monkeypatch, credit=95):
    state = {'stage':'RUNNING','instance_id':123,'remote_started':True,'ssh_key_attached':True,
             'initial_credit':97,'started_at':time.time(),'hourly_rate':1.23}
    monkeypatch.setattr(supervisor,'ROOT',tmp_path)
    monkeypatch.setattr(supervisor,'STATE',tmp_path/'state.json')
    supervisor.STATE.write_text(json.dumps(state))
    events = []
    def vast(*args):
        events.append(args[:2])
        if args[:2] == ('show','instances'):
            return [{'id':123,'actual_status':'running','ssh_host':'host','ssh_port':22}]
        if args[:2] == ('show','user'):
            return {'credit':credit}
        return {'success':True}
    monkeypatch.setattr(supervisor,'vast',vast)
    monkeypatch.setattr(supervisor,'ssh',lambda *args,**kwargs: json.dumps({'state':'DONE','alive':False,'log_age':0}))
    def interrupt_sleep(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(supervisor.time,'sleep',interrupt_sleep)
    return events


def test_destroy_requires_independent_verified_complete_checkpoint(tmp_path,monkeypatch):
    events = setup_run(tmp_path,monkeypatch)
    def verify(state):
        events.append(('verified','checkpoint'))
        return {'checkpoint_metadata':{'phase_completed':True,'tokens_seen':1000000000}}
    monkeypatch.setattr(supervisor,'recover_and_verify',verify)
    supervisor.main()
    assert events.index(('verified','checkpoint')) < events.index(('destroy','instance'))
    assert json.loads(supervisor.STATE.read_text())['stage'] == 'COMPLETE'


def test_failed_export_never_destroys_instance(tmp_path,monkeypatch):
    events = setup_run(tmp_path,monkeypatch)
    def fail(state):
        raise ValueError('remote hash mismatch')
    monkeypatch.setattr(supervisor,'recover_and_verify',fail)
    with pytest.raises(KeyboardInterrupt):
        supervisor.main()
    assert ('destroy','instance') not in events


def test_done_with_partial_checkpoint_never_destroys_instance(tmp_path,monkeypatch):
    events = setup_run(tmp_path,monkeypatch)
    monkeypatch.setattr(supervisor,'recover_and_verify',lambda state:
            {'checkpoint_metadata':{'phase_completed':False,'tokens_seen':100000000}})
    with pytest.raises(KeyboardInterrupt):
        supervisor.main()
    assert ('destroy','instance') not in events


def test_budget_boundary_stops_gpu_and_retains_disk_for_recovery(tmp_path,monkeypatch):
    events = setup_run(tmp_path,monkeypatch,credit=6)
    supervisor.main()
    assert ('stop','instance') in events and ('destroy','instance') not in events
    assert json.loads(supervisor.STATE.read_text())['stage'] == 'PAUSED_RECOVERY_REQUIRED'


def mock_cli(tmp_path,monkeypatch,response,instances):
    from types import SimpleNamespace
    key = tmp_path/'api-key'
    key.write_text('test-key')
    monkeypatch.setattr(supervisor,'KEY',key)
    def run(command,**kwargs):
        output = json.dumps(instances) if command[3:5] == ['show','instances'] else response
        return SimpleNamespace(returncode=0,stdout=output)
    monkeypatch.setattr(supervisor.subprocess,'run',run)
    monkeypatch.setattr(supervisor.time,'sleep',lambda seconds:None)


def test_already_attached_key_is_idempotent_success(tmp_path,monkeypatch):
    mock_cli(tmp_path,monkeypatch,"{'success': False, 'msg': 'SSH key already associated with instance.'}",[])
    assert supervisor.vast('attach','ssh',123,'public-key')['success']


def test_empty_destroy_cli_response_requires_instance_disappearance(tmp_path,monkeypatch):
    mock_cli(tmp_path,monkeypatch,'',[])
    assert supervisor.vast('destroy','instance',123,'-y')['success']


def test_stop_cli_response_requires_confirmed_target_state(tmp_path,monkeypatch):
    mock_cli(tmp_path,monkeypatch,'stopping instance 123.',[{'id':123,'intended_status':'stopped'}])
    assert supervisor.vast('stop','instance',123)['success']


def test_zero_cli_exit_without_requested_state_is_failure(tmp_path,monkeypatch):
    mock_cli(tmp_path,monkeypatch,'',[{'id':123,'intended_status':'running'}])
    with pytest.raises(RuntimeError,match='requested state'):
        supervisor.vast('destroy','instance',123,'-y')
