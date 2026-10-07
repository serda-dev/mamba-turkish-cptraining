"""One GPU run: baseline, 100M check, exact resume, final evaluation, export."""
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path('/mnt/home_extra')
STATUS = ROOT / 'run_status.json'


def status(state, **fields):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps({'state':state, **fields}, indent=2))
    temp.replace(STATUS)
    print('RUN_STATE', state, flush=True)


def command(args):
    subprocess.run([sys.executable, *args], check=True)


def evaluate(model, label, revision=None):
    args = ['-m', 'scripts.cpt_evaluate', '--model', model, '--cache',
            str(ROOT/'cache/token_cache/phase_1_validation/manifest.json'), '--output', str(ROOT/f'eval_{label}.json')]
    if revision:
        args += ['--revision', revision]
    command(args)
    return json.loads((ROOT/f'eval_{label}.json').read_text())


def latest():
    return json.loads((ROOT/'checkpoint/latest.json').read_text())


def export_reports(config):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    repo = config['checkpointing']['hub_repo']
    for path in [STATUS, ROOT/'run.yaml', *ROOT.glob('eval_*.json'), *ROOT.glob('log/*.json*')]:
        if path.is_file():
            api.upload_file(repo_id=repo, path_or_fileobj=str(path), path_in_repo='run/'+path.name)


def main():
    os.environ['HF_TOKEN'] = (ROOT/'hf_token').read_text().strip()
    config = yaml.safe_load((ROOT/'run.yaml').read_text())
    try:
        if (ROOT/'STOP').exists():
            status('STOPPED_BEFORE_TRAINING')
            return
        status('BASELINE_EVALUATION')
        baseline = evaluate(config['model']['base_model'], 'baseline', config['model']['revision'])
        config['training']['stop_after_steps'] = 763
        (ROOT/'first100m.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        status('TRAINING_FIRST_100M')
        command(['-m','src.cli','train','--config',str(ROOT/'first100m.yaml'),'--resume','none'])
        checkpoint = latest()
        if checkpoint['stop_reason'] != 'operator_limit':
            status('STOPPED', checkpoint=checkpoint)
            export_reports(config)
            return
        status('EVALUATING_100M', checkpoint=checkpoint)
        scored = evaluate(checkpoint['latest_checkpoint'], '100m')
        checks = {name: scored['scores'][name]['mean_nll'] <= baseline['scores'][name]['mean_nll'] * tolerance
                  for name, tolerance in [('turkish',1.05),('english',1.10)]}
        checks['contract_diagnostic'] = scored['contract_correct'] >= baseline['contract_correct'] - 2
        export_reports(config)
        if not all(checks.values()) or (ROOT/'STOP').exists():
            status('QUALITY_GATE_STOPPED', checks=checks, checkpoint=checkpoint)
            export_reports(config)
            return
        status('TRAINING_REMAINDER', checks=checks)
        command(['-m','src.cli','train','--config',str(ROOT/'run.yaml'),'--resume','auto'])
        checkpoint = latest()
        if checkpoint['phase_completed'] and checkpoint['tokens_seen'] == 1000000000:
            status('FINAL_EVALUATION', checkpoint=checkpoint)
            evaluate(checkpoint['latest_checkpoint'], 'final')
            status('DONE', checkpoint=checkpoint)
        else:
            status('STOPPED', checkpoint=checkpoint)
        export_reports(config)
    except BaseException as exc:
        status('FAILED', error=type(exc).__name__)
        # Do not mark DONE or discard the local checkpoint after an upload error.
        try:
            export_reports(config)
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
