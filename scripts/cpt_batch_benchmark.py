"""Compare execution layouts from one immutable full training checkpoint."""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
import yaml

ROOT=Path('/mnt/home_extra')


def main():
    latest=json.loads((ROOT/'checkpoint/latest.json').read_text())
    checkpoint=Path(latest['latest_checkpoint'])
    base=yaml.safe_load((ROOT/'run.yaml').read_text())
    results=[]
    for batch,checkpointing in [(4,True),(8,True),(4,False),(8,False)]:
        name=f'batch{batch}-gc{int(checkpointing)}'
        folder=ROOT/'batch-benchmark'/name
        folder.mkdir(parents=True,exist_ok=True)
        cfg=copy.deepcopy(base)
        cfg['training'].update(micro_batch_size=batch,gradient_accumulation_steps=64//batch,
            gradient_checkpointing=checkpointing,allow_batch_regrouping=True,stop_file=None,
            stop_after_steps=latest['step']+3)
        cfg['checkpointing'].update(checkpoint_every_steps=0,save_final=False)
        cfg['checkpointing'].pop('hub_repo',None)
        cfg['paths'].update(checkpoint_dir=str(folder/'checkpoint'),log_dir=str(folder/'log'))
        cfg['logging']['log_every_steps']=1
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        start=time.time()
        with (folder/'run.log').open('w') as log:
            p=subprocess.run([sys.executable,'-m','scripts.cpt_batch_benchmark','--case',str(path),
                              '--checkpoint',str(checkpoint),'--memory-report',str(folder/'memory.json')],
                             stdout=log,stderr=subprocess.STDOUT,timeout=240)
        metrics=[]
        for f in (folder/'log').glob('*.jsonl'):
            metrics.extend(json.loads(line) for line in f.read_text().splitlines() if line.strip())
        valid=[m for m in metrics if 'tokens_per_sec' in m]
        row={'name':name,'batch':batch,'accumulation':64//batch,'gradient_checkpointing':checkpointing,
             'returncode':p.returncode,'elapsed_seconds':time.time()-start,'metrics':valid}
        if p.returncode==0 and len(valid)==3:
            row['steady_tokens_per_second']=sum(m['tokens_per_sec'] for m in valid[1:])/2
            row.update(json.loads((folder/'memory.json').read_text()))
        else:
            row['error']='CUDA_OUT_OF_MEMORY' if 'out of memory' in (folder/'run.log').read_text().lower() else 'INCOMPLETE_RUN'
        results.append(row)
        (ROOT/'batch-benchmark/results.json').write_text(json.dumps({'checkpoint':latest,'cases':results},indent=2))
        print('BENCHMARK_CASE',json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)


def case():
    import argparse
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--case');parser.add_argument('--checkpoint');parser.add_argument('--memory-report')
    args=parser.parse_args()
    # Run the real CLI/trainer in this process, retaining peak CUDA statistics.
    from src.cli import main as cli_main
    torch.cuda.reset_peak_memory_stats()
    sys.argv=['src.cli','train','--config',args.case,'--resume',args.checkpoint]
    exit_code=cli_main()
    if exit_code:
        raise SystemExit(exit_code)
    Path(args.memory_report).write_text(json.dumps({
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}))


if __name__=='__main__':
    if '--case' in sys.argv:case()
    else:main()
