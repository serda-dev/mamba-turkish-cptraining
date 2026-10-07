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
                    save(state,'PAUSED_RECOVERY_REQUIRED',error='…21003 tokens truncated…end(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.optimizer_name == "adamw":
            self.optimizer = AdamW(
                param_groups,
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
            )
            logger.info("Optimizer: AdamW (torch)")
        elif self.optimizer_name == "adamw_8bit":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "optimizer=adamw_8bit requires bitsandbytes. "
                    "Install the pinned package from requirements.txt."
                ) from exc

            self.optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
            )
            logger.info("Optimizer: AdamW8bit (bitsandbytes)")
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        logger.info(
            "Optimizer settings: lr=%s, weight_decay=%s, betas=(%s, %s), eps=%s",
            self.lr,
            self.weight_decay,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        if self.warmup_ratio is not None:
            self.warmup_steps = int(max(0, self.max_steps * float(self.warmup_ratio)))

        if self.warmup_steps <= 0:
            if self.scheduler_name == "cosine":
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, self.max_steps))
            elif self.scheduler_name in ("constant", "linear_constant"):
                self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=max(1, self.max_steps))
            else:
                raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
            return

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        remaining_steps = max(1, self.max_steps - self.warmup_steps)
        if self.scheduler_name == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=0.0,
            )
        elif self.scheduler_name in ("constant", "linear_constant"):
            main_scheduler = ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=remaining_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.warmup_steps],
        )

        logger.info(
            "Scheduler: %s with linear warmup for %s steps",
            self.scheduler_name,
            self.warmup_steps,
        )

    def _setup_scaler(self):
        """Setup dtype for mixed precision (bf16 doesn't need GradScaler)."""
        if self.mixed_precision and self.device.type == "cuda":
            # Determine amp dtype
            if self.amp_dtype in ("bfloat16", "bf16"):
                self.amp_torch_dtype = torch.bfloat16
                self.use_grad_scaler = False  # bf16 doesn't need scaling
                logger.info("Mixed precision: enabled (bfloat16, no GradScaler)")
            else:
                self.amp_torch_dtype = torch.float16
                self.use_grad_scaler = True
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
                logger.info("Mixed precision: enabled (fp16 with GradScaler)")
        else:
            self.amp_torch_dtype = None
            self.use_grad_scaler = False
            logger.info("Mixed precision: disabled")

    def _setup_logger(self):
        """Setup metrics logger."""
        self.metrics_logger = MetricsLogger(
            log_dir=self.log_dir,
            log_json=self.config.get("logging", {}).get("log_json", True),
        )

    def _get_gpu_memory(self) -> Optional[float]:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return None

    def _capture_rng(self):
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": self.train_loader.generator.get_state() if self.train_loader.generator is not None else None,
        }

    def _restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        if state["loader"] is not None:
            self.train_loader.generator.set_state(state["loader"].cpu())

    def _data_iterator(self):
        if self._epoch_rng is None:
            self._epoch_rng = self._capture_rng()
        else:
            self._restore_rng(self._epoch_rng)
        iterator = iter(self.train_loader)
        for _ in range(self.batches_consumed):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError("Resume cursor exceeds dataset") from exc
        if self._resume_rng is not None:
            self._restore_rng(self._resume_rng)
            self._resume_rng = None
        return iterator

    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Publish complete checkpoints before updating the latest pointer."""
        ckpt_name = "final" if is_final and self.phase_completed else (
            f"stopped_step_{step:06d}" if is_final else f"step_{step:06d}"
        )
        ckpt_path = self.checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            # Never overwrite a published checkpoint in place.
            ckpt_path = self.checkpoint_dir / f"{ckpt_name}_{time.time_ns()}"
        checkpoint_rng = self._capture_rng()
        temporary = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=self.checkpoint_dir))
        try:
            self.model.save_pretrained(temporary)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(temporary)
            state = {
                "state_version": 2, "step": step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "tokens_seen": self.tokens_seen, "loss_tokens_seen": self.loss_tokens_seen,
                "phase": self.phase_id, "phase_name": self.phase_name,
                "source_sample_counts": self.source_sample_counts,
                "source_token_counts": self.source_token_counts,
                "epoch": self.epoch, "batches_consumed": self.batches_consumed,
                "repeated_tokens": self.repeated_tokens,
                "epoch_rng": self._epoch_rng, "rng": checkpoint_rng,
                "data_signature": self.data_signature,
                "training_signature": self.training_signature,
                "phase_completed": self.phase_completed, "stop_reason": self.stop_reason,
            }
            if self.use_grad_scaler:
                state["scaler_state_dict"] = self.scaler.state_dict()
            torch.save(state, temporary / "training_state.pt")
            (temporary / "checkpoint_metadata.json").write_text(json.dumps({
                "phase": self.phase_id, "phase_completed": self.phase_completed,
                "stop_reason": self.stop_reason, "step": step, "tokens_seen": self.tokens_seen,
                "data_signature": self.data_signature, "training_signature": self.training_signature,
            }, indent=2))
            from ..config import save_yaml_config
            save_yaml_config(self.config, str(temporary / "resolved_config.yaml"))
            temporary.rename(ckpt_path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        hub_repo = self.config.get("checkpointing", {}).get("hub_repo")
        if hub_repo:
            from .hub import upload_checkpoint
            upload_checkpoint(ckpt_path, hub_repo)
        write_latest_metadata(str(self.checkpoint_root), {
            "latest_checkpoint": str(ckpt_path.resolve()),
            "phase": self.phase_id, "phase_name": self.phase_name,
            "step": step, "tokens_seen": self.tokens_seen,
            "phase_completed": self.phase_completed, "stop_reason": self.stop_reason,
        })
        if not is_final:
            self._cleanup_checkpoints()
        if hub_repo:
            from .hub import prune_remote_checkpoints
            try:
                prune_remote_checkpoints(hub_repo, keep=self.save_total_limit)
            except Exception as exc:
                logger.warning("Remote checkpoint pruning failed: %s", type(exc).__name__)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted([
            d for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        ], key=lambda x: x.stat().st_mtime)

        if self.save_total_limit is None:
            return

        while len(checkpoints) > self.save_total_limit:
            old_ckpt = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {old_ckpt}")
            import shutil
            shutil.rmtree(old_ckpt)

    def train(self) -> Dict[str, Any]:
        """Train a finite pass unless repetition is explicitly enabled.

        Checkpoints are taken only at optimizer boundaries. A partial final
        accumulation is normalized by its actual supervised token count.
        """
        self.model.train()
        self.start_time = time.time()
        budget = TimeBudget(self.time_budget_seconds)
        iterator = self._data_iterator()
        self.optimizer.zero_grad(set_to_none=True)
        count = supervised = step_tokens = 0
        loss_sum = 0.0
        step_start = time.time()
        normalization = max(1, self.seq_len * self.micro_batch_size * self.gradient_accumulation_steps)
        pbar = tqdm(total=self.max_steps, initial=self.global_step, desc="Training", unit="step")

        def update():
            nonlocal count, supervised, step_tokens, loss_sum, step_start
            if self.use_grad_scaler:
                self.scaler.unscale_(self.optimizer)
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(normalization / supervised)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            if self.use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            duration = max(time.time() - step_start, 1e-9)
            if self.log_every_steps and self.global_step % self.log_every_steps == 0:
                elapsed = time.time() - self.start_time
                self.metrics_logger.log({
                    "step": self.global_step, "phase": self.phase_id, "phase_name": self.phase_name,
                    "loss": loss_sum / supervised, "lr": self.scheduler.get_last_lr()[0],
                    "tokens_per_sec": step_tokens / duration, "step_time": duration,
                    "gpu_memory_gb": self._get_gpu_memory(), "elapsed_seconds": elapsed,
                    "total_tokens": self.tokens_seen, "loss_tokens": self.loss_tokens_seen,
                    "repeated_tokens": self.repeated_tokens,
                    "source_token_counts": dict(self.source_token_counts),
                })
            if self.checkpoint_every_steps and self.global_step % self.checkpoint_every_steps == 0:
                self._save_checkpoint(self.global_step)
            if self.empty_cache_every_steps and self.global_step % self.empty_cache_every_steps == 0:
                torch.cuda.empty_cache()
            count = supervised = step_tokens = 0
            loss_sum = 0.0
            step_start = time.time()
            pbar.update(1)

        try:
            while self.global_step < self.max_steps:
                if self.stop_file and Path(self.stop_file).exists():
                    self.stop_reason = "budget_or_operator_stop"
                    break
                if self.stop_after_steps is not None and self.global_step >= self.stop_after_steps:
                    self.stop_reason = "operator_limit"
                    break
                if self.max_tokens is not None and self.tokens_seen >= self.max_tokens:
                    self.stop_reason = "max_tokens"
                    break
                if budget.is_expired():
                    self.stop_reason = "time_budget"
                    break
                try:
                    batch = next(iterator)
                except StopIteration:
                    if not self.repeat_dataset:
                        self.stop_reason = "dataset_exhausted"
                        break
                    if self.batches_consumed == 0:
                        raise ValueError("Training dataset is empty")
                    self.epoch += 1
                    self.batches_consumed = 0
                    self._epoch_rng = None
                    iterator = self._data_iterator()
                    continue
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device).clone()
                mask = batch["attention_mask"].to(self.device).clone()
                if self.max_tokens is not None:
                    remaining = self.max_tokens - self.tokens_seen
                    keep = mask.bool() & (mask.reshape(-1).cumsum(0).reshape_as(mask) <= remaining)
                    mask = keep.to(mask.dtype)
                labels.masked_fill_(~mask.bool(), -100)
                # HF causal losses shift labels by one position.
                n_supervised = int((labels[:, 1:] != -100).sum().item())
                n_tokens = int(mask.sum().item())
                if n_supervised == 0:
                    if self.max_tokens is not None and remaining == 1:
                        self.stop_reason = "unsupervised_token_remainder"
                        break
                    raise ValueError("Batch/token remainder contains no causal training targets")
                with autocast(device_type=self.device.type, dtype=self.amp_torch_dtype, enabled=self.amp_torch_dtype is not None):
                    outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=labels)
                    raw_loss = outputs.loss
                if not torch.isfinite(raw_loss).all():
                    raise FloatingPointError("Non-finite training loss; checkpoint was not published")
                scaled_loss = raw_loss * (n_supervised / normalization)
                if self.use_grad_scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                count += 1
                supervised += n_supervised
                loss_sum += raw_loss.detach().item() * n_supervised
                step_tokens += n_tokens
                self.tokens_seen += n_tokens
                self.loss_tokens_seen += n_supervised
                self.batches_consumed += 1
                if self.epoch > 0:
                    self.repeated_tokens += n_tokens
                token_sources = batch.get("token_source_ids")
                names = batch.get("source")
                if token_sources is not None:
                    source_map = getattr(self.train_loader.dataset, "source_id_to_name", {})
                    for source_row, valid_row in zip(token_sources, mask.cpu()):
                        for source_id in source_row[valid_row.bool()].unique().tolist():
                            if source_id == 0:
                                continue
                            name = source_map.get(int(source_id), str(source_id))
                            tokens = int(((source_row == source_id) & valid_row.bool()).sum().item())
                            self.source_sample_counts[name] = self.source_sample_counts.get(name, 0) + 1
                            self.source_token_counts[name] = self.source_token_counts.get(name, 0) + tokens
                elif names is not None:
                    names = [names] if isinstance(names, str) else names
                    for name, row in zip(names, mask):
                        tokens = int(row.sum().item())
                        if tokens:
                            self.source_sample_counts[name] = self.source_sample_counts.get(name, 0) + 1
                            self.source_token_counts[name] = self.source_token_counts.get(name, 0) + tokens
                if count >= self.gradient_accumulation_steps:
                    update()
            if self.stop_reason is None:
                self.stop_reason = "max_tokens" if self.max_tokens is not None and self.tokens_seen >= self.max_tokens else "max_steps"
            if count:
                update()
            self.phase_completed = self.stop_reason == "max_tokens" or (
                self.max_tokens is None and self.stop_reason in ("max_steps", "dataset_exhausted")
            )
            if self.save_final:
                self._save_checkpoint(self.global_step, is_final=True)
        finally:
            pbar.close()
        return {
            "final_step": self.global_step, "total_tokens": self.tokens_seen,
            "loss_tokens": self.loss_tokens_seen, "repeated_tokens": self.repeated_tokens,
            "epoch": self.epoch, "batches_consumed": self.batches_consumed,
            "total_time_seconds": time.time() - self.start_time,
            "stop_reason": self.stop_reason, "phase_completed": self.phase_completed,
            "phase": self.phase_id, "phase_name": self.phase_name,
            "source_sample_counts": self.source_sample_counts,
            "source_token_counts": self.source_token_counts,
        }
