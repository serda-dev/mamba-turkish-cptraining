# CPT log preservation

The October 11 run archives GPU console logs, JSONL training metrics, evaluation
reports, run configuration, checkpoint metadata, batch benchmark output, and VPS
preparation/supervisor logs. Model weights and credential files are excluded.

Run manually from the repository:

```bash
LINGUAI_CPT_RUN_ROOT=/home/serda/cpt-run-20261011 /home/serda/.local/share/linguai-training/venv/bin/python -u -m scripts.cpt_log_backup
```

The VPS user timer `linguai-cpt-log-backup.timer` runs every ten minutes. Copies
are retained under `/home/serda/cpt-run-20261011/log-backup/`, including individual
files, a compressed archive, and `receipt.json`. The archive is uploaded to the
private model repository configured in `run.yaml`, at
`run/log-archives/cpt-run-20261011.tar.gz`. Hub commit history retains previous
versions. The receipt records the independently verified revision, size and hash.

The supervisor requires a verified final log archive, in addition to a verified
checkpoint, before deleting an instance that completed or stopped training. An
archive failure retains the instance for retry. The backup lock serializes timer
and supervisor uploads. Periodic backups after a terminal state leave existing
archives intact.
