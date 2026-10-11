# CPT continuation: 2026-10-11

The October 7 A100 instance 54662760 was automatically deleted when CPU
preparation failed. No training checkpoint was produced. Its partial
334,508,883-token cache is not a complete training corpus.

The failure was real source identity corruption after student annotation
resumes, not a reason to disable CPT's document ID or content hash checks.
Academic parent ordinal 100000 authenticates the full text at canonical raw
ordinal 2247. Academic 200000 and 300000 repeat that same text. Ozenli parent
200000 matches canonical 451; parent 300000 matches canonical 100451.
The exact library-level cause of the producer's resume behavior has not been
reproduced. Other source exports have not been comprehensively audited yet.

`cpt_repair_label_identity.py` joins the existing selected Academic/Ozenli
annotations with continuously streamed pinned source text by full UTF-8 SHA256.
It authenticates the original document ID and byte count, retains every score,
and assigns canonical source coordinates. Conflicting annotations for the same
content are excluded, as are parent hashes without a canonical raw match.
Identical content is emitted only once per source, at its earliest raw ordinal.
Original Hub exports and October 7 artifacts remain immutable.

`cpt_after_identity_repair.py` waits for both repaired sources, verifies artifact
checksums, and publishes new label files and separate parent provenance under
`repairs/identity-20261011` in the original dataset repository. It independently
verifies remote file sizes/hashes and pins the resulting commit in the run
configuration. It replaces only the two affected label file lists.
It then starts CPU token preparation and the existing rental supervisor.

New run directory: `/home/serda/cpt-run-20261011`.
Repair directory: `/home/serda/cpt-identity-repair-20261011`.
User services: `linguai-cpt-identity-repair`, `linguai-cpt-after-repair`,
`linguai-cpt-prepare`, `linguai-cpt-supervisor`.
The supervisor receives the new root through `LINGUAI_CPT_RUN_ROOT` and labels
its instance `linguai-cpt-1b-20261011`. The existing private model output
repository name ending in 20261007 is retained; it has no prior CPT checkpoint.

The 1B-token full-weight Jamba2-3B plan is unchanged: 80% Turkish, 20% English;
canonical accepted Turkish token quotas; 2M-token held-out validation; a 100M
checkpoint/evaluation gate before continuation with the same optimizer and
scheduler. No GPU may be rented until full train/validation cache counts,
checksums, language mix and source quotas pass validation. Preparation failure
must leave rental disabled. Offers are rechecked at rental time and must meet
the existing A100 SXM4 80GB, 180GB disk, <=$1.25/hour and >=72-hour availability
limits. Budget stop, verified checkpoint export, and automatic deletion remain
enabled. The October 11 repair covers the selected CPT label pool, not all
214 million original annotations.

## Transfer incident and recovery

The complete frozen caches were ready at 03:18 UTC. Instance 55328346 was
rented successfully, but recursive SCP through the Vast SSH proxy repeatedly
hit its 30-minute timeout and restarted the cache copy. At 10:03 UTC the GPU
had no compute process and only 958 MiB of the 4.7 GiB cache had arrived.
The team balance was $88.62, down from $96.89 including the idle rental.

The supervisor now probes the host's published direct SSH port, prefers that
connection when accessible, and transfers directories with checksummed,
resumable rsync. Proxy SSH remains a connection fallback. Every remote cache
shard is authenticated against its manifest before evaluation/training can
start. An unsuccessful transfer lasting over 45 minutes stops the GPU and
retains the instance disk for recovery rather than retrying indefinitely.
The October 11 incident was recovered on the same instance; no duplicate GPU
rental was requested.
