"""Reconstruct classified text without storing the raw corpus.

Identity contract is linguai_quality.contracts.document_id (source plan repo,
revision, config, split, zero-based raw row ordinal and SHA256 of UTF-8 text).
The SQLite index preserves *every* label, including excluded routes; it grows
with label metadata, not raw text. Exact duplicate filtering is global across
sources. Near duplicates are not claimed. Run preparation to exhaustion before
using a manifest as evidence of a complete join. Early generator closure records
an incomplete audit. Concurrent writers to the same audit_db are unsupported.
"""
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterator

ROUTES = {"PREMIUM", "KEEP", "REVIEW", "DETERMINISTIC_REPAIR", "MODEL_REPAIR", "DROP"}
CLEAN_ROUTES = {"PREMIUM", "KEEP"}
SHA = re.compile(r"^[0-9a-f]{40}$")


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id(source: dict, ordinal: int, text: str) -> str:
    coordinates = [source["repo"], source["revision"], source["config"],
                   source["split"], ordinal, document_hash(text)]
    encoded = json.dumps(coordinates, ensure_ascii=False, separators=(",", ":"))
    return "doc_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def content_split(digest: str, fraction: float, seed: int) -> str:
    # Use content alone (rather than source coordinates) to prevent cross-source leakage.
    value = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest(), 16)
    return "validation" if value < int(fraction * (1 << 256)) else "train"


def _settings(config):
    tr = config.get("datasets", {}).get("turkish", {})
    return tr.get("classified", tr)


def _source_plan(settings):
    plan = settings["source_plan"]
    if isinstance(plan, (str, os.PathLike)):
        plan = json.loads(Path(plan).read_text(encoding="utf-8"))
    sources = plan["sources"]
    seen = set()
    for src in sources:
        for field in ("id", "repo", "revision", "config", "split", "text_field", "license"):
            if not src.get(field):
                raise ValueError(f"Source requires {field}")
        if not SHA.fullmatch(src["revision"]):
            raise ValueError("Raw source revision must be an immutable 40-character commit SHA")
        if src["id"] in seen:
            raise ValueError("Duplicate source_id in source plan")
        seen.add(src["id"])
    return plan


def _load_stream(repo, subset=None, **kwargs):
    from datasets import load_dataset
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return load_dataset(repo, subset, streaming=True, token=token, **kwargs)


def _iter_labels(settings):
    local = settings.get("labels_local_path")
    if local:
        path = Path(local)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        else:
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                yield from batch.to_pylist()
        return
    from huggingface_hub import HfApi
    revision = settings.get("labels_revision", "")
    if not SHA.fullmatch(revision):
        raise ValueError("labels_revision must be an immutable 40-character commit SHA")
    repo = settings["labels_repo"]
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    files = settings.get("labels_files")
    if not files:
        pattern = settings.get("labels_pattern", "*.parquet")
        files = sorted(p for p in HfApi(token=token).list_repo_files(
            repo, repo_type="dataset", revision=revision) if fnmatch.fnmatch(p, pattern))
    if not files:
        raise ValueError("No label Parquet files found")
    urls = [f"hf://datasets/{repo}@{revision}/{p}" for p in files]
    yield from _load_stream("parquet", data_files={"train": urls}, split="train")


def _iter_source(source):
    # Local raw fixtures/export mirrors must preserve original row order and coordinates.
    if source.get("local_path"):
        with Path(source["local_path"]).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    yield from _load_stream(source["repo"], source["config"],
                            revision=source["revision"], split=source["split"])


def _fingerprint(settings, plan):
    identity = {"settings": settings, "source_plan": plan}
    local = settings.get("labels_local_path")
    if local:
        digest = hashlib.sha256()
        with open(local, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        identity["local_labels_sha256"] = digest.hexdigest()
    return document_hash(json.dumps(identity, sort_keys=True, ensure_ascii=False))


def _index_labels(conn, settings, sources):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS labels(source_id TEXT, ordinal INTEGER, payload TEXT NOT NULL,
          PRIMARY KEY(source_id, ordinal)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS decisions(source_id TEXT, ordinal INTEGER, route TEXT,
          decision TEXT, content_hash TEXT, split TEXT, PRIMARY KEY(source_id, ordinal)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS seen(hash TEXT PRIMARY KEY, source_id TEXT, ordinal INTEGER) WITHOUT ROWID;
    """)
    if conn.execute("SELECT value FROM meta WHERE key='labels_complete'").fetchone():
        return
    conn.execute("DELETE FROM labels")
    count = 0
    for row in _iter_labels(settings):
        src, ordinal = row.get("source_id"), row.get("row_ordinal")
        if src not in sources or isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("Invalid source_id or zero-based row_ordinal in labels")
        if row.get("predicted_route") not in ROUTES:
            raise ValueError(f"Unknown predicted_route at {src}:{ordinal}")
        for field in ("doc_id", "document_hash", "input_bytes", "input_truncated", "predicted_label_json"):
            if field not in row:
                raise ValueError(f"Missing label field {field} at {src}:{ordinal}")
        if not isinstance(row["input_truncated"], bool):
            raise ValueError("input_truncated must be a boolean")
        if isinstance(row["input_bytes"], bool) or not isinstance(row["input_bytes"], int) or row["input_bytes"] < 0:
            raise ValueError("input_bytes must be a nonnegative integer")
        label = row["predicted_label_json"]
        if isinstance(label, str):
            label = json.loads(label)
        if not isinstance(label, dict):
            raise ValueError("predicted_label_json must contain a JSON object")
        try:
            conn.execute("INSERT INTO labels VALUES(?,?,?)", (src, ordinal, json.dumps(row, ensure_ascii=False)))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate label coordinates {src}:{ordinal}") from exc
        count += 1
        if count > int(settings.get("max_labels", 2000000)):
            raise ValueError("Label-index budget exceeded; select representative labels_files and a new audit_db")
        if count % 10000 == 0:
            conn.commit()
            used_bytes = conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute("PRAGMA page_size").fetchone()[0]
            if used_bytes > int(settings.get("max_audit_bytes", 10 * 1024**3)):
                raise ValueError("Audit disk budget exceeded; reduce labels_files or explicitly raise budget")
    if not count:
        raise ValueError("Label dataset is empty")
    conn.execute("INSERT OR REPLACE INTO meta VALUES('labels_complete',?)", (str(count),))
    conn.commit()


def _joined_rows(conn, source):
    sid = source["id"]
    labels = iter(conn.execute("SELECT ordinal,payload FROM labels WHERE source_id=? ORDER BY ordinal", (sid,)))
    pending = next(labels, None)
    if pending is None:
        return
    for ordinal, raw in enumerate(_iter_source(source)):
        if pending is None:
            break
        target, payload = pending
        if ordinal < target:
            continue
        yield source, ordinal, raw, json.loads(payload)
        pending = next(labels, None)
    if pending is not None:
        raise ValueError(f"Source ended before labeled row {sid}:{pending[0]}")


def _interleaved_rows(conn, plan, settings):
    # Smooth weighted round robin guarantees every active source makes progress.
    # Seven source streams and label cursors are bounded; no source text is buffered.
    active = [(src, iter(_joined_rows(conn, src))) for src in plan["sources"]]
    weights = settings.get("source_weights", {})
    scores = {src["id"]: 0.0 for src, _ in active}
    for src, _ in active:
        weight = float(weights.get(src["id"], 1.0))
        if not 0 < weight < float("inf"):
            raise ValueError("source_weights must be finite and positive")
    try:
        while active:
            total = sum(float(weights.get(src["id"], 1.0)) for src, _ in active)
            for src, _ in active:
                scores[src["id"]] += float(weights.get(src["id"], 1.0))
            chosen = max(range(len(active)), key=lambda idx: scores[active[idx][0]["id"]])
            source, stream = active[chosen]
            scores[source["id"]] -= total
            try:
                yield next(stream)
            except StopIteration:
                active.pop(chosen)
    finally:
        for _, stream in active:
            stream.close()


def iter_classified_texts(config: dict, split: str = "train") -> Iterator[str]:
    """Yield verified PREMIUM/KEEP text; audit all routes and exact duplicates.

    A fresh pass reuses immutable label metadata but reconstructs source text again.
    A missing/changed source row fails closed. Labels marked input_truncated are
    audited and excluded: their full text was not evaluated by the classifier.
    Repair routes cannot be enabled by simply changing accepted_routes.
    """
    if split not in {"train", "validation", "all"}:
        raise ValueError("split must be train, validation or all")
    settings = _settings(config)
    plan = _source_plan(settings)
    sources = {s["id"]: s for s in plan["sources"]}
    accepted = set(settings.get("accepted_routes", ["PREMIUM", "KEEP"]))
    if not accepted or not accepted <= CLEAN_ROUTES:
        raise ValueError("Only PREMIUM/KEEP are clean CPT inputs; repair/review require a separate verified artifact")
    fraction = float(settings.get("validation_fraction", .01))
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if settings.get("allow_cosmos") and not settings.get("cosmos_review_evidence"):
        raise ValueError("Cosmos opt-in requires cosmos_review_evidence")
    db = Path(settings.get("audit_db", "artifacts/classified/audit.sqlite"))
    manifest_path = Path(settings.get("manifest_path", db.with_suffix(".manifest.json")))
    db.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA cache_size=-16384")
    conn.execute("PRAGMA temp_store=FILE")
    fingerprint = _fingerprint(settings, plan)
    complete, failure, processed = False, None, 0
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        prior = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        if prior and prior[0] != fingerprint:
            raise ValueError("Audit database belongs to different labels/source plan/policy; choose a new audit_db")
        conn.execute("INSERT OR REPLACE INTO meta VALUES('fingerprint',?)", (fingerprint,))
        _index_labels(conn, settings, sources)
        conn.execute("DELETE FROM decisions")
        conn.execute("DELETE FROM seen")
        conn.commit()
        joined = _interleaved_rows(conn, plan, settings)
        try:
            for source, ordinal, raw, row in joined:
                sid = source["id"]
                text = raw.get(source["text_field"])
                if not isinstance(text, str) or not text:
                    raise ValueError(f"Raw text missing at {sid}:{ordinal}")
                digest = document_hash(text)
                if digest != row["document_hash"] or document_id(source, ordinal, text) != row["doc_id"]:
                    raise ValueError(f"Identity/hash mismatch at {sid}:{ordinal}; do not change hash contract to bypass")
                if not row["input_truncated"] and row["input_bytes"] != len(text.encode("utf-8")):
                    raise ValueError(f"input_bytes mismatch at {sid}:{ordinal}")
                route = row["predicted_route"]
                chosen_split = content_split(digest, fraction, int(settings.get("split_seed", 42)))
                if route not in accepted:
                    decision = "excluded_route"
                elif sid == "cosmos" and not settings.get("allow_cosmos", False):
                    decision = "cosmos_review_gate"
                elif row["input_truncated"]:
                    decision = "truncated_input_review"
                else:
                    inserted = conn.execute("INSERT OR IGNORE INTO seen VALUES(?,?,?)", (digest, sid, ordinal))
                    decision = "accepted" if inserted.rowcount else "duplicate"
                conn.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?)", (sid, ordinal, route, decision, digest, chosen_split))
                processed += 1
                if processed % 10000 == 0:
                    conn.commit()
                if decision == "accepted" and split in {chosen_split, "all"}:
                    yield text
        finally:
            joined.close()
        complete = True
    except GeneratorExit:
        raise
    except BaseException as exc:
        failure = str(exc)
        raise
    finally:
        conn.commit()
        # Do not claim full coverage after a failed index build or a partial consumer.
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = [dict(zip(("route", "decision", "split", "count"), r)) for r in conn.execute(
            "SELECT route,decision,split,count(*) FROM decisions GROUP BY route,decision,split ORDER BY route,decision,split")] if "decisions" in tables else []
        manifest = {"version": 1, "fingerprint": fingerprint, "complete": complete,
                    "error": failure, "labels": conn.execute("SELECT count(*) FROM labels").fetchone()[0] if "labels" in tables else 0,
                    "verified_rows": processed, "decisions": counts, "sources": plan["sources"],
                    "labels_repo": settings.get("labels_repo"), "labels_revision": settings.get("labels_revision"),
                    "labels_files": settings.get("labels_files"), "labels_pattern": settings.get("labels_pattern", "*.parquet"),
                    "label_scope": "explicit_shard_subset" if settings.get("labels_files") else "all_matching_label_shards",
                    "source_weights": settings.get("source_weights", "equal_source_round_robin"),
                    "audit_disk_bytes": db.stat().st_size,
                    "audit_db": str(db), "requested_split": split,
                    "split_algorithm": "SHA256(seed:UTF8-content-SHA256), 256-bit threshold",
                    "validation_fraction": fraction, "split_seed": int(settings.get("split_seed", 42)),
                    "identity_contract": "linguai_quality.contracts.document_id at 8ce7c69855e018fb697c7c6872347a4d354d171a; full raw UTF8 SHA256",
                    "accepted_routes": sorted(accepted),
                    "cosmos_review_evidence": settings.get("cosmos_review_evidence"),
                    "deduplication": "global exact UTF8 SHA256; no near-duplicate guarantee"}
        temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(manifest_path)
        conn.close()
