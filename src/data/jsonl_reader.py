"""JSONL reader with streaming and validation."""

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class _TempFileWrapper:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "r", encoding="utf-8")
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.f)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()
        import os
        try:
            os.remove(self.path)
            logger.info(f"Removed temporary extracted file {self.path}")
        except Exception as e:
            logger.warning(f"Could not remove temp file {self.path}: {e}")

def open_text_maybe_gzip(file_path: Union[str, Path]):
    """Open plain JSONL or gzip-compressed JSONL as a text stream.
    For .gz files, extracts entirely to /tmp first to avoid Python gzip IO bottlenecks."""
    file_path = Path(file_path)
    if file_path.suffix == ".gz":
        import tempfile
        import subprocess
        import os
        
        logger.info(f"Extracting {file_path} entirely before processing...")
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", dir="/tmp")
        os.close(tmp_fd)
        
        try:
            # Use native gunzip to extract to the temporary file
            with open(tmp_path, "wb") as f_out:
                subprocess.run(["gunzip", "-c", str(file_path)], stdout=f_out, check=True)
            logger.info(f"Extraction complete to {tmp_path}. Reading...")
            return _TempFileWrapper(tmp_path)
        except Exception as e:
            logger.error(f"Failed to extract {file_path}: {e}")
            try:
                os.remove(tmp_path)
            except:
                pass
            # Fallback to python gzip
            return gzip.open(file_path, "rt", encoding="utf-8")
            
    return open(file_path, "r", encoding="utf-8")


def read_jsonl_files(
    file_paths: Union[str, Path, List[Union[str, Path]]],
    text_field: str = "text",
    max_samples: Optional[int] = None,
) -> Iterator[str]:
    """
    Stream text from JSONL files with validation.
    
    Args:
        file_paths: Single path or list of paths to JSONL files
        text_field: JSON field containing the text content
        max_samples: Optional limit on total samples to yield
        
    Yields:
        Text strings from valid JSONL lines
    """
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    
    file_paths = [Path(p) for p in file_paths]
    
    total_lines = 0
    valid_lines = 0
    invalid_json = 0
    missing_field = 0
    empty_text = 0
    
    for file_path in file_paths:
        if not file_path.exists():
            logger.warning(f"File not found, skipping: {file_path}")
            continue
            
        logger.info(f"Reading: {file_path}")
        
        with open_text_maybe_gzip(file_path) as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                
                if not line:
                    continue
                
                # Parse JSON
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    invalid_json += 1
                    if invalid_json <= 5:  # Log first few errors
                        logger.debug(f"Invalid JSON at {file_path}:{line_num}: {e}")
                    continue
                
                # Extract text field
                if text_field not in data:
                    missing_field += 1
                    if missing_field <= 5:
                        logger.debug(f"Missing '{text_field}' field at {file_path}:{line_num}")
                    continue
                
                text = data[text_field]
                
                # Validate text
                if not isinstance(text, str) or not text.strip():
                    empty_text += 1
                    continue
                
                valid_lines += 1
                yield text
                
                if max_samples and valid_lines >= max_samples:
                    logger.info(f"Reached max_samples limit: {max_samples}")
                    break
            
            if max_samples and valid_lines >= max_samples:
                break
    
    # Log summary
    logger.info(
        f"JSONL reading complete: {valid_lines}/{total_lines} valid lines "
        f"(invalid_json={invalid_json}, missing_field={missing_field}, empty={empty_text})"
    )


def count_jsonl_lines(file_paths: Union[str, Path, List[Union[str, Path]]]) -> int:
    """Quick line count for progress estimation."""
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    
    total = 0
    for file_path in file_paths:
        file_path = Path(file_path)
        if file_path.exists():
            with open_text_maybe_gzip(file_path) as f:
                total += sum(1 for _ in f)
    return total
