import time
import gzip
import json
import re
from pathlib import Path

# Try to find a file to test
files = list(Path('/mnt/volume-nbg1-1/cache/').rglob('*.jsonl.gz'))
if not files:
    print("No gzip files found for testing.")
    exit(0)
    
test_file = files[0]
print(f"Profiling on {test_file}")

def test_gzip_only():
    t0 = time.time()
    lines = 0
    with gzip.open(test_file, 'rt', encoding='utf-8') as f:
        for _ in f:
            lines += 1
    t1 = time.time()
    print(f"Gzip read: {lines} lines in {t1-t0:.2f}s -> {lines/(t1-t0):.2f} lines/s")

def test_gzip_json():
    t0 = time.time()
    lines = 0
    with gzip.open(test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                json.loads(line)
                lines += 1
            except:
                pass
    t1 = time.time()
    print(f"Gzip + JSON: {lines} lines in {t1-t0:.2f}s -> {lines/(t1-t0):.2f} lines/s")

def test_regex():
    # Read a few lines to get text
    texts = []
    with gzip.open(test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            texts.append(json.loads(line).get('text', ''))
            if len(texts) >= 1000: break
    
    t0 = time.time()
    for text in texts:
        text = text.strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[^\S\n]+', ' ', text)
    t1 = time.time()
    print(f"Regex: 1000 lines in {t1-t0:.2f}s -> {1000/(t1-t0):.2f} lines/s")

test_gzip_only()
test_gzip_json()
test_regex()
