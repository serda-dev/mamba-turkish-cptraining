import re
import time

texts = ["This is a test " * 1000 + "\n\n\n\n\n" + "test " * 1000] * 5000

t0 = time.time()
for t in texts:
    t = t.strip()
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[^\S\n]+', ' ', t)
t1 = time.time()

_RE_NEWLINES = re.compile(r'\n{3,}')
_RE_SPACES = re.compile(r'[^\S\n]+')

t2 = time.time()
for t in texts:
    t = t.strip()
    t = _RE_NEWLINES.sub('\n\n', t)
    t = _RE_SPACES.sub(' ', t)
t3 = time.time()

print(f"Without precompile: {t1-t0:.4f}s")
print(f"With precompile: {t3-t2:.4f}s")
