"""Pre-publish guard. Non-ASCII bytes have twice slipped into the page and
rendered as mojibake; a byte check is cheaper than another screenshot."""
import re, sys
from pathlib import Path

html = Path(__file__).with_name("index.html")
raw = html.read_bytes()
fails = []

bad = [(i, b) for i, b in enumerate(raw) if b > 127]
if bad:
    line = raw[:bad[0][0]].count(b"\n") + 1
    fails.append(f"{len(bad)} non-ASCII byte(s), first at line {line}")

text = raw.decode("utf-8")
if "<title>" not in text[:8192]:
    fails.append("no <title> in the first 8KB")

for host in re.findall(r'<script[^>]+src="https://([^/"]+)', text):
    if host not in {"cdnjs.cloudflare.com", "cdn.jsdelivr.net", "code.jquery.com"}:
        fails.append(f"script host not on the CSP allowlist: {host}")

if re.search(r'src="https://cdn\.jsdelivr\.net/npm/[^"@]+"', text):
    fails.append("jsdelivr script without a pinned version")

print("FAIL: " + "; ".join(fails) if fails else f"OK - {len(raw)//1024} KB, pure ASCII")
sys.exit(1 if fails else 0)
