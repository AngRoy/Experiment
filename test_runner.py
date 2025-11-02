#!/usr/bin/env python3

import argparse
import sys
import json
from typing import Any, Dict, List
try:
    import requests
except ImportError:
    print("Please install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

def pretty(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)

def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)

def post(url, payload):
    r = requests.post(url, headers={"Content-Type":"application/json"}, data=json.dumps(payload), timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()

def count_segments(segments: List[Dict[str, Any]]):
    mermaids = sum(1 for s in segments if isinstance(s.get("mermaid"), str) and s["mermaid"].strip())
    images   = sum(1 for s in segments if isinstance(s.get("image_prompt"), str) and s["image_prompt"].strip())
    return mermaids, images

def count_rendered(segments: List[Dict[str, Any]]):
    dp = sum(1 for s in segments if s.get("diagram_path"))
    ip = sum(1 for s in segments if s.get("image_path"))
    return dp, ip

def has_markdown_headings(segments: List[Dict[str, Any]]):
    for s in segments:
        txt = s.get("text") or ""
        if any(h in txt for h in ["# ", "## ", "### ", "- ", "* "]):
            return True
    return False

def test_min_defaults(base):
    print("TEST: defaults → expect ≥2 diagrams + ≥2 images (JSON only) ...", end="", flush=True)
    j = post(f"{base}/lesson", {"chat":"teach me routing protocols"})
    m,i = count_segments(j["segments"])
    assert_true(m >= 2, f"expected >=2 mermaid, got {m}")
    assert_true(i >= 2, f"expected >=2 image_prompts, got {i}")
    assert_true(has_markdown_headings(j["segments"]), "expected markdown-rich text")
    print(" OK")

def test_min_defaults_rendered(base):
    print("TEST: defaults rendered → expect ≥2 diagram PNGs + ≥2 image PNGs ...", end="", flush=True)
    j = post(f"{base}/lesson_rendered", {"chat":"teach me routing protocols"})
    dp, ip = count_rendered(j["segments"])
    assert_true(dp >= 2, f"expected >=2 diagram PNGs, got {dp}")
    assert_true(ip >= 2, f"expected >=2 image PNGs, got {ip}")
    # also ensure each URL present when path is present
    for s in j["segments"]:
        if s.get("diagram_path"):
            assert_true(bool(s.get("diagram_url")), "diagram_url missing")
        if s.get("image_path"):
            assert_true(bool(s.get("image_url")), "image_url missing")
    print(" OK")

def test_custom_counts(base):
    print("TEST: custom counts → 3 diagrams + 4 images (JSON only) ...", end="", flush=True)
    j = post(f"{base}/lesson", {"chat":"teach BFS with 3 diagrams and 4 images"})
    m,i = count_segments(j["segments"])
    assert_true(m >= 3, f"expected >=3 mermaid, got {m}")
    assert_true(i >= 4, f"expected >=4 image_prompts, got {i}")
    print(" OK")

def test_custom_counts_rendered(base):
    print("TEST: custom counts rendered → 3 diagram PNGs + 4 image PNGs ...", end="", flush=True)
    j = post(f"{base}/lesson_rendered", {"chat":"teach BFS with 3 diagrams and 4 images"})
    dp, ip = count_rendered(j["segments"])
    assert_true(dp >= 3, f"expected >=3 diagram PNGs, got {dp}")
    assert_true(ip >= 4, f"expected >=4 image PNGs, got {ip}")
    print(" OK")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000", help="API base URL")
    args = ap.parse_args()

    try:
        test_min_defaults(args.base)
        test_min_defaults_rendered(args.base)
        test_custom_counts(args.base)
        test_custom_counts_rendered(args.base)
        print("\nAll tests passed ✅")
    except Exception as e:
        print("\nFAILED ❌:", e, file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()