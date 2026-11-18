"""vLLM's Prometheus metrics around one benchmark invocation.

The client sees only the whole TTFT. The server also splits it: how long each request waited
before it started running (queue time) and how long its prefill took. scripts/benchmark.py
reads these histograms from every engine's /metrics before and after an invocation and keeps
the difference, so each point records what its own requests saw.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.request import urlopen

# Request-latency histograms, by the name the summary uses.
HISTOGRAMS = {
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "ttft": "vllm:time_to_first_token_seconds",
}
COUNTERS = {"prefix_cache_queries": "vllm:prefix_cache_queries", "prefix_cache_hits": "vllm:prefix_cache_hits"}
_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})? (\S+)$")
_LE = re.compile(r'le="([^"]+)"')


# The engines whose metrics count: the server itself, or both PD roles (the proxy has none).
def endpoints(base_url: str) -> dict[str, str]:
    if os.environ.get("SERVING_MODE") == "disaggregated":
        roles = {}
        for role in ("prefill", "decode"):
            head = os.environ.get(f"{role.upper()}_HEAD_IP")
            port = os.environ.get(f"{role.upper()}_API_PORT")
            if head and port:
                roles[role] = f"http://{head}:{port}"
        if roles:
            return roles
    return {"server": base_url}


# Sum every engine's series of the histograms and counters above; None if unreachable.
def scrape(url: str) -> dict[str, Any] | None:
    try:
        with urlopen(f"{url.rstrip('/')}/metrics", timeout=10) as response:
            text = response.read().decode()
    except (OSError, ValueError):
        return None
    histograms = {key: {"sum": 0.0, "count": 0.0, "buckets": {}} for key in HISTOGRAMS}
    counters = {key: 0.0 for key in COUNTERS}
    names = {name: key for key, name in HISTOGRAMS.items()}
    counter_names = {name: key for key, name in COUNTERS.items()}
    for line in text.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        name, labels, value = match[1], match[2] or "", float(match[3])
        for suffix in ("_sum", "_count", "_bucket"):
            base = name.removesuffix(suffix)
            if base != name and base in names:
                entry = histograms[names[base]]
                if suffix == "_bucket":
                    le = _LE.search(labels)
                    if le:
                        bound = float(le[1])
                        entry["buckets"][bound] = entry["buckets"].get(bound, 0.0) + value
                else:
                    entry[suffix[1:]] += value
                break
        else:
            base = name.removesuffix("_total")
            if base in counter_names:
                counters[counter_names[base]] += value
    return {"histograms": histograms, "counters": counters}


def _quantile(buckets: dict[float, float], count: float, q: float) -> float | None:
    # Linear interpolation inside the cumulative bucket that holds the quantile.
    if count <= 0:
        return None
    target, low, below = q * count, 0.0, 0.0
    for bound in sorted(buckets):
        cumulative = buckets[bound]
        if cumulative >= target:
            if bound == float("inf"):
                return low
            share = (target - below) / (cumulative - below) if cumulative > below else 1.0
            return low + share * (bound - low)
        low, below = bound, cumulative
    return low


# What one invocation's requests saw: mean and estimated p50/p90/p99 in ms per histogram, and
# counter increases. The percentiles come from vLLM's buckets, so they are approximate.
def difference(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    result: dict[str, Any] = {}
    for key in HISTOGRAMS:
        a, b = before["histograms"][key], after["histograms"][key]
        count = b["count"] - a["count"]
        buckets = {bound: b["buckets"][bound] - a["buckets"].get(bound, 0.0) for bound in b["buckets"]}
        result[key] = {
            "requests": count,
            "mean_ms": (b["sum"] - a["sum"]) / count * 1000 if count > 0 else None,
            **{
                f"p{p}_ms": (lambda v: None if v is None else v * 1000)(_quantile(buckets, count, p / 100))
                for p in (50, 90, 99)
            },
        }
    result["counters"] = {key: after["counters"][key] - before["counters"][key] for key in COUNTERS}
    return result


def scrape_all(urls: dict[str, str]) -> dict[str, Any]:
    return {role: scrape(url) for role, url in urls.items()}


def difference_all(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {role: difference(before.get(role), after.get(role)) for role in after}
