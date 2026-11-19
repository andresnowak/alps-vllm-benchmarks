from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import server_metrics


def exposition(queue_sum: float, queue_count: int, buckets: dict[str, int], hits: float) -> bytes:
    lines = ["# HELP vllm:request_queue_time_seconds Time spent waiting."]
    for engine in ("0", "1"):
        for le, count in buckets.items():
            lines.append(f'vllm:request_queue_time_seconds_bucket{{engine="{engine}",le="{le}"}} {count}')
        lines.append(f'vllm:request_queue_time_seconds_sum{{engine="{engine}"}} {queue_sum}')
        lines.append(f'vllm:request_queue_time_seconds_count{{engine="{engine}"}} {queue_count}')
        lines.append(f'vllm:prefix_cache_queries_total{{engine="{engine}"}} 1000.0')
        lines.append(f'vllm:prefix_cache_hits_total{{engine="{engine}"}} {hits}')
    return ("\n".join(lines) + "\n").encode()


class ServerMetricsTests(unittest.TestCase):
    def test_scrape_sums_every_engine(self) -> None:
        body = exposition(2.0, 4, {"0.5": 2, "1.0": 4, "+Inf": 4}, 300.0)
        with mock.patch.object(server_metrics, "urlopen", return_value=io.BytesIO(body)):
            data = server_metrics.scrape("http://server:8000")

        queue = data["histograms"]["queue"]
        self.assertEqual((queue["sum"], queue["count"]), (4.0, 8.0))
        self.assertEqual(queue["buckets"][0.5], 4.0)
        self.assertEqual(data["counters"], {"prefix_cache_queries": 2000.0, "prefix_cache_hits": 600.0})

    def test_difference_keeps_only_this_invocation(self) -> None:
        empty = {"0.5": 0, "1.0": 0, "+Inf": 0}
        before_body = exposition(0.0, 0, empty, 0.0)
        after_body = exposition(2.0, 4, {"0.5": 2, "1.0": 4, "+Inf": 4}, 500.0)
        with mock.patch.object(server_metrics, "urlopen", side_effect=[io.BytesIO(before_body), io.BytesIO(after_body)]):
            before = server_metrics.scrape("http://server:8000")
            after = server_metrics.scrape("http://server:8000")

        result = server_metrics.difference(before, after)

        self.assertEqual(result["queue"]["requests"], 8.0)
        self.assertAlmostEqual(result["queue"]["mean_ms"], 500.0)
        # Half the requests are below 0.5 s, so the median sits at the bucket edge.
        self.assertAlmostEqual(result["queue"]["p50_ms"], 500.0)
        self.assertAlmostEqual(result["queue"]["p99_ms"], 990.0)
        self.assertEqual(result["counters"]["prefix_cache_hits"], 1000.0)

    def test_unreachable_endpoint_gives_none(self) -> None:
        with mock.patch.object(server_metrics, "urlopen", side_effect=OSError("no route")):
            self.assertIsNone(server_metrics.scrape("http://proxy:8000"))
        self.assertIsNone(server_metrics.difference(None, None))

    def test_pd_reads_both_engines_not_the_proxy(self) -> None:
        env = {"SERVING_MODE": "disaggregated", "PREFILL_HEAD_IP": "10.0.0.1", "PREFILL_API_PORT": "8100",
               "DECODE_HEAD_IP": "10.0.0.5", "DECODE_API_PORT": "8200"}
        with mock.patch.dict("os.environ", env, clear=False):
            urls = server_metrics.endpoints("http://10.0.0.1:8000")

        self.assertEqual(urls, {"prefill": "http://10.0.0.1:8100", "decode": "http://10.0.0.5:8200"})

    def test_single_mode_reads_the_server(self) -> None:
        with mock.patch.dict("os.environ", {"SERVING_MODE": "single"}, clear=False):
            self.assertEqual(server_metrics.endpoints("http://h:8000"), {"server": "http://h:8000"})


class ServerSummaryTests(unittest.TestCase):
    @staticmethod
    def role(queue: float, prefill: float, decode: float | None) -> dict:
        histogram = lambda mean: {"mean_ms": mean, "p50_ms": None, "p90_ms": None, "p99_ms": mean}
        row = {"queue": histogram(queue), "prefill": histogram(prefill)}
        if decode is not None:
            row["decode"] = histogram(decode)
        return row

    def test_pd_takes_prompt_times_from_prefill_and_decode_time_from_decode(self) -> None:
        from benchmark import server_summary

        records = [
            {"prefill": self.role(10, 100, None), "decode": self.role(0, 1, 2000)},
            {"prefill": self.role(30, 300, None), "decode": self.role(0, 1, 4000)},
        ]

        summary = server_summary(records, "prefill", "decode")

        self.assertEqual(summary["server_mean_queue_ms"], 20)
        self.assertEqual(summary["server_mean_prefill_ms"], 200)
        self.assertEqual(summary["server_mean_decode_ms"], 3000)

    def test_runs_without_the_decode_histogram_leave_it_blank(self) -> None:
        from benchmark import server_summary

        summary = server_summary([{"server": self.role(5, 50, None)}, None], "server", "server")

        self.assertEqual(summary["server_mean_queue_ms"], 5)
        self.assertIsNone(summary["server_mean_decode_ms"])

    def test_the_decode_histogram_is_scraped(self) -> None:
        self.assertEqual(server_metrics.HISTOGRAMS["decode"], "vllm:request_decode_time_seconds")


if __name__ == "__main__":
    unittest.main()
