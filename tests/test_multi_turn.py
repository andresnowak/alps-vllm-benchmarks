from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import multi_turn
from benchmark import load_suite


class MultiTurnSuiteTests(unittest.TestCase):
    def test_suite_points_carry_the_multi_turn_columns(self) -> None:
        points = load_suite(ROOT / "benchmarks" / "multi-turn.tsv")

        self.assertTrue(all(point["driver"] == multi_turn.DRIVER for point in points))
        for point in points:
            for field in multi_turn.FIELDS:
                self.assertIsInstance(point[field], int)
            # Every request stays inside the 32K context of the scenarios it targets.
            context = point["common_prefix_len"] + point["prefix_len"]
            context += point["turns"] // 2 * (point["input_len"] + point["output_len"])
            self.assertLess(context, 32768)

    def test_serve_points_keep_their_identity(self) -> None:
        points = load_suite(ROOT / "benchmarks" / "standard.tsv")

        self.assertFalse(any("driver" in point for point in points))

    def test_rejects_too_few_conversations_for_the_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "suite.tsv"
            suite.write_text(
                "name\tscenario\tdriver\tdataset\tinput_len\toutput_len\tprompts\t"
                "warmup_prompts\tconcurrency\tturns\tprefix_len\tcommon_prefix_len\t"
                "conversations\tactive\n"
                "chat\tmulti-turn\tmulti-turn\tbook\t64\t16\t100\t4\t4\t4\t100\t0\t10\t4\n"
            )

            with self.assertRaisesRegex(ValueError, "at least twice prompts"):
                load_suite(suite)

    def test_rejects_unknown_driver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "suite.tsv"
            suite.write_text(
                "name\tscenario\tdriver\tinput_len\toutput_len\tprompts\t"
                "warmup_prompts\tconcurrency\n"
                "x\ttest\tnonsense\t32\t8\t100\t4\t4\n"
            )

            with self.assertRaisesRegex(ValueError, "unknown driver"):
                load_suite(suite)


class MultiTurnRunTests(unittest.TestCase):
    def test_rotated_text_differs_by_seed_and_keeps_every_character(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "book.txt"
            source.write_text(" ".join(f"word{index}" for index in range(5000)))

            first, first_offset = multi_turn.rotated_text(source, 1, directory)
            again, again_offset = multi_turn.rotated_text(source, 1, directory)
            other, other_offset = multi_turn.rotated_text(source, 2, directory)

            self.assertEqual(first_offset, again_offset)
            self.assertNotEqual(first_offset, other_offset)
            self.assertNotEqual(first.read_text()[:200], other.read_text()[:200])
            self.assertEqual(
                sorted(first.read_text().split()), sorted(source.read_text().split())
            )

    def test_config_fixes_every_length_from_the_point(self) -> None:
        point = {
            "conversations": 8,
            "turns": 10,
            "common_prefix_len": 0,
            "prefix_len": 100,
            "input_len": 64,
            "output_len": 16,
        }

        config = multi_turn.conversations_config(point, Path("/data/book.txt"))

        self.assertEqual(config["num_conversations"], 8)
        self.assertEqual(config["text_files"], ["/data/book.txt"])
        self.assertEqual(config["prompt_input"]["num_turns"]["value"], 10)
        self.assertEqual(config["prompt_output"]["num_tokens"]["value"], 16)

    def test_result_has_the_metrics_benchmark_aggregates(self) -> None:
        stats = [
            {
                "ttft_ms": 100.0 + index,
                "tpot_ms": 20.0,
                "latency_ms": 1000.0,
                "start_time_ms": 500.0 * index,
                "input_num_tokens": 2000,
                "output_num_tokens": 256,
                "approx_cached_percent": 90.0,
            }
            for index in range(100)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result_file = Path(temporary) / "result.json"
            result_file.write_text(json.dumps(stats))

            result, total_input = multi_turn.load_result(
                result_file, {"name": "chat", "prompts": 100, "output_len": 256}
            )

        # The last request starts at 49.5 s and takes 1 s.
        self.assertAlmostEqual(result["duration"], 50.5)
        self.assertAlmostEqual(result["request_throughput"], 100 / 50.5)
        self.assertEqual(total_input, 200000)
        self.assertAlmostEqual(result["mean_ttft_ms"], 149.5)
        self.assertAlmostEqual(result["p99_ttft_ms"], 198.01)
        self.assertEqual(result["mean_tpot_ms"], 20.0)

    def test_result_rejects_missing_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_file = Path(temporary) / "result.json"
            result_file.write_text("[]")

            with self.assertRaisesRegex(RuntimeError, "completed=0/100"):
                multi_turn.load_result(
                    result_file, {"name": "chat", "prompts": 100, "output_len": 256}
                )


    def test_result_rejects_answers_shorter_than_output_len(self) -> None:
        # A client that misses reasoning tokens counts fewer than the server generated.
        stats = [
            {"ttft_ms": 1.0, "tpot_ms": 1.0, "latency_ms": 10.0, "start_time_ms": float(index),
             "input_num_tokens": 100, "output_num_tokens": 256 if index else 140,
             "approx_cached_percent": 0.0}
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result_file = Path(temporary) / "result.json"
            result_file.write_text(json.dumps(stats))
            with self.assertRaisesRegex(RuntimeError, "1 of 4 requests .* 256 output tokens .first: 140"):
                multi_turn.load_result(
                    result_file, {"name": "chat", "prompts": 4, "output_len": 256}
                )

if __name__ == "__main__":
    unittest.main()
