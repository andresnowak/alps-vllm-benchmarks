#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets>=3", "transformers>=5.15", "jinja2"]
# ///
"""Build the real-text prompt files for suite rows with dataset=mixed.

Expert routing depends on the text, so random token IDs do not load the experts like real
traffic. This script takes prompts from seven public sources in turn and writes
OUT_DIR/mixed-in<length>.jsonl, or with --source NAME only from that source, into
OUT_DIR/NAME-in<length>.jsonl (suite rows with dataset=NAME). Each prompt has exactly <length> tokens under the chat
template, minus the template's fixed overhead; benchmark.py checks this count on the server.

Run on a login node (needs internet):
  scripts/make-dataset.py --tokenizer models/GLM-5.3 --lengths 256 1024 4096 16384 --out-dir datasets
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

# name -> (Hugging Face dataset, config, text of one record). None of them is gated.
SOURCES = {
    "web_en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", lambda r: r["text"]),
    "web_de": ("HuggingFaceFW/fineweb-2", "deu_Latn", lambda r: r["text"]),
    "web_fr": ("HuggingFaceFW/fineweb-2", "fra_Latn", lambda r: r["text"]),
    "web_zh": ("HuggingFaceFW/fineweb-2", "cmn_Hani", lambda r: r["text"]),
    "code": (
        "likaixin/InstructCoder",
        None,
        lambda r: "\n\n".join(
            filter(None, (r["instruction"], r["input"], r["output"]))
        ),
    ),
    "math": (
        "AI-MO/NuminaMath-CoT",
        None,
        lambda r: f"{r['problem']}\n\n{r['solution']}",
    ),
    "chat": (
        "allenai/WildChat-1M",
        None,
        lambda r: "\n\n".join(turn["content"] for turn in r["conversation"]),
    ),
}
SHUFFLE_BUFFER = 1000


def documents(name: str, seed: int) -> Iterator[str]:
    repo, config, text = SOURCES[name]
    stream = load_dataset(repo, config, split="train", streaming=True)
    for record in stream.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER):
        if document := text(record).strip():
            yield document


class Counter:
    """Count prompt tokens the way the server does, without the chat template overhead."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        probe = "token alignment probe"
        self.overhead = self.templated(probe) - len(self.encode(probe))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def templated(self, text: str) -> int:
        messages = [{"role": "user", "content": text}]
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return len(self.encode(rendered))

    def prompt(self, stream: Iterator[str], target: int) -> str:
        while True:
            ids: list[int] = []
            while len(ids) < target + max(16, target // 50):
                ids += self.encode(next(stream) + "\n\n")
            # Decoding and re-encoding can shift the count, so move the cut until it matches.
            # A cut inside a multi-byte character decodes to U+FFFD; strip it.
            cut = target
            for _ in range(8):
                text = self.tokenizer.decode(ids[:cut]).strip().strip("�").strip()
                length = self.templated(text) - self.overhead
                if length == target:
                    return text
                cut += target - length
                if not 0 < cut <= len(ids):
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True, help="model directory or HF id")
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", choices=sorted(SOURCES),
                        help="take every prompt from this source (default: all sources in turn)")
    args = parser.parse_args()
    sources = [args.source] if args.source else list(SOURCES)
    stem = args.source or "mixed"

    counter = Counter(AutoTokenizer.from_pretrained(args.tokenizer))
    streams = {name: documents(name, args.seed) for name in sources}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for length in args.lengths:
        path = args.out_dir / f"{stem}-in{length}.jsonl"
        names = itertools.islice(itertools.cycle(sources), args.num_prompts)
        lines = [
            json.dumps(
                {"prompt": counter.prompt(streams[name], length), "source": name},
                ensure_ascii=False,
            )
            for name in names
        ]
        path.with_suffix(".tmp").write_text("\n".join(lines) + "\n")
        path.with_suffix(".tmp").replace(path)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"wrote {path}")

    build = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": str(args.tokenizer),
        "chat_template_overhead": counter.overhead,
        "sources": {name: SOURCES[name][:2] for name in sources},
        "seed": args.seed,
        "num_prompts": args.num_prompts,
        "files": files,
    }
    (args.out_dir / f"{stem}.build.json").write_text(json.dumps(build, indent=2) + "\n")


if __name__ == "__main__":
    main()
