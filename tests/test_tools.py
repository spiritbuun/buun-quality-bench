from __future__ import annotations

import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRANSITIONS = ("fp16-t8", "t8-t4", "t4-t3", "t3-t2", "t2-t1")


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def write_dump(path: Path, values: np.ndarray, last_k: int = 0) -> None:
    values = np.asarray(values, dtype="<f4")
    n_chunk, n_pos = values.shape
    path.write_bytes(struct.pack("<ii", n_pos, n_chunk) + values.tobytes())
    Path(str(path) + ".meta").write_text(
        f"format_version=2\nn_pos={n_pos}\nn_chunk={n_chunk}\nscore_last_k={last_k}\n"
    )


def write_logits_base(path: Path) -> None:
    n_ctx, n_vocab, n_chunk = 6, 4, 2
    payload = bytearray(b"_logits_")
    payload.extend(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
    payload.extend(np.arange(n_ctx * n_chunk, dtype="<i4").tobytes())
    float_header = np.array([1.0, -4.0], dtype="<f4").view("<u2")
    row = np.concatenate([float_header, np.array([0, 1, 2, 3], dtype="<u2")])
    for _ in range(n_chunk * 2):
        payload.extend(row.tobytes())
    path.write_bytes(payload)


def kld_log(value: float) -> str:
    return f"""Mean PPL(Q)                   :   5.000000 ±   0.010000
Mean ln(PPL(Q)/PPL(base))     :   0.000000 ±   0.000100
Mean    KLD: {value:.6f} ± 0.000001
Maximum KLD: {value + 0.5:.6f}
99.9%   KLD: {value + 0.4:.6f}
99.0%   KLD: {value + 0.3:.6f}
95.0%   KLD: {value + 0.2:.6f}
Median  KLD: {value + 0.1:.6f}
RMS Δp    :  0.123 ± 0.001 %
Same top p: 99.500 ± 0.010 %
"""


class PricingTests(unittest.TestCase):
    def test_order_is_runtime_parseable_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "order.txt"
            result = run(
                "layer-pricing/build_degrade_order.py",
                "--cells",
                "layer-pricing/examples/panel_example.tsv",
                "--reliability",
                "layer-pricing/examples/reliability_example.tsv",
                "--tag",
                "demo",
                "--out",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            tokens = output.read_text().split()
            self.assertEqual(len(tokens), 40)
            self.assertTrue(all(token[0].isdigit() and ":" in token for token in tokens))

    def make_panel_dumps(self, directory: Path, fp16_anchor: float = 0.0) -> None:
        shape = (4, 8)
        grid = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 1e-7
        for transition_index, transition in enumerate(TRANSITIONS):
            high = transition.split("-", 1)[0]
            anchor_value = fp16_anchor if high == "fp16" else (transition_index + 1) * 1e-4
            anchor = np.full(shape, anchor_value, dtype=np.float32)
            write_dump(directory / f"pxr_demo_base_{high}.kld", anchor)
            for layer in range(3):
                for side_index, side in enumerate(("k", "v")):
                    excess = (layer + 1) * (side_index + 1) * (transition_index + 1) * 1e-5
                    write_dump(
                        directory / f"pxr_demo_{transition}_l{layer}_{side}.kld",
                        anchor + excess + grid,
                    )

    def test_dump_reducer_builds_complete_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_panel_dumps(directory)
            cells = directory / "cells.tsv"
            reliability = directory / "reliability.tsv"
            result = run(
                "layer-pricing/build_price_panel.py",
                "--dumps",
                str(directory),
                "--tag",
                "demo",
                "--campaign-id",
                "test-campaign",
                "--out-cells",
                str(cells),
                "--out-reliability",
                str(reliability),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with cells.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 30)
            with reliability.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 40)

    def test_dump_reducer_rejects_nonzero_self_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_panel_dumps(directory, fp16_anchor=1e-7)
            result = run(
                "layer-pricing/build_price_panel.py",
                "--dumps",
                str(directory),
                "--tag",
                "demo",
                "--campaign-id",
                "test-campaign",
                "--out-cells",
                str(directory / "cells.tsv"),
                "--out-reliability",
                str(directory / "reliability.tsv"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not exactly zero", result.stderr)

    def test_last_k_metadata_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_panel_dumps(directory)
            result = run(
                "layer-pricing/build_price_panel.py",
                "--dumps",
                str(directory),
                "--tag",
                "demo",
                "--campaign-id",
                "test-campaign",
                "--last-k",
                "4",
                "--out-cells",
                str(directory / "cells.tsv"),
                "--out-reliability",
                str(directory / "reliability.tsv"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("score_last_k=0, requested 4", result.stderr)


class ParserTests(unittest.TestCase):
    def test_raw_anchor_rejects_any_nonzero_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "anchor.kld"
            write_dump(path, np.array([[0.0, 1e-12]], dtype=np.float32))
            result = run("kld-panel/validate_kld_dump.py", "--exact-zero", str(path))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("nonzero values", result.stderr)

    def test_kld_parser_infers_names_and_depths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for candidate, offset in (("turbo3", 0.01), ("turbo3_tcq", 0.005)):
                for depth in (4096, 12288):
                    (directory / f"{candidate}_ctx{depth}.log").write_text(kld_log(offset))
            result = run("kld-panel/parse_kld.py", "--log-dir", str(directory))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("4096, 12288", result.stdout)
            self.assertIn("turbo3_tcq", result.stdout)

    def test_kld_parser_fails_on_incomplete_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "a_ctx4096.log").write_text(kld_log(0.01))
            (directory / "a_ctx8192.log").write_text(kld_log(0.02))
            (directory / "b_ctx4096.log").write_text(kld_log(0.03))
            result = run("kld-panel/parse_kld.py", "--log-dir", str(directory))
            self.assertEqual(result.returncode, 2)
            self.assertIn("incomplete candidates", result.stdout)


class HazardTests(unittest.TestCase):
    def test_offline_hazard_uses_probability_margin_and_bound_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dumps = directory / "dumps"
            dumps.mkdir()
            base = directory / "base.kld"
            margins = directory / "margins.f32"
            write_logits_base(base)
            anchor = np.zeros((2, 2), dtype=np.float32)
            write_dump(dumps / "pxr_demo_base_fp16.kld", anchor)
            write_dump(dumps / "pxr_demo_fp16-t8_l0_k.kld", anchor + 0.001)
            result = run(
                "hazard-bench/hazard_metrics.py",
                "--base",
                str(base),
                "--margins",
                str(margins),
                "--dumps",
                str(dumps),
                "--tag",
                "demo",
                "--campaign-id",
                "test-campaign",
                "--out",
                str(directory / "hazard"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((directory / "hazard_cells.tsv").is_file())
            metadata = json.loads(Path(str(margins) + ".meta").read_text())
            self.assertEqual(metadata["margin"], "top2_probability_gap")
            self.assertIn("sha256", metadata["base"])


class SweepTests(unittest.TestCase):
    def test_sweep_runs_anchor_and_refuses_changed_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base_directory = directory / "bases"
            base_directory.mkdir()
            base = base_directory / "base.kld"
            base.write_bytes(b"base")
            model = directory / "model.gguf"
            model.write_bytes(b"model")
            dataset = directory / "data.txt"
            dataset.write_text("data")
            run_directory = directory / "run"
            environment = {
                **os.environ,
                "BIN_DIR": str(ROOT / "tests"),
                "PPL_BIN": str(ROOT / "tests/fake_llama_tool.py"),
                "BENCH_BIN": str(ROOT / "tests/fake_llama_tool.py"),
                "MODEL": str(model),
                "DATASET": str(dataset),
                "BASE_DIR": str(base_directory),
                "KV_TIERS": "2048:base.kld:1",
                "TYPES": "turbo4",
            }
            first = subprocess.run(
                ["bash", "kld-panel/kv_kld_sweep.sh", str(run_directory)],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            rows = (run_directory / "results.tsv").read_text().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertIn("anchor_f16\t2048\t1\tOK\t0.000000", rows[1])

            changed = subprocess.run(
                ["bash", "kld-panel/kv_kld_sweep.sh", str(run_directory)],
                cwd=ROOT,
                env={**environment, "FA": "off"},
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(changed.returncode, 2)
            self.assertIn("Refusing to resume", changed.stderr)


class MarginTests(unittest.TestCase):
    def test_exact_outcomes_precede_confidence_margin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            files = []
            for label, exact, margin in (("wrong", False, 9.0), ("right", True, 1.0)):
                path = directory / f"{label}.jsonl"
                records = []
                for case_id in ("a", "b"):
                    records.append(
                        {
                            "id": case_id,
                            "label": label,
                            "case_sha256": f"fingerprint-{case_id}",
                            "expected_target": "TARGET",
                            "exact": exact,
                            "lp": [["TARGET", margin, 0.0]],
                        }
                    )
                path.write_text("".join(json.dumps(record) + "\n" for record in records))
                files.append(str(path))
            result = run("margin-bench/paired_margins.py", *files)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("B wins exact 2-0", result.stdout)
            self.assertNotIn("9.0000", result.stdout)

    def test_analyzer_reports_unscorable_exact_case_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "a",
                        "expected_target": "TARGET",
                        "exact": True,
                        "lp": [["FINAL_TARGET=TARGET", -0.1, None]],
                    }
                )
                + "\n"
            )
            result = run("margin-bench/router_probe/analyze_margins.py", str(path))
            self.assertEqual(result.returncode, 2)
            self.assertIn("results", result.stdout)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
