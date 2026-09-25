"""Paths, seed and the shared CLI surface.

Every path is derived from the repo root, which is found relative to this file rather
than the cwd, so the pipeline behaves the same whether it is run from student_resource/
locally or from a clone on Kaggle. No absolute path is ever hard-coded.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

SEED = 42

# src/ -> business_entity_resolution/ -> code/ -> repo root (student_resource/)
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATA_DIR = REPO_ROOT / "dataset"
DEFAULT_OUT_DIR = REPO_ROOT / "output"


@dataclass(frozen=True)
class Paths:
    """Resolved locations for one run. Nothing else builds a dataset path by hand."""

    data_dir: Path
    out_dir: Path
    repo_root: Path = REPO_ROOT

    # --- given data -------------------------------------------------------
    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def train_s1(self) -> Path:
        return self.train_dir / "train_source1.tsv"

    @property
    def train_s2(self) -> Path:
        return self.train_dir / "train_source2.tsv"

    @property
    def train_s3(self) -> Path:
        return self.train_dir / "train_source3.tsv"

    @property
    def train_gt(self) -> Path:
        return self.train_dir / "train_ground_truth.tsv"

    @property
    def test_s1(self) -> Path:
        return self.test_dir / "test_source1.tsv"

    @property
    def test_s2(self) -> Path:
        return self.test_dir / "test_source2.tsv"

    @property
    def test_s3(self) -> Path:
        return self.test_dir / "test_source3.tsv"

    # --- produced ---------------------------------------------------------
    @property
    def artifacts(self) -> Path:
        return self.repo_root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.repo_root / "reports"

    @property
    def matching_results(self) -> Path:
        return self.out_dir / "matching_results.tsv"

    @property
    def candidate_pairs(self) -> Path:
        return self.out_dir / "candidate_pairs.tsv"

    @property
    def validator(self) -> Path:
        return self.repo_root / "utils" / "validate_submission.py"

    def all_sources(self) -> dict[str, Path]:
        """The 7 given files, in the order the load-check reports them."""
        return {
            "train_source1": self.train_s1,
            "train_source2": self.train_s2,
            "train_source3": self.train_s3,
            "train_ground_truth": self.train_gt,
            "test_source1": self.test_s1,
            "test_source2": self.test_s2,
            "test_source3": self.test_s3,
        }

    def ensure_dirs(self) -> None:
        for d in (self.out_dir, self.artifacts, self.reports):
            d.mkdir(parents=True, exist_ok=True)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Give every entry point the same flags, so steps stay interchangeable."""
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory holding train/ and test/ (default: <repo>/dataset).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where submission TSVs are written (default: <repo>/output).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Smoke-run on N Source-1 entities instead of the full data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cached artifacts instead of reusing them.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def paths_from_args(args: argparse.Namespace) -> Paths:
    return Paths(data_dir=Path(args.data_dir), out_dir=Path(args.out_dir))


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we might touch. numpy is imported lazily to keep imports cheap."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
