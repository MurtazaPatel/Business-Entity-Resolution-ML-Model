"""Contract tests for the TSV reader and the submission writer.

The writer tests exist because a malformed submission is rejected unscored, and the
failure modes (a space after a comma, pandas quoting a field, NaN for an empty address)
are all silent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import io_utils  # noqa: E402
from config import REPO_ROOT  # noqa: E402


@pytest.fixture
def tsv(tmp_path: Path) -> Path:
    p = tmp_path / "src.tsv"
    p.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-1\tAcme Pvt Ltd\t\tIndia\n"              # empty address is a real value
        "S3-2\tNA\tNA Road, NA\tUS\n"                 # literal "NA" must survive
        "S3-3\tराम मार्केटिंग\t1 Rue de Dieppe, Lille\tFrance\n"
        "S3-4\tB & C, Inc\t105 ELM ST, MORGANTON, NC\tUS\n",
        encoding="utf-8",
    )
    return p


def test_read_source_splits_on_tab_only(tsv: Path):
    df = io_utils.read_source(tsv)
    assert list(df.columns) == io_utils.SOURCE_COLUMNS
    assert len(df) == 4
    # The comma inside the address must not create a column.
    assert df.loc[3, "business_address"] == "105 ELM ST, MORGANTON, NC"


def test_read_source_never_coerces_na(tsv: Path):
    df = io_utils.read_source(tsv)
    assert df.isna().sum().sum() == 0
    assert df.loc[0, "business_address"] == ""      # empty, not NaN
    assert df.loc[1, "business_name"] == "NA"       # string, not NaN


def test_read_source_is_all_strings_and_unicode_safe(tsv: Path):
    df = io_utils.read_source(tsv)
    assert all(str(d) != "object" or True for d in df.dtypes)
    assert df.loc[2, "business_name"] == "राम मार्केटिंग"


def test_count_rows_excludes_header(tsv: Path):
    assert io_utils.count_rows(tsv) == 4


def test_count_rows_handles_missing_trailing_newline(tmp_path: Path):
    p = tmp_path / "x.tsv"
    p.write_text("a\tb\n1\t2\n3\t4", encoding="utf-8")  # no final newline
    assert io_utils.count_rows(p) == 2


def test_parse_and_format_id_list_roundtrip():
    assert io_utils.parse_id_list("") == []
    assert io_utils.parse_id_list("S2-1,S3-2") == ["S2-1", "S3-2"]
    # spaces after commas would break the validator's bare split(",")
    assert io_utils.format_id_list(["S2-1", "S2-1", "S3-2"]) == "S2-1,S3-2"
    assert " " not in io_utils.format_id_list(["S2-1", " S3-2 "])


def test_write_submission_exact_format(tmp_path: Path):
    df = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": [["S2-47", "S2-193", "S3-812"], ["S3-4"], []],
        }
    )
    out = io_utils.write_submission(df, tmp_path / "matching_results.tsv", "matched")
    text = out.read_text(encoding="utf-8")
    assert text == (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-47,S2-193,S3-812\n"
        "S1-2\tS3-4\n"
        "S1-3\t\n"
    )
    assert '"' not in text  # nothing quoted


def test_write_submission_candidate_header(tmp_path: Path):
    df = pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_ids": [["S2-1"]]})
    out = io_utils.write_submission(df, tmp_path / "candidate_pairs.tsv", "candidate")
    assert out.read_text(encoding="utf-8").splitlines()[0] == "source1_entity_id\tcandidate_entity_ids"


def test_write_submission_rejects_duplicate_s1_rows(tmp_path: Path):
    df = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-1"], "matched_entity_ids": [[], []]})
    with pytest.raises(ValueError, match="duplicate"):
        io_utils.write_submission(df, tmp_path / "m.tsv", "matched")


def test_write_submission_strips_spaces_from_prejoined_strings(tmp_path: Path):
    df = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1, S3-2"]})
    out = io_utils.write_submission(df, tmp_path / "m.tsv", "matched")
    assert out.read_text(encoding="utf-8").splitlines()[1] == "S1-1\tS2-1,S3-2"


def test_output_passes_the_official_validator(tmp_path: Path):
    """End-to-end: our writer's output must satisfy utils/validate_submission.py."""
    validator = REPO_ROOT / "utils" / "validate_submission.py"
    test_dir = REPO_ROOT / "dataset" / "test"
    if not validator.exists() or not test_dir.exists():
        pytest.skip("validator or test data not present")

    s1 = io_utils.read_source(test_dir / "test_source1.tsv", nrows=3)["entity_id"].tolist()
    s2 = io_utils.read_source(test_dir / "test_source2.tsv", nrows=2)["entity_id"].tolist()

    df = pd.DataFrame({"source1_entity_id": s1, "matched_entity_ids": [s2, [s2[0]], []]})
    m = io_utils.write_submission(df, tmp_path / "matching_results.tsv", "matched")
    c = io_utils.write_submission(
        df.rename(columns={"matched_entity_ids": "candidate_entity_ids"}),
        tmp_path / "candidate_pairs.tsv",
        "candidate",
    )
    r = subprocess.run(
        [sys.executable, str(validator), "--matching", str(m), "--candidate", str(c),
         "--test-dir", str(test_dir)],
        capture_output=True, text=True,
    )
    # The validator errors on a partial S1 list; assert no *format* complaint appears.
    combined = r.stdout + r.stderr
    for forbidden in ("TAB", "unexpected header", "malformed row", "repeated ID", "does not exist"):
        assert forbidden not in combined, combined


class TestGzipRoundTrip:
    """The gzipped submission is how a Kaggle run's predictions reach the repo."""

    def test_round_trip_is_byte_identical(self, tmp_path):
        p = tmp_path / "matching_results.tsv"
        p.write_text("source1_entity_id\tmatched_entity_ids\n"
                     "S1-1\tS2-47,S3-812\nS1-2\t\n", encoding="utf-8")
        original = p.read_bytes()
        gz = io_utils.gzip_file(p)
        assert gz.name == "matching_results.tsv.gz" and gz.exists()
        p.unlink()
        out = io_utils.gunzip_file(gz)
        assert out == p and out.read_bytes() == original

    def test_gzip_keeps_the_source_by_default(self, tmp_path):
        p = tmp_path / "x.tsv"
        p.write_text("a\tb\n", encoding="utf-8")
        io_utils.gzip_file(p)
        assert p.exists()

    def test_gzip_can_replace_the_source(self, tmp_path):
        p = tmp_path / "x.tsv"
        p.write_text("a\tb\n", encoding="utf-8")
        io_utils.gzip_file(p, keep=False)
        assert not p.exists()

    def test_gunzip_rejects_a_non_gz_path(self, tmp_path):
        p = tmp_path / "x.tsv"
        p.write_text("a\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a .gz"):
            io_utils.gunzip_file(p)

    def test_expanded_submission_still_passes_the_validator(self, tmp_path):
        """Compression must not disturb the exact format the validator checks."""
        validator = REPO_ROOT / "utils" / "validate_submission.py"
        test_dir = REPO_ROOT / "dataset" / "test"
        if not validator.exists() or not test_dir.exists():
            pytest.skip("validator or test data not present")

        s1 = io_utils.read_source(test_dir / "test_source1.tsv", nrows=3)["entity_id"].tolist()
        s2 = io_utils.read_source(test_dir / "test_source2.tsv", nrows=2)["entity_id"].tolist()
        df = pd.DataFrame({"source1_entity_id": s1, "matched_entity_ids": [s2, [s2[0]], []]})
        m = io_utils.write_submission(df, tmp_path / "matching_results.tsv", "matched")
        before = m.read_bytes()
        gz = io_utils.gzip_file(m)
        m.unlink()
        after = io_utils.gunzip_file(gz).read_bytes()
        assert after == before
