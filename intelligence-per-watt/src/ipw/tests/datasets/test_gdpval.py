"""Tests for datasets/gdpval.py — GDPvalDataset."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ipw.core.types import DatasetRecord


def _sample_row(task_id: str = "t1", with_rubric: bool = True) -> dict:
    return {
        "task_id": task_id,
        "sector": "Professional Services",
        "occupation": "Accountant",
        "prompt": "Prepare a Q4 tax summary based on the attached statements.",
        "reference_files": [],
        "reference_file_urls": [
            "https://example.com/q4_statements.pdf",
        ],
        "reference_file_hf_uris": [
            "hf://datasets/openai/gdpval/files/q4_statements.pdf",
        ],
        "deliverable_files": [],
        "deliverable_file_urls": [],
        "deliverable_file_hf_uris": [],
        "rubric_pretty": "Tax summary rubric",
        "rubric_json": json.dumps(
            [
                {
                    "rubric_item_id": "r1",
                    "criterion": "Includes total revenue.",
                    "score": 2,
                    "required": True,
                },
                {
                    "rubric_item_id": "r2",
                    "criterion": "Includes deductions.",
                    "score": 1,
                    "required": False,
                },
            ]
        ) if with_rubric else "",
    }


class TestGDPvalDataset:
    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_strict_references_fail_on_download_and_colliding_names(
        self, mock_load_dataset: MagicMock, mock_hf_download: MagicMock,
        tmp_path, monkeypatch,
    ) -> None:
        import pytest

        from ipw.datasets.gdpval import GDPvalDataset

        monkeypatch.setenv("IPW_REQUIRE_GDPVAL_REFERENCES", "1")
        row = _sample_row()
        mock_load_dataset.return_value.to_list.return_value = [row]
        mock_hf_download.side_effect = OSError("fixture unavailable")
        with pytest.raises(RuntimeError, match="Failed to download GDPVal reference"):
            GDPvalDataset(cache_dir=str(tmp_path / "cache"), revision="fixed-commit")

        source = tmp_path / "fixture.pdf"
        source.write_bytes(b"fixture")
        mock_hf_download.side_effect = None
        mock_hf_download.return_value = str(source)
        row["reference_file_hf_uris"] = [
            "hf://datasets/openai/gdpval/files/one/fixture.pdf",
            "hf://datasets/openai/gdpval/files/two/fixture.pdf",
        ]
        with pytest.raises(ValueError, match="duplicated"):
            GDPvalDataset(cache_dir=str(tmp_path / "cache"), revision="fixed-commit")

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_strict_references_replace_stale_cache_from_pinned_snapshot(
        self, mock_load_dataset: MagicMock, mock_hf_download: MagicMock,
        tmp_path, monkeypatch,
    ) -> None:
        import hashlib

        from ipw.datasets.gdpval import GDPvalDataset

        monkeypatch.setenv("IPW_REQUIRE_GDPVAL_REFERENCES", "1")
        mock_load_dataset.return_value.to_list.return_value = [_sample_row()]
        source = tmp_path / "fixture.pdf"
        source.write_bytes(b"pinned snapshot")
        mock_hf_download.return_value = str(source)
        cache = tmp_path / "cache"
        stale = cache / "t1" / "inputs" / "q4_statements.pdf"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale snapshot")

        record = next(GDPvalDataset(cache_dir=str(cache), revision="fixed-commit").iter_records())

        assert stale.read_bytes() == b"pinned snapshot"
        assert record.dataset_metadata["gdpval_reference_sha256"] == {
            stale.name: hashlib.sha256(b"pinned snapshot").hexdigest()
        }

    @patch("ipw.datasets.gdpval.load_dataset")
    def test_task_instance_occupation_matches_pinned_row(
        self, mock_load_dataset: MagicMock, tmp_path
    ) -> None:
        import pytest

        from ipw.datasets.gdpval import GDPvalDataset

        mock_load_dataset.return_value.to_list.return_value = [_sample_row("t1")]
        common = {"cache_dir": str(tmp_path / "cache"), "download_files": False}
        valid = GDPvalDataset(
            **common,
            task_instances=[{"task_id": "t1", "occupation": "Accountant"}],
        )
        assert len(list(valid.iter_records())) == 1

        for scheduled_occupation in ("Engineer", "", None):
            with pytest.raises(ValueError, match="occupation does not match pinned"):
                GDPvalDataset(
                    **common,
                    task_instances=[
                        {"task_id": "t1", "occupation": scheduled_occupation}
                    ],
                )

    @patch("ipw.datasets.gdpval.load_dataset")
    def test_gemini_judge_accepts_provider_key(
        self, mock_load_dataset: MagicMock, tmp_path, monkeypatch
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_load_dataset.return_value.to_list.return_value = []
        monkeypatch.delenv("IPW_EVAL_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "fixture-key")
        dataset = GDPvalDataset(cache_dir=str(tmp_path / "cache"), download_files=False)
        dataset.eval_client = "gemini"
        assert dataset.verify_requirements() == []

    @patch("ipw.datasets.gdpval.load_dataset")
    def test_revision_pins_hugging_face_dataset(
        self, mock_load_dataset: MagicMock, tmp_path
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row()]
        mock_load_dataset.return_value = mock_dataset

        GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            download_files=False,
            revision="fixed-commit",
        )

        mock_load_dataset.assert_called_once_with(
            "openai/gdpval", name="default", split="train", revision="fixed-commit"
        )

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_revision_pins_unversioned_reference_files(
        self, mock_load_dataset: MagicMock, mock_hf_download: MagicMock, tmp_path
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_load_dataset.return_value.to_list.return_value = [_sample_row()]
        source = tmp_path / "fixture.pdf"
        source.write_bytes(b"fixture")
        mock_hf_download.return_value = str(source)

        GDPvalDataset(cache_dir=str(tmp_path / "cache"), revision="fixed-commit")

        assert mock_hf_download.call_count == 1
        assert mock_hf_download.call_args.kwargs["revision"] == "fixed-commit"

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_revision_overrides_mutable_main_reference_uris(
        self, mock_load_dataset: MagicMock, mock_hf_download: MagicMock, tmp_path
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        row = _sample_row()
        row["reference_file_hf_uris"] = [
            "hf://datasets/openai/gdpval@main/reference_files/example.pdf"
        ]
        mock_load_dataset.return_value.to_list.return_value = [row]
        source = tmp_path / "fixture.pdf"
        source.write_bytes(b"fixture")
        mock_hf_download.return_value = str(source)

        GDPvalDataset(cache_dir=str(tmp_path / "cache"), revision="fixed-commit")

        assert mock_hf_download.call_args.kwargs["revision"] == "fixed-commit"

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_iter_records_yields_dataset_records(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1"), _sample_row("t2")]
        mock_load_dataset.return_value = mock_dataset

        # Pretend the HF download returns a stub file we can copy
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"stub")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(cache_dir=str(tmp_path / "cache"))
        records = list(dataset.iter_records())

        assert len(records) == 2
        assert all(isinstance(r, DatasetRecord) for r in records)
        assert "Q4 tax summary" in records[0].problem
        assert records[0].subject == "Accountant"
        # Each task's inputs dir should be unique
        d1 = records[0].dataset_metadata["gdpval_inputs_dir"]
        d2 = records[1].dataset_metadata["gdpval_inputs_dir"]
        assert d1 != d2

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_max_samples_limits_records(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row(f"t{i}") for i in range(10)]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            max_samples=3,
        )
        assert dataset.size() == 3

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_metadata_carries_rubric_and_occupation(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(cache_dir=str(tmp_path / "cache"))
        record = list(dataset.iter_records())[0]
        meta = record.dataset_metadata
        assert meta["dataset_name"] == "GDPval"
        assert meta["task_id"] == "t1"
        assert meta["occupation"] == "Accountant"
        assert meta["sector"] == "Professional Services"
        assert meta["rubric_json"]  # not empty
        assert meta["include_rubric_in_prompt"] is False
        assert "## Evaluation rubric" not in record.problem
        assert "Tax summary rubric" not in record.problem

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_include_rubric_in_prompt_adds_rubric_section(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            include_rubric_in_prompt=True,
        )
        record = list(dataset.iter_records())[0]

        assert record.dataset_metadata["include_rubric_in_prompt"] is True
        assert "## Evaluation rubric" in record.problem
        assert "Tax summary rubric" in record.problem
        assert "reference files as the source of truth" in record.problem
        assert "do not invent data" in record.problem
        assert "Keep verification compact" in record.problem
        assert "repair only that concrete failure" in record.problem
        assert "compatible filenames" in record.problem
        assert "compatibility artifact" in record.problem
        assert "sampled rows and full population/reference rows clearly separated" in record.problem
        assert "## Task" in record.problem
        assert "Run-specific deliverable rescue instructions" not in record.problem

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_deliverable_rescue_instructions_are_opt_in(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
        monkeypatch,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)
        monkeypatch.setenv(
            "IPW_GDPVAL_DELIVERABLE_RESCUE_INSTRUCTIONS",
            "Create the workbook before further analysis.",
        )

        dataset = GDPvalDataset(cache_dir=str(tmp_path / "cache"))
        record = list(dataset.iter_records())[0]

        assert "## Run-specific deliverable rescue instructions" in record.problem
        assert "Create the workbook before further analysis." in record.problem
        assert record.dataset_metadata["deliverable_rescue_instructions"] is True

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_task_rescue_hints_are_keyed_by_task_id(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
        monkeypatch,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1"), _sample_row("t2")]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)
        hints = tmp_path / "rescue_hints.json"
        hints.write_text(
            json.dumps(
                {
                    "t2": {
                        "failure_cause": "Previous run only created probe.py.",
                        "required_action": "Create the requested workbook first.",
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("IPW_GDPVAL_TASK_RESCUE_HINTS_FILE", str(hints))

        dataset = GDPvalDataset(cache_dir=str(tmp_path / "cache"))
        records = list(dataset.iter_records())

        assert "Task-specific rescue hint from prior run" not in records[0].problem
        assert records[0].dataset_metadata["task_rescue_hint"] is False
        assert "Task-specific rescue hint from prior run" in records[1].problem
        assert "Previous run only created probe.py." in records[1].problem
        assert "Create the requested workbook first." in records[1].problem
        assert records[1].dataset_metadata["task_rescue_hint"] is True

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_terminal_tool_preference_removes_file_editor_from_prompt(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            tool_preference="terminal",
        )
        record = list(dataset.iter_records())[0]

        assert "You have `terminal` (bash)." in record.problem
        assert "`file_editor` is not available in this run" in record.problem
        assert record.dataset_metadata["tool_preference"] == "terminal"

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_prompt_lists_expected_deliverable_filenames(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        row = _sample_row("t1")
        row["deliverable_file_hf_uris"] = [
            "hf://datasets/openai/gdpval@main/deliverables/t1/Sample%20Workbook.xlsx",
            "hf://datasets/openai/gdpval@main/deliverables/t1/Briefing%20Note.docx",
        ]
        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [row]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(cache_dir=str(tmp_path / "cache"))
        record = list(dataset.iter_records())[0]

        assert "## Expected output deliverables" in record.problem
        assert "Sample Workbook.xlsx" in record.problem
        assert "Briefing Note.docx" in record.problem
        assert "case/spacing variants or similarly descriptive filenames" in record.problem
        assert "valid starter file before broad exploration" in record.problem
        assert "create all required deliverables in one compact script" in record.problem
        assert record.dataset_metadata["deliverable_files"] == [
            "Sample Workbook.xlsx",
            "Briefing Note.docx",
        ]

    def test_hf_uri_parser_handles_revision_and_urlencoding(self) -> None:
        from ipw.datasets.gdpval import _hf_uri_to_repo_path

        # @revision + URL-encoded filename (real gdpval URI shape)
        repo, path, rev = _hf_uri_to_repo_path(
            "hf://datasets/openai/gdpval@main/reference_files/abc/Population%20v2.xlsx"
        )
        assert repo == "openai/gdpval"
        assert rev == "main"
        assert path == "reference_files/abc/Population v2.xlsx"

        # No revision, no encoding
        repo, path, rev = _hf_uri_to_repo_path(
            "hf://datasets/openai/gdpval/files/x.pdf"
        )
        assert repo == "openai/gdpval"
        assert rev is None
        assert path == "files/x.pdf"

        # Non-HF URIs return None
        assert _hf_uri_to_repo_path("https://example.com/x.pdf") is None
        assert _hf_uri_to_repo_path("") is None

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_skip_download_when_disabled(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset

        dataset = GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            download_files=False,
        )
        records = list(dataset.iter_records())
        mock_hf_download.assert_not_called()
        # Falls back to URLs in the prompt
        assert "example.com" in records[0].problem

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_task_instances_preserve_duplicates_and_unique_instance_ids(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1"), _sample_row("t2")]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)

        dataset = GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            task_instances=[
                {
                    "task_id": "t2",
                    "instance_id": "t2::rep00",
                    "repeat_index": 0,
                    "repeat_count": 4,
                    "occupation_batch_index": 7,
                    "occupation_batch_id": "occ_b07",
                    "occupation_task_position": 1,
                    "source_task_index": 12,
                },
                {
                    "task_id": "t2",
                    "instance_id": "t2::rep01",
                    "repeat_index": 1,
                    "repeat_count": 4,
                    "occupation_batch_index": 7,
                    "occupation_batch_id": "occ_b07",
                    "occupation_task_position": 1,
                    "source_task_index": 12,
                },
                {
                    "task_id": "t1",
                    "instance_id": "t1::rep00",
                    "repeat_index": 0,
                    "repeat_count": 4,
                    "occupation_batch_index": 0,
                    "occupation_batch_id": "occ_b00",
                    "occupation_task_position": 0,
                    "source_task_index": 0,
                },
            ],
        )

        records = list(dataset.iter_records())
        assert [record.dataset_metadata["task_id"] for record in records] == [
            "t2",
            "t2",
            "t1",
        ]
        assert [record.dataset_metadata["instance_id"] for record in records] == [
            "t2::rep00",
            "t2::rep01",
            "t1::rep00",
        ]
        assert records[0].problem == records[1].problem
        assert records[0].dataset_metadata["repeat_index"] == 0
        assert records[1].dataset_metadata["repeat_index"] == 1
        assert records[0].dataset_metadata["repeat_count"] == 4
        assert records[0].dataset_metadata["occupation_batch_id"] == "occ_b07"
        assert records[0].dataset_metadata["repeated_instance"] is True
        assert records[0].dataset_metadata["task_instance_position"] == 0

    @patch("ipw.datasets.gdpval.load_dataset")
    def test_task_instances_cannot_be_combined_with_task_ids(
        self,
        mock_load_dataset: MagicMock,
        tmp_path,
    ) -> None:
        import pytest

        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset

        with pytest.raises(ValueError, match="task_instances cannot be combined"):
            GDPvalDataset(
                cache_dir=str(tmp_path / "cache"),
                task_ids=["t1"],
                task_instances=[{"task_id": "t1"}],
            )

    @patch("ipw.datasets.gdpval.hf_hub_download")
    @patch("ipw.datasets.gdpval.load_dataset")
    def test_task_instances_file_loads_repeated_instances(
        self,
        mock_load_dataset: MagicMock,
        mock_hf_download: MagicMock,
        tmp_path,
    ) -> None:
        from ipw.datasets.gdpval import GDPvalDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [_sample_row("t1")]
        mock_load_dataset.return_value = mock_dataset
        stub = tmp_path / "_stub.pdf"
        stub.write_bytes(b"x")
        mock_hf_download.return_value = str(stub)
        instances = tmp_path / "instances.json"
        instances.write_text(
            json.dumps(
                [
                    {"task_id": "t1", "instance_id": "t1::rep00"},
                    {"task_id": "t1", "instance_id": "t1::rep01"},
                ]
            ),
            encoding="utf-8",
        )

        dataset = GDPvalDataset(
            cache_dir=str(tmp_path / "cache"),
            task_instances_file=str(instances),
        )
        records = list(dataset.iter_records())

        assert [record.dataset_metadata["instance_id"] for record in records] == [
            "t1::rep00",
            "t1::rep01",
        ]
