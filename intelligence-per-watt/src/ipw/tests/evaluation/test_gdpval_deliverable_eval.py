"""Tests for deliverable-aware GDPval judging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class _JudgeClient:
    def __init__(self, response: str | list[str]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.last_user_prompt = ""
        self.calls = 0

    def chat(self, system_prompt: str, user_prompt: str, **kwargs):
        self.last_user_prompt = user_prompt
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class _FakeVisionClient:
    model = "vision-test"

    def _chat_completion(self, model: str, prompt: str, **kwargs):
        class Response:
            content = "A red square frame with no visible text."

        return Response()


def _write_workbook(path, value: str = "candidate deliverable") -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active["A1"] = value
    wb.save(path)


def test_gdpval_deliverable_registered_by_evaluation_package_import() -> None:
    import ipw.evaluation  # noqa: F401
    from ipw.core.registry import EvaluationRegistry

    assert EvaluationRegistry.has("gdpval-deliverable")
    assert EvaluationRegistry.has("gdpval-aa-single")


def test_deliverable_judge_reads_output_files(tmp_path) -> None:
    from ipw.evaluation.gdpval_deliverable import GdpvalDeliverableHandler

    output = tmp_path / "summary.txt"
    output.write_text("Revenue total: 123\nDeductions: 45\n", encoding="utf-8")

    client = _JudgeClient(
        json.dumps(
            {
                "verdict": "pass",
                "score": 1.0,
                "missing_or_broken_deliverables": [],
                "criteria": [
                    {
                        "rubric_item_id": "r1",
                        "criterion": "Includes total revenue.",
                        "satisfied": True,
                        "score_awarded": 2,
                        "score_possible": 2,
                        "evidence": "Revenue total is present.",
                    }
                ],
                "notes": "Good.",
            }
        )
    )
    handler = GdpvalDeliverableHandler(client=client)

    is_correct, meta = handler.evaluate(
        problem="Prepare a summary.",
        reference="",
        model_answer="Created summary.txt",
        metadata={
            "gdpval_submitted_files": [str(output)],
            "rubric_json": json.dumps(
                [{"rubric_item_id": "r1", "criterion": "Includes total revenue.", "score": 2}]
            ),
        },
    )

    assert is_correct is True
    assert meta["score"] == 1.0
    assert "Revenue total" in client.last_user_prompt
    assert meta["deliverables"][0]["sha256"]


def test_deliverable_judge_fails_when_no_files() -> None:
    from ipw.evaluation.gdpval_deliverable import GdpvalDeliverableHandler

    handler = GdpvalDeliverableHandler(client=_JudgeClient("{}"))
    is_correct, meta = handler.evaluate(
        problem="Task",
        reference="",
        model_answer="No files",
        metadata={
            "gdpval_submitted_files": [],
            "rubric_json": json.dumps([{"criterion": "Has output", "score": 1}]),
        },
    )

    assert is_correct is False
    assert meta["reason"] == "no_deliverables_found"


def test_deliverable_judge_repairs_non_json_response(tmp_path) -> None:
    from ipw.evaluation.gdpval_deliverable import GdpvalDeliverableHandler

    output = tmp_path / "summary.txt"
    output.write_text("Revenue total: 123\n", encoding="utf-8")

    client = _JudgeClient(
        [
            "This submission passes. It includes the revenue total.",
            json.dumps(
                {
                    "verdict": "pass",
                    "score": 1.0,
                    "missing_or_broken_deliverables": [],
                    "criteria": [
                        {
                            "rubric_item_id": "r1",
                            "criterion": "Includes total revenue.",
                            "satisfied": True,
                            "score_awarded": 1,
                            "score_possible": 1,
                            "evidence": "Revenue total is present.",
                        }
                    ],
                    "notes": "Pass.",
                }
            ),
        ]
    )

    handler = GdpvalDeliverableHandler(client=client)
    is_correct, meta = handler.evaluate(
        problem="Prepare a summary.",
        reference="",
        model_answer="Created summary.txt",
        metadata={
            "gdpval_submitted_files": [str(output)],
            "rubric_json": json.dumps([{"rubric_item_id": "r1", "criterion": "Includes total revenue.", "score": 1}]),
        },
    )

    assert is_correct is True
    assert client.calls == 2
    assert meta["repair_output"]


def test_deliverable_judge_refuses_blind_visual_scoring(tmp_path) -> None:
    from ipw.evaluation.gdpval_deliverable import GdpvalDeliverableHandler

    pillow = pytest.importorskip("PIL.Image")

    image_path = tmp_path / "visual.png"
    image = pillow.new("RGB", (24, 16), color="white")
    image.save(image_path)

    client = _JudgeClient("{}")
    handler = GdpvalDeliverableHandler(client=client)
    is_correct, meta = handler.evaluate(
        problem="Create a visual design as a PNG.",
        reference="",
        model_answer="Created visual.png",
        metadata={
            "gdpval_submitted_files": [str(image_path)],
            "rubric_json": json.dumps(
                [
                    {
                        "rubric_item_id": "r1",
                        "criterion": "Uses the requested visual design.",
                        "score": 1,
                    }
                ]
            ),
        },
    )

    assert is_correct is None
    assert meta["reason"] == "visual_evidence_unavailable"
    assert meta["failure_type"] == "harness_validation"
    assert client.calls == 0


def test_deliverable_extractor_handles_image_and_audio_metadata(tmp_path) -> None:
    from ipw.evaluation.deliverables import inspect_file

    pillow = pytest.importorskip("PIL.Image")

    png = tmp_path / "chart.png"
    image = pillow.new("RGB", (24, 16), color="white")
    image.save(png)

    wav = tmp_path / "clip.wav"
    import wave

    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\x00\x00" * 800)

    image_evidence = inspect_file(png)
    audio_evidence = inspect_file(wav)

    assert image_evidence.kind == "image"
    assert image_evidence.metadata["width"] == 24
    assert image_evidence.metadata["height"] == 16
    assert audio_evidence.kind == "audio"
    assert audio_evidence.metadata["duration_seconds"] > 0
    assert audio_evidence.metadata["sample_rate"] == 8000
    assert audio_evidence.metadata["channels"] == 1
    assert audio_evidence.metadata["bits_per_sample"] == 16
    assert audio_evidence.metadata["pcm_stats"]["measured_frames"] == 800


def test_deliverable_extractor_adds_vision_description(tmp_path) -> None:
    from ipw.evaluation.deliverables import inspect_file

    pillow = pytest.importorskip("PIL.Image")

    png = tmp_path / "frame.png"
    image = pillow.new("RGB", (12, 12), color="red")
    image.save(png)

    evidence = inspect_file(png, vision_client=_FakeVisionClient())

    assert evidence.kind == "image"
    assert "A red square frame" in evidence.text
    assert evidence.metadata["vision_description"] == "included"


def test_deliverable_extractor_recurses_into_zip(tmp_path) -> None:
    import zipfile

    from ipw.evaluation.deliverables import inspect_file

    archive = tmp_path / "backend.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("main.tf", "resource \"aws_lambda_function\" \"contact\" {}\n")
        zf.writestr("exports.js", "export async function handler() {}\n")

    evidence = inspect_file(archive)

    assert evidence.kind == "archive"
    assert evidence.metadata["entry_count"] == 2
    assert {child.name for child in evidence.children} == {"exports.js", "main.tf"}
    assert any("aws_lambda_function" in child.text for child in evidence.children)


def test_deliverable_extractor_reads_step_files_as_text(tmp_path) -> None:
    from ipw.evaluation.deliverables import inspect_file

    step = tmp_path / "assembly.step"
    step.write_text(
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('toasty assembly'),'2;1');\n"
        "ENDSEC;\n"
        "DATA;\n"
        "#1=CARTESIAN_POINT('',(0.,0.,0.));\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    evidence = inspect_file(step)

    assert evidence.kind == "text"
    assert evidence.errors == []
    assert "CARTESIAN_POINT" in evidence.text


def test_deliverable_extractor_reads_notebook_cells(tmp_path) -> None:
    from ipw.evaluation.deliverables import inspect_file

    notebook = tmp_path / "americanoptionpricing.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.11"},
                },
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": ["# American option pricing\n", "Binomial lattice implementation."],
                    },
                    {
                        "cell_type": "code",
                        "source": ["def price_american_put(S, K, r, sigma, T, steps):\n", "    return 1.23\n"],
                        "outputs": [{"data": {"text/plain": ["1.23"]}}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    evidence = inspect_file(notebook)

    assert evidence.kind == "notebook"
    assert evidence.errors == []
    assert evidence.metadata["cell_count"] == 2
    assert evidence.metadata["kernelspec"]["name"] == "python3"
    assert "American option pricing" in evidence.text
    assert "def price_american_put" in evidence.text
    assert "1.23" in evidence.text


def test_deliverable_contract_detects_missing_expected_type(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    (tmp_path / "guide.html").write_text("<h1>Draft</h1>", encoding="utf-8")

    validation = validate_deliverable_contract(
        paths=[tmp_path],
        problem="Create a guide delivered as a PDF file.",
        metadata={},
    )

    assert validation["status"] == "invalid"
    assert any(
        issue["code"] == "expected_file_type_missing"
        for issue in validation["issues"]
    )


def test_deliverable_contract_uses_task_section_for_expected_type(tmp_path) -> None:
    from PIL import Image

    from ipw.evaluation.deliverables import validate_deliverable_contract

    Image.new("RGB", (20, 20), color="purple").save(tmp_path / "moodboard.png")

    validation = validate_deliverable_contract(
        paths=[tmp_path],
        problem=(
            "## Evaluation rubric\n"
            "Generic wrapper mentions PDF, DOCX, XLSX, PPTX, and MP4.\n\n"
            "## Task\n"
            "Create a moodboard (in PNG format) based on the meeting notes.\n\n"
            "## Tools and workflow\n"
            "Use available tools."
        ),
        metadata={},
    )

    assert validation["status"] == "ok"
    assert validation["expected_extensions"] == [".png"]
    assert validation["primary_files"] == [str(tmp_path / "moodboard.png")]


def test_deliverable_contract_still_rejects_html_for_excel_task(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    (tmp_path / "sp500.html").write_text("<table><tr><td>MMM</td></tr></table>", encoding="utf-8")

    validation = validate_deliverable_contract(
        paths=[tmp_path],
        problem=(
            "## Evaluation rubric\n"
            "Generic wrapper mentions PNG and PDF.\n\n"
            "## Task\n"
            "Create a detailed Excel output outlining all S&P500 companies.\n\n"
            "## Tools and workflow\n"
            "Use available tools."
        ),
        metadata={},
    )

    assert validation["status"] == "invalid"
    assert any(
        issue["code"] == "expected_file_type_missing"
        for issue in validation["issues"]
    )


def test_deliverable_contract_ignores_reference_inspection_warnings(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    outputs = tmp_path / "outputs"
    inputs = tmp_path / "inputs"
    outputs.mkdir()
    inputs.mkdir()
    (outputs / "summary.txt").write_text("Finished deliverable", encoding="utf-8")
    (inputs / "brief.psd").write_bytes(b"not a submitted deliverable")

    validation = validate_deliverable_contract(
        paths=[outputs, inputs],
        problem="Create a written summary.",
        metadata={},
    )

    assert validation["status"] == "ok"
    assert validation["files"] == [str(outputs / "summary.txt")]
    assert not any(issue["code"] == "file_inspection_errors" for issue in validation["issues"])


def test_deliverable_paths_resolve_relative_submitted_files(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    outputs = tmp_path / "workspace"
    outputs.mkdir()
    workbook = outputs / "final.xlsx"
    _write_workbook(workbook)

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create an Excel workbook.",
        metadata={
            "gdpval_outputs_dir": str(outputs),
            "gdpval_submitted_files": ["final.xlsx"],
        },
    )

    assert str(workbook) in validation["files"]
    assert not any(issue["code"] == "submitted_file_missing" for issue in validation["issues"])


def test_deliverable_contract_rejects_reference_audio_as_fallback_output(tmp_path) -> None:
    import wave

    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with wave.open(str(workspace / "ref_48k.wav"), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(3)
        f.setframerate(48000)
        f.writeframes(b"\x00\x00\x00" * 2 * 480)
    (workspace / "process_mix.py").write_text("print('scratch')\n", encoding="utf-8")

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the final mix as TAVARUA_FINAL_MIX.wav.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": ["TAVARUA_FINAL_MIX.wav"],
        },
    )

    assert validation["status"] == "invalid"
    assert validation["primary_files"] == []
    assert any(
        issue["code"] == "expected_named_file_missing"
        for issue in validation["issues"]
    )


def test_deliverable_contract_excludes_staged_reference_aliases(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inputs = workspace / "inputs"
    inputs.mkdir()
    reference = inputs / "source.mp4"
    reference.write_bytes(b"reference")
    alias = workspace / "source.mp4"
    try:
        alias.symlink_to(reference)
    except OSError:
        alias.write_bytes(reference.read_bytes())

    validation = validate_deliverable_contract(
        paths=[workspace],
        problem="Create the final edited MP4.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "gdpval_reference_files": [str(reference), str(alias)],
        },
    )

    assert validation["status"] == "invalid"
    assert validation["files"] == []
    assert any(issue["code"] == "no_deliverables_found" for issue in validation["issues"])


def test_deliverable_contract_accepts_expected_final_audio(tmp_path) -> None:
    import wave

    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    final_mix = workspace / "TAVARUA_FINAL_MIX.wav"
    with wave.open(str(final_mix), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(3)
        f.setframerate(48000)
        f.writeframes(b"\x00\x00\x00" * 2 * 480)

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the final mix as TAVARUA_FINAL_MIX.wav.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": ["TAVARUA_FINAL_MIX.wav"],
        },
    )

    assert validation["status"] == "ok"
    assert validation["primary_files"] == [str(final_mix)]
    assert not any(
        issue["code"] == "media_inspection_unavailable"
        for issue in validation["issues"]
    )


def test_deliverable_contract_rejects_corrupt_video_as_invalid_media(tmp_path) -> None:
    from ipw.evaluation.deliverables import _ffmpeg_executable, validate_deliverable_contract

    if _ffmpeg_executable() is None:
        pytest.skip("ffmpeg unavailable")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    video = workspace / "final.mp4"
    video.write_bytes(
        b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
        b"\x00\x00\x00\x08free"
        b"\x00\x00\x00\x10mdatnot-enough-media"
    )

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the final edited MP4.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": ["final.mp4"],
        },
    )

    assert validation["status"] == "invalid"
    assert any(issue["code"] == "invalid_media" for issue in validation["issues"])
    assert not any(
        issue["code"] == "media_inspection_unavailable"
        for issue in validation["issues"]
    )


def test_deliverable_contract_ignores_corrupt_scratch_video_when_final_video_is_valid(tmp_path) -> None:
    import subprocess

    from ipw.evaluation.deliverables import _ffmpeg_executable, validate_deliverable_contract

    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        pytest.skip("ffmpeg unavailable")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    final_video = workspace / "campaign_final.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x32:d=1",
            "-pix_fmt",
            "yuv420p",
            str(final_video),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (workspace / "clip_1.mp4").write_bytes(b"not a real video")
    (workspace / "test.mp4").write_bytes(b"")

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create a final campaign video as an MP4.",
        metadata={"gdpval_outputs_dir": str(workspace)},
    )

    assert validation["status"] == "ok"
    assert validation["primary_files"] == [str(final_video)]
    assert not any(issue["code"] == "invalid_media" for issue in validation["issues"])


def test_media_validation_marks_decoder_failure_as_invalid_media() -> None:
    from ipw.evaluation.deliverables import FileEvidence, _media_validation_issues

    evidence = FileEvidence(
        path="/tmp/final.mp4",
        name="final.mp4",
        suffix=".mp4",
        mime_type="video/mp4",
        size_bytes=128,
        sha256="0" * 64,
        kind="video",
        metadata={"ffmpeg": "/usr/bin/ffmpeg", "sampled_frames": 0},
        errors=["video frame extraction failed: moov atom not found"],
    )

    issues = _media_validation_issues([evidence])

    assert not issues["unavailable"]
    assert issues["invalid"]
    assert issues["invalid"][0]["file"] == "final.mp4"


def test_deliverable_contract_decodes_expected_filename(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workbook = workspace / "UAT Plan.xlsx"
    _write_workbook(workbook)

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the UAT plan as an Excel workbook.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": ["UAT%20Plan.xlsx"],
        },
    )

    assert validation["status"] != "invalid"
    assert validation["primary_files"] == [str(workbook)]
    assert validation["expected_filenames"] == ["uat plan.xlsx"]


def test_deliverable_contract_accepts_normalized_expected_filename(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workbook = workspace / "cover_plate_manufacturing_steps.xlsx"
    _write_workbook(workbook)

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the cover plate manufacturing steps workbook.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": ["cover plate manufacturing steps.xlsx"],
        },
    )

    assert validation["status"] != "invalid"
    assert validation["primary_files"] == [str(workbook)]
    assert not any(
        issue["code"] == "expected_named_file_missing"
        for issue in validation["issues"]
    )


def test_deliverable_contract_warns_on_plausible_filename_mismatch(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workbook = workspace / "Monthly Tracker Patient Lab Results - Completed.xlsx"
    _write_workbook(workbook)

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the completed monthly tracker workbook.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": ["monthly tracker patient lab results data1.xlsx"],
        },
    )

    assert validation["status"] == "warning"
    assert validation["primary_files"] == [str(workbook)]
    assert any(
        issue["code"] == "expected_named_file_mismatch"
        for issue in validation["issues"]
    )
    assert not any(
        issue["code"] == "expected_file_type_missing"
        for issue in validation["issues"]
    )


def test_deliverable_contract_blocks_missing_multi_file_named_outputs(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference_pdf = workspace / "Nurturing_Program.pdf"
    reference_pdf.write_bytes(b"%PDF-1.4\n% placeholder reference\n")

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create two visually engaging PowerPoint presentations.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": [
                "Topic 13 Edited.pptx",
                "NP-Topic 14 Edited.pptx",
            ],
        },
    )

    assert validation["status"] == "invalid"
    assert any(
        issue["code"] == "expected_named_file_missing"
        for issue in validation["issues"]
    )


def test_deliverable_contract_warns_on_similar_multi_file_name_when_type_present(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workbook = workspace / "Spring Bazaar Vendor Assignments.xlsx"
    _write_workbook(workbook)

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the updated vendor assignment workbooks.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": [
                "Spring Bazaar 2025 Vendors List & Table Assignments-v2.xlsx",
                "Spring Bazaar 2025 Vendor Summary-v2.xlsx",
            ],
        },
    )

    assert validation["status"] == "warning"
    assert validation["primary_files"] == [str(workbook)]
    assert any(
        issue["code"] == "expected_named_file_mismatch"
        for issue in validation["issues"]
    )
    assert not any(
        issue["code"] == "expected_named_file_missing"
        for issue in validation["issues"]
    )


def test_deliverable_contract_accepts_expected_overpassql_text_file(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instructions = workspace / "abq okc query instructions.md"
    query = workspace / "abq okc query.overpassql"
    instructions.write_text(
        "Run abq okc query.overpassql against Overpass Turbo and export JSON.",
        encoding="utf-8",
    )
    query.write_text(
        "[out:json][timeout:180];\n"
        "way[\"highway\"~\"^motorway(_link)?$\"][\"network\"=\"US:I\"]"
        "[\"ref\"~\"^I[ -]?40$\"](35.0,-107.2,36.0,-97.5);\n"
        "out body; >; out skel qt;\n",
        encoding="utf-8",
    )

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create OverpassQL instructions and a query file for the I-40 corridor.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": [
                "abq okc query instructions.md",
                "abq okc query.overpassql",
            ],
        },
    )

    assert validation["status"] == "ok"
    assert validation["expected_extensions"] == [".md", ".overpassql"]
    assert validation["primary_files"] == [str(instructions), str(query)]


def test_deliverable_contract_warns_on_partial_mixed_type_package(tmp_path) -> None:
    from pypdf import PdfWriter

    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    partial_pdf = workspace / "nih_checklist.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with partial_pdf.open("wb") as file:
        writer.write(file)

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the ergonomics action items document and checklist PDF.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": [
                "Organizational Action Items_Ergonomics edits.docx",
                "Workstation Ergonomics Checklist edits.pdf",
            ],
        },
    )

    assert validation["status"] == "warning"
    assert validation["primary_files"] == [str(partial_pdf)]
    assert any(
        issue["code"] == "expected_named_file_missing"
        and issue.get("missing_expected_suffixes") == [".docx"]
        for issue in validation["issues"]
    )
    assert any(
        issue["code"] == "expected_named_file_mismatch"
        for issue in validation["issues"]
    )


def test_deliverable_contract_excludes_copied_reference_files_by_content(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True)
    reference = inputs / "COA.xlsx"
    copied_reference = workspace / "COA.xlsx"
    _write_workbook(reference, "reference chart of accounts")
    copied_reference.write_bytes(reference.read_bytes())

    validation = validate_deliverable_contract(
        paths=[],
        problem="Create the Aurisic amortization workbook.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": ["Aurisic_Amortization_4-25.xlsx"],
        },
    )

    assert validation["status"] == "invalid"
    assert validation["primary_files"] == []
    assert copied_reference.name not in {
        Path(path).name for path in validation["files"]
    }
    assert any(issue["code"] == "no_deliverables_found" for issue in validation["issues"])


def test_deliverable_contract_blocks_unreadable_zip(tmp_path) -> None:
    from ipw.evaluation.deliverables import validate_deliverable_contract

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive = workspace / "deliverable.zip"
    archive.write_bytes(b"")

    validation = validate_deliverable_contract(
        paths=[],
        problem="Provide the final deliverable in a zip folder.",
        metadata={
            "gdpval_outputs_dir": str(workspace),
            "deliverable_files": ["Deja Vu Stems.zip"],
        },
    )

    assert validation["status"] == "invalid"
    assert any(issue["code"] == "unreadable_archive" for issue in validation["issues"])


def test_deliverable_contract_blocks_unreadable_primary_pdf(tmp_path) -> None:
    from ipw.evaluation.deliverables import FileEvidence, _unreadable_deliverable_issues

    pdf = tmp_path / "mtbi soap.pdf"
    pdf.write_bytes(b"not a pdf")
    evidence = FileEvidence(
        path=str(pdf),
        name=pdf.name,
        suffix=".pdf",
        mime_type="application/pdf",
        size_bytes=pdf.stat().st_size,
        sha256="0" * 64,
        kind="pdf",
        errors=[
            "pdf text parse error: No /Root object! - Is this really a PDF?",
            "pypdf text parse error: Stream has ended unexpectedly",
        ],
    )

    issues = _unreadable_deliverable_issues(
        [evidence],
        primary_path_keys={str(pdf.resolve())},
    )

    assert issues
    assert issues[0]["file"] == "mtbi soap.pdf"


def test_deliverable_contract_does_not_block_when_pdf_parser_unavailable(tmp_path) -> None:
    from ipw.evaluation.deliverables import FileEvidence, _unreadable_deliverable_issues

    pdf = tmp_path / "mtbi soap.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    evidence = FileEvidence(
        path=str(pdf),
        name=pdf.name,
        suffix=".pdf",
        mime_type="application/pdf",
        size_bytes=pdf.stat().st_size,
        sha256="0" * 64,
        kind="pdf",
        errors=[
            "pdfplumber is not installed",
            "pypdf is not installed",
            "pdftotext failed: [Errno 2] No such file or directory: 'pdftotext'",
        ],
    )

    issues = _unreadable_deliverable_issues(
        [evidence],
        primary_path_keys={str(pdf.resolve())},
    )

    assert issues == []


def test_deliverable_extractor_samples_video_frames_and_audio(tmp_path) -> None:
    from ipw.evaluation.deliverables import _ffmpeg_executable, inspect_file

    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        pytest.skip("ffmpeg unavailable")

    video = tmp_path / "clip.mp4"
    import subprocess

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    evidence = inspect_file(video)

    assert evidence.kind == "video"
    assert evidence.metadata["sampled_frames"] >= 1
    assert evidence.metadata["video_stream"]["width"] == 160
    assert evidence.metadata["video_stream"]["height"] == 120
    assert evidence.metadata["extracted_audio"]["sample_rate"] == 16000
    assert any(child.kind == "video_frame" for child in evidence.children)
