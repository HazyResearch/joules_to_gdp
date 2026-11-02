"""Tests for evaluation/gdpval.py — GdpvalHandler (rubric LLM-as-judge)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock


def _rubric(items):
    return json.dumps(items)


class TestGdpvalHandler:
    def test_strict_judge_rejects_outage_and_invalid_verdict(self, monkeypatch) -> None:
        import pytest

        from ipw.evaluation.gdpval import GdpvalHandler

        monkeypatch.setenv("IPW_REQUIRE_GDPVAL_JUDGE", "1")
        monkeypatch.setenv("IPW_GDPVAL_JUDGE_MAX_ATTEMPTS", "1")
        client = MagicMock()
        handler = GdpvalHandler(client=client)
        request = {
            "problem": "task", "reference": "", "model_answer": "answer",
            "metadata": {"rubric_json": _rubric([{"criterion": "Includes totals", "score": 1}])},
        }
        client.chat.side_effect = RuntimeError("fabricated service outage")
        with pytest.raises(RuntimeError, match="rubric judge failed"):
            handler.evaluate(**request)
        client.chat.side_effect = None
        client.chat.return_value = "perhaps"
        with pytest.raises(RuntimeError, match="invalid verdict"):
            handler.evaluate(**request)
        client.chat.return_value = "The example says verdict: no.\nVERDICT: yes"
        assert handler.evaluate(**request)[0] is True
        client.chat.return_value = "VERDICT: yes\nVERDICT: no"
        with pytest.raises(RuntimeError, match="invalid verdict"):
            handler.evaluate(**request)

    def test_returns_none_when_no_rubric(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        handler = GdpvalHandler(client=MagicMock())
        is_correct, meta = handler.evaluate(
            problem="task", reference="", model_answer="answer", metadata={}
        )
        assert is_correct is None
        assert meta["reason"] == "no_rubric"

    def test_empty_response_is_false(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        handler = GdpvalHandler(client=MagicMock())
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="",
            metadata={"rubric_json": _rubric([{"criterion": "x", "score": 1}])},
        )
        assert is_correct is False
        assert meta["reason"] == "empty_response"

    def test_empty_response_with_deliverables_is_judged(self, tmp_path) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        (tmp_path / "summary.txt").write_text("final analysis", encoding="utf-8")
        client = MagicMock()
        client.chat.return_value = "verdict: yes\nreason: file evidence satisfies it"
        handler = GdpvalHandler(client=client)

        is_correct, meta = handler.evaluate(
            problem="Create a summary deliverable.",
            reference="",
            model_answer="",
            metadata={
                "rubric_json": _rubric(
                    [{"criterion": "Includes final analysis", "score": 1}]
                ),
                "gdpval_outputs_dir": str(tmp_path),
            },
        )

        assert is_correct is True
        assert meta["synthesized_empty_response"] is True
        assert meta["score"] == 1.0
        assert client.chat.call_count == 1
        prompt = client.chat.call_args.kwargs["user_prompt"]
        assert "No textual finish response was captured" in prompt
        assert "summary.txt" in prompt

    def test_judges_primary_file_when_scratch_image_is_broken(self, tmp_path) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        (tmp_path / "final.txt").write_text("finished deliverable", encoding="utf-8")
        (tmp_path / "downloaded_reference.jpg").write_text(
            "<!DOCTYPE html>not an image",
            encoding="utf-8",
        )
        client = MagicMock()
        client.chat.return_value = "verdict: yes\nreason: primary file satisfies it"
        handler = GdpvalHandler(client=client)

        is_correct, meta = handler.evaluate(
            problem="Create the final deliverable as final.txt.",
            reference="",
            model_answer="done",
            metadata={
                "rubric_json": _rubric(
                    [{"criterion": "Includes final deliverable", "score": 1}]
                ),
                "gdpval_outputs_dir": str(tmp_path),
                "deliverable_files": ["final.txt"],
            },
        )

        assert is_correct is True
        assert meta["score"] == 1.0
        assert meta["deliverable_validation"]["status"] == "warning"
        assert client.chat.call_count == 1
        prompt = client.chat.call_args.kwargs["user_prompt"]
        assert "final.txt" in prompt
        assert "downloaded_reference.jpg" not in prompt

    def test_all_pass_required_criteria(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        client.chat.return_value = "verdict: yes\nreason: looks good"
        handler = GdpvalHandler(client=client)

        rubric = [
            {"rubric_item_id": "a", "criterion": "Has totals", "score": 2, "required": True},
            {"rubric_item_id": "b", "criterion": "Has detail", "score": 1, "required": False},
        ]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="my answer with totals and detail",
            metadata={"rubric_json": _rubric(rubric)},
        )
        assert is_correct is True
        assert meta["match_type"] == "rubric_llm_judge"
        assert meta["achieved_points"] == 3.0
        assert meta["max_points"] == 3.0
        assert meta["required_failures"] == 0

    def test_judge_call_retries_transient_failure(self, monkeypatch) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        monkeypatch.setenv("IPW_GDPVAL_JUDGE_MAX_ATTEMPTS", "2")
        monkeypatch.setenv("IPW_GDPVAL_JUDGE_RETRY_SECONDS", "0")
        client = MagicMock()
        client.chat.side_effect = [
            RuntimeError("502 Bad Gateway"),
            "verdict: yes\nreason: retry succeeded",
        ]
        handler = GdpvalHandler(client=client)

        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="answer",
            metadata={
                "rubric_json": _rubric(
                    [{"criterion": "Has totals", "score": 1, "required": True}]
                )
            },
        )

        assert is_correct is True
        assert meta["required_failures"] == 0
        assert "error" not in meta["per_criterion"][0]
        assert client.chat.call_count == 2

    def test_required_failure_fails_task(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        # First criterion is required and judge says no
        client.chat.side_effect = [
            "verdict: no\nreason: missing totals",
            "verdict: yes\nreason: has detail",
        ]
        handler = GdpvalHandler(client=client)

        rubric = [
            {"rubric_item_id": "a", "criterion": "Has totals", "score": 2, "required": True},
            {"rubric_item_id": "b", "criterion": "Has detail", "score": 1, "required": False},
        ]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="answer",
            metadata={"rubric_json": _rubric(rubric)},
        )
        assert is_correct is False
        assert meta["required_failures"] == 1
        assert meta["achieved_points"] == 1.0

    def test_score_below_threshold_fails(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        # Two of five criteria pass; no required → score 2/5 = 0.4 < 0.5
        client.chat.side_effect = [
            "verdict: yes",
            "verdict: yes",
            "verdict: no",
            "verdict: no",
            "verdict: no",
        ]
        handler = GdpvalHandler(client=client)
        rubric = [
            {"criterion": f"c{i}", "score": 1, "required": False} for i in range(5)
        ]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="answer",
            metadata={"rubric_json": _rubric(rubric)},
        )
        assert is_correct is False
        assert meta["score"] == 0.4

    def test_handles_malformed_judge_output(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        client.chat.return_value = "garbled response with no verdict line"
        handler = GdpvalHandler(client=client)

        rubric = [{"criterion": "c", "score": 1, "required": False}]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="answer",
            metadata={"rubric_json": _rubric(rubric)},
        )
        # Malformed → treat as not satisfied; score 0 → False
        assert is_correct is False
        assert meta["achieved_points"] == 0.0

    def test_negative_rubric_points_are_penalties_not_denominator(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        client.chat.side_effect = [
            "verdict: yes\nreason: has the required deliverable",
            "verdict: yes\nreason: bad thing is present",
        ]
        handler = GdpvalHandler(client=client)

        rubric = [
            {"criterion": "Has output", "score": 10},
            {"criterion": "Contains forbidden artifact", "score": -4},
        ]
        is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="answer",
            metadata={"rubric_json": _rubric(rubric)},
        )

        assert is_correct is True
        assert meta["max_points"] == 10.0
        assert meta["penalty_possible"] == 4.0
        assert meta["penalty_applied"] == 4.0
        assert meta["achieved_points"] == 6.0
        assert meta["score"] == 0.6

    def test_negative_rubric_prompt_asks_whether_penalty_applies(self) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        client = MagicMock()
        client.chat.return_value = "verdict: no\nreason: the defect is absent"
        handler = GdpvalHandler(client=client)

        rubric = [{"criterion": "Fails to include the required workbook", "score": -5}]
        _is_correct, meta = handler.evaluate(
            problem="task",
            reference="",
            model_answer="Created workbook.xlsx",
            metadata={"rubric_json": _rubric(rubric)},
        )

        prompt = client.chat.call_args.kwargs["user_prompt"]
        assert "negative-point penalty criterion" in prompt
        assert "Answer yes only if the defect" in prompt
        assert meta["penalty_applied"] == 0.0

    def test_contract_validation_blocks_script_only_workspace(self, tmp_path) -> None:
        from ipw.evaluation.gdpval import GdpvalHandler

        (tmp_path / "build_report.py").write_text("print('draft')", encoding="utf-8")
        client = MagicMock()
        handler = GdpvalHandler(client=client)

        rubric = [{"criterion": "Delivers an Excel workbook", "score": 1}]
        is_correct, meta = handler.evaluate(
            problem="Create an Excel workbook in .xlsx format.",
            reference="",
            model_answer="Created build_report.py",
            metadata={
                "rubric_json": _rubric(rubric),
                "gdpval_outputs_dir": str(tmp_path),
            },
        )

        assert is_correct is False
        assert meta["reason"] == "deliverable_contract_invalid"
        assert meta["failure_type"] == "harness_validation"
        assert meta["max_points"] == 1.0
        assert client.chat.call_count == 0
