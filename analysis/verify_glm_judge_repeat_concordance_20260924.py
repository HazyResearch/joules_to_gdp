"""Calculate agreement between repeated GLM 5.2 Gemini judgments."""

from collections import Counter, defaultdict
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/gdpval_pairwise_gemini31_flash_lite_20260716/judgments.raw.unblinded.jsonl"


def main() -> None:
    records = [json.loads(line) for line in SOURCE.open() if line.strip()]
    records = [
        row for row in records
        if row["condition"] == "without_rubric" and row["baseline"] == "human"
    ]
    by_task = defaultdict(dict)
    for row in records:
        assert row["repeat"] not in by_task[row["task_id"]]
        by_task[row["task_id"]][row["repeat"]] = row

    assert len(by_task) == 220
    assert all(set(runs) == {0, 1, 2, 3} for runs in by_task.values())
    assert all(Counter(run["agent_position"] for run in runs.values()) == {"A": 2, "B": 2}
               for runs in by_task.values())

    exact = sum(runs[0]["agent_outcome"] == runs[1]["agent_outcome"]
                for runs in by_task.values())
    binary = sum((runs[0]["agent_outcome"] in {"win", "tie"}) ==
                 (runs[1]["agent_outcome"] in {"win", "tie"})
                 for runs in by_task.values())
    print(json.dumps({"tasks": len(by_task), "judgments_compared": 2 * len(by_task),
                      "exact_outcome_agreement": exact, "exact_percent": 100 * exact / len(by_task),
                      "win_tie_agreement": binary, "win_tie_percent": 100 * binary / len(by_task)},
                     indent=2))


if __name__ == "__main__":
    main()
