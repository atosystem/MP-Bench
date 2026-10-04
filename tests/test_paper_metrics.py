"""Offline checks against the paper's 927-task set and GPT-Realtime Table 3 numbers."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mpbench_eval import benchmark, metrics  # noqa: E402
from mpbench_eval.cli import cmd_detect  # noqa: E402
from mpbench_eval.transcripts import Detected  # noqa: E402

ASR_GPT = Path(
    "/Users/ericsunkuan/Desktop/NTUGICE/114-1/bd/mpbench-final-postDiagQltyChck_tasks_only"
    "/asr_results/elevenlabs_greet_v1prompt/gpt"
)
RERUN_GPT = Path(
    "/Users/ericsunkuan/Desktop/NTUGICE/114-1/bd/mpbench-final-postDiagQltyChck_tasks_only"
    "/asr_results/elevenlabs_greet_v1prompt_argument_t2_rerun/gpt"
)


class TestBenchmark(unittest.TestCase):
    def test_paper_task_counts(self):
        tasks = benchmark.load_tasks()
        self.assertEqual(len(tasks), 927)
        fam = {name: sum(t.family == name for t in tasks.values()) for name in ("understanding", "discussion", "turnbased")}
        self.assertEqual(fam, {"understanding": 543, "discussion": 128, "turnbased": 256})
        self.assertEqual(sum(t.is_speaker_name for t in tasks.values()), 396)
        tb = [t for t in tasks.values() if t.family == "turnbased"]
        self.assertEqual(sum(t.should_speak for t in tb), 128)

    def test_discussion_uses_argument_t2_closing(self):
        tasks = benchmark.load_tasks()
        disc = [t for t in tasks.values() if t.family == "discussion"]
        self.assertEqual(len(disc), 128)
        sample = json.loads(next(t.metatranscript for t in disc if t.task_dir == "scenario_argument-diag_174-tier_2-task_0"))
        self.assertEqual(
            sample["metatranscript"][-1]["content"],
            "AI Assistant, weigh in on this. What is the answer and who's got it right?",
        )


@unittest.skipUnless(ASR_GPT.exists(), "GPT-Realtime ASR dump not on this machine")
class TestGptRealtimePaperTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Args:
            result_dir = str(ASR_GPT)
            out_dir = "/tmp/mpbench_eval_gpt_test"
            name = "GPT-Realtime"
            tasks = None

        cls.detected = cmd_detect(Args())
        cls.tasks = benchmark.load_tasks()

    def test_turn_taking_matches_table3(self):
        disc = [r for r in self.detected if r.family == "discussion"]
        tb = [r for r in self.detected if r.family == "turnbased"]
        self.assertEqual(len(disc), 128)
        self.assertEqual(sum(r.tt_correct for r in disc), 128)  # 100%
        self.assertEqual(len(tb), 256)
        self.assertEqual(sum(r.tt_correct for r in tb), 124)  # 48.44%

    def test_app_acc_from_saved_paper_judges(self):
        judgments = {}
        t1 = ASR_GPT / "tier1_task_acc_results_v2-answer-only.csv"
        for row in csv.DictReader(open(t1), delimiter="\t"):
            judgments[f"understanding:{row['task_dir']}"] = row["correctness"].strip().lower() == "true"
        spk = ASR_GPT / "speaker_name_llm_eval.csv"
        for row in csv.DictReader(open(spk), delimiter="\t"):
            judgments[f"speaker_name:{row['task_dir']}"] = row["correct"].strip().lower() == "true"
        disc = RERUN_GPT / "tier2_task_acc_results_post_input_v2-answer-only.csv"
        if not disc.exists():
            self.skipTest("post-input discussion labels missing")
        for row in csv.DictReader(open(disc), delimiter="\t"):
            judgments[f"discussion:{row['task_dir']}"] = row["consistency"].strip().lower() == "true"
        tb = ASR_GPT / "tier2_task_acc_results_turnbased-v3-cot.csv"
        for row in csv.DictReader(open(tb), delimiter="\t"):
            judgments[f"turnbased:{row['task_dir']}"] = row["consistency"].strip().lower() == "true"

        result = metrics.score(self.detected, self.tasks, judgments)
        self.assertEqual(result["understanding"]["overall_acc"]["pct"], 14.0)
        self.assertEqual(result["understanding"]["spk_name_acc"]["pct"], 16.67)
        self.assertEqual(result["discussion"]["app_acc"]["count"], 68)
        self.assertEqual(result["turnbased"]["app_acc"]["pct"], 28.12)


if __name__ == "__main__":
    unittest.main()
