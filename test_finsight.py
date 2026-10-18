"""Tests for core FinSight workflows and safety rules; no network needed."""
import tempfile
import unittest
from pathlib import Path

from finsight_core import (
    AnalysisMemory,
    InvestmentResearchAgent,
    SemanticRetriever,
    classify_news,
    demo_documents,
    detect_specialist,
    finqa_answer_accuracy,
    finqa_to_document,
    news_prompt_chain,
    normalize_numeric_answer,
    safe_calculate,
)


class TestFinSight(unittest.TestCase):
    def setUp(self):
        self.docs = demo_documents()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agent = InvestmentResearchAgent(
            self.docs, f"{self.temp.name}/notes.json",
            quotes={"DEMO": {"price": 52.30, "as_of": "fictional"}},
        )

    def test_safe_calculator(self):
        self.assertAlmostEqual(safe_calculate("(250-200)/200*100"), 25)
        for expr in ("__import__('os').system('echo x')", "x+1", "1/0", "3**1000"):
            with self.assertRaises((ValueError, SyntaxError, ZeroDivisionError)):
                safe_calculate(expr)

    def test_routing(self):
        self.assertEqual(detect_specialist("What is the stock price?"), "market")
        self.assertEqual(detect_specialist("Summarize company news"), "news")
        self.assertEqual(detect_specialist("Compute revenue growth"), "earnings")

    def test_prompt_chaining(self):
        x = news_prompt_chain(self.docs)
        self.assertEqual([v["stage"] for v in x["trace"]], [
            "Ingest News", "Preprocess", "Classify", "Extract", "Summarize"])
        self.assertGreater(len(x["summary"]), 30)

    def test_retrieval(self):
        retriever = SemanticRetriever(self.docs)
        hits = retriever.retrieve("revenue from 2023 to 2024", symbol="DEMO", kind="earnings")
        self.assertEqual(hits[0]["document"].doc_id, "DEMO-10K-2024")

    def test_agent_math_tool(self):
        result = self.agent.run("DEMO", "What was the percentage change in revenue from 2023 to 2024?")
        self.assertIn("25.00%", result["answer"])
        self.assertTrue(result["evaluation"]["passed"])
        self.assertTrue(any(e["tool"] == "calculator" for e in result["tool_trace"]))

    def test_agent_news_and_market(self):
        a = self.agent.run("DEMO", "Summarize the latest news")
        b = self.agent.run("DEMO", "What is the stock price?")
        self.assertEqual(a["specialist"], "news")
        self.assertEqual(b["specialist"], "market")
        self.assertTrue(a["evaluation"]["passed"])
        self.assertTrue(b["evaluation"]["passed"])

    def test_abstention_iteration_and_memory(self):
        q = "What are the CEO qualifications?"
        before = self.agent.run("DEMO", q)
        after = self.agent.run("DEMO", q)
        self.assertFalse(before["evaluation"]["passed"])
        self.assertIn("Insufficient evidence", before["answer"])
        self.assertEqual(len(before["iterations"]), 2)
        self.assertGreater(len(after["memory_before"]), 0)
        self.assertTrue(any("top-" in step for step in after["plan"]))
        self.assertTrue(Path(self.temp.name, "notes.json").exists())

    def test_finqa_schema_does_not_leak_gold_answer(self):
        row = {"id": "example-1", "pre_text": ["revenue grew"],
               "post_text": ["the company"], "table": [["year", "2024"]],
               "qa": {"question": "growth?", "exe_ans": "SECRET_ANSWER", "program": "SECRET_PROGRAM"}}
        doc = finqa_to_document(row, "dev")
        self.assertIn("revenue grew", doc.content)
        self.assertNotIn("SECRET_ANSWER", doc.content)
        self.assertNotIn("SECRET_PROGRAM", doc.content)

    def test_numeric_evaluation(self):
        result = finqa_answer_accuracy([{"answer": "25%"}],
                                       [{"id": "x", "gold_answer": "25%"}])
        self.assertEqual(result["numeric_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
