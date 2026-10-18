"""FinSight educational agentic investment research prototype.

The default demonstration uses synthetic classroom data, and all agents are read-only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ResearchDocument:
    doc_id: str
    symbol: str
    kind: str
    title: str
    content: str
    source: str
    published: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def display(self) -> str:
        return f"[{self.doc_id}] ({self.kind}; {self.source}) {self.title}: {self.content}"

@dataclass
class ToolEvent:
    tool: str
    parameters: Dict[str, Any]
    outcome: str

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.events: List[ToolEvent] = []

    def register(self, name: str, func: Any) -> None:
        self.tools[name] = func

    def call(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        try:
            result = self.tools[name](**kwargs)
            preview = str(result)
            self.events.append(ToolEvent(name, kwargs, preview[:160]))
            return result
        except Exception as exc:
            self.events.append(ToolEvent(name, kwargs, f"ERROR: {exc}"))
            raise

class SemanticRetriever:
    """Vector-space semantic retriever: sentence embeddings if available; TF-IDF otherwise.

    No benchmark claim is made about TF-IDF: it is an offline lexical baseline.
    """
    def __init__(self, docs: List[ResearchDocument], embedding_backend="tfidf",
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.docs = list(docs)
        self.backend = "tfidf"
        self.encoder = None
        self.vectorizer = None
        texts = [f"{d.title}. {d.content}" for d in docs]
        if embedding_backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name)
            self.vectors = np.asarray(self.encoder.encode(texts, normalize_embeddings=True))
            self.backend = f"sentence_transformer ({model_name})"
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.vectors = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3,
                 symbol: Optional[str] = None, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        eligible = [i for i, d in enumerate(self.docs)
                    if (not symbol or d.symbol.upper() == symbol.upper())
                    and (not kind or d.kind == kind)]
        if not eligible:
            return []
        if self.encoder is not None:
            q = self.encoder.encode([query], normalize_embeddings=True)
            scores = (self.vectors[eligible] @ q[0]).tolist()
        else:
            q = self.vectorizer.transform([query])
            scores = cosine_similarity(q, self.vectors[eligible]).ravel().tolist()
        ranked = sorted(zip(eligible, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"document": self.docs[idx], "score": float(score)} for idx, score in ranked]

_ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Pow: operator.pow}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def safe_calculate(expression: str) -> float:
    """Evaluate arithmetic only: no function calls, variables, attributes, or Python execution."""
    if len(expression) > 120:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval")

    def walk(node, depth=0):
        if depth > 14:
            raise ValueError("Expression nesting too deep")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            if abs(node.value) > 1e12:
                raise ValueError("Literal magnitude too large")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = walk(node.left, depth + 1), walk(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1e6):
                raise ValueError("Unsafe exponentiation")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](walk(node.operand, depth + 1))
        raise ValueError("Unsafe calculation")

    result = float(walk(tree))
    if not math.isfinite(result):
        raise ValueError("Nonfinite result")
    return result

def ingest_news(documents: Iterable[ResearchDocument]) -> List[ResearchDocument]:
    return [d for d in documents if d.kind == "news"]

def preprocess_news(documents: Iterable[ResearchDocument]) -> List[ResearchDocument]:
    return [ResearchDocument(**{**d.__dict__, "content": " ".join(d.content.split())})
            for d in documents]

def classify_news(documents: Iterable[ResearchDocument]) -> List[Dict[str, Any]]:
    labels = {"earnings": ["earnings", "revenue", "profit", "quarter", "results"],
              "market": ["price", "share", "stock", "market", "volume"],
              "risk": ["recall", "lawsuit", "investigation", "risk", "regulator"]}
    classified = []
    for d in documents:
        t = (d.title + " " + d.content).lower()
        counts = {k: sum(w in t for w in v) for k, v in labels.items()}
        cls = max(counts, key=counts.get) if max(counts.values()) else "general"
        classified.append({"document": d, "category": cls})
    return classified

def extract_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    extracted = []
    for item in items:
        text = item["document"].content
        figures = re.findall(r"(?<![\w])(?:\$\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|million|billion)?", text, flags=re.I)
        extracted.append({**item, "figures": figures[:12]})
    return extracted

def summarize_news(items: List[Dict[str, Any]], max_items=3) -> str:
    if not items:
        return "No relevant news articles are available in the supplied corpus."
    summaries = []
    for item in items[:max_items]:
        d = item["document"]
        first = re.split(r"(?<=[.!?])\s+", d.content, maxsplit=1)[0]
        summaries.append(f"[{d.doc_id}] {d.title}: {first} (category: {item['category']})")
    return "\n".join(summaries)

def news_prompt_chain(documents: List[ResearchDocument]) -> Dict[str, Any]:
    """Required workflow: ingest -> preprocess -> classify -> extract -> summarize."""
    trace = []
    x = ingest_news(documents)
    trace.append({"stage": "Ingest News", "count": len(x)})
    x = preprocess_news(x)
    trace.append({"stage": "Preprocess", "count": len(x)})
    y = classify_news(x)
    trace.append({"stage": "Classify", "labels": [r["category"] for r in y]})
    z = extract_news(y)
    trace.append({"stage": "Extract", "figure_counts": [len(r["figures"]) for r in z]})
    final = summarize_news(z)
    trace.append({"stage": "Summarize", "chars": len(final)})
    return {"summary": final, "items": z, "trace": trace}

def detect_specialist(question: str, preferred_kind: Optional[str] = None) -> str:
    """Content routing (not fixed routing for every query)."""
    if preferred_kind in {"earnings", "news", "market"}:
        return preferred_kind
    q = question.lower()
    if any(w in q for w in ("news", "headline", "article", "announcement")):
        return "news"
    if any(w in q for w in ("share price", "stock price", "market price", "volatility", "quote", "volume")):
        return "market"
    return "earnings"

class AnalysisMemory:
    """Short local notes, with timestamps and no storage of retrieved document text."""
    def __init__(self, path: str):
        self.path = Path(path)
        if self.path.exists():
            try:
                self.notes = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.notes = {}
        else:
            self.notes = {}

    def recall(self, symbol: str) -> List[Dict[str, Any]]:
        return list(self.notes.get(symbol.upper(), []))[-3:]

    def store(self, symbol: str, specialist: str, outcome: str, feedback: str) -> None:
        key = symbol.upper()
        self.notes.setdefault(key, []).append({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "specialist": specialist,
            "outcome": outcome[:80],
            "feedback": feedback[:140],
        })
        self.notes[key] = self.notes[key][-12:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.notes, indent=2), encoding="utf-8")

class LocalLLM:
    """Optional genuine local HF encoder-decoder LLM. Run in a networked Colab environment."""
    def __init__(self, model_name="google/flan-t5-small"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens=150) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=512).to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(**encoded, max_new_tokens=max_new_tokens,
                                         do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

class InvestmentResearchAgent:
    """Planner + dynamic tools + specialist routing + evaluator-optimizer + memory."""
    def __init__(self, documents: List[ResearchDocument], memory_path: str,
                 embedding_backend="tfidf", llm: Optional[Any] = None,
                 quotes: Optional[Dict[str, Dict[str, Any]]] = None):
        self.documents = list(documents)
        self.retriever = SemanticRetriever(self.documents, embedding_backend=embedding_backend)
        self.memory = AnalysisMemory(memory_path)
        self.llm = llm
        self.quotes = quotes or {}
        self.registry = ToolRegistry()
        self.registry.register("retrieve", self.retriever.retrieve)
        self.registry.register("news_chain", lambda symbol: news_prompt_chain(
            [d for d in self.documents if d.symbol.upper() == symbol.upper()]))
        self.registry.register("quote_lookup", lambda symbol: self.quotes.get(symbol.upper()))
        self.registry.register("calculator", safe_calculate)

    @staticmethod
    def plan(symbol: str, question: str, specialist: str, previous_notes: List[Dict[str, Any]]) -> List[str]:
        steps = [f"Classify task and route to {specialist} specialist",
                 f"Retrieve grounded evidence for {symbol} and question",
                 "Extract relevant numerical facts and consult tools if needed",
                 "Generate concise answer with explicit evidence identifiers",
                 "Evaluate grounding, question coverage, and arithmetic consistency"]
        if previous_notes:
            steps.append("Use previous feedback notes to improve result")
        steps.append("Refine once on failure; retain short feedback memory")
        return steps

    def _evidence_answer(self, question: str, hits: List[Dict[str, Any]], specialist: str,
                         symbol: str) -> Dict[str, Any]:
        if not hits or (hits[0]["score"] < 0.045 and specialist != "news"):
            return {"answer": "Insufficient evidence in the available sources.",
                    "source_ids": [], "calculation": None, "abstained": True}
        content = "\n".join(row["document"].display()[:850] for row in hits[:3])
        source_ids = [h["document"].doc_id for h in hits[:3]]

        # A model can synthesize from retrieved text, but source IDs are enforced downstream.
        if self.llm is not None:
            prompt = ("You are a cautious financial-research assistant. Answer only from the "
                      "EVIDENCE. If the evidence is inadequate, say 'Insufficient evidence'. "
                      "Do not invent figures, dates, source IDs, or citations. "
                      "Show a short computation if applicable.\n"
                      f"QUESTION: {question}\nEVIDENCE:\n{content}\nANSWER:")
            answer = self.llm.generate(prompt)
            abstained = "insufficient evidence" in answer.lower() or not answer.strip()
            calculation = None
            # A second, constrained tool-use pass is attempted for numerical questions.
            # The expression is NOT trusted until a restricted calculator evaluates it.
            if not abstained and re.search(r"\b(percentage|percent|how much|ratio|change)\b", question, re.I):
                eq_prompt = ("From the evidence below, return only one numerical Python arithmetic "
                             "expression that answers the question, such as "
                             "(250-200)/200*100. No variables or function calls. "
                             "Return NONE if not enough evidence.\n" 
                             f"QUESTION: {question}\nEVIDENCE: {content[:1200]}\nEXPRESSION:")
                candidate = self.llm.generate(eq_prompt, max_new_tokens=55).strip()
                candidate = candidate.removeprefix("EXPRESSION:").strip().splitlines()[0]
                candidate = candidate.rstrip(". ")
                if candidate != "NONE":
                    try:
                        value = self.registry.call("calculator", expression=candidate)
                        calculation = {"expression": candidate, "value": value}
                        answer += f" [Calculator result: {value:.4g} from {candidate}]"
                    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                        pass
            return {"answer": answer or "Insufficient evidence in the available sources.",
                    "source_ids": [] if abstained else source_ids,
                    "calculation": calculation, "abstained": abstained}

        # Fully labeled synthetic demonstration only; NOT FinQA benchmark prediction.
        primary = hits[0]["document"]
        if primary.metadata.get("demo_arithmetic") and ("percent" in question.lower() or "percentage" in question.lower()):
            v = primary.metadata["demo_arithmetic"]
            expr = f"({v['later']}-{v['earlier']})/{v['earlier']}*100"
            result = self.registry.call("calculator", expression=expr)
            text = (f"Revenue increased from ${v['earlier']:,.0f} million in {v['earlier_year']} "
                    f"to ${v['later']:,.0f} million in {v['later_year']}, "
                    f"a change of {result:.2f}% (calculated as {expr}).")
            return {"answer": text, "source_ids": [primary.doc_id],
                    "calculation": {"expression": expr, "value": result}, "abstained": False}
        if specialist == "news":
            chain = self.registry.call("news_chain", symbol=symbol)
            if chain["items"]:
                return {"answer": chain["summary"],
                        "source_ids": [r["document"].doc_id for r in chain["items"][:3]],
                        "calculation": None, "abstained": False}
        if specialist == "market":
            quote = self.registry.call("quote_lookup", symbol=symbol)
            if quote:
                return {"answer": f"Reference market snapshot: {quote} (offline snapshot; not live pricing).",
                        "source_ids": [f"snapshot-{symbol}"], "calculation": None,
                        "abstained": False}
        # An extractive fallback can present evidence, but MUST NOT invent computed answers.
        return {"answer": f"Evidence retrieved; numerical conclusion not independently verified: "
                          f"{primary.content[:250]}",
                "source_ids": [primary.doc_id], "calculation": None, "abstained": False}

    @staticmethod
    def evaluate(question: str, answer: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        if answer["abstained"]:
            return {"score": 0.5, "passed": False,
                    "feedback": "No supported answer; retrieve more evidence or request review."}
        valid_ids = {row["document"].doc_id for row in hits}
        cited_ids = set(answer.get("source_ids", []))
        citation_ok = bool(cited_ids) and cited_ids.issubset(valid_ids.union(
            {f"snapshot-{d['document'].symbol}" for d in hits}))
        calculation = answer.get("calculation")
        arithmetic_ok = True
        if calculation:
            try:
                recomputed = safe_calculate(calculation["expression"])
                arithmetic_ok = math.isclose(recomputed, calculation["value"], abs_tol=1e-6)
            except (ValueError, SyntaxError, ZeroDivisionError):
                arithmetic_ok = False
        # Requirements requiring numeric results are not passed by mere quoting.
        numeric_question = bool(re.search(r"\b(percentage|percent|how much|ratio|change)\b", question, re.I))
        numeric_ok = not numeric_question or calculation is not None
        supported = bool(citation_ok and arithmetic_ok and numeric_ok)
        score = (int(citation_ok) + int(arithmetic_ok) + int(bool(numeric_ok))) / 3
        feedback = "Evidence checks passed." if supported else (
            "Check citations, verified arithmetic, and question coverage before relying on the answer.")
        return {"score": round(score, 3), "passed": supported, "feedback": feedback,
                "citation_ok": citation_ok, "arithmetic_ok": arithmetic_ok,
                "numeric_coverage_ok": bool(numeric_ok)}

    def run(self, symbol: str, question: str, top_k=3) -> Dict[str, Any]:
        """Plan -> use retrieved tools -> route -> evaluate -> refine -> store feedback.

        Note: evaluation is a structural check, not a guarantee of factual truth.
        """
        event_start = len(self.registry.events)
        specialist = detect_specialist(question)
        notes_before = self.memory.recall(symbol)
        steps = self.plan(symbol, question, specialist, notes_before)
        effective_k = min(6, top_k + 2) if any(n.get("outcome") == "review_needed" for n in notes_before) else top_k
        steps.append(f"Select retrieval depth: top-{effective_k} based on prior review notes")
        hits = self.registry.call("retrieve", query=question, symbol=symbol,
                                  top_k=effective_k,
                                  kind="news" if specialist == "news" else
                                  ("market" if specialist == "market" else "earnings"))
        routed_trace = None
        if specialist == "news":
            routed_trace = self.registry.call("news_chain", symbol=symbol)["trace"]
        elif specialist == "market":
            quote = self.registry.call("quote_lookup", symbol=symbol)
            if quote is not None:
                steps.append("Use reference market snapshot tool")
        draft = self._evidence_answer(question, hits, specialist, symbol)
        initial_eval = self.evaluate(question, draft, hits)
        iteration_trace = [{"iteration": 1, "feedback": initial_eval["feedback"],
                            "score": initial_eval["score"]}]
        current = draft
        current_eval = initial_eval
        if not initial_eval["passed"]:
            wider_hits = self.registry.call("retrieve", query=question, symbol=symbol,
                                           top_k=min(6, max(effective_k + 2, 4)),
                                           kind="news" if specialist == "news" else
                                           ("market" if specialist == "market" else "earnings"))
            if wider_hits:
                hits = wider_hits
                new_answer = self._evidence_answer(question, hits, specialist, symbol)
                new_eval = self.evaluate(question, new_answer, hits)
                iteration_trace.append({"iteration": 2, "feedback": new_eval["feedback"],
                                        "score": new_eval["score"]})
                if new_eval["score"] >= current_eval["score"]:
                    current, current_eval = new_answer, new_eval
        self.memory.store(symbol, specialist,
                          "passed" if current_eval["passed"] else "review_needed",
                          current_eval["feedback"])
        return {
            "symbol": symbol, "question": question, "specialist": specialist,
            "plan": steps, "backend": "local_hf_llm" if self.llm else "offline_extract",
            "retrieved": [{"id": x["document"].doc_id, "score": round(x["score"], 4),
                           "kind": x["document"].kind, "source": x["document"].source}
                          for x in hits],
            "answer": current["answer"], "source_ids": current["source_ids"],
            "evaluation": current_eval, "iterations": iteration_trace,
            "memory_before": notes_before, "memory_after": self.memory.recall(symbol),
            "news_chain": routed_trace,
            "tool_trace": [e.__dict__ for e in self.registry.events[event_start:]],
        }

def finqa_to_document(row: Dict[str, Any], split="unknown") -> ResearchDocument:
    """Convert one official FinQA JSON record to a document, excluding ground truth answers."""
    table = row.get("table") or []
    table_lines = [" | ".join(map(str, r)) if isinstance(r, list) else str(r)
                   for r in table]
    pre = row.get("pre_text") or []
    post = row.get("post_text") or []
    chunks = ["PRE-TABLE: " + " ".join(map(str, pre)),
              "FINANCIAL TABLE:\n" + "\n".join(table_lines),
              "POST-TABLE: " + " ".join(map(str, post))]
    ident = str(row.get("id") or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:14])
    return ResearchDocument(doc_id=ident, symbol="FINQA", kind="earnings",
                            title="FinQA report extract " + ident,
                            content="\n".join(chunks), source=f"FinQA {split} (Chen et al., 2021)",
                            metadata={"split": split})

def fetch_finqa(split="dev", limit=200, timeout=30) -> tuple:
    """Download original public FinQA JSON from the project repository.

    Separates the gold Q/A examples from retrieval passages, preventing leakage.
    """
    import requests
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train/dev/test")
    url = f"https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    raw = response.json()
    rows = raw[:limit]
    docs = [finqa_to_document(row, split=split) for row in rows]
    queries = [{"id": str(row.get("id")), "question": row.get("qa", {}).get("question", ""),
                "gold_answer": row.get("qa", {}).get("exe_ans"),
                "gold_program": row.get("qa", {}).get("program"),
                "doc_id": docs[i].doc_id}
               for i, row in enumerate(rows)]
    return docs, queries, {"source_url": url, "split": split,
                           "subset": len(rows), "note": "Contiguous subset for lab runtime, not full benchmark"}

def retrieval_metrics(retriever: SemanticRetriever, queries: List[Dict[str, Any]], k=3):
    """Evaluate whether source report appears in top-k, not generation correctness."""
    hits, reciprocal = [], []
    for row in queries:
        ranked = retriever.retrieve(row["question"], top_k=k, symbol="FINQA")
        ids = [x["document"].doc_id for x in ranked]
        target = row["doc_id"]
        hits.append(int(target in ids))
        reciprocal.append(1.0 / (ids.index(target) + 1) if target in ids else 0.0)
    return {"num_questions": len(queries), f"hit_at_{k}": sum(hits) / max(1,len(hits)),
            f"mrr_at_{k}": sum(reciprocal) / max(1,len(reciprocal))}

def normalize_numeric_answer(answer: Any) -> Optional[float]:
    """Parse a numeric final answer; percentages remain numerical percent values."""
    if answer is None:
        return None
    text = str(answer).lower().replace(",", "").replace("$", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None

def finqa_answer_accuracy(results: List[Dict[str, Any]], queries: List[Dict[str, Any]],
                          tolerance: float = 0.02) -> Dict[str, Any]:
    """Simple numeric accuracy on aligned samples; this is NOT official FinQA evaluation."""
    n = min(len(results), len(queries))
    matched = 0
    abstained = 0
    details = []
    for prediction, gold in zip(results[:n], queries[:n]):
        raw = str(prediction.get("answer", ""))
        pred = normalize_numeric_answer(raw)
        target = normalize_numeric_answer(gold.get("gold_answer"))
        correct = bool(pred is not None and target is not None and
                       math.isclose(pred, target, rel_tol=tolerance, abs_tol=tolerance))
        abstain = "insufficient evidence" in raw.lower()
        matched += int(correct)
        abstained += int(abstain)
        details.append({"id": gold.get("id"), "predicted": pred,
                        "gold": target, "correct": correct, "abstained": abstain})
    return {"n": n, "numeric_accuracy": matched / max(1,n),
            "abstention_rate": abstained/max(1,n), "details": details}

def fetch_yahoo_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    """Optional read-only live market tool; no trading or brokerage access."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", auto_adjust=False)
    if hist.empty:
        return None
    latest = hist.iloc[-1]
    return {"price": round(float(latest["Close"]), 4),
            "volume": int(latest["Volume"]),
            "as_of": str(hist.index[-1]), "source": "Yahoo Finance via yfinance",
            "note": "May be delayed; informational only"}

def demo_documents() -> List[ResearchDocument]:
    """Small clearly synthetic dataset for reproducible architecture tests."""
    return [
        ResearchDocument("DEMO-10K-2024", "DEMO", "earnings", "DEMO hypothetical revenue filing",
            "DEMO Corporation reported revenue of $200 million in 2023 and $250 million in 2024. "
            "Operating costs were $150 million in 2024. This is fictional educational data.",
            "Synthetic classroom filing", "2024-12-31",
            {"demo_arithmetic": {"earlier": 200, "later": 250,
                                 "earlier_year": 2023, "later_year": 2024}}),
        ResearchDocument("DEMO-N-1", "DEMO", "news", "DEMO announces new plant",
            "DEMO Corporation announced a proposed $20 million expansion for a new plant. "
            "Management expects construction to begin next year.",
            "Synthetic news release", "2025-01-11"),
        ResearchDocument("DEMO-N-2", "DEMO", "news", "DEMO quarterly earnings update",
            "DEMO Corporation reported 5% higher quarterly revenue in its illustrative update.",
            "Synthetic classroom news", "2025-02-14"),
        ResearchDocument("DEMO-M-1", "DEMO", "market", "DEMO reference market quote",
            "The illustrative quote for DEMO was $52.30 on a fictional reference date; "
            "this is not an actual traded stock.",
            "Synthetic market snapshot", "2025-02-14"),
    ]
