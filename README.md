# FinSight — Multi-Agent Financial Analysis System

**AAI-520 — University of San Diego · Team 3**

**Members:** Hivanshu Kapania, Karthik K and Navneet Kumar

## Final submission

- [Executed final notebook](FinSight_Team3_Final_Notebook.ipynb) — **real FinQA development subset, MiniLM embeddings, FLAN-T5-Small model, 20 executed Python cells, synthetic architecture demonstration and live read-only quote**.
- [**Final Canvas HTML**](FinSight_Team3_Final_Submission.html) — code, real outputs, analysis, and AI assistance disclosure. Submit **this HTML** or an exported PDF on Canvas.

**GitHub repository:** https://github.com/karthik1997/finsight-multi-agent-financial-analysis

## Project requirements implemented

- Autonomous investment-research **planning**
- Bounded dynamic **tool invocation**: report retrieval, classification, extraction, restricted calculator, optional read-only quote
- Self-reflection/evaluator–optimizer with one evidence refinement step
- Persistent feedback memory with retrieval-depth adjustment
- **Prompt chaining:** Ingest News → Preprocess → Classify → Extract → Summarize
- **Routing:** earnings, news, market specialists
- **Evaluator–optimizer:** generate → evaluate → refine

## Observed REAL data results (educational, not official benchmark)

- Downloaded: **100 FinQA `dev` records** from the [official repository](https://github.com/czyssrs/FinQA)
- Dense retrieval: **all-MiniLM-L6-v2**
- Generation: **google/flan-t5-small**
- Retrieval **Hit@3 = 56.7%**, **MRR@3 = 0.5111** over the first **30** questions
- Numeric answer accuracy **0/8 = 0%**, abstention **1/8 = 12.5%** over eight samples
- Structural `Verified=True` **does not** establish numeric correctness, and the internal calculator produced some invalid arithmetic outputs

**These are honest findings, not an investment-ready agent.** A separate fictional DEMO corpus tests architecture and should not be mistaken for real benchmark outcomes.

## Test and development

Core: `finsight_core.py`, `test_finsight.py` and `requirements.txt`. Run `python -m unittest -v test_finsight.py`. Nine core local tests passed in the working environment. The real HF evaluation requires a networked environment with model downloads.

A GitHub Actions workflow is provided under `.github/workflows/python-tests.yml`. Actual contributions and teammate commits must be documented separately.

## AI-use disclosure

AI tools were used to assist with project scaffolding, coding, test design, explanation, documentation, and revisions. All final results and limitations must be reviewed by the team; this repository does not claim that autonomous model numerical reasoning was successful.
