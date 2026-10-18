# FinSight — Multi-Agent Financial Analysis System

**University of San Diego · AAI-520 · Team 3**

**Team:** Hivanshu Kapania · Karthik K · Navneet Kumar

## Repository URL

https://github.com/karthik1997/finsight-multi-agent-financial-analysis

## What was built

FinSight is an educational, read-only financial research prototype that combines:

- **Planner:** selects a research plan based on symbol, question, prior notes.
- **Tool use:** vector-space retrieval, news classification/extraction, arithmetic calculator, and read-only market quote lookup.
- **Memory:** JSON-based run notes with feedback; a previous failed review increases subsequent retrieval depth.
- **Routing:** earnings, news, and market specialist pathways.
- **Prompt chaining:** **Ingest News → Preprocess → Classify → Extract → Summarize**.
- **Evaluator–optimizer:** creates an answer, checks evidence and arithmetic, widens retrieval once when necessary, and records the result.
- **Optional Hugging Face LLM:** `google/flan-t5-small` generated answers from retrieved passages. The offline demonstration does not claim to use a model.
- **Official FinQA evaluation path:** fetch original JSON from the authors' repository; retain answers separately from retrieved documents; measure hit@3, MRR@3, simple numeric answer accuracy, and abstention.

## Files

- `FinSight_Team3_Final_Notebook.ipynb` — self-contained, source-code-inclusive notebook.
- `FinSight_Team3_Executed_Demo.html` — executed, self-contained HTML **of the synthetic offline architecture demonstration**, clearly labeled.
- `finsight_core.py` — PEP 8-oriented reference code for repository collaboration.
- `test_finsight.py` — reproducible test suite.
- `requirements.txt` — dependencies; most are only needed for the Colab real-data path.
- `team_contributions.md` — responsibilities from the status update.

## Run the offline demonstrator

```bash
python -m pip install numpy pandas scikit-learn requests nbformat nbconvert
python -m unittest -v test_finsight.py
jupyter notebook FinSight_Team3_Final_Notebook.ipynb
```

Run All. On a regular environment without Colab, the notebook runs the fully labeled synthetic example so the agent patterns can be verified without data or model downloads.

## Run the real FinQA + Hugging Face mode (recommended for final academic evaluation)

1. Open the `.ipynb` in Google Colab.
2. Set `RUN_REAL_FINQA = True`, `USE_SENTENCE_EMBEDDINGS = True`, and `USE_LOCAL_HF_LLM = True` in **Configuration**.
3. Enable Internet and optionally a GPU in Runtime settings.
4. Run `pip install -q sentence-transformers transformers sentencepiece yfinance` if the notebook offers installation or the packages are absent.
5. Run all cells, inspect both retrieval and model outputs, and export to HTML or PDF.
6. Confirm the verified repository URL is visible in the exported report, and add team-validated contribution and evaluation observations.

**FinQA:** official `https://github.com/czyssrs/FinQA` dataset, using author-provided train/dev/test files. The project reads `dev.json` without putting answer labels or oracle programs in the retrieval context. FinQA does not expose current stock prices or tickers in each QA row, so these are separate functionality demonstrations.

## Academic integrity and validation

The HTML produced here contains *real executed outputs of the synthetic classroom DEMO company*, not actual FinQA benchmark or live Yahoo Finance market results. Do not present that HTML as proof of real-data model performance. To meet all project requirements rigorously, the team should execute and export the FinQA + Hugging Face mode.

AI tools were used to support drafting, coding, testing, notebook layout, and documentation; the team must review, understand, modify, cite, and own its actual contributions. No claims are made about fabricated Git commits or team actions.

### Project limitations

- No investment advice, no trading access, no write privileges to brokerages.
- The feedback evaluator is structural and can miss semantic hallucinations; human oversight remains important.
- Miniature FinQA subset metrics are **not** official leaderboard metrics.
- The local `flan-t5-small` baseline is lightweight, not a specialized financial numeric reasoner.
- The synthetic offline example is not a real company.