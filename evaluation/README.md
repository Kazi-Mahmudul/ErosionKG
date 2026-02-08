# ErosionKG GraphRAG Evaluation System

## Quick Start

### 1. Prerequisites
Ensure dependencies are installed:
```bash
pip install -r requirements-eval.txt
```

### 2. Start the ErosionKG API
```bash
cd api
python index.py
```

The API should be running at `http://localhost:8000`

### 3. Run Evaluation

**Test with first 3 questions:**
```bash
python evaluate_graphrag.py --questions 1-3
```

**Run full evaluation (all 20 questions):**
```bash
python evaluate_graphrag.py --all
```

**Custom options:**
```bash
python evaluate_graphrag.py \
  --dataset api/data/golden_dataset.txt \
  --api-url http://localhost:8000 \
  --output-dir evaluation_results \
  --all
```

### 4. Cloud Run Deployment
If your API is deployed to Google Cloud Run:
```bash
python evaluate_graphrag.py \
  --api-url https://your-service-url.run.app \
  --all
```

## Expected Output

The evaluation will generate 4 files in `evaluation_results/`:

1. **`evaluation_results.json`** - Detailed JSON with all metrics per question
2. **`results_summary.md`** - Publication-ready table comparing ErosionKG vs Baseline
3. **`evaluation_full_log.csv`** - Complete Q&A log for appendix
4. **`case_studies.md`** - Top 3 qualitative examples for paper

## Terminal Output Example

```
================================================================================
EROSIONKG GRAPHRAG EVALUATION SYSTEM
================================================================================
Dataset: api/data/golden_dataset.txt
API URL: http://localhost:8000
Output Dir: evaluation_results
================================================================================

[STEP 1] Parsing golden dataset...
✅ Loaded 20 questions

[STEP 2] Initializing systems...
✅ Baseline RAG initialized
✅ ErosionKG API endpoint ready

[STEP 3-7] Processing 20 questions...
================================================================================
[Q1/20] Complexity: Low
Question: What methodology is the 'Options by Context' (OxC) approach based on?...
  -> Querying ErosionKG... [4.2s]
  -> Querying Baseline RAG... [2.8s]
  -> Calculating RAGAS metrics (ErosionKG)...
  -> Calculating RAGAS metrics (Baseline)...
     - Faithfulness: ErosionKG=1.00, Baseline=0.87
     - Citation Accuracy: ErosionKG=100%, Baseline=0%
================================================================================
[Q13/20] Complexity: Medium (2 Sources)
Question: How do Val d'Arda climate factors interact with Shainberg's flume findings?...
  -> Querying ErosionKG... [7.8s]
  -> Querying Baseline RAG... [3.1s]
  -> Calculating RAGAS metrics (ErosionKG)...
  -> Calculating RAGAS metrics (Baseline)...
  -> Judge LLM synthesis review...
     - Faithfulness: ErosionKG=0.95, Baseline=0.68
     - Citation Accuracy: ErosionKG=100%, Baseline=50%
     - Synthesis: ErosionKG=PASS, Baseline=FAIL
================================================================================

[STEP 7] Generating reports...
✅ Saved detailed results to: evaluation_results/evaluation_results.json
✅ Saved summary table to: evaluation_results/results_summary.md
✅ Saved full log to: evaluation_results/evaluation_full_log.csv
✅ Saved case studies to: evaluation_results/case_studies.md

================================================================================
✅ EVALUATION COMPLETE!
================================================================================
Results saved to: evaluation_results/
  - evaluation_results.json (detailed)
  - results_summary.md (publication table)
  - evaluation_full_log.csv (all responses)
  - case_studies.md (qualitative examples)
```

## Evaluation Metrics Explained

### RAGAS Metrics
1. **Faithfulness** (0-1): Is the answer grounded in retrieved evidence?
2. **Answer Relevancy** (0-1): Does the answer address the question?
3. **Context Recall** (0-1): Did retrieval find chunks with ground truth?
4. **Context Precision** (0-1): Are retrieved chunks relevant?

### Custom Metrics
5. **Citation Accuracy** (%): Percentage of expected DOIs cited correctly
6. **Synthesis Success** (%): Judge LLM evaluation of multi-hop synthesis

## Using Results in Your Paper

### Publication Table
Copy from `results_summary.md`:
```markdown
| Evaluation Metric      | Simple Queries (n=10) | Multi-Hop Queries (n=10) | Overall Performance |
|------------------------|----------------------------------|---------------------------------------|---------------------|
| **ErosionKG (GraphRAG)**                                                                                     |
| Faithfulness           | 0.980                        | 0.910                           | 0.945              |
| ...
```

### Qualitative Examples
Use top 3 case studies from `case_studies.md` to demonstrate:
1. **Hallucination Prevention** - Where baseline invents facts
2. **Cross-Document Synthesis** - Where ErosionKG connects multiple papers
3. **Citation Grounding** - Where ERosionKG provides precise DOIs

### Full Response Log
Attach `evaluation_full_log.csv` as supplementary material for reviewers

## Troubleshooting

### API Connection Error
```python
Error: Connection refused to http://localhost:8000
```
**Solution:** Make sure the API is running:
```bash
cd api
python index.py
```

### RAGAS Timeout
```python
Error: RAGAS calculation timeout
```
**Solution:** Increase timeout in `evaluate_graphrag.py` or use smaller batch

### Memory Issues
```python
Error: Out of memory
```
**Solution:** Run in smaller batches:
```bash
python evaluate_graphrag.py --questions 1-5
python evaluate_graphrag.py --questions 6-10
# ... then merge results
```

## Customization

### Change Judge LLM
Edit `evaluate_graphrag.py`:
```python
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Use Pro instead of Flash
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
```

### Adjust Retrieval Parameters
Edit `baseline_rag.py` or API call:
```python
# More chunks
top_k=5  # instead of 3

# More graph hops (ErosionKG only)
retrieval_result = retriever.retrieve(query, top_k=5, hops=2)
```

### Add Custom Metrics
Add to `evaluate_graphrag.py`:
```python
def calculate_custom_metric(question, answer, ground_truth):
    # Your custom logic
    return score
```

## Files Overview

```
erosion-kg/
├── api/
│   ├── index.py                  # [MODIFIED] Added /api/chat_eval endpoint
│   └── data/
│       └── golden_dataset.txt    # 20 evaluation questions
├── baseline_rag.py               # [NEW] Vector-only RAG for comparison
├── evaluate_graphrag.py          # [NEW] Main 7-step evaluation script
├── requirements-eval.txt         # [NEW] Evaluation dependencies
└── evaluation_results/           # [GENERATED] Output directory
    ├── evaluation_results.json
    ├── results_summary.md
    ├── evaluation_full_log.csv
    └── case_studies.md
```

## Next Steps

1. **Run Test Evaluation**: Start with 3 questions to verify everything works
2. **Review Baseline Results**: Ensure baseline RAG is working correctly
3. **Run Full Evaluation**: Execute all 20 questions (~15-20 minutes)
4. **Analyze Results**: Review `results_summary.md` for publication table
5. **Select Case Studies**: Use `case_studies.md` examples in your paper
6. **Iterate if Needed**: Adjust parameters and re-run for optimal results
