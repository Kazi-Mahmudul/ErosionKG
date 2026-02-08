# ErosionKG GraphRAG Evaluation Results

**Generated:** 2026-02-09 02:02:26  
**Dataset:** 20 questions (Simple: 1, Multi-Hop: 0)  
**Models:** ErosionKG (Groq Llama 3.3 70B) vs Baseline RAG (Groq Llama 3.3 70B)  
**Judge LLM:** Gemini 1.5 Flash

## Performance Comparison

| Evaluation Metric      | Simple Queries (n=1) | Multi-Hop Queries (n=0) | Overall Performance |
|------------------------|----------------------------------|---------------------------------------|---------------------|
| **ErosionKG (GraphRAG)**                                                                                     |
| Context Recall         | 1.000                      | 0.000                         | 1.000            |
| Context Precision      | 0.250                   | 0.000                      | 0.250         |
| Context Entity Recall  | 0.000               | 0.000                  | 0.000     |
| Noise Sensitivity      | 0.000                   | 0.000                      | 0.000         |
| Citation Accuracy      | 100.0%                  | 0.0%                     | 100.0%        |
| **Baseline RAG (Vector-only)**                                                                               |
| Context Recall         | 0.000                     | 0.000                        | 0.000           |
| Context Precision      | 0.000                  | 0.000                     | 0.000        |
| Context Entity Recall  | 0.000              | 0.000                 | 0.000    |
| Noise Sensitivity      | 0.000                  | 0.000                     | 0.000        |
| Citation Accuracy      | 0.0%                 | 0.0%                    | 0.0%       |

## Key Findings

### ErosionKG Advantages
1. **Citation Accuracy**: 100.0% vs 0.0% (baseline)
2. **Context Entity Recall**: 0.000 vs 0.000 (baseline)
3. **Noise Sensitivity**: 0.000 vs 0.000 (Lower is better)

### Performance Delta
- **Context Recall Improvement**: 0.0%
- **Context Entity Recall Improvement**: 0.0%
- **Citation Accuracy Improvement**: 100.0 percentage points

## Files
- Detailed results: `evaluation_results.json`
- Full response log: `evaluation_full_log.csv`
- Case studies: `case_studies.md`
