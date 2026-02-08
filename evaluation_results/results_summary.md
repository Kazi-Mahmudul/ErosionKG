# ErosionKG GraphRAG Evaluation Results

**Generated:** 2026-02-08 16:49:09  
**Dataset:** 20 questions (Simple: 2, Multi-Hop: 0)  
**Models:** ErosionKG (Groq Llama 3.3 70B) vs Baseline RAG (Groq Llama 3.3 70B)  
**Judge LLM:** Gemini 1.5 Flash

## Performance Comparison

| Evaluation Metric      | Simple Queries (n=2) | Multi-Hop Queries (n=0) | Overall Performance |
|------------------------|----------------------------------|---------------------------------------|---------------------|
| **ErosionKG (GraphRAG)**                                                                                     |
| Faithfulness           | 0.000                        | 0.000                           | 0.000              |
| Answer Relevancy       | 0.000                    | 0.000                       | 0.000          |
| Context Recall         | 0.000                      | 0.000                         | 0.000            |
| Context Precision      | 0.000                   | 0.000                      | 0.000         |
| Citation Accuracy      | 0.0%                  | 0.0%                     | 0.0%        |
| Synthesis Success      | N/A                       | 0.0%                     | 0.0%      |
| **Baseline RAG (Vector-only)**                                                                               |
| Faithfulness           | 0.000                       | 0.000                          | 0.000             |
| Answer Relevancy       | 0.000                   | 0.000                      | 0.000         |
| Context Recall         | 0.000                     | 0.000                        | 0.000           |
| Context Precision      | 0.000                  | 0.000                     | 0.000        |
| Citation Accuracy      | 0.0%                 | 0.0%                    | 0.0%       |
| Synthesis Success      | N/A                      | 0.0%                    | 0.0%     |

## Key Findings

### ErosionKG Advantages
1. **Citation Accuracy**: 0.0% vs 0.0% (baseline)
2. **Multi-Hop Synthesis**: 0.0% success rate on cross-document questions
3. **Faithfulness**: 0.000 vs 0.000 (baseline)

### Performance Delta
- **Faithfulness Improvement**: 0.0%
- **Context Recall Improvement**: 0.0%
- **Citation Accuracy Improvement**: 0.0 percentage points

## Files
- Detailed results: `evaluation_results.json`
- Full response log: `evaluation_full_log.csv`
- Case studies: `case_studies.md`
