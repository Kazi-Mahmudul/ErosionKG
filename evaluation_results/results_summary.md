# ErosionKG GraphRAG Evaluation Results

**Generated:** 2026-02-08 17:26:41  
**Dataset:** 20 questions (Simple: 1, Multi-Hop: 0)  
**Models:** ErosionKG (Groq Llama 3.3 70B) vs Baseline RAG (Groq Llama 3.3 70B)  
**Judge LLM:** Gemini 1.5 Flash

## Performance Comparison

| Evaluation Metric      | Simple Queries (n=1) | Multi-Hop Queries (n=0) | Overall Performance |
|------------------------|----------------------------------|---------------------------------------|---------------------|
| **ErosionKG (GraphRAG)**                                                                                     |
| Faithfulness           | nan                        | 0.000                           | nan              |
| Answer Relevancy       | nan                    | 0.000                       | nan          |
| Context Recall         | nan                      | 0.000                         | nan            |
| Context Precision      | nan                   | 0.000                      | nan         |
| Citation Accuracy      | 100.0%                  | 0.0%                     | 100.0%        |
| Synthesis Success      | N/A                       | 0.0%                     | 0.0%      |
| **Baseline RAG (Vector-only)**                                                                               |
| Faithfulness           | nan                       | 0.000                          | nan             |
| Answer Relevancy       | nan                   | 0.000                      | nan         |
| Context Recall         | nan                     | 0.000                        | nan           |
| Context Precision      | nan                  | 0.000                     | nan        |
| Citation Accuracy      | 0.0%                 | 0.0%                    | 0.0%       |
| Synthesis Success      | N/A                      | 0.0%                    | 0.0%     |

## Key Findings

### ErosionKG Advantages
1. **Citation Accuracy**: 100.0% vs 0.0% (baseline)
2. **Multi-Hop Synthesis**: 0.0% success rate on cross-document questions
3. **Faithfulness**: nan vs nan (baseline)

### Performance Delta
- **Faithfulness Improvement**: 0.0%
- **Context Recall Improvement**: 0.0%
- **Citation Accuracy Improvement**: 100.0 percentage points

## Files
- Detailed results: `evaluation_results.json`
- Full response log: `evaluation_full_log.csv`
- Case studies: `case_studies.md`
