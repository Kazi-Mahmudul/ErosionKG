"""
Comprehensive GraphRAG Evaluation System
Implements rigorous 7-step methodology for publication-ready metrics
"""
import re
import json
import time
import csv
import os
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# RAGAS imports - v0.4+ compatibility
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._context_precision import ContextPrecision
from ragas import evaluate
from datasets import Dataset

# LangChain for judge LLM and embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Baseline RAG
from baseline_rag import BaselineRAG

# Load env
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# STEP 1: Data Parsing & Prep
# ============================================================================

def parse_golden_dataset(filepath: str) -> List[Dict]:
    """
    Parse golden_dataset.txt and extract questions with metadata
    Returns list of dicts with: id, question, ground_truth, expected_dois, complexity
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to extract each question block
    pattern = r'\[(\d+)\] =+\s+(.*?)(?=\[\d+\] =+|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    questions = []
    for question_id, block in matches:
        # Extract question
        q_match = re.search(r'Question:\s*(.*?)(?=\n\nGround Truth:)', block, re.DOTALL)
        question_text = q_match.group(1).strip() if q_match else ""
        
        # Extract ground truth
        gt_match = re.search(r'Ground Truth:\s*(.*?)(?=\n\nContext Source:)', block, re.DOTALL)
        ground_truth = gt_match.group(1).strip() if gt_match else ""
        
        # Extract DOIs from context source
        doi_match = re.search(r'Context Source:\s*(.*?)(?=\n\nComplexity:)', block, re.DOTALL)
        doi_text = doi_match.group(1).strip() if doi_match else ""
        expected_dois = re.findall(r'https://doi\.org/[^\s,)]+', doi_text)
        
        # Extract complexity
        complexity_match = re.search(r'Complexity:\s*(.*?)(?=\n|$)', block, re.DOTALL)
        complexity = complexity_match.group(1).strip() if complexity_match else "Unknown"
        
        questions.append({
            "id": int(question_id),
            "question": question_text,
            "ground_truth": ground_truth,
            "expected_dois": expected_dois,
            "complexity": complexity,
            "num_sources_required": len(expected_dois)
        })
    
    return questions


# ============================================================================
# STEP 2: API Invocation
# ============================================================================

def query_erosionkg(question: str, api_url: str = None) -> Dict:
    """
    Query ErosionKG GraphRAG system via /api/chat_eval endpoint
    Returns: answer, contexts, dois, response_time
    """
    import requests
    
    if api_url is None:
        # Use NEXT_PUBLIC_API_URL from .env
        api_url = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")
    
    start_time = time.time()
    
    response = requests.post(
        f"{api_url}/api/chat_eval",
        json={"query": question, "stream": False},
        timeout=60
    )
    
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        return {
            "answer": data.get("response", ""),
            "contexts": data.get("contexts", []),
            "dois": data.get("dois", []),
            "response_time_sec": end_time - start_time,
            "num_chunks": data.get("num_chunks", 0),
            "num_triplets": data.get("num_triplets", 0)
        }
    else:
        return {
            "answer": f"ERROR: {response.status_code}",
            "contexts": [],
            "dois": [],
            "response_time_sec": end_time - start_time,
            "error": response.text
        }


def query_baseline(question: str, baseline_rag: BaselineRAG) -> Dict:
    """
    Query baseline vector-only RAG
    Returns: answer, contexts, dois, response_time
    """
    result = baseline_rag.query(question, top_k=3)
    return {
        "answer": result["answer"],
        "contexts": result["contexts"],
        "dois": result["dois"],
        "response_time_sec": result["response_time_sec"]
    }


# ============================================================================
# STEP 4: RAGAS Metric Calculation
# ============================================================================

def calculate_ragas_metrics(question: str, answer: str, ground_truth: str, contexts: List[str]) -> Dict:
    """
    Calculate RAGAS metrics using Gemini as judge LLM
    Returns: faithfulness, answer_relevancy, context_recall, context_precision
    """
    # Initialize Gemini judge LLM
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    
    # Initialize Google embeddings for RAGAS
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create dataset in RAGAS format
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    }
    dataset = Dataset.from_dict(data)
    
    # Calculate metrics
    try:
        result = evaluate(
            dataset,
            metrics=[
                Faithfulness(llm=judge_llm),
                AnswerRelevancy(llm=judge_llm, embeddings=gemini_embeddings),
                ContextRecall(llm=judge_llm),
                ContextPrecision(llm=judge_llm)
            ],
            llm=judge_llm,
            embeddings=gemini_embeddings
        ).to_pandas().iloc[0] # Fix: RAGAS evaluate returns a Dataset, convert to pandas and get the first row
        
        return {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_recall": result["context_recall"],
            "context_precision": result["context_precision"]
        }
    except Exception as e:
        print(f"RAGAS error: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "error": str(e)
        }


# ============================================================================
# STEP 5: Citation Accuracy Verification
# ============================================================================

def extract_dois_from_text(text: str) -> List[str]:
    """Extract all DOIs from response text"""
    return re.findall(r'https://doi\.org/[^\s,)]+', text)


def calculate_citation_accuracy(response_text: str, expected_dois: List[str]) -> Dict:
    """
    Calculate citation accuracy percentage
    Returns: accuracy, found_dois, missing_dois
    """
    if not expected_dois:
        return {
            "accuracy": 1.0,
            "found_dois": [],
            "missing_dois": [],
            "hallucinated_dois": []
        }
    
    found_dois = extract_dois_from_text(response_text)
    
    # Normalize DOIs for comparison (remove trailing punctuation)
    def normalize_doi(doi):
        return doi.rstrip('.,;:)')
    
    found_normalized = set(normalize_doi(d) for d in found_dois)
    expected_normalized = set(normalize_doi(d) for d in expected_dois)
    
    # Calculate matches
    matches = found_normalized.intersection(expected_normalized)
    missing = expected_normalized - found_normalized
    hallucinated = found_normalized - expected_normalized
    
    accuracy = len(matches) / len(expected_normalized) if expected_normalized else 1.0
    
    return {
        "accuracy": accuracy,
        "found_dois": list(matches),
        "missing_dois": list(missing),
        "hallucinated_dois": list(hallucinated)
    }


# ============================================================================
# STEP 6: Judge LLM Final Review (Multi-Hop Synthesis)
# ============================================================================

JUDGE_PROMPT = """You are evaluating a GraphRAG system's ability to synthesize information across multiple research papers.

Question (requires {num_sources} sources): {question}

Expected sources: {expected_dois}

Bot's Answer: {answer}

Evaluation Criteria:
1. Does the answer reference findings from ALL {num_sources} required papers?
2. Does it explicitly connect or compare information across papers?
3. Are the connections logical and accurate based on the ground truth?

Ground Truth (for reference): {ground_truth}

Output ONLY in this format:
RESULT: PASS or FAIL
REASONING: [1-2 sentences explaining your decision]
"""

def judge_multi_hop_synthesis(question: str, answer: str, ground_truth: str, expected_dois: List[str], num_sources_required: int) -> Dict:
    """
    Use judge LLM to evaluate multi-hop synthesis quality
    Returns: passed (bool), reasoning (str)
    """
    if num_sources_required < 2:
        return {"applicable": False, "passed": None, "reasoning": "N/A - single source question"}
    
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    
    prompt = JUDGE_PROMPT.format(
        num_sources=num_sources_required,
        question=question,
        expected_dois=", ".join(expected_dois),
        answer=answer,
        ground_truth=ground_truth
    )
    
    try:
        response = judge_llm.invoke(prompt)
        response_text = response.content
        
        # Parse result
        passed = "PASS" in response_text.upper().split("REASONING:")[0]
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*)', response_text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text
        
        return {
            "applicable": True,
            "passed": passed,
            "reasoning": reasoning
        }
    except Exception as e:
        return {
            "applicable": True,
            "passed": False,
            "reasoning": f"Error: {str(e)}"
        }


# ============================================================================
# STEP 7: Automated Report Generation
# ============================================================================

def save_detailed_results(results: List[Dict], output_path: str):
    """Save detailed JSON results"""
    metadata = {
        "metadata": {
            "dataset": "golden_dataset.txt",
            "total_questions": len(results),
            "timestamp": datetime.now().isoformat(),
            "models": {
                "erosionkg_llm": "llama-3.3-70b-versatile (Groq)",
                "baseline_llm": "llama-3.3-70b-versatile (Groq)",
                "judge_llm": "gemini-1.5-flash (Google)"
            }
        },
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved detailed results to: {output_path}")


def save_summary_table(results: List[Dict], output_path: str):
    """Generate publication-ready summary table in Markdown"""
    # Separate by complexity
    simple = [r for r in results if "Low" in r.get("complexity", "")]
    multi_hop = [r for r in results if "Medium" in r.get("complexity", "") or "High" in r.get("complexity", "")]
    
    def avg(lst, key, subkey=None):
        if not lst:
            return 0.0
        if subkey:
            values = [r[key].get(subkey, 0) for r in lst if key in r]
        else:
            values = [r[key] for r in lst if key in r]
        return sum(values) / len(values) if values else 0.0
    
    # Calculate averages
    metrics = {
        "ErosionKG (GraphRAG)": {
            "simple": {
                "faithfulness": avg(simple, "erosionkg", "faithfulness"),
                "answer_relevancy": avg(simple, "erosionkg", "answer_relevancy"),
                "context_recall": avg(simple, "erosionkg", "context_recall"),
                "context_precision": avg(simple, "erosionkg", "context_precision"),
                "citation_accuracy": avg(simple, "erosionkg_citation", "accuracy") * 100,
                "synthesis_success": "N/A"
            },
            "multi_hop": {
                "faithfulness": avg(multi_hop, "erosionkg", "faithfulness"),
                "answer_relevancy": avg(multi_hop, "erosionkg", "answer_relevancy"),
                "context_recall": avg(multi_hop, "erosionkg", "context_recall"),
                "context_precision": avg(multi_hop, "erosionkg", "context_precision"),
                "citation_accuracy": avg(multi_hop, "erosionkg_citation", "accuracy") * 100,
                "synthesis_success": sum(1 for r in multi_hop if r.get("erosionkg_synthesis", {}).get("passed", False)) / len(multi_hop) * 100 if multi_hop else 0
            }
        },
        "Baseline RAG (Vector-only)": {
            "simple": {
                "faithfulness": avg(simple, "baseline", "faithfulness"),
                "answer_relevancy": avg(simple, "baseline", "answer_relevancy"),
                "context_recall": avg(simple, "baseline", "context_recall"),
                "context_precision": avg(simple, "baseline", "context_precision"),
                "citation_accuracy": avg(simple, "baseline_citation", "accuracy") * 100,
                "synthesis_success": "N/A"
            },
            "multi_hop": {
                "faithfulness": avg(multi_hop, "baseline", "faithfulness"),
                "answer_relevancy": avg(multi_hop, "baseline", "answer_relevancy"),
                "context_recall": avg(multi_hop, "baseline", "context_recall"),
                "context_precision": avg(multi_hop, "baseline", "context_precision"),
                "citation_accuracy": avg(multi_hop, "baseline_citation", "accuracy") * 100,
                "synthesis_success": sum(1 for r in multi_hop if r.get("baseline_synthesis", {}).get("passed", False)) / len(multi_hop) * 100 if multi_hop else 0
            }
        }
    }
    
    # Calculate overall
    for system in metrics:
        metrics[system]["overall"] = {
            "faithfulness": (metrics[system]["simple"]["faithfulness"] * len(simple) + metrics[system]["multi_hop"]["faithfulness"] * len(multi_hop)) / len(results),
            "answer_relevancy": (metrics[system]["simple"]["answer_relevancy"] * len(simple) + metrics[system]["multi_hop"]["answer_relevancy"] * len(multi_hop)) / len(results),
            "context_recall": (metrics[system]["simple"]["context_recall"] * len(simple) + metrics[system]["multi_hop"]["context_recall"] * len(multi_hop)) / len(results),
            "context_precision": (metrics[system]["simple"]["context_precision"] * len(simple) + metrics[system]["multi_hop"]["context_precision"] * len(multi_hop)) / len(results),
            "citation_accuracy": (metrics[system]["simple"]["citation_accuracy"] * len(simple) + metrics[system]["multi_hop"]["citation_accuracy"] * len(multi_hop)) / len(results),
            "synthesis_success": metrics[system]["multi_hop"]["synthesis_success"]
        }
    
    # Generate markdown table
    markdown = f"""# ErosionKG GraphRAG Evaluation Results

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Dataset:** 20 questions (Simple: {len(simple)}, Multi-Hop: {len(multi_hop)})  
**Models:** ErosionKG (Groq Llama 3.3 70B) vs Baseline RAG (Groq Llama 3.3 70B)  
**Judge LLM:** Gemini 1.5 Flash

## Performance Comparison

| Evaluation Metric      | Simple Queries (n={len(simple)}) | Multi-Hop Queries (n={len(multi_hop)}) | Overall Performance |
|------------------------|----------------------------------|---------------------------------------|---------------------|
| **ErosionKG (GraphRAG)**                                                                                     |
| Faithfulness           | {metrics["ErosionKG (GraphRAG)"]["simple"]["faithfulness"]:.3f}                        | {metrics["ErosionKG (GraphRAG)"]["multi_hop"]["faithfulness"]:.3f}                           | {metrics["ErosionKG (GraphRAG)"]["overall"]["faithfulness"]:.3f}              |
| Answer Relevancy       | {metrics["ErosionKG (GraphRAG)"]["simple"]["answer_relevancy"]:.3f}                    | {metrics["ErosionKG (GraphRAG)"]["multi_hop"]["answer_relevancy"]:.3f}                       | {metrics["ErosionKG (GraphRAG)"]["overall"]["answer_relevancy"]:.3f}          |
| Context Recall         | {metrics["ErosionKG (GraphRAG)"]["simple"]["context_recall"]:.3f}                      | {metrics["ErosionKG (GraphRAG)"]["multi_hop"]["context_recall"]:.3f}                         | {metrics["ErosionKG (GraphRAG)"]["overall"]["context_recall"]:.3f}            |
| Context Precision      | {metrics["ErosionKG (GraphRAG)"]["simple"]["context_precision"]:.3f}                   | {metrics["ErosionKG (GraphRAG)"]["multi_hop"]["context_precision"]:.3f}                      | {metrics["ErosionKG (GraphRAG)"]["overall"]["context_precision"]:.3f}         |
| Citation Accuracy      | {metrics["ErosionKG (GraphRAG)"]["simple"]["citation_accuracy"]:.1f}%                  | {metrics["ErosionKG (GraphRAG)"]["multi_hop"]["citation_accuracy"]:.1f}%                     | {metrics["ErosionKG (GraphRAG)"]["overall"]["citation_accuracy"]:.1f}%        |
| Synthesis Success      | {metrics["ErosionKG (GraphRAG)"]["simple"]["synthesis_success"]}                       | {metrics["ErosionKG (GraphRAG)"]["multi_hop"]["synthesis_success"]:.1f}%                     | {metrics["ErosionKG (GraphRAG)"]["multi_hop"]["synthesis_success"]:.1f}%      |
| **Baseline RAG (Vector-only)**                                                                               |
| Faithfulness           | {metrics["Baseline RAG (Vector-only)"]["simple"]["faithfulness"]:.3f}                       | {metrics["Baseline RAG (Vector-only)"]["multi_hop"]["faithfulness"]:.3f}                          | {metrics["Baseline RAG (Vector-only)"]["overall"]["faithfulness"]:.3f}             |
| Answer Relevancy       | {metrics["Baseline RAG (Vector-only)"]["simple"]["answer_relevancy"]:.3f}                   | {metrics["Baseline RAG (Vector-only)"]["multi_hop"]["answer_relevancy"]:.3f}                      | {metrics["Baseline RAG (Vector-only)"]["overall"]["answer_relevancy"]:.3f}         |
| Context Recall         | {metrics["Baseline RAG (Vector-only)"]["simple"]["context_recall"]:.3f}                     | {metrics["Baseline RAG (Vector-only)"]["multi_hop"]["context_recall"]:.3f}                        | {metrics["Baseline RAG (Vector-only)"]["overall"]["context_recall"]:.3f}           |
| Context Precision      | {metrics["Baseline RAG (Vector-only)"]["simple"]["context_precision"]:.3f}                  | {metrics["Baseline RAG (Vector-only)"]["multi_hop"]["context_precision"]:.3f}                     | {metrics["Baseline RAG (Vector-only)"]["overall"]["context_precision"]:.3f}        |
| Citation Accuracy      | {metrics["Baseline RAG (Vector-only)"]["simple"]["citation_accuracy"]:.1f}%                 | {metrics["Baseline RAG (Vector-only)"]["multi_hop"]["citation_accuracy"]:.1f}%                    | {metrics["Baseline RAG (Vector-only)"]["overall"]["citation_accuracy"]:.1f}%       |
| Synthesis Success      | {metrics["Baseline RAG (Vector-only)"]["simple"]["synthesis_success"]}                      | {metrics["Baseline RAG (Vector-only)"]["multi_hop"]["synthesis_success"]:.1f}%                    | {metrics["Baseline RAG (Vector-only)"]["multi_hop"]["synthesis_success"]:.1f}%     |

## Key Findings

### ErosionKG Advantages
1. **Citation Accuracy**: {metrics["ErosionKG (GraphRAG)"]["overall"]["citation_accuracy"]:.1f}% vs {metrics["Baseline RAG (Vector-only)"]["overall"]["citation_accuracy"]:.1f}% (baseline)
2. **Multi-Hop Synthesis**: {metrics["ErosionKG (GraphRAG)"]["multi_hop"]["synthesis_success"]:.1f}% success rate on cross-document questions
3. **Faithfulness**: {metrics["ErosionKG (GraphRAG)"]["overall"]["faithfulness"]:.3f} vs {metrics["Baseline RAG (Vector-only)"]["overall"]["faithfulness"]:.3f} (baseline)

### Performance Delta
- **Faithfulness Improvement**: {((metrics["ErosionKG (GraphRAG)"]["overall"]["faithfulness"] - metrics["Baseline RAG (Vector-only)"]["overall"]["faithfulness"]) / metrics["Baseline RAG (Vector-only)"]["overall"]["faithfulness"] * 100 if metrics["Baseline RAG (Vector-only)"]["overall"]["faithfulness"] > 0 else 0):.1f}%
- **Context Recall Improvement**: {((metrics["ErosionKG (GraphRAG)"]["overall"]["context_recall"] - metrics["Baseline RAG (Vector-only)"]["overall"]["context_recall"]) / metrics["Baseline RAG (Vector-only)"]["overall"]["context_recall"] * 100 if metrics["Baseline RAG (Vector-only)"]["overall"]["context_recall"] > 0 else 0):.1f}%
- **Citation Accuracy Improvement**: {(metrics["ErosionKG (GraphRAG)"]["overall"]["citation_accuracy"] - metrics["Baseline RAG (Vector-only)"]["overall"]["citation_accuracy"]):.1f} percentage points

## Files
- Detailed results: `evaluation_results.json`
- Full response log: `evaluation_full_log.csv`
- Case studies: `case_studies.md`
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"✅ Saved summary table to: {output_path}")


def save_full_log_csv(results: List[Dict], output_path: str):
    """Save complete evaluation log as CSV for appendix"""
    rows = []
    for r in results:
        rows.append({
            "question_id": r["id"],
            "complexity": r["complexity"],
            "question": r["question"][:100] + "..." if len(r["question"]) > 100 else r["question"],
            "ground_truth": r["ground_truth"][:200] + "..." if len(r["ground_truth"]) > 200 else r["ground_truth"],
            "erosionkg_answer": r["erosionkg_answer"][:200] + "..." if len(r["erosionkg_answer"]) > 200 else r["erosionkg_answer"],
            "baseline_answer": r["baseline_answer"][:200] + "..." if len(r["baseline_answer"]) > 200 else r["baseline_answer"],
            "erosionkg_faithfulness": r["erosionkg"].get("faithfulness", 0),
            "baseline_faithfulness": r["baseline"].get("faithfulness", 0),
            "erosionkg_citation_accuracy": r["erosionkg_citation"].get("accuracy", 0),
            "baseline_citation_accuracy": r["baseline_citation"].get("accuracy", 0),
            "synthesis_passed": r.get("erosionkg_synthesis", {}).get("passed", "N/A")
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Saved full log to: {output_path}")


def generate_case_studies(results: List[Dict], output_path: str, top_n: int = 3):
    """Select and format top 3 case studies for publication"""
    # Criteria for selection:
    # 1. Hallucination prevention (baseline low faithfulness, ErosionKG high)
    # 2. Cross-document synthesis (multi-hop where ErosionKG passed, baseline failed)
    # 3. Citation grounding (high citation delta)
    
    # Find best examples
    hallucination_examples = sorted(
        [r for r in results if r["baseline"].get("faithfulness", 1) < 0.7],
        key=lambda x: x["erosionkg"].get("faithfulness", 0) - x["baseline"].get("faithfulness", 0),
        reverse=True
    )[:1]
    
    synthesis_examples = sorted(
        [r for r in results if r.get("erosionkg_synthesis", {}).get("applicable", False)],
        key=lambda x: (
            x.get("erosionkg_synthesis", {}).get("passed", False) 
            and not x.get("baseline_synthesis", {}).get("passed", False)
        ),
        reverse=True
    )[:1]
    
    citation_examples = sorted(
        results,
        key=lambda x: x["erosionkg_citation"].get("accuracy", 0) - x["baseline_citation"].get("accuracy", 0),
        reverse=True
    )[:1]
    
    selected = (hallucination_examples + synthesis_examples + citation_examples)[:top_n]
    
    # Generate markdown
    markdown = f"""# Qualitative Case Studies

**Purpose:** Demonstrate the practical advantages of ErosionKG GraphRAG over baseline vector-only RAG through representative examples.

"""
    
    for i, r in enumerate(selected, 1):
        category = "Hallucination Prevention" if r in hallucination_examples else \
                   "Cross-Document Synthesis" if r in synthesis_examples else \
                   "Citation Grounding"
        
        markdown += f"""## Case Study {i}: {category}

**Question (ID {r["id"]}, Complexity: {r["complexity"]})**  
{r["question"]}

**Expected Sources:**  
{chr(10).join(f"- {doi}" for doi in r["expected_dois"])}

---

### Standard RAG Response
{r["baseline_answer"]}

**Metrics:**
- Faithfulness: {r["baseline"].get("faithfulness", 0):.3f}
- Citation Accuracy: {r["baseline_citation"].get("accuracy", 0)*100:.1f}%
{f"- Synthesis: {'PASS' if r.get('baseline_synthesis', {}).get('passed', False) else 'FAIL'}" if r.get("baseline_synthesis", {}).get("applicable", False) else ""}

---

### ErosionKG GraphRAG Response
{r["erosionkg_answer"]}

**Metrics:**
- Faithfulness: {r["erosionkg"].get("faithfulness", 0):.3f}
- Citation Accuracy: {r["erosionkg_citation"].get("accuracy", 0)*100:.1f}%
{f"- Synthesis: {'PASS' if r.get('erosionkg_synthesis', {}).get('passed', False) else 'FAIL'}" if r.get("erosionkg_synthesis", {}).get("applicable", False) else ""}

---

### Analysis
**Advantage:** ErosionKG demonstrates {category.lower()} by {"providing verifiable citations with DOIs" if category == "Citation Grounding" else "staying grounded in retrieved evidence" if category == "Hallucination Prevention" else "successfully connecting findings across multiple papers"}.

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"✅ Saved case studies to: {output_path}")


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def run_evaluation(
    dataset_path: str = "api/data/golden_dataset.txt",
    api_url: str = None,
    output_dir: str = "evaluation_results",
    question_range: Tuple[int, int] = None
):
    """
    Run complete 7-step evaluation pipeline
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("EROSIONKG GRAPHRAG EVALUATION SYSTEM")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"API URL: {api_url or 'http://localhost:8000'}")
    print(f"Output Dir: {output_dir}")
    print("="*80)
    
    # Step 1: Parse golden dataset
    print("\n[STEP 1] Parsing golden dataset...")
    questions = parse_golden_dataset(dataset_path)
    print(f"✅ Loaded {len(questions)} questions")
    
    # Filter by range if specified
    if question_range:
        start, end = question_range
        questions = [q for q in questions if start <= q["id"] <= end]
        print(f"   Filtered to questions {start}-{end}: {len(questions)} questions")
    
    # Initialize baseline RAG
    print("\n[STEP 2] Initializing systems...")
    baseline_rag = BaselineRAG()
    print("✅ Baseline RAG initialized")
    print("✅ ErosionKG API endpoint ready")
    
    # Step 3-7: Process each question
    print(f"\n[STEP 3-7] Processing {len(questions)} questions...")
    results = []
    
    for q in tqdm(questions, desc="Evaluating"):
        print(f"\n{'='*80}")
        print(f"[Q{q['id']}/{len(questions)}] Complexity: {q['complexity']}")
        print(f"Question: {q['question'][:80]}...")
        
        result = {
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "expected_dois": q["expected_dois"],
            "complexity": q["complexity"],
            "num_sources_required": q["num_sources_required"]
        }
        
        # Query ErosionKG
        print(f"  -> Querying ErosionKG...", end=" ")
        erosionkg_result = query_erosionkg(q["question"], api_url)
        print(f"[{erosionkg_result['response_time_sec']:.1f}s]")
        
        result["erosionkg_answer"] = erosionkg_result["answer"]
        result["erosionkg_response_time"] = erosionkg_result["response_time_sec"]
        
        # Short delay between queries (separate API keys now)
        time.sleep(5)
        
        # Query Baseline
        print(f"  -> Querying Baseline RAG...", end=" ")
        baseline_result = query_baseline(q["question"], baseline_rag)
        print(f"[{baseline_result['response_time_sec']:.1f}s]")
        
        result["baseline_answer"] = baseline_result["answer"]
        result["baseline_response_time"] = baseline_result["response_time_sec"]
        
        # RAGAS metrics for ErosionKG
        print(f"  -> Calculating RAGAS metrics (ErosionKG)...")
        if erosionkg_result["contexts"]:
            result["erosionkg"] = calculate_ragas_metrics(
                q["question"],
                erosionkg_result["answer"],
                q["ground_truth"],
                erosionkg_result["contexts"]
            )
        else:
            result["erosionkg"] = {"error": "No contexts available"}
        
        # Add delay between RAGAS calculations to prevent rate limiting
        time.sleep(3)
        
        # RAGAS metrics for Baseline
        print(f"  -> Calculating RAGAS metrics (Baseline)...")
        if baseline_result["contexts"]:
            result["baseline"] = calculate_ragas_metrics(
                q["question"],
                baseline_result["answer"],
                q["ground_truth"],
                baseline_result["contexts"]
            )
        else:
            result["baseline"] = {"error": "No contexts available"}
        
        # Citation accuracy
        result["erosionkg_citation"] = calculate_citation_accuracy(
            erosionkg_result["answer"],
            q["expected_dois"]
        )
        result["baseline_citation"] = calculate_citation_accuracy(
            baseline_result["answer"],
            q["expected_dois"]
        )
        
        print(f"     - Faithfulness: ErosionKG={result['erosionkg'].get('faithfulness', 0):.2f}, Baseline={result['baseline'].get('faithfulness', 0):.2f}")
        print(f"     - Citation Accuracy: ErosionKG={result['erosionkg_citation']['accuracy']*100:.0f}%, Baseline={result['baseline_citation']['accuracy']*100:.0f}%")
        
        # Multi-hop synthesis check
        if q["num_sources_required"] >= 2:
            print(f"  -> Judge LLM synthesis review...")
            result["erosionkg_synthesis"] = judge_multi_hop_synthesis(
                q["question"],
                erosionkg_result["answer"],
                q["ground_truth"],
                q["expected_dois"],
                q["num_sources_required"]
            )
            result["baseline_synthesis"] = judge_multi_hop_synthesis(
                q["question"],
                baseline_result["answer"],
                q["ground_truth"],
                q["expected_dois"],
                q["num_sources_required"]
            )
            print(f"     - Synthesis: ErosionKG={'PASS' if result['erosionkg_synthesis']['passed'] else 'FAIL'}, Baseline={'PASS' if result['baseline_synthesis']['passed'] else 'FAIL'}")
        
        results.append(result)
    
    # Close baseline RAG
    baseline_rag.close()
    
    # Step 7: Generate reports
    print(f"\n{'='*80}")
    print("[STEP 7] Generating reports...")
    
    save_detailed_results(results, os.path.join(output_dir, "evaluation_results.json"))
    save_summary_table(results, os.path.join(output_dir, "results_summary.md"))
    save_full_log_csv(results, os.path.join(output_dir, "evaluation_full_log.csv"))
    generate_case_studies(results, os.path.join(output_dir, "case_studies.md"))
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}/")
    print(f"  - evaluation_results.json (detailed)")
    print(f"  - results_summary.md (publication table)")
    print(f"  - evaluation_full_log.csv (all responses)")
    print(f"  - case_studies.md (qualitative examples)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ErosionKG GraphRAG system")
    parser.add_argument("--dataset", default="api/data/golden_dataset.txt", help="Path to golden dataset")
    parser.add_argument("--api-url", default=None, help="ErosionKG API URL (default: http://localhost:8000)")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--questions", default=None, help="Question range (e.g., '1-5')")
    parser.add_argument("--all", action="store_true", help="Run all 20 questions")
    
    args = parser.parse_args()
    
    question_range = None
    if args.questions:
        start, end = map(int, args.questions.split('-'))
        question_range = (start, end)
    
    run_evaluation(
        dataset_path=args.dataset,
        api_url=args.api_url,
        output_dir=args.output_dir,
        question_range=question_range
    )
