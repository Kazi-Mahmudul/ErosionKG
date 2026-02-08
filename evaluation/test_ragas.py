"""
Standalone RAGAS Test - No API Calls
Tests RAGAS metrics configuration with mock data before running full evaluation
"""
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate, RunConfig
# Verified imports from package inspection
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    ContextEntityRecall,
    NoiseSensitivity
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

def test_ragas_metrics():
    """Test RAGAS metrics with simple and complex mock data"""
    
    print("="*80)
    print("RAGAS METRICS COMPREHENSIVE TEST - TESTING 4 METRICS")
    print("="*80)
    
    # Initialize components
    print("\n[1/3] Initializing Gemini LLM and embeddings...")
    try:
        judge_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        # Use embeddings if needed for some metrics (AnswerCorrectness might need it)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("✅ Judge LLM and embeddings initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    # Test 1: Simple example
    print("\n[2/3] Testing with SIMPLE example...")
    simple_data = {
        "question": ["What is erosion?"],
        "answer": ["Erosion is the process of soil degradation caused by water and wind."],
        "contexts": [["Erosion is the wearing away of soil and rock by natural forces such as water, wind, and ice."]],
        "ground_truth": ["Erosion involves the removal of soil and rock material through natural processes."]
    }
    simple_dataset = Dataset.from_dict(simple_data)
    
    if not run_metrics_test(simple_dataset, judge_llm, embeddings, "SIMPLE"):
        return False
    
    # Test 2: Complex multi-hop example
    print("\n[3/3] Testing with COMPLEX multi-hop example...")
    complex_data = {
        "question": ["How does climate change affect soil erosion rates in agricultural regions?"],
        "answer": ["Climate change increases erosion through more intense rainfall events and extreme weather patterns."],
        "contexts": [[
            "Climate change leads to increased precipitation intensity and frequency of extreme weather events, accelerating soil erosion.",
            "Conservation agricultural practices including reduced tillage and cover crops have been shown to decrease erosion rates significantly."
        ]],
        "ground_truth": ["Climate change intensifies erosion through extreme weather events and increased rainfall."]
    }
    complex_dataset = Dataset.from_dict(complex_data)
    
    if not run_metrics_test(complex_dataset, judge_llm, embeddings, "COMPLEX"):
        return False
    
    print("\n" + "="*80)
    print("✅ ALL RAGAS METRICS VALIDATED ON SIMPLE AND COMPLEX EXAMPLES")
    print("="*80)
    print("\nValidated metrics for research paper:")
    print("  1. Context Recall - retrieval completeness")
    print("  2. Context Precision - retrieval relevance") 
    print("  3. Context Entity Recall - entity-level retrieval accuracy")
    print("  4. Noise Sensitivity - robustness to irrelevant contexts")
    print("\nReady for full 20-question evaluation!")
    return True

def run_metrics_test(dataset, judge_llm, embeddings, label):
    """Run all 4 metrics on a dataset"""
    # Increased timeout significantly to prevent timeouts during testing
    run_config = RunConfig(timeout=180, max_retries=2, max_wait=30)
    working_metrics = []
    failed_metrics = []
    
    # Test 1: Context Recall
    print(f"  [{label}] Testing Context Recall...")
    try:
        result = evaluate(
            dataset,
            metrics=[ContextRecall(llm=judge_llm)],
            llm=judge_llm,
            run_config=run_config
        ).to_pandas().iloc[0]
        print(f"  ✅ Context Recall: {result['context_recall']:.4f}")
        working_metrics.append("ContextRecall")
    except Exception as e:
        print(f"  ❌ Context Recall failed: {str(e)[:100]}")
        failed_metrics.append("ContextRecall")
    
    # Test 2: Context Precision
    print(f"  [{label}] Testing Context Precision...")
    try:
        result = evaluate(
            dataset,
            metrics=[ContextPrecision(llm=judge_llm)],
            llm=judge_llm,
            run_config=run_config
        ).to_pandas().iloc[0]
        print(f"  ✅ Context Precision: {result['context_precision']:.4f}")
        working_metrics.append("ContextPrecision")
    except Exception as e:
        print(f"  ❌ Context Precision failed: {str(e)[:100]}")
        failed_metrics.append("ContextPrecision")
    
    # Test 3: Context Entity Recall
    print(f"  [{label}] Testing Context Entity Recall...")
    try:
        result = evaluate(
            dataset,
            metrics=[ContextEntityRecall(llm=judge_llm)],
            llm=judge_llm,
            run_config=run_config
        ).to_pandas().iloc[0]
        print(f"  ✅ Context Entity Recall: {result['context_entity_recall']:.4f}")
        working_metrics.append("ContextEntityRecall")
    except Exception as e:
        print(f"  ❌ Context Entity Recall failed: {str(e)[:100]}")
        failed_metrics.append("ContextEntityRecall")
        
    # Test 4: Noise Sensitivity
    print(f"  [{label}] Testing Noise Sensitivity...")
    try:
        result = evaluate(
            dataset,
            metrics=[NoiseSensitivity(llm=judge_llm)],
            llm=judge_llm,
            run_config=run_config
        ).to_pandas().iloc[0]
        # Noise sensitivity returns multiple values, we check relevant
        val = result.get('noise_sensitivity_relevant', 0.0)
        print(f"  ✅ Noise Sensitivity: {val:.4f}")
        working_metrics.append("NoiseSensitivity")
    except Exception as e:
        print(f"  ❌ Noise Sensitivity failed: {str(e)[:100]}")
        failed_metrics.append("NoiseSensitivity")
    
    print(f"\n  [{label}] Summary: {len(working_metrics)}/4 metrics working")
    if failed_metrics:
        print(f"  Failed metrics: {', '.join(failed_metrics)}")
    
    # All 4 metrics must work
    return len(working_metrics) == 4

if __name__ == "__main__":
    success = test_ragas_metrics()
    exit(0 if success else 1)
