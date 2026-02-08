"""
Golden Dataset Generator for GraphRAG Evaluation
Generates 20 high-quality questions from ErosionKG chunks using Gemini API
"""
import os
import json
import logging
import time
import re
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import random

from llama_index.llms.gemini import Gemini

# Configuration
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate-limited Gemini
class RateLimitedGemini(Gemini):
    """Gemini with rate limiting."""
    def complete(self, *args, **kwargs):
        time.sleep(5)  # Rate limit
        return super().complete(*args, **kwargs)


def has_quantitative_content(text: str) -> bool:
    """Check if chunk contains quantitative measurements."""
    patterns = [
        r'\d+\.?\d*\s*(mm|cm|m|km)',  # Distance measurements
        r'\d+\.?\d*\s*(ha|hectare)',  # Area
        r'\d+\.?\d*\s*(ton|t|kg)',     # Weight
        r'\d+\.?\d*\s*(year|yr)',      # Time
        r'\d+\.?\d*\s*%',              # Percentages
        r'\d+\.?\d*\s*degrees?',       # Degrees
        r'\d+\.?\d*\s*(mm/year|t/ha)', # Rates
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def load_chunks(filepath: str) -> List[Dict]:
    """Load chunks from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('chunks', [])


def get_valid_dois(chunks: List[Dict]) -> set:
    """Extract all valid DOIs from chunks."""
    dois = set()
    for chunk in chunks:
        doi = chunk.get('metadata', {}).get('doi_url', '')
        if doi and doi != 'N/A':
            dois.add(doi)
    return dois


def sample_chunks_by_complexity(chunks: List[Dict], target_counts: Dict[str, int]) -> Dict[str, List[Dict]]:
    """
    Sample chunks for different complexity levels.
    - Simple: Single chunk with quantitative data
    - Medium: Pairs of chunks from same paper
    - Complex: Pairs of chunks from different papers
    """
    # Filter chunks with sufficient content
    valid_chunks = [c for c in chunks if len(c.get('content', '')) > 200]
    
    # Prioritize chunks with quantitative content
    quant_chunks = [c for c in valid_chunks if has_quantitative_content(c.get('content', ''))]
    
    # If not enough quantitative chunks, use regular chunks
    if len(quant_chunks) < sum(target_counts.values()):
        logger.warning(f"Only {len(quant_chunks)} quantitative chunks found, using all valid chunks")
        quant_chunks = valid_chunks
    
    # Group by DOI for medium/complex sampling
    chunks_by_doi = {}
    for chunk in quant_chunks:
        doi = chunk.get('metadata', {}).get('doi_url', 'N/A')
        if doi not in chunks_by_doi:
            chunks_by_doi[doi] = []
        chunks_by_doi[doi].append(chunk)
    
    sampled = {
        'simple': [],
        'medium': [],
        'complex': []
    }
    
    # Sample simple (single chunks)
    sampled['simple'] = random.sample(quant_chunks, min(target_counts['simple'], len(quant_chunks)))
    
    # Sample medium (two chunks from same paper)
    remaining_dois = [doi for doi in chunks_by_doi.keys() if len(chunks_by_doi[doi]) >= 2]
    medium_count = min(target_counts['medium'], len(remaining_dois))
    for _ in range(medium_count):
        doi = random.choice(remaining_dois)
        remaining_dois.remove(doi)
        pair = random.sample(chunks_by_doi[doi], 2)
        sampled['medium'].append(pair)
    
    # Sample complex (two chunks from different papers)
    available_dois = list(chunks_by_doi.keys())
    complex_count = min(target_counts['complex'], len(available_dois) // 2)
    for _ in range(complex_count):
        if len(available_dois) < 2:
            break
        doi1, doi2 = random.sample(available_dois, 2)
        chunk1 = random.choice(chunks_by_doi[doi1])
        chunk2 = random.choice(chunks_by_doi[doi2])
        sampled['complex'].append([chunk1, chunk2])
        available_dois.remove(doi1)
        available_dois.remove(doi2)
    
    return sampled


def create_prompt(chunks: List[Dict], complexity: str) -> str:
    """Create prompt with scientific guardrails."""
    
    base_prompt = """You are an expert professor in Geomorphology. Based STRICTLY on the following text chunk(s), create 1 high-quality question and its ground truth answer.

SCIENTIFIC GUARDRAILS:
1. NO SHORTCUTS: Do not include the answer in the question itself. Ensure the question requires understanding of relationships between variables (e.g., 'What is the impact of rainfall intensity on soil erosion rates?' NOT 'Is rainfall related to erosion?').

2. SPECIFIC VALUES: Prioritize questions about specific measurements (mm/year, degrees, percentages, ton/ha). The answer MUST include precise numeric values from the text.

3. EXACT CITATIONS: You MUST copy the DOI exactly as provided below. Do not modify or create DOIs.

"""
    
    # Add chunk information
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get('metadata', {})
        citation = metadata.get('citation_str', 'Unknown')
        doi = metadata.get('doi_url', 'N/A')
        content = chunk.get('content', '')[:2000]  # Limit content length
        
        base_prompt += f"\nChunk {i} (from '{citation}', DOI: {doi}):\n{content}\n"
    
    base_prompt += """
Format your response EXACTLY as:
Question: [specific, relationship-based question]
Ground Truth: [detailed answer with specific numeric values and relationships]
Context Source: [exact DOI(s) from chunks provided above]
Complexity: """ + ("Low" if complexity == "simple" else "Medium" if complexity == "medium" else "High") + "\n"
    
    return base_prompt


def validate_doi(generated_text: str, valid_dois: set) -> bool:
    """Check if DOIs in generated text are from the valid set (Hallucination Filter)."""
    # Extract DOIs from generated text
    doi_pattern = r'https?://doi\.org/[\w\.\-/]+'
    found_dois = re.findall(doi_pattern, generated_text)
    
    if not found_dois:
        logger.warning("No DOI found in generated answer")
        return False
    
    # Check all found DOIs are valid
    for doi in found_dois:
        if doi not in valid_dois:
            logger.warning(f"Hallucinated DOI detected: {doi}")
            return False
    
    return True


def parse_qa(response_text: str) -> Optional[Dict]:
    """Parse the LLM response into structured format."""
    lines = response_text.strip().split('\n')
    
    qa = {
        'question': '',
        'ground_truth': '',
        'context_source': '',
        'complexity': ''
    }
    
    current_field = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Question:'):
            current_field = 'question'
            qa['question'] = line.replace('Question:', '').strip()
        elif line.startswith('Ground Truth:'):
            current_field = 'ground_truth'
            qa['ground_truth'] = line.replace('Ground Truth:', '').strip()
        elif line.startswith('Context Source:'):
            current_field = 'context_source'
            qa['context_source'] = line.replace('Context Source:', '').strip()
        elif line.startswith('Complexity:'):
            current_field = 'complexity'
            qa['complexity'] = line.replace('Complexity:', '').strip()
        elif current_field and line:
            # Continue previous field
            qa[current_field] += ' ' + line
    
    # Validate all fields are present
    if all(qa.values()):
        return qa
    else:
        logger.warning(f"Incomplete Q&A: {qa}")
        return None


def generate_question(llm, chunks: List[Dict], complexity: str, valid_dois: set, max_retries: int = 3) -> Optional[Dict]:
    """Generate a single question with validation."""
    
    for attempt in range(max_retries):
        try:
            prompt = create_prompt(chunks, complexity)
            response = llm.complete(prompt)
            response_text = response.text.strip()
            
            # Validate DOIs (Hallucination Filter)
            if not validate_doi(response_text, valid_dois):
                logger.warning(f"Attempt {attempt + 1}: DOI validation failed, retrying...")
                continue
            
            # Parse response
            qa = parse_qa(response_text)
            if qa:
                return qa
            else:
                logger.warning(f"Attempt {attempt + 1}: Failed to parse response, retrying...")
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: Error generating question: {e}")
    
    logger.error(f"Failed to generate valid question after {max_retries} attempts")
    return None


def save_dataset(questions: List[Dict], output_file: str):
    """Save golden dataset to text file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("GOLDEN DATASET FOR GRAPHRAG EVALUATION\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Model: gemini-3-flash-preview\n")
        f.write(f"Total Questions: {len(questions)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Questions
        for i, qa in enumerate(questions, 1):
            f.write(f"[{i}] " + "=" * 75 + "\n\n")
            f.write(f"Question: {qa['question']}\n\n")
            f.write(f"Ground Truth: {qa['ground_truth']}\n\n")
            f.write(f"Context Source: {qa['context_source']}\n\n")
            f.write(f"Complexity: {qa['complexity']}\n\n")
            f.write("=" * 80 + "\n\n")
    
    logger.info(f"Dataset saved to: {output_file}")


def main():
    """Main execution function."""
    logger.info("Starting Golden Dataset Generation...")
    
    # Configuration
    input_file = "api/data/extracted_chunks.json"
    output_file = "api/data/golden_dataset.txt"
    
    target_counts = {
        'simple': 10,
        'medium': 7,
        'complex': 3
    }
    
    # Load chunks
    logger.info(f"Loading chunks from {input_file}...")
    chunks = load_chunks(input_file)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Get valid DOIs for hallucination filter
    valid_dois = get_valid_dois(chunks)
    logger.info(f"Found {len(valid_dois)} unique DOIs")
    
    # Sample chunks
    logger.info("Sampling chunks by complexity...")
    sampled = sample_chunks_by_complexity(chunks, target_counts)
    logger.info(f"Sampled: {len(sampled['simple'])} simple, {len(sampled['medium'])} medium, {len(sampled['complex'])} complex")
    
    # Initialize LLM
    logger.info("Initializing Gemini...")
    llm = RateLimitedGemini(model="models/gemini-3-flash-preview", api_key=GOOGLE_API_KEY)
    
    # Generate questions
    all_questions = []
    
    # Simple questions
    logger.info("Generating simple questions...")
    for chunk in tqdm(sampled['simple'], desc="Simple"):
        qa = generate_question(llm, [chunk], 'simple', valid_dois)
        if qa:
            all_questions.append(qa)
    
    # Medium questions
    logger.info("Generating medium questions...")
    for chunk_pair in tqdm(sampled['medium'], desc="Medium"):
        qa = generate_question(llm, chunk_pair, 'medium', valid_dois)
        if qa:
            all_questions.append(qa)
    
    # Complex questions
    logger.info("Generating complex questions...")
    for chunk_pair in tqdm(sampled['complex'], desc="Complex"):
        qa = generate_question(llm, chunk_pair, 'complex', valid_dois)
        if qa:
            all_questions.append(qa)
    
    # Save results
    logger.info(f"Generated {len(all_questions)} valid questions")
    save_dataset(all_questions, output_file)
    
    # Summary
    complexity_counts = {}
    for qa in all_questions:
        comp = qa['complexity']
        complexity_counts[comp] = complexity_counts.get(comp, 0) + 1
    
    logger.info("Summary:")
    for comp, count in complexity_counts.items():
        logger.info(f"  {comp}: {count}")
    
    logger.info("✅ Golden dataset generation complete!")


if __name__ == "__main__":
    main()
