"""
Manual KG Extraction Pipeline - Direct Neo4j Upsert
This version extracts triplets manually and writes directly to Neo4j
to avoid LlamaIndex async cleanup issues.
"""
import os
import json
import logging
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

from llama_index.llms.gemini import Gemini
from llama_index.core import Document

# Configuration
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GOOGLE_API_KEY]):
    logger.error("Missing environment variables. Please check .env")
    exit(1)

# Initialize LLM with rate limiting
class RateLimitedGemini(Gemini):
    def complete(self, *args, **kwargs):
        time.sleep(3)  # Rate limit
        return super().complete(*args, **kwargs)

TRIPLET_EXTRACTION_PROMPT = '''
Extract knowledge graph triplets from the following text about landscape erosion.
Each triplet should be in the format: (subject, relationship, object)

Focus on extracting entities like:
- Erosion processes (e.g., sheet erosion, gully erosion, rill erosion)
- Landforms (e.g., slopes, valleys, watersheds)
- Factors (e.g., rainfall, soil type, vegetation cover)
- Metrics (e.g., soil loss rate, erosion index, R-factor)
- Regions (e.g., geographic locations, study areas)

Use relationships like:
- CAUSES, AFFECTS, MODULATES, MEASURES, OCCURS_IN, ACTS_ON, COMPOSED_OF

Return ONLY a JSON array of triplets. Example format:
[{{"subject": "rainfall intensity", "relationship": "CAUSES", "object": "sheet erosion"}}]

If no clear triplets can be extracted, return an empty array: []

TEXT:
{text}

JSON TRIPLETS:
'''

def extract_triplets(llm, text: str) -> list:
    """Extract triplets from text using LLM."""
    try:
        # Truncate very long texts
        truncated_text = text[:4000] if len(text) > 4000 else text
        prompt = TRIPLET_EXTRACTION_PROMPT.format(text=truncated_text)
        response = llm.complete(prompt)
        response_text = response.text.strip()
        
        # Try to find JSON array in response
        import re
        # Look for JSON array pattern
        json_match = re.search(r'\[[\s\S]*?\]', response_text)
        if json_match:
            response_text = json_match.group(0)
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        triplets = json.loads(response_text)
        if isinstance(triplets, list):
            # Validate triplet structure
            valid_triplets = []
            for t in triplets:
                if isinstance(t, dict) and "subject" in t and "object" in t:
                    valid_triplets.append(t)
            return valid_triplets
        return []
    except json.JSONDecodeError as e:
        # Log only if verbose
        return []
    except Exception as e:
        return []

def upsert_triplet_to_neo4j(driver, subject: str, relationship: str, obj: str, source_file: str):
    """Upsert a single triplet to Neo4j."""
    query = """
    MERGE (s:Entity {name: $subject})
    MERGE (o:Entity {name: $object})
    MERGE (s)-[r:RELATES {type: $relationship}]->(o)
    SET r.source = $source_file
    RETURN s, r, o
    """
    try:
        driver.execute_query(query, subject=subject, object=obj, relationship=relationship, source_file=source_file)
    except Exception as e:
        logger.warning(f"Failed to upsert triplet: {e}")

def run_kg_extraction(input_file: str = "data/extracted_chunks.json"):
    # 1. Connect to Neo4j
    logger.info("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    logger.info("Connected to Neo4j successfully!")

    # 2. Initialize LLM
    logger.info("Initializing Gemini...")
    llm = RateLimitedGemini(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)

    # 3. Load Data
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data.get("chunks", [])
    logger.info(f"Loaded {len(chunks)} chunks for KG extraction.")

    # 4. Extract and Upsert Triplets
    total_triplets = 0
    processed = 0
    for i, chunk in enumerate(tqdm(chunks, desc="Extracting triplets")):
        content = chunk.get("content", "")
        source_file = chunk.get("metadata", {}).get("source_file", "unknown")
        
        if not content or len(content) < 50:  # Skip very short chunks
            continue
        
        processed += 1
        triplets = extract_triplets(llm, content)
        
        if triplets:
            logger.info(f"Chunk {i}: Found {len(triplets)} triplets")
        
        for triplet in triplets:
            subject = triplet.get("subject", "").strip()
            relationship = triplet.get("relationship", "RELATES").strip()
            obj = triplet.get("object", "").strip()
            
            if subject and obj:
                upsert_triplet_to_neo4j(driver, subject, relationship, obj, source_file)
                total_triplets += 1
        
        # Log progress every 50 chunks
        if processed % 50 == 0:
            logger.info(f"Processed {processed} chunks, extracted {total_triplets} triplets so far")

    logger.info(f"Extraction complete! Upserted {total_triplets} triplets to Neo4j.")
    
    # 5. Verify
    result = driver.execute_query("MATCH (n:Entity) RETURN count(n) as count")
    count = result.records[0]["count"]
    logger.info(f"Total entities in Neo4j: {count}")
    
    driver.close()
    logger.info("Done!")

if __name__ == "__main__":
    run_kg_extraction()
