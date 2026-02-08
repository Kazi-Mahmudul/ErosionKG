"""
Semantic Entity Resolution Module
Deduplicates entities in Neo4j using fuzzy matching and LLM verification.
"""
import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from rapidfuzz import fuzz, process
from tqdm import tqdm

from llama_index.llms.gemini import Gemini

# Configuration
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MERGE_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "merge_log.json")


@dataclass
class MergeAction:
    """Record of a merge action."""
    timestamp: str
    master_name: str
    duplicate_name: str
    similarity_score: float
    llm_confirmed: bool
    merged: bool
    reason: str


class RateLimitedGemini(Gemini):
    """Gemini with rate limiting."""
    def complete(self, *args, **kwargs):
        time.sleep(3)  # Rate limit for free tier
        return super().complete(*args, **kwargs)


# --------------------------------------------------------------------------
# 1. CLUSTERING: Fetch all entities from Neo4j
# --------------------------------------------------------------------------
def get_all_entities(driver) -> List[Dict]:
    """Fetch all Entity nodes from Neo4j."""
    query = """
    MATCH (e:Entity)
    RETURN elementId(e) as id, e.name as name
    ORDER BY e.name
    """
    result = driver.execute_query(query)
    entities = [{"id": r["id"], "name": r["name"]} for r in result.records]
    logger.info(f"Fetched {len(entities)} entities from Neo4j")
    return entities


def is_numeric_entity(name: str) -> bool:
    """
    Check if entity is a numeric value/measurement that should NOT be merged.
    Examples: '10.2 ton ha¯¹ yr¯¹', '0 to 334.5 ton/ha/year', '1.9 mm year-1', '0.2'
    """
    import re
    
    # Skip if name is too short
    if len(name.strip()) < 2:
        return True
    
    # Patterns that indicate numeric/measurement entities
    numeric_patterns = [
        r'^\d',  # Starts with digit
        r'^-?\d+\.?\d*$',  # Pure number like "0.2", "10", "-5.3"
        r'\d+\s*(to|-)\s*\d+',  # Ranges like "0 to 334.5"
        r'\d+\s*(ton|t|kg|mm|cm|m|ha|yr|year|%)',  # Measurements
        r'ton\s*(per|/|ha)',  # Erosion rates
        r'ha.*yr',  # Hectare per year patterns
        r'mm.*year',  # mm per year patterns
        r'^\d+°',  # Degree measurements
        r'^\d+\s*m$',  # Meters like "681 m"
    ]
    
    name_lower = name.lower().strip()
    for pattern in numeric_patterns:
        if re.search(pattern, name_lower):
            return True
    
    return False


# --------------------------------------------------------------------------
# 2. CANDIDATE SELECTION: Fuzzy matching to find potential duplicates
# --------------------------------------------------------------------------
def find_duplicate_candidates(entities: List[Dict], threshold: float = 80.0) -> List[Tuple[Dict, Dict, float]]:
    """
    Find potential duplicate pairs using fuzzy string matching.
    Returns list of (entity1, entity2, similarity_score) tuples.
    """
    candidates = []
    
    # Filter out numeric entities BEFORE processing
    filtered_entities = [e for e in entities if not is_numeric_entity(e["name"])]
    logger.info(f"Filtered to {len(filtered_entities)} non-numeric entities (excluded {len(entities) - len(filtered_entities)} numeric values)")
    
    names = [e["name"].lower() for e in filtered_entities]
    seen_pairs = set()
    
    for i, entity in enumerate(tqdm(filtered_entities, desc="Finding candidates")):
        name = entity["name"].lower()
        
        # Find similar names
        matches = process.extract(
            name, 
            names, 
            scorer=fuzz.token_sort_ratio,
            limit=10
        )
        
        for match_name, score, match_idx in matches:
            if match_idx == i:  # Skip self
                continue
            if score < threshold:
                continue
            
            # Avoid duplicate pairs
            pair_key = tuple(sorted([i, match_idx]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            candidates.append((entity, filtered_entities[match_idx], score))
    
    logger.info(f"Found {len(candidates)} potential duplicate pairs")
    return candidates


# --------------------------------------------------------------------------
# 3. LLM VERIFICATION: Confirm duplicates with Gemini
# --------------------------------------------------------------------------
VERIFICATION_PROMPT = """You are an expert in landscape erosion research terminology.

Are these two entities identical or referring to the EXACT same concept?

Entity A: {name1}
Entity B: {name2}

IMPORTANT RULES:
1. NEVER merge different numeric values (e.g., "10.2 ton" and "12 ton" are DIFFERENT)
2. NEVER merge different measurements or ranges
3. Only merge if they are TRUE synonyms or spelling variations of the same concept
4. Abbreviation expansions are OK (e.g., "GIS" = "Geographic Information System")
5. Case variations are OK (e.g., "RUSLE" = "rusle")

Respond with ONLY "YES" or "NO".
"""

def verify_with_llm(llm, name1: str, name2: str) -> bool:
    """Use LLM to verify if two entities are truly duplicates."""
    try:
        prompt = VERIFICATION_PROMPT.format(name1=name1, name2=name2)
        response = llm.complete(prompt)
        answer = response.text.strip().upper()
        return answer == "YES"
    except Exception as e:
        logger.warning(f"LLM verification failed: {e}")
        return False


# --------------------------------------------------------------------------
# 4. MERGING LOGIC: Merge duplicate nodes in Neo4j
# --------------------------------------------------------------------------
def merge_entities(driver, master_id: str, duplicate_id: str) -> bool:
    """
    Merge duplicate entity into master entity.
    - Redirects all relationships from duplicate to master
    - Deletes duplicate node
    """
    try:
        # Transfer incoming relationships
        query_incoming = """
        MATCH (dup) WHERE elementId(dup) = $dup_id
        MATCH (master) WHERE elementId(master) = $master_id
        MATCH (other)-[r]->(dup)
        WHERE other <> master
        CREATE (other)-[r2:RELATES]->(master)
        SET r2 = properties(r)
        DELETE r
        """
        driver.execute_query(query_incoming, dup_id=duplicate_id, master_id=master_id)
        
        # Transfer outgoing relationships
        query_outgoing = """
        MATCH (dup) WHERE elementId(dup) = $dup_id
        MATCH (master) WHERE elementId(master) = $master_id
        MATCH (dup)-[r]->(other)
        WHERE other <> master
        CREATE (master)-[r2:RELATES]->(other)
        SET r2 = properties(r)
        DELETE r
        """
        driver.execute_query(query_outgoing, dup_id=duplicate_id, master_id=master_id)
        
        # Delete duplicate node
        query_delete = """
        MATCH (dup) WHERE elementId(dup) = $dup_id
        DETACH DELETE dup
        """
        driver.execute_query(query_delete, dup_id=duplicate_id)
        
        return True
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return False


# --------------------------------------------------------------------------
# 5. LOGGING: Store merge actions
# --------------------------------------------------------------------------
def load_merge_log() -> List[Dict]:
    """Load existing merge log."""
    if os.path.exists(MERGE_LOG_PATH):
        with open(MERGE_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_merge_log(log: List[Dict]):
    """Save merge log to file."""
    os.makedirs(os.path.dirname(MERGE_LOG_PATH), exist_ok=True)
    with open(MERGE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def log_merge_action(action: MergeAction):
    """Append a merge action to the log."""
    log = load_merge_log()
    log.append(asdict(action))
    save_merge_log(log)


# --------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# --------------------------------------------------------------------------
def run_resolution(dry_run: bool = False, similarity_threshold: float = 80.0):
    """
    Run the full entity resolution pipeline.
    
    Args:
        dry_run: If True, only find candidates without merging.
        similarity_threshold: Minimum fuzzy match score (0-100).
    """
    logger.info("Starting Entity Resolution Pipeline...")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    logger.info("Connected to Neo4j")
    
    # Initialize LLM
    llm = RateLimitedGemini(model="models/gemini-3-flash-preview", api_key=GOOGLE_API_KEY)
    
    # 1. Fetch entities
    entities = get_all_entities(driver)
    
    if len(entities) < 2:
        logger.info("Not enough entities for resolution")
        driver.close()
        return
    
    # 2. Find candidates
    candidates = find_duplicate_candidates(entities, threshold=similarity_threshold)
    
    if not candidates:
        logger.info("No duplicate candidates found")
        driver.close()
        return
    
    # 3. Verify and merge
    merged_count = 0
    for entity1, entity2, score in tqdm(candidates, desc="Verifying duplicates"):
        name1, name2 = entity1["name"], entity2["name"]
        
        # LLM verification
        is_duplicate = verify_with_llm(llm, name1, name2)
        
        if dry_run:
            action = MergeAction(
                timestamp=datetime.now().isoformat(),
                master_name=name1,
                duplicate_name=name2,
                similarity_score=score,
                llm_confirmed=is_duplicate,
                merged=False,
                reason="Dry run - no merge performed"
            )
            log_merge_action(action)
            if is_duplicate:
                logger.info(f"[DRY RUN] Would merge: '{name2}' -> '{name1}' (score: {score:.1f})")
            continue
        
        if is_duplicate:
            # Perform merge
            success = merge_entities(driver, entity1["id"], entity2["id"])
            
            action = MergeAction(
                timestamp=datetime.now().isoformat(),
                master_name=name1,
                duplicate_name=name2,
                similarity_score=score,
                llm_confirmed=True,
                merged=success,
                reason="LLM confirmed duplicate" if success else "Merge failed"
            )
            log_merge_action(action)
            
            if success:
                merged_count += 1
                logger.info(f"Merged: '{name2}' -> '{name1}'")
        else:
            # Log rejection
            action = MergeAction(
                timestamp=datetime.now().isoformat(),
                master_name=name1,
                duplicate_name=name2,
                similarity_score=score,
                llm_confirmed=False,
                merged=False,
                reason="LLM rejected as non-duplicate"
            )
            log_merge_action(action)
    
    # Final stats
    result = driver.execute_query("MATCH (n:Entity) RETURN count(n) as count")
    final_count = result.records[0]["count"]
    
    logger.info(f"Resolution complete! Merged {merged_count} entity pairs.")
    logger.info(f"Final entity count: {final_count}")
    logger.info(f"Merge log saved to: {MERGE_LOG_PATH}")
    
    driver.close()


def execute_merges_from_log():
    """Execute merges from the existing log file."""
    logger.info(f"Loading merge log from {MERGE_LOG_PATH}")
    log = load_merge_log()
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    
    merged_count = 0
    skipped_count = 0
    
    # Filter for confirmed but unmerged entries
    to_merge = [entry for entry in log if entry.get("llm_confirmed") and not entry.get("merged")]
    logger.info(f"Found {len(to_merge)} confirmed merges to execute")
    
    for entry in tqdm(to_merge, desc="Executing merges"):
        master_name = entry["master_name"]
        dup_name = entry["duplicate_name"]
        
        # Look up current IDs (names might refer to nodes that were already merged/deleted)
        # We need to find the node with this Name property
        # Note: If multiple nodes have the same name (which is the problem we are solving),
        # we pick one as master and one as dup.
        
        try:
            # Find master node(s)
            result_m = driver.execute_query("MATCH (n:Entity {name: $name}) RETURN elementId(n) as id LIMIT 1", name=master_name)
            if not result_m.records:
                skipped_count += 1
                continue
            master_id = result_m.records[0]["id"]
            
            # Find duplicate node(s)
            # Make sure it's not the same node as master
            result_d = driver.execute_query("MATCH (n:Entity {name: $name}) WHERE elementId(n) <> $master_id RETURN elementId(n) as id LIMIT 1", name=dup_name, master_id=master_id)
            if not result_d.records:
                skipped_count += 1
                continue
            dup_id = result_d.records[0]["id"]
            
            # Execute merge
            success = merge_entities(driver, master_id, dup_id)
            
            if success:
                merged_count += 1
                entry["merged"] = True
                entry["reason"] = "Executed from log"
                entry["timestamp"] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Error merging {dup_name} -> {master_name}: {e}")
            skipped_count += 1
            
    # Save updated log
    save_merge_log(log)
    
    # Final stats
    result = driver.execute_query("MATCH (n:Entity) RETURN count(n) as count")
    final_count = result.records[0]["count"]
    
    logger.info(f"Execution complete! Merged {merged_count} pairs. Skipped {skipped_count}.")
    logger.info(f"Final entity count: {final_count}")
    driver.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entity Resolution for ErosionKG")
    parser.add_argument("--dry-run", action="store_true", help="Find candidates without merging")
    parser.add_argument("--threshold", type=float, default=80.0, help="Similarity threshold (0-100)")
    parser.add_argument("--from-log", action="store_true", help="Execute merges from existing log file")
    args = parser.parse_args()
    
    if args.from_log:
        execute_merges_from_log()
    else:
        run_resolution(dry_run=args.dry_run, similarity_threshold=args.threshold)
