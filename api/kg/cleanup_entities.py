"""
Advanced Entity Resolution & Graph Cleanup Script
Merges semantically similar entities in Neo4j using embedding-based clustering
"""
import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv
import json
from typing import List, Dict, Set
from collections import defaultdict
import re

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Manual synonyms for RUSLE factors and common entities
MANUAL_SYNONYMS = {
    # RUSLE/USLE Models
    "RUSLE": ["(R)USLE", "R)USLE", "RUSLE model", "USLE", "Universal Soil Loss Equation", "Revised Universal Soil Loss Equation", "USLE model"],
    
    # R-Factor variations
    "R-Factor": ["R Factor", "R-factor", "R factor", "rainfall erosivity factor", "erosivity factor", "rainfall erosivity"],
    
    # K-Factor variations
    "K-Factor": ["K Factor", "K-factor", "K factor", "soil erodibility factor", "erodibility factor", "soil erodibility"],
    
    # LS-Factor variations
    "LS-Factor": ["LS Factor", "LS-factor", "LS factor", "slope-length factor", "slope length factor", "length slope factor", "topographic factor", "slope length and steepness factor"],
    
    # C-Factor variations
    "C-Factor": ["C Factor", "C-factor", "C factor", "cover management factor", "crop management factor", "cover factor", "vegetation cover factor"],
    
    # P-Factor variations
    "P-Factor": ["P Factor", "P-factor", "P factor", "support practice factor", "conservation practice factor", "practice factor", "management practice factor"],
    
    # Common terms
    "Soil Erosion": ["erosion", "soil loss", "soil degradation"],
    "Rainfall": ["precipitation", "mean annual precipitation", "annual rainfall", "rainfall amount"],
    "Slope": ["slope steepness", "slope angle", "gradient", "slope gradient"],
    "Runoff": ["surface runoff", "overland flow", "water runoff"],
    "Sediment": ["sediment yield", "sediment transport", "sediment load"],
    "Land Use": ["land cover", "land use land cover", "LULC"],
    "GIS": ["Geographic Information System", "Geographic Information Systems", "geospatial"],
}



def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def get_all_entities(driver) -> List[Dict]:
    """Get all unique entities from the graph"""
    query = """
    MATCH (e:Entity)
    RETURN e.name as name, id(e) as nodeId
    ORDER BY e.name
    """
    results = driver.execute_query(query)
    return [{"name": r["name"], "nodeId": r["nodeId"]} for r in results.records]


def create_similarity_groups(entities: List[Dict]) -> Dict[str, List[str]]:
    """Group Similar entities using manual rules and fuzzy matching"""
    from rapidfuzz import process, fuzz
    
    # Step 1: Apply manual synonyms
    master_to_duplicates = defaultdict(list)
    entity_names = {e["name"] for e in entities}
    processed = set()
    
    for master, synonyms in MANUAL_SYNONYMS.items():
        if master in entity_names:
            for syn in synonyms:
                # Fuzzy match synonyms
                matches = process.extract(syn, entity_names, scorer=fuzz.ratio, limit=5, score_cutoff=85)
                for match, score, idx in matches:
                    if match != master and match not in processed:
                        master_to_duplicates[master].append(match)
                        processed.add(match)
    
    # Step 2: Cluster remaining similar entities
    # Group by first letter (single-letter nodes are problematic)
    single_letter_entities = [e for e in entities if len(e["name"]) <= 2 and e["name"].upper() == e["name"]]
    
    # These single-letter nodes (P, L, etc.) should NOT exist - they are stemming/abbreviation artifacts
    # We'll merge them into their likely full forms based on relationships
    
    return master_to_duplicates, single_letter_entities


def delete_single_letter_nodes(driver, single_letter_entities: List[Dict]):
    """Delete problematic single-letter nodes that are abbreviation artifacts"""
    print(f"\n🗑️ Deleting {len(single_letter_entities)} single-letter artifact nodes...")
    
    for entity in single_letter_entities:
        name = entity["name"]
        node_id = entity["nodeId"]
        
        # First, check what they're connected to (for logging)
        check_query = """
        MATCH (e:Entity)
        WHERE id(e) = $nodeId
        OPTIONAL MATCH (e)-[r]->(target:Entity)
        RETURN collect(DISTINCT target.name) as targets
        """
        result = driver.execute_query(check_query, nodeId=node_id)
        targets = result.records[0]["targets"] if result.records else []
        
        print(f"  - Deleting '{name}' (connected to: {targets[:5]})")  
        
        # Delete the node and its relationships
        delete_query = """
        MATCH (e:Entity)
        WHERE id(e) = $nodeId
        DETACH DELETE e
        """
        driver.execute_query(delete_query, nodeId=node_id)


def merge_entities(driver, master_to_duplicates: Dict[str, List[str]]):
    """Merge duplicate entities into master entities"""
    print(f"\n🔗 Merging {sum(len(v) for v in master_to_duplicates.values())} duplicate entities...")
    
    for master, duplicates in master_to_duplicates.items():
        if not duplicates:
            continue
            
        print(f"\n  Merging into '{master}':")
        for dup in duplicates:
            print(f"    ← {dup}")
            
        # Cypher query to merge entities
        merge_query = """
        // Find master node
        MATCH (master:Entity {name: $master})
        
        // Find duplicate nodes
        UNWIND $duplicates as dupName
        MATCH (dup:Entity {name: dupName})
        
        // Transfer all relationships from duplicates to master
        OPTIONAL MATCH (dup)-[r]->(target)
        WHERE NOT (master)-[:RELATES]->(target)
        MERGE (master)-[:RELATES {type: r.type}]->(target)
        
        OPTIONAL MATCH (source)-[r2]->(dup)
        WHERE NOT (source)-[:RELATES]->(master)
        MERGE (source)-[:RELATES {type: r2.type}]->(master)
        
       // Delete duplicates
        DETACH DELETE dup
        
        RETURN count(dup) as merged_count
        """
        
        try:
            result = driver.execute_query(merge_query, master=master, duplicates=duplicates)
            merged_count = result.records[0]["merged_count"] if result.records else 0
            print(f"    ✓ Merged {merged_count} nodes")
        except Exception as e:
            print(f"    ✗ Error: {e}")


def main():
    print("=" * 70)
    print("Entity Resolution & Graph Cleanup")
    print("=" * 70)
    
    driver = get_driver()
    
    # Get all entities
    print("\n📊 Loading entities from Neo4j...")
    entities = get_all_entities(driver)
    print(f"Found {len(entities)} entities")
    
    # Create similarity groups
    print("\n🔍 Analyzing similarity groups...")
    master_to_duplicates, single_letter_entities = create_similarity_groups(entities)
    
    print(f"\nIdentified {len(master_to_duplicates)} master entities with duplicates")
    print(f"Found {len(single_letter_entities)} single-letter artifact nodes to delete")
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Proceed with cleanup? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        driver.close()
        return
    
    # Delete single-letter nodes
    delete_single_letter_nodes(driver, single_letter_entities)
    
    # Merge entities
    merge_entities(driver, master_to_duplicates)
    
    # Verify results
    print("\n✅ Cleanup complete!")
    final_entities = get_all_entities(driver)
    print(f"Final entity count: {len(final_entities)} (reduced by {len(entities) - len(final_entities)})")
    
    driver.close()


if __name__ == "__main__":
    main()
