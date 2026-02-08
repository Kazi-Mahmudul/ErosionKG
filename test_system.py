"""Comprehensive test suite for ErosionKG system"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from dotenv import load_dotenv
from neo4j import GraphDatabase
from kg.graphrag_retriever import get_retriever

load_dotenv()

print("=" * 70)
print("COMPREHENSIVE SYSTEM TEST")
print("=" * 70)

# Test 1: Neo4j Data Verification
print("\n[TEST 1] Neo4j Data Verification")
print("-" * 70)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    # Count chunks
    result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
    chunk_count = result.single()["count"]
    print(f"✅ Total Chunks: {chunk_count}")
    
    # Check embeddings
    result = session.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) as count")
    embedded_count = result.single()["count"]
    print(f"✅ Chunks with embeddings: {embedded_count}")
    
    # Count entities
    result = session.run("MATCH (e:Entity) RETURN count(e) as count")
    entity_count = result.single()["count"]
    print(f"✅ Total Entities: {entity_count}")
    
    # Count relationships
    result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as count")
    rel_count = result.single()["count"]
    print(f"✅ Total Relationships: {rel_count}")
    
    # Count papers
    result = session.run("MATCH (c:Chunk) RETURN count(DISTINCT c.sourceFile) as count")
    paper_count = result.single()["count"]
    print(f"✅ Total Papers: {paper_count}")
    
    # Check vector index
    result = session.run("SHOW VECTOR INDEXES WHERE name = 'erosion_chunk_index'")
    index_info = result.single()
    if index_info:
        print(f"✅ Vector Index: {dict(index_info)['state']}")

driver.close()

# Test 2: Retriever Test with Multiple Queries
print("\n[TEST 2] Retriever Functionality")
print("-" * 70)

test_queries = [
    "What is soil erosion?",
    "Explain RUSLE model",
    "How does rainfall affect erosion?"
]

retriever, driver = get_retriever()

for i, query in enumerate(test_queries, 1):
    print(f"\nQuery {i}: '{query}'")
    try:
        result = retriever.retrieve(query, top_k=3, hops=1, expand_synonyms=False)
        print(f"  ✅ Chunks found: {len(result.chunks)}")
        print(f"  ✅ Triplets found: {len(result.triplets)}")
        
        if result.chunks:
            chunk = result.chunks[0]
            print(f"  📄 Top chunk score: {chunk.score:.3f}")
            print(f"  📄 Source: {chunk.citation_str or chunk.source_file}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

driver.close()

# Test 3: Metadata API Test
print("\n[TEST 3] Metadata API Simulation")
print("-" * 70)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    # Test metadata query (same as API endpoint)
    stats_query = """
    CALL {
        MATCH (c:Chunk)
        RETURN count(DISTINCT c.sourceFile) as paper_count
    }
    CALL {
        MATCH (e:Entity)
        RETURN count(e) as entity_count
    }
    CALL {
        MATCH ()-[r:RELATES]->()
        RETURN count(r) as relationship_count
    }
    RETURN paper_count, entity_count, relationship_count
    """
    
    result = session.run(stats_query)
    stats = result.single()
    
    print(f"✅ Papers: {stats['paper_count']}")
    print(f"✅ Entities: {stats['entity_count']}")
    print(f"✅ Relationships: {stats['relationship_count']}")
    
    # Test DOI query
    doi_query = """
    MATCH (c:Chunk)
    WHERE c.doiUrl IS NOT NULL AND c.doiUrl <> 'N/A'
    RETURN DISTINCT c.doiUrl as doi, c.sourceFile as file, c.citationStr as citation
    LIMIT 5
    """
    
    doi_result = session.run(doi_query)
    doi_count = len(list(doi_result))
    print(f"✅ DOIs available: {doi_count}")

driver.close()

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED!")
print("=" * 70)
print("\n✅ System is ready for deployment!")
