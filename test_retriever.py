"""Test the retriever with actual queries"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from kg.graphrag_retriever import get_retriever

print("=== TESTING RETRIEVER ===\n")

# Initialize retriever
print("Initializing retriever...")
retriever, driver = get_retriever()

# Test query
query = "What is soil erosion?"
print(f"\nQuery: '{query}'")
print("-" * 60)

try:
    result = retriever.retrieve(query, top_k=3, hops=1)
    
    print(f"\n✅ SUCCESS!")
    print(f"Found {len(result.chunks)} chunks")
    print(f"Found {len(result.triplets)} graph triplets")
    
    if result.chunks:
        print(f"\n📄 Sample chunk:")
        chunk = result.chunks[0]
        print(f"   Content: {chunk.content[:150]}...")
        print(f"   Score: {chunk.score:.3f}")
        print(f"   Source: {chunk.citation_str or chunk.source_file}")
    
    if result.triplets:
        print(f"\n🔗 Sample triplet:")
        t = result.triplets[0]
        print(f"   ({t.subject}) -[{t.relationship}]-> ({t.obj})")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

driver.close()
print("\n" + "=" * 60)
