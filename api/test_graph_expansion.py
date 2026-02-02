"""Test script for Graph Expansion with DOIs"""
from kg.graphrag_retriever import get_retriever

def test_expansion():
    print("=" * 70)
    print("Testing Graph Expansion with DOIs")
    print("=" * 70)
    
    retriever, driver = get_retriever()
    expander = retriever.graph_expander
    
    # Test entities (use something common like 'Rainfall' or 'Erosion')
    entities = ["Rainfall", "Soil Loss"]
    print(f"\nExpanding entities: {entities}")
    
    # Expand
    triplets = expander.expand(entities, hops=1)
    
    print(f"\n📊 Results:")
    print(f"  - Triplets found: {len(triplets)}")
    
    if triplets:
        print(f"\n🔗 Triplet Details (First 5):")
        for i, t in enumerate(triplets[:5], 1):
            print(f"\n  Triplet {i}:")
            print(f"    - Subject: {t.subject}")
            print(f"    - Rel: {t.relationship}")
            print(f"    - Object: {t.obj}")
            print(f"    - Source: {t.source_file}")
            print(f"    - DOI: {t.doi_url}")
            print(f"    - Page: {t.page_number}")
            
            # Check if DOI is valid
            if t.doi_url and t.doi_url.startswith("http"):
                print("    ✅ Valid DOI found")
            else:
                print("    ⚠️  Missing or invalid DOI")
    else:
        print("\n❌ NO TRIPLETS FOUND!")
    
    driver.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_expansion()
