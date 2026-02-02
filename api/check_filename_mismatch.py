"""Check relationship vs chunk source filename matching"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'), 
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

print("=" * 70)
print("Checking Filename Mismatches")
print("=" * 70)

# Get some relationship sources
result = driver.execute_query("""
    MATCH ()-[r]->() 
    WHERE r.source IS NOT NULL
    RETURN DISTINCT r.source as source
    LIMIT 10
""")
print("\nRelationship sources (r.source):")
rel_sources = [rec['source'] for rec in result.records]
for src in rel_sources:
    print(f"  - {src}")

# Get chunk sources  
result2 = driver.execute_query("""
    MATCH (c:Chunk)
    RETURN DISTINCT c.sourceFile as source
    LIMIT 10
""")
print("\nChunk sources (c.sourceFile):")
chunk_sources = [rec['source'] for rec in result2.records]
for src in chunk_sources:
    print(f"  - {src}")

# Check for the specific problem files
problem_files = [
    "2020_Asia_soil_erosion_review.pdf",
    "2019_RUSLE_Model_SriLanka.pdf",
    "2023_Soil_erosion_Bangladesh.pdf"
]

print("\n" + "=" * 70)
print("Checking Problem Files")
print("=" * 70)

for pfile in problem_files:
    print(f"\n{pfile}:")
    
    # Check in relationships
    rel_check = driver.execute_query("""
        MATCH ()-[r]->()
        WHERE r.source CONTAINS $filename
        RETURN count(r) as count, collect(DISTINCT r.source)[0] as example
    """, filename=pfile.replace(".pdf", ""))
    
    rel_rec = rel_check.records[0] if rel_check.records else None
    print(f"  Relationships: {rel_rec['count'] if rel_rec else 0} found")
    if rel_rec and rel_rec['example']:
        print(f"    Example: {rel_rec['example']}")
    
    # Check in chunks
    chunk_check = driver.execute_query("""
        MATCH (c:Chunk)
        WHERE c.sourceFile CONTAINS $filename
        RETURN count(c) as count, c.sourceFile as example, c.doiUrl as doi
        LIMIT 1
    """, filename=pfile.replace(".pdf", ""))
    
    chunk_rec = chunk_check.records[0] if chunk_check.records else None
    print(f"  Chunks: {chunk_rec['count'] if chunk_rec else 0} found")
    if chunk_rec and chunk_rec['example']:
        print(f"    Example: {chunk_rec['example']}")
        print(f"    DOI: {chunk_rec['doi']}")

driver.close()
print("\n" + "=" * 70)
