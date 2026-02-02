from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'), 
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

files = [
    "2020_Asia_soil_erosion_review.pdf",
    "2019_RUSLE_Model_SriLanka.pdf",
    "2023_Soil_erosion_Bangladesh.pdf"
]

for f in files:
    print(f"\nChecking DOIs for {f}:")
    res = driver.execute_query("MATCH (c:Chunk {sourceFile: $f}) RETURN DISTINCT c.doiUrl as doi", f=f)
    dois = [r['doi'] for r in res.records]
    print(f"  DOIs: {dois}")

driver.close()
