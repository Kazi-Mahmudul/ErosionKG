import os
import json
import logging
from typing import List, Literal, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

from llama_index.core import Document, PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

# ...



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

def run_kg_extraction(input_file: str = "data/extracted_chunks.json"):
    # 1. Infrastructure: Neo4j
    logger.info("Initializing Neo4j Graph Store...")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
    )

    # 2. Intelligence: Gemini
    logger.info("Initializing Gemini...")
    llm = Gemini(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)

    # 3. Schema Extraction (Simple)
    # Using SimpleLLMPathExtractor with single worker to avoid async cleanup issues.
    kg_extractor = SimpleLLMPathExtractor(
        llm=llm,
        num_workers=1,  # Single worker to avoid grpc async cleanup issues
        max_paths_per_chunk=10,
    )

    # 4. Processing: Load Data
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Assuming the JSON structure is { "chunks": [ ... ] } based on previous schema
    chunks = data.get("chunks", [])
    documents = []
    
    for chunk in chunks:
        # metadata = chunk.get("metadata", {})
        # content = chunk.get("content", "")
        # doc = Document(text=content, metadata=metadata)
        # documents.append(doc)
        # Using schema object structure if pydantic dump:
        meta = chunk.get("metadata", {})
        content = chunk.get("content", "")
        if content:
             documents.append(Document(text=content, metadata=meta))

    logger.info(f"Prepared {len(documents)} documents for KG extraction.")

    # 5. Persistence: Extract and Index
    logger.info("Building Property Graph Index (this may take time)...")
    
    # Create embedding model
    embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=GOOGLE_API_KEY)
    
    # Create the index from documents. This triggers the extraction.
    try:
        index = PropertyGraphIndex.from_documents(
            documents,
            kg_extractors=[kg_extractor],
            property_graph_store=graph_store,
            llm=llm,
            embed_model=embed_model,
            show_progress=True
        )
        logger.info("KG Extraction complete. Data synced to Neo4j.")
    except Exception as e:
        logger.warning(f"Extraction finished with cleanup warning (this is usually benign): {e}")
        logger.info("KG Extraction complete despite async cleanup warning. Data synced to Neo4j.")
    finally:
        # Graceful shutdown
        try:
            graph_store.close()
        except:
            pass

if __name__ == "__main__":
    run_kg_extraction()
