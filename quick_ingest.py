"""Quick test ingestion of first 5 chunks"""
import os
import json
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from llama_index.embeddings.gemini import GeminiEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

embed_model = GeminiEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Load first 5 chunks
with open("api/data/extracted_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    chunks = data["chunks"][:5]

logger.info(f"Testing with {len(chunks)} chunks")

# Generate embeddings
texts = [c["content"] for c in chunks]
embeddings = embed_model.get_text_embedding_batch(texts)

logger.info(f"Generated {len(embeddings)} embeddings")
logger.info(f"Embedding dimension: {len(embeddings[0])}")

# Ingest
with driver.session() as session:
    for chunk, embedding in zip(chunks, embeddings):
        session.run("""
        CREATE (c:Chunk {
            chunkId: $chunkId,
            content: $content,
            sourceFile: $sourceFile,
            pageNumber: $pageNumber,
            headerPath: $headerPath,
            lastModified: $lastModified,
            doiUrl: coalesce($doiUrl, "N/A"),
            citationStr: coalesce($citationStr, $sourceFile),
            embedding: $embedding
        })
        """,
        chunkId=chunk["metadata"]["chunk_id"],
        content=chunk["content"],
        sourceFile=chunk["metadata"]["source_file"],
        pageNumber=chunk["metadata"]["page_number"],
        headerPath=chunk["metadata"]["header_path"],
        lastModified=chunk["metadata"]["last_modified"],
        doiUrl=chunk["metadata"].get("doi_url"),
        citationStr=chunk["metadata"].get("citation_str"),
        embedding=embedding
        )

logger.info("✅ Ingested 5 chunks successfully!")

# Verify
with driver.session() as session:
    result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
    count = result.single()["count"]
    logger.info(f"Total chunks in Neo4j: {count}")

driver.close()
