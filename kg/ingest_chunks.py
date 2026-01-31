import os
import json
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class ChunkIngester:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE")
            logger.info("Constraints created/verified.")

    def ingest_chunks(self, chunks_file):
        if not os.path.exists(chunks_file):
            logger.error(f"File not found: {chunks_file}")
            return

        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chunks = data.get("chunks", [])

        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")

        with self.driver.session() as session:
            # Clear existing chunks (optional: remove this if we want append-only or need detailed merge)
            # For now, we want the graph to reflect the latest extraction, so we delete old chunks.
            logger.info("Deleting existing Chunk nodes...")
            session.run("MATCH (c:Chunk) DETACH DELETE c")
            
            # Batch ingestion
            batch_size = 500
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                self._write_batch(session, batch)
                logger.info(f"Ingested batch {i}-{i+len(batch)}")

    def _write_batch(self, session, batch):
        query = """
        UNWIND $batch AS item
        CREATE (c:Chunk {
            chunkId: item.metadata.chunk_id,
            content: item.content,
            sourceFile: item.metadata.source_file,
            pageNumber: item.metadata.page_number,
            headerPath: item.metadata.header_path,
            lastModified: item.metadata.last_modified,
            doiUrl: coalesce(item.metadata.doi_url, "N/A"),
            citationStr: coalesce(item.metadata.citation_str, item.metadata.source_file)
        })
        """
        session.run(query, batch=batch)

if __name__ == "__main__":
    if not NEO4J_PASSWORD:
        logger.error("NEO4J_PASSWORD not set in .env")
        exit(1)

    ingester = ChunkIngester(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        ingester.create_constraints()
        ingester.ingest_chunks("data/extracted_chunks.json")
    finally:
        ingester.close()
