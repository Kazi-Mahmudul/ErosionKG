import os
import json
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Import Embedding Model
from llama_index.embeddings.gemini import GeminiEmbedding

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VECTOR_INDEX_NAME = "erosion_chunk_index"
EMBEDDING_DIMENSION = 768  # gemini-embedding-001

class ChunkIngester:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            raise ValueError("GOOGLE_API_KEY is required for embedding generation")
            
        self.embed_model = GeminiEmbedding(
            model_name="models/gemini-embedding-001", 
            api_key=GOOGLE_API_KEY
        )

    def close(self):
        self.driver.close()

    def create_constraints_and_index(self):
        with self.driver.session() as session:
            # 1. Unique Constraint
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE")
            logger.info("Constraints created/verified.")
            
            # 2. Vector Index
            # check if index exists
            result = session.run("SHOW VECTOR INDEXES YIELD name WHERE name = $name", name=VECTOR_INDEX_NAME)
            if not result.peek():
                logger.info(f"Creating vector index: {VECTOR_INDEX_NAME}")
                session.run(f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIMENSION},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """)
            else:
                logger.info(f"Vector index {VECTOR_INDEX_NAME} already exists.")

    def ingest_chunks(self, chunks_file):
        if not os.path.exists(chunks_file):
            logger.error(f"File not found: {chunks_file}")
            return

        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chunks = data.get("chunks", [])

        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")

        with self.driver.session() as session:
            # Clear existing chunks to avoid duplicates/stale data
            logger.info("Deleting existing Chunk nodes...")
            session.run("MATCH (c:Chunk) DETACH DELETE c")
            
            # Batch ingestion
            batch_size = 50
            total_chunks = len(chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i+batch_size]
                
                # Generate embeddings for the batch
                texts = [item["content"] for item in batch]
                try:
                    embeddings = self.embed_model.get_text_embedding_batch(texts)
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {i}: {e}")
                    continue
                
                # Add embeddings to batch items
                for item, embedding in zip(batch, embeddings):
                    item["embedding"] = embedding

                self._write_batch(session, batch)
                logger.info(f"Ingested batch {i}-{min(i+batch_size, total_chunks)} / {total_chunks}")

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
            citationStr: coalesce(item.metadata.citation_str, item.metadata.source_file),
            embedding: item.embedding
        })
        """
        session.run(query, batch=batch)

if __name__ == "__main__":
    if not NEO4J_PASSWORD:
        logger.error("NEO4J_PASSWORD not set in .env")
        exit(1)

    ingester = ChunkIngester(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        ingester.create_constraints_and_index()
        ingester.ingest_chunks("api/data/extracted_chunks.json")
    finally:
        ingester.close()
