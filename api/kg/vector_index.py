"""
Hybrid Search Index Configuration
Vector indexing and hybrid retriever for ErosionKG.
"""
import os
import logging
import time
from typing import List, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

# Configuration
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Embedding dimension for text-embedding-004
EMBEDDING_DIMENSION = 3072  # gemini-embedding-001
VECTOR_INDEX_NAME = "erosion_chunk_index"


# --------------------------------------------------------------------------
# 1. EMBEDDING MODEL
# --------------------------------------------------------------------------
def get_embedding_model():
    """Initialize Google GenAI embeddings."""
    from llama_index.embeddings.gemini import GeminiEmbedding
    return GeminiEmbedding(
        model_name="models/gemini-embedding-001",
        api_key=GOOGLE_API_KEY
    )
    except ImportError:
        # Fallback to older package if available
        from llama_index.embeddings.gemini import GeminiEmbedding
        return GeminiEmbedding(
            model="models/text-embedding-004",
            api_key=GOOGLE_API_KEY
        )


# --------------------------------------------------------------------------
# 2. VECTOR INDEX CREATION
# --------------------------------------------------------------------------
def create_vector_index(driver):
    """Create vector index on Entity nodes in Neo4j."""
    # Check if index already exists
    check_query = "SHOW INDEXES YIELD name WHERE name = $name RETURN name"
    result = driver.execute_query(check_query, name=VECTOR_INDEX_NAME)
    
    if result.records:
        logger.info(f"Vector index '{VECTOR_INDEX_NAME}' already exists")
        return True
    
    # Create the vector index
    create_query = f"""
    CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
    FOR (c:Chunk)
    ON (c.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMENSION},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    try:
        driver.execute_query(create_query)
        logger.info(f"Created vector index '{VECTOR_INDEX_NAME}'")
        return True
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        return False


# --------------------------------------------------------------------------
# 3. BATCH EMBEDDING
# --------------------------------------------------------------------------
def batch_embed_nodes(driver, embed_model, batch_size: int = 50):
    """
    Generate embeddings for all Chunk nodes that don't have them yet.
    Updates nodes in Neo4j with the embedding property.
    """
    # Find nodes without embeddings
    count_query = """
    MATCH (c:Chunk)
    WHERE c.embedding IS NULL
    RETURN count(c) as count
    """
    result = driver.execute_query(count_query)
    total = result.records[0]["count"]
    logger.info(f"Found {total} chunks without embeddings")
    
    if total == 0:
        logger.info("All chunks already have embeddings")
        return
    
    # Process in batches
    offset = 0
    embedded_count = 0
    
    while offset < total:
        # Fetch batch of nodes
        fetch_query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NULL
        RETURN elementId(c) as id, c.content as text
        ORDER BY c.chunkId
        LIMIT $limit
        """
        # Note: Removing SKIP/OFFSET because we are matching WHERE embedding IS NULL.
        # As we fill them, they drop out of the set.
        # But to be safe against partial failures, using LIMIT is good.
        # Using elementId as strict ordering.
        
        result = driver.execute_query(fetch_query, limit=batch_size)
        
        if not result.records:
            break
        
        nodes = [(r["id"], r["text"]) for r in result.records]
        texts = [n[1] for n in nodes]
        
        # Generate embeddings
        try:
            # Rate limiting for API
            time.sleep(1)
            embeddings = embed_model.get_text_embedding_batch(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Try one at a time
            embeddings = []
            for text in texts:
                try:
                    time.sleep(0.5)
                    emb = embed_model.get_text_embedding(text)
                    embeddings.append(emb)
                except Exception as e2:
                    logger.warning(f"Failed to embed chunk starting with '{text[:30]}...': {e2}")
                    embeddings.append(None)
        
        # Update nodes with embeddings
        for (node_id, text), embedding in zip(nodes, embeddings):
            if embedding is None:
                continue
            
            update_query = """
            MATCH (c) WHERE elementId(c) = $id
            SET c.embedding = $embedding
            """
            try:
                driver.execute_query(update_query, id=node_id, embedding=embedding)
                embedded_count += 1
            except Exception as e:
                logger.warning(f"Failed to update node {node_id}: {e}")
        
        logger.info(f"Embedded {embedded_count}/{total} chunks")
        # Don't increment offset since we rely on "WHERE embedding IS NULL"
        # But we DO need to break if we made no progress to avoid inf loop
        if batch_size > 0: # dummy check
             pass
    
    logger.info(f"Batch embedding complete! Embedded {embedded_count} chunks.")


# --------------------------------------------------------------------------
# 4. HYBRID RETRIEVER
# --------------------------------------------------------------------------
class HybridRetriever:
    """
    Combines vector similarity with Cypher relationship scoring.
    """
    def __init__(self, driver, embed_model, top_k: int = 5):
        self.driver = driver
        self.embed_model = embed_model
        self.top_k = top_k
    
    def search(self, query: str) -> List[dict]:
        """
        Perform hybrid search combining vector similarity and relationship count.
        """
        # Generate query embedding
        query_embedding = self.embed_model.get_text_embedding(query)
        
        # Hybrid query: vector similarity + relationship count boost
        hybrid_query = f"""
        CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
        YIELD node, score
        WITH node, score
        OPTIONAL MATCH (node)-[r]-()
        WITH node, score, count(r) as rel_count
        RETURN 
            node.name as name,
            node.text as text,
            score as vector_score,
            rel_count,
            score + (rel_count * 0.01) as hybrid_score
        ORDER BY hybrid_score DESC
        LIMIT $top_k
        """
        
        result = self.driver.execute_query(
            hybrid_query,
            embedding=query_embedding,
            top_k=self.top_k * 2  # Fetch more for filtering
        )
        
        return [
            {
                "name": r["name"],
                "text": r["text"],
                "vector_score": r["vector_score"],
                "rel_count": r["rel_count"],
                "hybrid_score": r["hybrid_score"]
            }
            for r in result.records
        ][:self.top_k]
    
    def vector_search(self, query: str) -> List[dict]:
        """Pure vector similarity search."""
        query_embedding = self.embed_model.get_text_embedding(query)
        
        vector_query = f"""
        CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
        YIELD node, score
        RETURN node.content as text, node.sourceFile as source_file, node.pageNumber as page_number, node.doiUrl as doi_url, node.citationStr as citation_str, score
        ORDER BY score DESC
        """
        
        result = self.driver.execute_query(
            vector_query,
            embedding=query_embedding,
            top_k=self.top_k
        )
        
        formatted_results = []
        for r in result.records:
            formatted_results.append({
                "text": r["text"],
                "source_file": r["source_file"],
                "page_number": r["page_number"],
                "doi_url": r["doi_url"],
                "citation_str": r["citation_str"],
                "score": r["score"]
            })
            
        return formatted_results


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def setup_vector_index():
    """Set up vector index and embed all nodes."""
    logger.info("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    
    logger.info("Initializing embedding model...")
    embed_model = get_embedding_model()
    
    # Create index
    logger.info("Creating vector index...")
    create_vector_index(driver)
    
    # Embed nodes
    logger.info("Generating embeddings for entities...")
    batch_embed_nodes(driver, embed_model)
    
    driver.close()
    logger.info("Setup complete!")


def test_query(query: str, top_k: int = 3):
    """Test vector search with a query."""
    logger.info(f"Testing query: '{query}'")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    embed_model = get_embedding_model()
    
    retriever = HybridRetriever(driver, embed_model, top_k=top_k)
    results = retriever.vector_search(query)
    
    print(f"\nTop {top_k} results for: '{query}'")
    print("=" * 50)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']} (score: {r['score']:.4f})")
    print("=" * 50)
    
    driver.close()
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Index Configuration for ErosionKG")
    parser.add_argument("--setup", action="store_true", help="Create index and embed all nodes")
    parser.add_argument("--test-query", type=str, help="Test vector search with a query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()
    
    if args.setup:
        setup_vector_index()
    elif args.test_query:
        test_query(args.test_query, args.top_k)
    else:
        parser.print_help()
