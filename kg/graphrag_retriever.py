"""
Hybrid GraphRAG Retriever
Combines vector similarity search with knowledge graph traversal for contextual RAG.
"""
import os
import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from neo4j import GraphDatabase

from llama_index.llms.gemini import Gemini

# Configuration
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vector index name (from vector_index.py)
VECTOR_INDEX_NAME = "erosion_vector_index"

# Data paths
CHUNKS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "extracted_chunks.json")


@dataclass
class RetrievedChunk:
    """A text chunk retrieved via vector search."""
    content: str
    source_file: str
    page_number: Optional[int] = None
    score: float = 0.0
    entities: List[str] = field(default_factory=list)


@dataclass
class GraphTriplet:
    """A triplet from the knowledge graph."""
    subject: str
    relationship: str
    obj: str
    source_file: Optional[str] = None


@dataclass
class RetrievalResult:
    """Combined result from hybrid retrieval."""
    query: str
    expanded_terms: List[str]
    chunks: List[RetrievedChunk]
    triplets: List[GraphTriplet]
    context: str


# --------------------------------------------------------------------------
# 1. SYNONYM EXPANDER
# --------------------------------------------------------------------------
class SynonymExpander:
    """Expand query terms using LLM for higher recall."""
    
    EXPANSION_PROMPT = """You are an expert in landscape erosion terminology.
Given a query about erosion, generate 3-5 synonyms or related terms that might appear in scientific papers.

Query: {query}

Return ONLY a JSON array of terms, e.g.: ["term1", "term2", "term3"]
Do not include the original query terms.
"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def expand(self, query: str) -> List[str]:
        """Expand query to include synonyms."""
        try:
            prompt = self.EXPANSION_PROMPT.format(query=query)
            response = self.llm.complete(prompt)
            text = response.text.strip()
            
            # Parse JSON array
            import re
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                terms = json.loads(match.group(0))
                return [t.strip().lower() for t in terms if isinstance(t, str)]
            return []
        except Exception as e:
            logger.warning(f"Synonym expansion failed: {e}")
            return []


# --------------------------------------------------------------------------
# 2. VECTOR SEARCH (reusing vector_index.py logic)
# --------------------------------------------------------------------------
class VectorSearcher:
    """Search text chunks using vector similarity."""
    
    def __init__(self, driver, embed_model):
        self.driver = driver
        self.embed_model = embed_model
        self.chunks_cache = None
    
    def _load_chunks(self) -> Dict[str, dict]:
        """Load chunks from JSON file."""
        if self.chunks_cache is not None:
            return self.chunks_cache
        
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Index by content hash for lookup
                self.chunks_cache = {
                    chunk.get("content", "")[:100]: chunk 
                    for chunk in data.get("chunks", [])
                }
        else:
            self.chunks_cache = {}
        return self.chunks_cache
    
    def search(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Search for relevant chunks using vector similarity on Entity nodes."""
        query_embedding = self.embed_model.get_text_embedding(query)
        
        # Search entities and return their names as "chunks"
        vector_query = f"""
        CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
        YIELD node, score
        RETURN node.name as name, node.text as text, score
        ORDER BY score DESC
        """
        
        result = self.driver.execute_query(
            vector_query,
            embedding=query_embedding,
            top_k=top_k
        )
        
        chunks = []
        for r in result.records:
            chunk = RetrievedChunk(
                content=r["text"] or r["name"],
                source_file="graph",
                score=r["score"],
                entities=[r["name"]]
            )
            chunks.append(chunk)
        
        return chunks


# --------------------------------------------------------------------------
# 3. ENTITY LINKER
# --------------------------------------------------------------------------
class EntityLinker:
    """Link chunk text to graph entities."""
    
    def __init__(self, driver):
        self.driver = driver
        self._entity_cache = None
    
    def _load_entities(self) -> List[str]:
        """Load all entity names from graph."""
        if self._entity_cache is not None:
            return self._entity_cache
        
        result = self.driver.execute_query("MATCH (e:Entity) RETURN e.name as name")
        self._entity_cache = [r["name"].lower() for r in result.records if r["name"]]
        return self._entity_cache
    
    def link(self, chunks: List[RetrievedChunk]) -> List[str]:
        """Find graph entities mentioned in chunks."""
        entities = self._load_entities()
        linked = set()
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            for entity in entities:
                if entity in content_lower:
                    linked.add(entity)
        
        return list(linked)[:20]  # Limit to avoid explosion


# --------------------------------------------------------------------------
# 4. GRAPH EXPANDER
# --------------------------------------------------------------------------
class GraphExpander:
    """Expand from entities to 1-2 hop neighbors."""
    
    def __init__(self, driver):
        self.driver = driver
    
    def expand(self, entity_names: List[str], hops: int = 1) -> List[GraphTriplet]:
        """Get relationships 1-2 hops from given entities."""
        if not entity_names:
            return []
        
        # Build case-insensitive match
        if hops == 1:
            query = """
            UNWIND $names as name
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower(name)
            MATCH (e)-[r]-(other:Entity)
            RETURN DISTINCT 
                e.name as subject,
                type(r) as rel_type,
                r.type as rel_subtype,
                other.name as object,
                r.source as source_file
            LIMIT 50
            """
        else:
            query = """
            UNWIND $names as name
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower(name)
            MATCH path = (e)-[*1..2]-(other:Entity)
            WITH e, other, relationships(path) as rels
            UNWIND rels as r
            RETURN DISTINCT 
                startNode(r).name as subject,
                type(r) as rel_type,
                r.type as rel_subtype,
                endNode(r).name as object,
                r.source as source_file
            LIMIT 100
            """
        
        result = self.driver.execute_query(query, names=entity_names)
        
        triplets = []
        for r in result.records:
            rel = r["rel_subtype"] or r["rel_type"]
            triplet = GraphTriplet(
                subject=r["subject"],
                relationship=rel,
                obj=r["object"],
                source_file=r["source_file"]
            )
            triplets.append(triplet)
        
        return triplets


# --------------------------------------------------------------------------
# 5. CONTEXT SYNTHESIZER
# --------------------------------------------------------------------------
class ContextSynthesizer:
    """Combine chunks and triplets into formatted context."""
    
    def synthesize(self, chunks: List[RetrievedChunk], triplets: List[GraphTriplet]) -> str:
        """Create a formatted context string with provenance."""
        parts = []
        
        # Text chunks section
        if chunks:
            parts.append("## Relevant Entities (from Vector Search)")
            for i, chunk in enumerate(chunks, 1):
                source = f"[Source: {chunk.source_file}]"
                parts.append(f"{i}. **{chunk.content}** (score: {chunk.score:.3f}) {source}")
        
        # Graph triplets section
        if triplets:
            parts.append("\n## Knowledge Graph Relationships")
            seen = set()
            for t in triplets:
                key = (t.subject, t.relationship, t.obj)
                if key in seen:
                    continue
                seen.add(key)
                source = f"[Source: {t.source_file}]" if t.source_file else ""
                parts.append(f"- ({t.subject}) -[{t.relationship}]-> ({t.obj}) {source}")
        
        return "\n".join(parts)


# --------------------------------------------------------------------------
# 6. HYBRID GRAPHRAG RETRIEVER
# --------------------------------------------------------------------------
class HybridGraphRAGRetriever:
    """Main orchestrator for hybrid vector + graph retrieval."""
    
    def __init__(self, driver, llm, embed_model):
        self.driver = driver
        self.llm = llm
        
        self.synonym_expander = SynonymExpander(llm)
        self.vector_searcher = VectorSearcher(driver, embed_model)
        self.entity_linker = EntityLinker(driver)
        self.graph_expander = GraphExpander(driver)
        self.synthesizer = ContextSynthesizer()
    
    def retrieve(self, query: str, top_k: int = 3, hops: int = 1, expand_synonyms: bool = True) -> RetrievalResult:
        """
        Perform hybrid retrieval:
        1. Expand query with synonyms
        2. Vector search for top-k entities
        3. Link entities from chunks
        4. Expand graph 1-2 hops
        5. Synthesize context
        """
        logger.info(f"Query: {query}")
        
        # 1. Synonym expansion
        expanded_terms = []
        if expand_synonyms:
            time.sleep(2)  # Rate limit
            expanded_terms = self.synonym_expander.expand(query)
            logger.info(f"Expanded terms: {expanded_terms}")
        
        # 2. Vector search
        chunks = self.vector_searcher.search(query, top_k=top_k)
        logger.info(f"Found {len(chunks)} chunks via vector search")
        
        # 3. Entity linking - use chunk entities + expanded terms
        linked_entities = []
        for chunk in chunks:
            linked_entities.extend(chunk.entities)
        
        # Also search for expanded terms directly
        for term in expanded_terms:
            linked_entities.append(term)
        
        linked_entities = list(set(linked_entities))
        logger.info(f"Linked entities: {linked_entities}")
        
        # 4. Graph expansion
        triplets = self.graph_expander.expand(linked_entities, hops=hops)
        logger.info(f"Found {len(triplets)} graph triplets")
        
        # 5. Context synthesis
        context = self.synthesizer.synthesize(chunks, triplets)
        
        return RetrievalResult(
            query=query,
            expanded_terms=expanded_terms,
            chunks=chunks,
            triplets=triplets,
            context=context
        )


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def get_retriever():
    """Initialize and return the retriever."""
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    
    llm = Gemini(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
    
    return HybridGraphRAGRetriever(driver, llm, embed_model), driver


def test_query(query: str, top_k: int = 3, hops: int = 1):
    """Test the retriever with a query."""
    retriever, driver = get_retriever()
    
    result = retriever.retrieve(query, top_k=top_k, hops=hops)
    
    print("\n" + "=" * 60)
    print(f"QUERY: {result.query}")
    print("=" * 60)
    
    if result.expanded_terms:
        print(f"\nExpanded Terms: {', '.join(result.expanded_terms)}")
    
    print(f"\nRetrieved {len(result.chunks)} chunks and {len(result.triplets)} triplets")
    print("\n" + "-" * 60)
    print("SYNTHESIZED CONTEXT:")
    print("-" * 60)
    print(result.context)
    print("=" * 60)
    
    driver.close()
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid GraphRAG Retriever")
    parser.add_argument("--query", type=str, required=True, help="Query to search")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--hops", type=int, default=1, help="Graph expansion hops (1 or 2)")
    args = parser.parse_args()
    
    test_query(args.query, args.top_k, args.hops)
