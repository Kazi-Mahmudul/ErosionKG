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
VECTOR_INDEX_NAME = "erosion_chunk_index"

# Data paths
CHUNKS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "extracted_chunks.json")


@dataclass
class RetrievedChunk:
    """A text chunk retrieved via vector search."""
    content: str
    source_file: str
    page_number: Optional[int] = None
    doi_url: Optional[str] = "N/A"
    citation_str: Optional[str] = None
    score: float = 0.0
    entities: List[str] = field(default_factory=list)


@dataclass
class GraphTriplet:
    """A triplet from the knowledge graph."""
    subject: str
    relationship: str
    obj: str
    source_file: Optional[str] = None
    doi_url: Optional[str] = "N/A"
    page_number: Optional[str] = "Unknown"


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
        """Search for relevant chunks using vector similarity on Chunk nodes."""
        query_embedding = self.embed_model.get_text_embedding(query)
        
        # Search chunks and return metadata
        vector_query = f"""
        CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
        YIELD node, score
        RETURN node.content as content, node.sourceFile as source_file, node.pageNumber as page_number, node.doiUrl as doi_url, node.citationStr as citation_str, score
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
                content=r["content"],
                source_file=r["source_file"],
                page_number=r["page_number"],
                doi_url=r["doi_url"],
                citation_str=r["citation_str"],
                score=r["score"],
                entities=[] # Entities will be linked by EntityLinker
            )
            chunks.append(chunk)

        # Keyword Search Supplement (for specific terms like "rill")
        # Extract keywords - exclude common stop words and punctuation
        stop_words = {'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'could', 'should', 'would', 'now'}
        
        raw_words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
        words = [w for w in raw_words if len(w) > 3 and w not in stop_words]
        
        if words:
            logger.info(f"Supplemental keyword search for: {words}")
            # Identify the most specific word (least common)
            # For this domain, we prioritize specific types of erosion
            priority_words = [w for w in words if w in {'rill', 'gully', 'ls-factor', 'rusle', 'musle', 'sediment', 'deposition', 'rill', 'inter-rill'}]
            search_words = priority_words if priority_words else words
            
            # Search for chunks containing specific keywords (must contain at least one priority word if present)
            keyword_query = """
            MATCH (c:Chunk)
            WHERE any(word IN $words WHERE toLower(c.content) CONTAINS word)
            RETURN c.content as content, c.sourceFile as source_file, c.pageNumber as page_number, 
                   c.doiUrl as doi_url, c.citationStr as citation_str, 0.9 as score
            ORDER BY size([word IN $words WHERE toLower(c.content) CONTAINS word]) DESC
            LIMIT $limit
            """
            kw_result = self.driver.execute_query(
                keyword_query,
                words=search_words,
                limit=5
            )
            
            seen_content = {c.content for c in chunks}
            for r in kw_result.records:
                if r["content"] not in seen_content:
                    chunks.append(RetrievedChunk(
                        content=r["content"],
                        source_file=r["source_file"],
                        page_number=r["page_number"],
                        doi_url=r["doi_url"],
                        citation_str=r["citation_str"],
                        score=r["score"],
                        entities=[]
                    ))
        
        # Final sort by score
        chunks.sort(key=lambda x: x.score, reverse=True)
        return chunks[:10] # Return up to 10 total


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
        """Find graph entities mentioned in chunks using fuzzy matching."""
        from rapidfuzz import process, fuzz
        
        entities = self._load_entities()
        linked = set()
        
        logger.info(f"Linking against {len(entities)} known knowledge graph entities")
        
        # 1. Direct substring match (fast)
        for chunk in chunks:
            chunk_content = chunk.content.lower()
            for entity in entities:
                # Require word boundaries for short entities to avoid matching inside other words
                if len(entity) <= 3:
                     if f" {entity} " in f" {chunk_content} ":
                         linked.add(entity)
                elif entity in chunk_content:
                    linked.add(entity)
        
        # 2. Fuzzy match chunks if few entities found
        if len(linked) < 3:
            logger.info("Few entities found via substring, trying fuzzy match on chunks...")
            for chunk in chunks:
                # Extract potential noun phrases or just match against whole chunk content window
                # For simplicity/speed, we match top entities against the chunk text
                # This is expensive O(N*M), so we limit to top vector search results
                
                # Better approach: Extract keywords from chunk and fuzzy match those
                pass 

        # 3. Always exact/fuzzy match the *query terms* against entities
        # (The retriever calls link with chunks, but we should also consider the query itself 
        # which is passed as expanded_terms in the main retrieve method. 
        # But here we only have access to chunks. 
        # The retrieve method handles expanded terms separately, so we are good.)
        
        result = list(linked)[:20]
        logger.info(f"Total unique entities linked from chunks: {len(result)} - {result}")
        return result


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
            OPTIONAL MATCH (c:Chunk) WHERE c.sourceFile = r.source
            WITH e, r, other, collect(c.doiUrl)[0] as chunk_doi, collect(c.pageNumber)[0] as chunk_page
            RETURN DISTINCT 
                e.name as subject,
                type(r) as rel_type,
                r.type as rel_subtype,
                other.name as object,
                r.source as source_file,
                coalesce(chunk_doi, "N/A") as doi_url,
                coalesce(toString(chunk_page), "Unknown") as page_number
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
            OPTIONAL MATCH (c:Chunk) WHERE c.sourceFile = r.source
            WITH startNode(r) as subt, r, endNode(r) as objt, collect(c.doiUrl)[0] as chunk_doi, collect(c.pageNumber)[0] as chunk_page
            RETURN DISTINCT 
                subt.name as subject,
                type(r) as rel_type,
                r.type as rel_subtype,
                objt.name as object,
                r.source as source_file,
                coalesce(chunk_doi, "N/A") as doi_url,
                coalesce(toString(chunk_page), "Unknown") as page_number
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
                source_file=r["source_file"],
                doi_url=r["doi_url"],
                page_number=r["page_number"]
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
                citation = chunk.citation_str or chunk.source_file
                source = f"[Source: {citation}, Page: {chunk.page_number}]"
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
                source_text = ""
                if t.doi_url and t.doi_url != "N/A":
                     source_text = f" [Source: {t.source_file}, Page: {t.page_number} | DOI: {t.doi_url}]"
                elif t.source_file:
                     source_text = f" [Source: {t.source_file}]"
                     
                parts.append(f"- ({t.subject}) -[{t.relationship}]-> ({t.obj}){source_text}")
        
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
        # Fix: Previously we were just extending empty lists. Now we call the linker.
        linked_entities = self.entity_linker.link(chunks)
        
        # Also fuzzy match the QUERY against entities to find entry points
        # This is critical if chunks don't contain exact entity names
        from rapidfuzz import process, fuzz
        all_entities = self.entity_linker._load_entities()
        
        # Match query directly
        query_matches = process.extract(query, all_entities, scorer=fuzz.partial_ratio, limit=5, score_cutoff=85)
        for match, score, idx in query_matches:
            logger.info(f"Fuzzy match query '{query}' -> '{match}' (score: {score})")
            linked_entities.append(match)
            
        # Match expanded terms
        for term in expanded_terms:
            # Fuzzy match expanded terms
            term_matches = process.extract(term, all_entities, scorer=fuzz.ratio, limit=2, score_cutoff=85)
            for match, score, idx in term_matches:
               linked_entities.append(match)
        
        linked_entities = list(set(linked_entities))
        logger.info(f"Final Linked entities for expansion: {linked_entities}")
        
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
    from llama_index.embeddings.gemini import GeminiEmbedding
    from llama_index.llms.groq import Groq
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    
    llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=GOOGLE_API_KEY)
    
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
