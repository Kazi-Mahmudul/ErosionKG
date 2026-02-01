"""
GraphRAG Generation Engine
Uses Groq LLM (Llama 3.1 70B) for response generation with DOI-linked citations.
"""
import os
import logging
import sys
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Configuration
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# RESPONSE TEMPLATE
# --------------------------------------------------------------------------
GRAPHRAG_RESPONSE_TEMPLATE = """You are a **Geomorphology Research Assistant** specializing in landscape erosion, soil science, and environmental hydrology.

## Your Role
- Provide expert, research-backed answers using ONLY the provided context
- If the answer is not in the context, clearly state: "I don't have sufficient information to answer this question."
- Be concise but thorough

## Citation Contract (MANDATORY)
For EVERY factual claim, you MUST include a citation in this exact format:
**Claim** (Source: [Filename], Page: [X] | [DOI Link])

Example:
"Rainfall intensity is the primary driver of sheet erosion (Source: 2023_Soil_erosion_Bangladesh.pdf, Page: 12 | [DOI](https://doi.org/10.1016/j.example))"

If DOI is not available, use: [DOI: N/A]

## Context
### Knowledge Graph Relationships
{graph_context}

### Retrieved Entities
{entity_context}

## User Question
{query}

## Your Response (with citations):
"""


# --------------------------------------------------------------------------
# GROQ LLM SETUP
# --------------------------------------------------------------------------
def get_groq_llm():
    """Initialize Groq LLM with Llama 3.3 70B."""
    from llama_index.llms.groq import Groq
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return Groq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.3,  # Lower for more factual responses
    )


# --------------------------------------------------------------------------
# METADATA HELPER
# --------------------------------------------------------------------------
def format_metadata(source_file: str, page: Optional[str] = "N/A", doi_url: Optional[str] = "N/A", citation_str: Optional[str] = None) -> str:
    """Format metadata for citation."""
    # Try to construct citation string from filename if not provided
    if not citation_str or citation_str == "N/A":
        citation_str = source_file
    
    parts = [f"Source: {citation_str}"]
    if page and page != "N/A":
        parts.append(f"Page: {page}")
    
    if doi_url and doi_url != "N/A":
        parts.append(f"[DOI]({doi_url})")
    else:
        parts.append("[DOI: N/A]")
    
    return " | ".join(parts)


# --------------------------------------------------------------------------
# CONTEXT FORMATTER (for DOI-aware context)
# --------------------------------------------------------------------------
def format_context_with_doi(retrieval_result) -> tuple:
    """
    Format retrieval result into context strings with DOI metadata.
    Returns (graph_context, entity_context).
    """
    
    # Format graph triplets
    graph_lines = []
    for t in retrieval_result.triplets:
        # In a real system, we'd lookup metadata for the source file
        # Here we just use the filename as we don't have node-level metadata easy access in triplets yet
        # But we can try to guess author/year from filename like in pipeline
        source = t.source_file or "Unknown"
        citation = source
        doi_url = "N/A" # Triplets currently don't carry DOI, would need graph property lookup
        
        line = f"- ({t.subject}) -[{t.relationship}]-> ({t.obj}) (Source: {citation}, Page: N/A | [DOI: {doi_url}])"
        graph_lines.append(line)
    
    graph_context = "\n".join(graph_lines) if graph_lines else "No graph relationships found."
    
    # Format entity chunks
    entity_lines = []
    for chunk in retrieval_result.chunks:
        # Check for metadata in chunk entities (RetrievedChunk needs to support this or we rely on logic here)
        # VectorSearcher in graphrag_retriever.py currently returns chunks with limited metadata.
        # We need to assume retrieval_result.chunks coming from there might NOT have doi_url yet 
        # unless we updated VectorSearcher. 
        # For now, let's just make sure we handle what we have.
        
        source = chunk.source_file
        page = getattr(chunk, 'page_number', 'N/A')
        doi_url = getattr(chunk, 'doi_url', 'N/A')
        citation = getattr(chunk, 'citation_str', source)
        
        line = f"- **{chunk.content}** (score: {chunk.score:.3f}) (Source: {citation}, Page: {page} | [DOI: {doi_url}])"
        entity_lines.append(line)
    
    entity_context = "\n".join(entity_lines) if entity_lines else "No entities found."
    
    return graph_context, entity_context


# --------------------------------------------------------------------------
# GENERATION ENGINE
# --------------------------------------------------------------------------
class GraphRAGGenerator:
    """
    Combines the retriever with Groq LLM for GraphRAG response generation.
    """
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def generate(self, query: str, top_k: int = 3, hops: int = 1, stream: bool = False) -> str:
        """
        Generate a response using hybrid retrieval + Groq LLM.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            hops: Graph expansion depth
            stream: Whether to stream the response
            
        Returns:
            Generated response with citations
        """
        # 1. Retrieve context
        logger.info(f"Retrieving context for: {query}")
        retrieval_result = self.retriever.retrieve(query, top_k=top_k, hops=hops)
        
        # 2. Format context with DOI metadata
        graph_context, entity_context = format_context_with_doi(retrieval_result)
        
        # 3. Build prompt
        prompt = GRAPHRAG_RESPONSE_TEMPLATE.format(
            graph_context=graph_context,
            entity_context=entity_context,
            query=query
        )
        
        logger.info(f"Generating response with Groq LLM...")
        
        # 4. Generate response
        if stream:
            return self._stream_response(prompt)
        else:
            response = self.llm.complete(prompt)
            return response.text
    
    def _stream_response(self, prompt: str):
        """Stream the response token by token."""
        response_stream = self.llm.stream_complete(prompt)
        
        full_response = []
        for chunk in response_stream:
            token = chunk.delta
            print(token, end="", flush=True)
            full_response.append(token)
        
        print()  # Newline at end
        return "".join(full_response)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def create_generator():
    """Initialize and return the generator."""
    # Add parent dir to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from kg.graphrag_retriever import get_retriever
    
    retriever, driver = get_retriever()
    llm = get_groq_llm()
    
    generator = GraphRAGGenerator(retriever, llm)
    
    return generator, driver


def run_query(query: str, top_k: int = 3, hops: int = 1, stream: bool = True):
    """Run a query through the full GraphRAG pipeline."""
    generator, driver = create_generator()
    
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70 + "\n")
    
    try:
        response = generator.generate(query, top_k=top_k, hops=hops, stream=stream)
        
        if not stream:
            print(response)
        
        print("\n" + "=" * 70)
        
    finally:
        driver.close()
    
    return response


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG Generation Engine")
    parser.add_argument("--query", type=str, required=True, help="Query to answer")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--hops", type=int, default=1, help="Graph expansion depth")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()
    
    run_query(args.query, args.top_k, args.hops, stream=not args.no_stream)
