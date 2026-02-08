"""
FastAPI Entry Point for ErosionKG GraphRAG Dashboard
Integrates Groq LLM with GraphRAG retriever for full citation support
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import json
import asyncio
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

app = FastAPI(title="ErosionKG GraphRAG API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever and LLM (lazy loading)
_retriever = None
_llm = None
_driver = None


def get_retriever():
    """Lazy init retriever"""
    global _retriever, _driver
    if _retriever is None:
        from kg.graphrag_retriever import get_retriever as init_retriever
        _retriever, _driver = init_retriever()
    return _retriever


def get_llm():
    """Lazy init Groq LLM"""
    global _llm
    if _llm is None:
        from llama_index.llms.groq import Groq
        _llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
    return _llm


class ChatRequest(BaseModel):
    query: str
    stream: bool = True


def extract_graph_data(retrieval_result) -> dict:
    """Extract nodes and edges from retrieval result"""
    nodes = []
    edges = []
    seen_nodes = set()
    
    # Extract entities from chunks (Fallback if no triplets)
    for chunk in retrieval_result.chunks:
        # Check if chunk has entities list (depends on retriever implementation)
        chunk_entities = getattr(chunk, 'entities', [])
        for entity in chunk_entities:
            if entity and entity not in seen_nodes:
                nodes.append({
                    "id": entity,
                    "label": entity,
                    "type": "entity",
                    "color": "#4299e1" # Default blue for entities
                })
                seen_nodes.add(entity)
    
    # Extract relationships from graph triplets
    for triplet in retrieval_result.triplets:
        subject = triplet.subject
        obj = triplet.obj
        rel_type = triplet.relationship
        
        # Add nodes if not present
        for node_name in [subject, obj]:
            if node_name and node_name not in seen_nodes:
                nodes.append({
                    "id": node_name,
                    "label": node_name,
                    "type": "entity"
                })
                seen_nodes.add(node_name)
        
        
        # Add edge with source metadata
        if subject and obj:
            edges.append({
                "source": subject,
                "target": obj,
                "type": rel_type,
                "sourceFile": getattr(triplet, 'source_file', 'Unknown'),
                "doiUrl": getattr(triplet, 'doi_url', 'N/A')
            })

    
    return {"nodes": nodes, "edges": edges}


def format_context_with_doi(retrieval_result) -> tuple:
    """Format retrieval result with DOI metadata"""
    # Format graph triplets with DOI metadata
    graph_lines = []
    for t in retrieval_result.triplets:
        doi = getattr(t, 'doi_url', 'N/A')
        pg = getattr(t, 'page_number', 'Unknown')
        source = t.source_file or "Unknown"
        
        # Format the triplet line with source info
        line = f"- ({t.subject}) -[{t.relationship}]-> ({t.obj}) (Source: {source}, Page: {pg} | DOI: {doi})"
        graph_lines.append(line)
    
    graph_context = "\n".join(graph_lines) if graph_lines else "No graph relationships found."
    
    # Format context with metadata - ONLY include chunks with valid DOI
    context_parts = []
    source_num = 1
    for chunk in retrieval_result.chunks:
        # Access dataclass attributes directly
        source_file = chunk.source_file or "Unknown"
        page_num = chunk.page_number or "Unknown"
        doi_url = chunk.doi_url or "N/A"
        citation_str = chunk.citation_str or source_file
        
        # SKIP chunks without valid DOI to prevent hallucinated citations
        if not doi_url or doi_url == "N/A" or not doi_url.startswith("http"):
            continue
        
        # Format with clear metadata for LLM to cite
        context_parts.append(
            f"[Source {source_num}]\n"
            f"File: {citation_str}\n"
            f"Page: {page_num}\n"
            f"DOI: {doi_url}\n"
            f"Content: {chunk.content}\n"
        )
        source_num += 1
    
    entity_context = "\n\n".join(context_parts) if context_parts else "No entities found."
    
    return graph_context, entity_context


GRAPHRAG_RESPONSE_TEMPLATE = """You are an expert research assistant specializing in soil erosion and land degradation.

Question: {query}

CRITICAL CITATION RULES:
- ONLY cite sources that are explicitly provided in the "Knowledge Graph Relationships" or "Retrieved Entities" sections below
- NEVER make up, invent, or fabricate any citations or DOI links
- NEVER cite a source that doesn't appear in the context with a valid DOI link
- Every citation MUST use this EXACT format: (Source: <filename>, Page: <page_number> | DOI: <doi_url>)
- If you cannot find relevant information WITH a valid DOI in the provided sources, say "I don't have enough information with verifiable sources to answer this question."

Citation Format Examples:
✓ CORRECT: "Rainfall erosivity is critical... (Source: A review of the (Revised) Universal Soil Loss Equation, Page: 7 | DOI: https://doi.org/10.5194/hess-22-6059-2018)"
✗ WRONG: Citing any source not in the "Knowledge Graph Relationships" or "Retrieved Entities" sections
✗ WRONG: Making up DOI links or page numbers
✗ WRONG: Using sources without DOI links

Answer:## Context
### Knowledge Graph Relationships
{graph_context}

### Retrieved Entities
{entity_context}
## User Question
{query}

## Your Response (with citations):
"""


RELATED_QUERIES_PROMPT = """Based on this Q&A about soil erosion research:

Question: {query}
Answer Summary: {response_summary}

Generate exactly 3 short, specific follow-up questions a researcher might ask next.
Keep each question under 60 characters. Focus on related concepts, causes, or solutions.

Output ONLY the 3 questions, one per line. No numbering, no bullets, no explanations."""


async def stream_chat_response(query: str):
    """Stream chat response with real GraphRAG data"""
    try:
        # Get retriever and LLM
        retriever = get_retriever()
        llm = get_llm()
        
        # Retrieve context
        retrieval_result = retriever.retrieve(query, top_k=3, hops=1)
        
        # Extract graph data for visualization
        graph_data = extract_graph_data(retrieval_result)
        
        # Send graph data first
        yield f"data: {json.dumps({'type': 'graph', 'data': graph_data})}\n\n"
        
        # Format context with DOI
        graph_context, entity_context = format_context_with_doi(retrieval_result)
        
        # Build prompt
        prompt = GRAPHRAG_RESPONSE_TEMPLATE.format(
            graph_context=graph_context,
            entity_context=entity_context,
            query=query
        )
        
        # Stream response from Groq and collect full text
        response_stream = llm.stream_complete(prompt)
        full_response = ""
        
        for chunk in response_stream:
            if chunk.delta:
                full_response += chunk.delta
                yield f"data: {json.dumps({'type': 'text', 'content': chunk.delta})}\n\n"
                await asyncio.sleep(0.01)
        
        # Send completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        # Generate related queries
        try:
            # Use first 500 chars of response as summary for prompt
            response_summary = full_response[:500].replace('\n', ' ')
            related_prompt = RELATED_QUERIES_PROMPT.format(
                query=query,
                response_summary=response_summary
            )
            
            related_response = llm.complete(related_prompt)
            related_text = related_response.text.strip()
            
            # Parse the 3 questions (one per line)
            related_queries = [q.strip() for q in related_text.split('\n') if q.strip()][:3]
            
            if related_queries:
                yield f"data: {json.dumps({'type': 'related_queries', 'queries': related_queries})}\n\n"
        except Exception as rq_error:
            # Don't fail the whole response if related queries fail
            print(f"Related queries generation failed: {rq_error}")
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"


@app.get("/")
async def root():
    return {"message": "ErosionKG GraphRAG API", "version": "1.0.0", "status": "running"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Streaming chat endpoint with full GraphRAG integration"""
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request.query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        # Non-streaming fallback
        retriever = get_retriever()
        llm = get_llm()
        
        retrieval_result = retriever.retrieve(request.query, top_k=3, hops=1)
        graph_data = extract_graph_data(retrieval_result)
        
        graph_context, entity_context = format_context_with_doi(retrieval_result)
        
        prompt = GRAPHRAG_RESPONSE_TEMPLATE.format(
            graph_context=graph_context,
            entity_context=entity_context,
            query=request.query
        )
        
        response = llm.complete(prompt)
        
        return {
            "response": response.text,
            "graph_data": graph_data
        }


@app.get("/api/metadata")
async def get_metadata():
    """Get research library metadata"""
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        
        # Get statistics using CALL subqueries for clean separation
        stats_query = """
        CALL {
            MATCH (c:Chunk)
            RETURN count(DISTINCT c.sourceFile) as paper_count
        }
        CALL {
            MATCH (e:Entity)
            RETURN count(e) as entity_count
        }
        CALL {
            MATCH ()-[r:RELATES]->()
            RETURN count(r) as relationship_count
        }
        RETURN paper_count, entity_count, relationship_count
        """
        
        result = driver.execute_query(stats_query)
        stats = result.records[0] if result.records else {}
        
        # Get unique DOIs
        doi_query = """
        MATCH (c:Chunk)
        WHERE c.doiUrl IS NOT NULL AND c.doiUrl <> 'N/A'
        RETURN DISTINCT c.doiUrl as doi, c.sourceFile as file, c.citationStr as citation
        ORDER BY c.sourceFile
        LIMIT 20
        """
        
        doi_result = driver.execute_query(doi_query)
        dois = [
            {
                "doi": r["doi"],
                "file": r["file"],
                "citation": r["citation"]
            } 
            for r in doi_result.records
        ]
        
        driver.close()
        
        return {
            "paper_count": stats.get("paper_count", 0),
            "entity_count": stats.get("entity_count", 0),
            "relationship_count": stats.get("relationship_count", 0),
            "dois": dois
        }
        
    except Exception as e:
        import traceback
        print(f"Metadata error: {e}")
        traceback.print_exc()
        # Return empty data on error
        return {
            "paper_count": 0,
            "entity_count": 0,
            "relationship_count": 0,
            "dois": []
        }


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with GraphRAG integration")
    print("http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
