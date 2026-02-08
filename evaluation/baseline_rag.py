"""
Baseline RAG (Vector-Only) for Fair Comparison
Uses same Neo4j vector index and Groq LLM, but WITHOUT graph traversal
"""
import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from llama_index.llms.groq import Groq
import time

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

load_dotenv()

class BaselineRAG:
    """Simple vector-only RAG for baseline comparison"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        
        self.llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
        
        self.embed_model_name = "gemini-embedding-001"  # Same as Eros ionKG main system
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Google Generative AI"""
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        result = genai.embed_content(
            model=f"models/{self.embed_model_name}",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def retrieve_contexts(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve chunks using ONLY vector similarity (no graph traversal)
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Vector similarity search (no graph traversal)
        cypher_query = """
        CALL db.index.vector.queryNodes('erosion_chunk_index', $top_k, $query_embedding)
        YIELD node AS chunk, score
        RETURN 
            chunk.content AS content,
            chunk.sourceFile AS source_file,
            chunk.pageNumber AS page_number,
            chunk.doiUrl AS doi_url,
            chunk.citationStr AS citation_str,
            score
        ORDER BY score DESC
        """
        
        result = self.driver.execute_query(
            cypher_query,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        contexts = []
        for record in result.records:
            contexts.append({
                "content": record["content"],
                "source_file": record["source_file"],
                "page_number": record["page_number"],
                "doi_url": record["doi_url"],
                "citation_str": record["citation_str"],
                "score": record["score"]
            })
        
        return contexts
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Query the baseline RAG system
        Returns: answer, contexts, DOIs, response time
        """
        start_time = time.time()
        
        # Retrieve contexts (vector-only)
        contexts = self.retrieve_contexts(question, top_k=top_k)
        
        # Build simple prompt (no citation formatting like GraphRAG)
        context_text = "\n\n".join([
            f"Context {i+1}:\n{ctx['content']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""You are an expert research assistant specializing in soil erosion and land degradation.

Based on the following context, answer the question below. Provide a clear and detailed answer.

Context:
{context_text}

Question: {question}

Answer:"""
        
        # Get response from LLM
        response = self.llm.complete(prompt)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Extract DOIs from contexts
        dois = [ctx["doi_url"] for ctx in contexts if ctx["doi_url"] != "N/A"]
        
        return {
            "answer": response.text,
            "contexts": [ctx["content"] for ctx in contexts],
            "full_contexts": contexts,
            "dois": dois,
            "response_time_sec": response_time
        }
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()


if __name__ == "__main__":
    # Test baseline RAG
    baseline = BaselineRAG()
    
    test_question = "What is the impact of rainfall intensity on soil erosion rates?"
    print(f"Question: {test_question}\n")
    
    result = baseline.query(test_question)
    
    print(f"Answer: {result['answer']}\n")
    print(f"Response Time: {result['response_time_sec']:.2f}s")
    print(f"Number of contexts retrieved: {len(result['contexts'])}")
    print(f"DOIs found: {result['dois']}")
    
    baseline.close()
