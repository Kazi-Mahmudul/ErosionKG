# Evaluation Module
# This folder contains the evaluation system for comparing ErosionKG GraphRAG vs Baseline RAG

from .baseline_rag import BaselineRAG
from .evaluate_graphrag import run_evaluation

__all__ = ['BaselineRAG', 'run_evaluation']
