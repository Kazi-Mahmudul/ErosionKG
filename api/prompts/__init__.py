# Prompts Module
# This module provides versioned prompt management for the KG extraction pipeline.

from .registry import PromptRegistry, PromptConfig, get_prompt

__all__ = ['PromptRegistry', 'PromptConfig', 'get_prompt']
