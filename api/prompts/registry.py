"""
Prompt Registry Module
Handles loading, validation, and versioning of prompt configurations.
"""
import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PromptMetadata:
    """Metadata for a prompt configuration."""
    version: str
    author: str
    model_target: str
    description: str
    created_at: Optional[str] = None


@dataclass
class Ontology:
    """Ontology definition with entities and relations."""
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)


@dataclass
class PromptConfig:
    """Complete prompt configuration including metadata, ontology, and template."""
    metadata: PromptMetadata
    ontology: Ontology
    template: str
    
    def format_prompt(self, text: str) -> str:
        """Format the prompt template with the given text."""
        return self.template.format(text=text)


class PromptRegistry:
    """
    Registry for managing versioned prompt configurations.
    
    Usage:
        registry = PromptRegistry()
        config = registry.get('v1')
        prompt = config.format_prompt("Some erosion text...")
    """
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the registry.
        
        Args:
            prompts_dir: Directory containing prompt YAML files.
                         Defaults to 'prompts/' relative to this file.
        """
        if prompts_dir is None:
            # Default to 'prompts/' directory relative to this file's parent
            prompts_dir = Path(__file__).parent
        
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, PromptConfig] = {}
        self._scan_prompts()
    
    def _scan_prompts(self) -> None:
        """Scan the prompts directory and index available versions."""
        self._available_versions: Dict[str, Path] = {}
        
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return
        
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data and 'metadata' in data:
                    version = data['metadata'].get('version')
                    if version:
                        self._available_versions[version] = yaml_file
                        logger.debug(f"Found prompt version: {version} at {yaml_file}")
            except Exception as e:
                logger.warning(f"Failed to scan {yaml_file}: {e}")
    
    def list_versions(self) -> List[str]:
        """Return a list of available prompt versions."""
        return list(self._available_versions.keys())
    
    def get(self, version_id: str) -> PromptConfig:
        """
        Load and return a prompt configuration by version ID.
        
        Args:
            version_id: The version identifier (e.g., 'v1')
            
        Returns:
            PromptConfig object with metadata, ontology, and template.
            
        Raises:
            ValueError: If the version is not found or validation fails.
        """
        # Check cache first
        if version_id in self._cache:
            return self._cache[version_id]
        
        # Find the file
        if version_id not in self._available_versions:
            available = ', '.join(self.list_versions()) or 'none'
            raise ValueError(f"Prompt version '{version_id}' not found. Available: {available}")
        
        yaml_path = self._available_versions[version_id]
        
        # Load and parse
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Validate and construct
        config = self._parse_config(data, yaml_path)
        
        # Cache it
        self._cache[version_id] = config
        
        return config
    
    def _parse_config(self, data: dict, source_path: Path) -> PromptConfig:
        """Parse and validate a prompt configuration from YAML data."""
        # Validate required sections
        required_sections = ['metadata', 'ontology', 'prompt_template']
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Missing required section '{section}' in {source_path}")
        
        # Parse metadata
        meta_data = data['metadata']
        required_meta = ['version', 'author', 'model_target', 'description']
        for field in required_meta:
            if field not in meta_data:
                raise ValueError(f"Missing required metadata field '{field}' in {source_path}")
        
        metadata = PromptMetadata(
            version=meta_data['version'],
            author=meta_data['author'],
            model_target=meta_data['model_target'],
            description=meta_data['description'],
            created_at=meta_data.get('created_at')
        )
        
        # Parse ontology
        onto_data = data['ontology']
        ontology = Ontology(
            entities=onto_data.get('entities', []),
            relations=onto_data.get('relations', [])
        )
        
        # Get template
        template = data['prompt_template']
        
        return PromptConfig(
            metadata=metadata,
            ontology=ontology,
            template=template
        )
    
    def reload(self) -> None:
        """Clear cache and rescan the prompts directory."""
        self._cache.clear()
        self._scan_prompts()


# Convenience function for quick access
def get_prompt(version_id: str = 'v1', prompts_dir: Optional[str] = None) -> PromptConfig:
    """
    Convenience function to get a prompt configuration.
    
    Args:
        version_id: The version to load (default: 'v1')
        prompts_dir: Optional custom prompts directory
        
    Returns:
        PromptConfig object
    """
    registry = PromptRegistry(prompts_dir)
    return registry.get(version_id)


if __name__ == "__main__":
    # Test the registry
    logging.basicConfig(level=logging.DEBUG)
    
    registry = PromptRegistry()
    print(f"Available versions: {registry.list_versions()}")
    
    if registry.list_versions():
        config = registry.get('v1')
        print(f"\nLoaded: {config.metadata.version}")
        print(f"Author: {config.metadata.author}")
        print(f"Model Target: {config.metadata.model_target}")
        print(f"Description: {config.metadata.description}")
        print(f"\nEntities: {config.ontology.entities}")
        print(f"Relations: {config.ontology.relations}")
        print(f"\nTemplate preview:\n{config.template[:200]}...")
