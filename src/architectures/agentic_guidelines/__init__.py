"""
5_agentic_guidelines Architecture Package
Pure LLM Self-Reflective Annotation & Dynamic Guideline Curation (Generator-Reflector-Curator Loop).
"""

from src.architectures.agentic_guidelines.generator import Generator
from src.architectures.agentic_guidelines.reflector import Reflector
from src.architectures.agentic_guidelines.curator import Curator
from src.architectures.agentic_guidelines.combined_agent import ReflectorCuratorCombined
from src.architectures.agentic_guidelines.manager import DynamicGuidebookManager, DynamicGuidelinesManager
from src.architectures.agentic_guidelines.schema_store import GuidebookStore, SchemaStore
from src.architectures.agentic_guidelines.pipeline import LSF_NER_Pipeline as AgenticNERPipeline

__all__ = [
    "Generator",
    "Reflector",
    "Curator",
    "ReflectorCuratorCombined",
    "DynamicGuidebookManager",
    "DynamicGuidelinesManager",
    "GuidebookStore",
    "SchemaStore",
    "AgenticNERPipeline",
]
