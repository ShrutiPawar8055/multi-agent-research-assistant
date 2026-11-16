"""ResearchCrew: a small multi-agent research assistant with grounded, cited answers."""

from .config import Settings
from .models import Evidence, Result
from .pipeline import ResearchCrew
from .retrieval import KnowledgeBase

__version__ = "0.1.0"
__all__ = ["Evidence", "KnowledgeBase", "ResearchCrew", "Result", "Settings"]
