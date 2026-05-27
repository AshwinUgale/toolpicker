"""ToolPicker - hybrid lexical + semantic tool selection for LLM agents.

See https://github.com/ashwinugale/toolpicker for usage.
"""

from toolpicker.embeddings import EmbeddingProvider, HashEmbedder, OpenAIEmbeddings
from toolpicker.fusion import reciprocal_rank_fusion
from toolpicker.retrievers import BM25Retriever, Retriever, SemanticRetriever
from toolpicker.router import ToolPicker
from toolpicker.sources import FunctionSchemaSource
from toolpicker.types import RetrievalHit, Tool, ToolSource

__version__ = "0.1.0"

__all__ = [
    "BM25Retriever",
    "EmbeddingProvider",
    "FunctionSchemaSource",
    "HashEmbedder",
    "OpenAIEmbeddings",
    "RetrievalHit",
    "Retriever",
    "SemanticRetriever",
    "Tool",
    "ToolPicker",
    "ToolSource",
    "__version__",
    "reciprocal_rank_fusion",
]
