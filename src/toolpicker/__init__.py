"""ToolPicker - hybrid lexical + semantic tool selection for LLM agents.

Pick K relevant tools out of N for an LLM agent. Three-stage router
(BM25 + embeddings + optional intent classifier) fused via Reciprocal
Rank Fusion, with a token-budget packer on top.

Public surface (stable from v1.0):

* `ToolPicker` - the facade. Construct with a `ToolSource`, an optional
  embedder, and an optional intent classifier. Call `select(query, k=...)`.
* Sources: `FunctionSchemaSource`, `OpenAPISource`, `MCPSource`,
  `MergedSource`.
* Embedders: `OpenAIEmbeddings`, `HashEmbedder`, `CachedEmbedder`. All
  satisfy the `EmbeddingProvider` Protocol.
* Intent: `IntentClassifier` Protocol + `EmbeddingNNIntent` reference impl
  + `IntentExample` for labelled training pairs.
* Retrievers (for custom fusion): `Retriever` Protocol, `BM25Retriever`,
  `SemanticRetriever`, `reciprocal_rank_fusion`.
* Packer: `pack_to_budget`, `count_tokens`, `default_serialise`.
* Core types: `Tool`, `ToolSource`.

Docs: https://ashwinugale.github.io/toolpicker/
Repo: https://github.com/ashwinugale/toolpicker
"""

from toolpicker.cache import CachedEmbedder
from toolpicker.embeddings import EmbeddingProvider, HashEmbedder, OpenAIEmbeddings
from toolpicker.fusion import reciprocal_rank_fusion
from toolpicker.intent import EmbeddingNNIntent, IntentClassifier, IntentExample
from toolpicker.packer import count_tokens, default_serialise, pack_to_budget
from toolpicker.retrievers import BM25Retriever, Retriever, SemanticRetriever
from toolpicker.router import ToolPicker
from toolpicker.sources import (
    FunctionSchemaSource,
    MCPSource,
    MergedSource,
    OpenAPISource,
)
from toolpicker.types import Tool, ToolSource

__version__ = "1.0.0"

__all__ = [
    "BM25Retriever",
    "CachedEmbedder",
    "EmbeddingNNIntent",
    "EmbeddingProvider",
    "FunctionSchemaSource",
    "HashEmbedder",
    "IntentClassifier",
    "IntentExample",
    "MCPSource",
    "MergedSource",
    "OpenAIEmbeddings",
    "OpenAPISource",
    "Retriever",
    "SemanticRetriever",
    "Tool",
    "ToolPicker",
    "ToolSource",
    "__version__",
    "count_tokens",
    "default_serialise",
    "pack_to_budget",
    "reciprocal_rank_fusion",
]
