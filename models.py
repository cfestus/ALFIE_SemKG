"""
models.py
---------
Core data models for the Semantic Knowledge Graph Pipeline.
Defines Entity, Relation, ProcessingResult, AnnotationBatch, and metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

def _looks_like_iri(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("urn:"))

# ============================================================
# ENTITY MODEL (UPDATED FOR LLM + ONTOLOGY MAPPING PIPELINE)
# ============================================================

@dataclass
class Entity:
    """
    Represents an extracted entity with:
    - canonical URI
    - LLM semantic type
    - ontology mapping
    - provenance (doc ID + text span)
    - extra metadata
    """

    uri: str                               # Canonical entity URI (etd:xxxxx)
    label: str                             # Human-readable surface form
    entity_type: str                       # LLM semantic type (Dataset, BiasType…)
    confidence: float = 0.85

    ontology_uri: Optional[str] = None     # Primary ontology class (VAIR / AIPO / etc.)

    # Provenance
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    doc_id: Optional[str] = None
    source: str = "llm"                    # "llm", "regex", "manual", etc.

    # For cross-ontology alignment (AIPO, VAIR, FMO…)
    additional_ontologies: Dict[str, str] = field(default_factory=dict)

    # Arbitrary metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Validation ---
    def __post_init__(self):
        try:
            self.confidence = float(self.confidence)
        except Exception as e:
            raise ValueError("confidence must be a float") from e
        if not self.uri:
            raise ValueError("Entity.uri must be a non-empty string")
        if not _looks_like_iri(self.uri):
            raise ValueError(f"Entity.uri must look like an IRI, got: {self.uri}")        
        if not self.label:
            raise ValueError("Entity.label must be a non-empty string")
        if not self.entity_type:
            raise ValueError("Entity.entity_type cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Entity.confidence must be between 0 and 1")

    # --- Serialization ---
    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "label": self.label,
            "entity_type": self.entity_type,
            "ontology_uri": self.ontology_uri,
            "confidence": round(self.confidence, 3),
            "start_char": self.start_char,
            "end_char": self.end_char,
            "doc_id": self.doc_id,
            "source": self.source,
            "additional_ontologies": self.additional_ontologies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            uri=data["uri"],
            label=data["label"],
            entity_type=data["entity_type"],
            ontology_uri=data.get("ontology_uri"),
            confidence=data.get("confidence", 0.85),
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            doc_id=data.get("doc_id"),
            source=data.get("source", "llm"),
            additional_ontologies=data.get("additional_ontologies", {}),
            metadata=data.get("metadata", {})
        )


# ============================================================
# RELATION MODEL (COMPATIBLE WITH LLM RELATION EXTRACTION)
# ============================================================

@dataclass
class Relation:
    """
    Represents a semantic relation between two entities.
    Uses canonical URIs, supports ontology alignment, and inference flags.
    """

    # --- REQUIRED (non-default) fields FIRST ---
    source: str                            # entity URI (subject)
    relation: str                          # LLM relation label
    target: str                            # entity URI (object)

    # --- OPTIONAL (defaulted) fields AFTER ---
    uri: Optional[str] = None              # canonical URI for the relation triple
    confidence: float = 0.75
    ontology_uri: Optional[str] = None     # mapped ontology class (e.g. rel:causes)

    is_inferred: bool = False
    context: Optional[str] = None          # sentence or doc segment
    doc_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)



    # --- Validation ---
    def __post_init__(self):
        try:
            self.confidence = float(self.confidence)
        except Exception as e:
            raise ValueError("confidence must be a float") from e        
        
        if not self.uri:
            # Allow late binding (pipeline may set uri after constructing the object)
            # Serializer/graph layer must ensure a URI exists before RDF export.
            self.uri = "" 
        if not self.source or not self.target:
            raise ValueError("Relation.source and Relation.target must be non-empty")
        if not self.relation:
            raise ValueError("Relation.relation must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Relation.confidence must be between 0 and 1")

    # --- Serialization ---
    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "source": self.source,
            "relation": self.relation,
            "ontology_uri": self.ontology_uri,
            "target": self.target,
            "confidence": round(self.confidence, 3),
            "is_inferred": self.is_inferred,
            "context": self.context,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        return cls(
            uri=data["uri"],
            source=data["source"],
            relation=data["relation"],
            target=data["target"],
            confidence=data.get("confidence", 0.75),
            ontology_uri=data.get("ontology_uri"),
            is_inferred=data.get("is_inferred", False),
            context=data.get("context"),
            doc_id=data.get("doc_id"),
            metadata=data.get("metadata", {})
        )


# ============================================================
# PROCESSING RESULT
# ============================================================

@dataclass
class ProcessingResult:
    """Result of processing one document/theme/question."""
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "metadata": self.metadata,
            "errors": self.errors
        }


# ============================================================
# BATCH ANNOTATION MODEL
# ============================================================

@dataclass
class AnnotationBatch:
    """Holds all processing results for all document categories."""

    themes: Dict[str, ProcessingResult] = field(default_factory=dict)
    questions: Dict[str, ProcessingResult] = field(default_factory=dict)
    answers: Dict[str, ProcessingResult] = field(default_factory=dict)
    votes: Dict[str, ProcessingResult] = field(default_factory=dict)
    experts: Dict[str, ProcessingResult] = field(default_factory=dict)
    documents: Dict[str, ProcessingResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "themes": {k: v.to_dict() for k, v in self.themes.items()},
            "questions": {k: v.to_dict() for k, v in self.questions.items()},
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
            "votes": {k: v.to_dict() for k, v in self.votes.items()},
            "experts": {k: v.to_dict() for k, v in self.experts.items()},
            "documents": {k: v.to_dict() for k, v in self.documents.items()},
        }

    def get_all_entities(self) -> List[Entity]:
        all_e = []
        for block in [self.themes, self.questions, self.answers, self.votes, self.experts, self.documents]:
            for result in block.values():
                all_e.extend(result.entities)
        return all_e

    def get_all_relations(self) -> List[Relation]:
        all_r = []
        for block in [self.themes, self.questions, self.answers, self.votes, self.experts, self.documents]:
            for result in block.values():
                all_r.extend(result.relations)
        return all_r


# ============================================================
# QUALITY METRICS
# ============================================================

@dataclass
class QualityMetrics:
    """Aggregated quality metrics for the entire knowledge graph."""

    total_entities: int = 0
    unique_entities: int = 0
    total_relations: int = 0
    entity_coverage: float = 0.0
    relation_coverage: float = 0.0
    avg_confidence: float = 0.0
    linking_rate: float = 0.0
    overall_quality_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "total_entities": self.total_entities,
            "unique_entities": self.unique_entities,
            "total_relations": self.total_relations,
            "entity_coverage": round(self.entity_coverage, 3),
            "relation_coverage": round(self.relation_coverage, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "linking_rate": round(self.linking_rate, 3),
            "overall_quality_score": round(self.overall_quality_score, 2),
            "timestamp": self.timestamp
        }
