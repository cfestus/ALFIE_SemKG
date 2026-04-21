"""
metrics/quality_metrics.py
--------------------------
Compute comprehensive quality metrics for the knowledge graph.

Production-grade updates:
- Robustly iterates annotations regardless of structure (dict-of-dicts, dict-of-lists, list)
- Entity totals based on GlobalEntityRegistry (true unique entities)
- Entity ontology alignment computed from GlobalEntityRegistry (no double counting)
- Relation metrics remain doc-based (counts what was emitted per discourse unit)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator

from models import QualityMetrics
from utils.helpers import GlobalEntityRegistry


class QualityMetricsComputer:
    """Compute quality metrics for knowledge graph."""

    def __init__(self, annotations: dict, global_registry: GlobalEntityRegistry):
        self.annotations = annotations or {}
        self.global_registry = global_registry

    # ============================================================
    # INTERNAL: robust iteration over documents in annotations
    # ============================================================
    def _iter_docs(self) -> Iterator[Dict[str, Any]]:
        """
        Yield each document payload in annotations robustly, regardless of shape:
        - {"themes": {...}, "questions": {...}}  (dict-of-dicts)
        - {"themes": [..], "questions": [..]}   (dict-of-lists)
        - {"themes": {...}} where values are dict OR model objects with to_dict()
        - [doc, doc, ...]                       (list)
        """
        ann = self.annotations

        if ann is None:
            return
        if isinstance(ann, list):
            containers: Iterable[Any] = [ann]
        elif isinstance(ann, dict):
            containers = ann.values()
        else:
            return

        for container in containers:
            if container is None:
                continue

            if isinstance(container, dict):
                it = container.values()
            elif isinstance(container, list):
                it = container
            else:
                continue

            for doc in it:
                if doc is None:
                    continue
                if isinstance(doc, dict):
                    yield doc
                elif hasattr(doc, "to_dict"):
                    try:
                        yield doc.to_dict()
                    except Exception:
                        continue

    # ============================================================
    # MASTER METRIC COMPUTATION
    # ============================================================
    def compute_all(self) -> Dict[str, Any]:
        """Compute comprehensive quality metrics."""
        return {
            "entity_metrics": self.compute_entity_metrics(),
            "relation_metrics": self.compute_relation_metrics(),
            "coverage_metrics": self.compute_coverage_metrics(),
            "quality_scores": self.compute_quality_scores(),
            "ontology_alignment": self.compute_ontology_alignment(),
        }

    # ============================================================
    # ENTITY METRICS
    # ============================================================
    def compute_entity_metrics(self) -> dict:
        """Compute entity-level metrics from the consolidated global registry."""
        total_entities = len(self.global_registry.entities)

        entity_types = defaultdict(int)
        confidence_sum = 0.0
        linked_count = 0

        for entity in self.global_registry.entities.values():
            etype = getattr(entity, "entity_type", None) or "unknown"
            if isinstance(etype, str):
                et = etype.strip()
                if ":" in et and " " not in et:
                    _, local = et.split(":", 1)
                    et = local or et
                etype = et.title().replace(" ", "") if et else "unknown"
            entity_types[etype] += 1

            try:
                confidence_sum += float(getattr(entity, "confidence", 0.0) or 0.0)
            except Exception:
                pass

            meta = getattr(entity, "metadata", {}) or {}
            if meta.get("wikidata_uri"):
                linked_count += 1

        return {
            "total_entities": total_entities,
            "unique_entities": total_entities,  # global registry already consolidates
            "entity_type_distribution": dict(entity_types),
            "avg_confidence": round(confidence_sum / total_entities, 3) if total_entities > 0 else 0,
            "linked_entities": linked_count,
            "linking_rate": round(linked_count / total_entities, 3) if total_entities > 0 else 0,
        }

    # ============================================================
    # RELATION METRICS
    # ============================================================
    def compute_relation_metrics(self) -> dict:
        """Compute relation-level metrics from annotations (per-discourse output)."""
        total_relations = 0
        inferred_relations = 0
        relation_types = defaultdict(int)

        for doc_data in self._iter_docs():
            rels = doc_data.get("relations", []) or []
            total_relations += len(rels)

            for r in rels:
                if isinstance(r, dict):
                    if r.get("is_inferred", False):
                        inferred_relations += 1
                    relation_types[r.get("relation", "unknown")] += 1
                else:
                    # model object
                    if getattr(r, "is_inferred", False):
                        inferred_relations += 1
                    relation_types[getattr(r, "relation", "unknown")] += 1

        denom = len(self.global_registry.entities) if self.global_registry and self.global_registry.entities else 0
        return {
            "total_relations": total_relations,
            "inferred_relations": inferred_relations,
            "relation_type_distribution": dict(relation_types),
            "extraction_rate": round(total_relations / denom, 2) if denom > 0 else 0,
        }

    # ============================================================
    # COVERAGE METRICS
    # ============================================================
    def compute_coverage_metrics(self) -> dict:
        """Compute coverage metrics over annotations (documents)."""
        docs_with_entities = 0
        docs_with_relations = 0
        total_docs = 0

        for doc_data in self._iter_docs():
            total_docs += 1

            entities = doc_data.get("entities", []) or []
            relations = doc_data.get("relations", []) or []

            if entities:
                docs_with_entities += 1
            if relations:
                docs_with_relations += 1

        return {
            "total_documents": total_docs,
            "documents_with_entities": docs_with_entities,
            "documents_with_relations": docs_with_relations,
            "entity_coverage": round(docs_with_entities / total_docs, 3) if total_docs > 0 else 0,
            "relation_coverage": round(docs_with_relations / total_docs, 3) if total_docs > 0 else 0,
        }

    # ============================================================
    # ONTOLOGY ALIGNMENT METRICS
    # ============================================================
    def compute_ontology_alignment(self) -> dict:
        """
        Compute ontology alignment metrics.

        Production rules:
        - Entity alignment computed from GlobalEntityRegistry to avoid double counting.
        - Relation alignment computed from annotations (per-discourse relations).
        """
        # -------------------------
        # ENTITY ALIGNMENT (registry-based)
        # -------------------------
        total_entities = len(self.global_registry.entities)
        aligned_entities = 0

        for e in self.global_registry.entities.values():
            meta = getattr(e, "metadata", {}) or {}
            # your pipeline stores expanded IRI into entity.ontology_uri
            # and also puts ontology_classes=[iri] into metadata for metrics/serializer
            if meta.get("ontology_classes"):
                aligned_entities += 1
            else:
                # fallback: accept direct field if present
                if getattr(e, "ontology_uri", None):
                    aligned_entities += 1

        # -------------------------
        # RELATION ALIGNMENT (doc-based)
        # -------------------------
        aligned_relations = 0
        total_relations = 0

        for doc_data in self._iter_docs():
            for rel in (doc_data.get("relations", []) or []):
                total_relations += 1
                if isinstance(rel, dict):
                    meta = rel.get("metadata", {}) or {}
                else:
                    meta = getattr(rel, "metadata", {}) or {}

                if meta.get("ontology_properties"):
                    aligned_relations += 1

        return {
            "total_entities": total_entities,
            "aligned_entities": aligned_entities,
            "entity_alignment_rate": round(aligned_entities / total_entities, 3) if total_entities > 0 else 0,
            "total_relations": total_relations,
            "aligned_relations": aligned_relations,
            "relation_alignment_rate": round(aligned_relations / total_relations, 3) if total_relations > 0 else 0,
        }

    # ============================================================
    # QUALITY SCORES (0–10)
    # ============================================================
    def compute_quality_scores(self) -> dict:
        """Compute overall quality scores (0–10)."""
        entity_m = self.compute_entity_metrics()
        coverage_m = self.compute_coverage_metrics()
        alignment_m = self.compute_ontology_alignment()

        alignment_rate = (alignment_m["entity_alignment_rate"] + alignment_m["relation_alignment_rate"]) / 2

        # Weighted score
        quality_score = (
            entity_m["avg_confidence"] * 0.27
            + entity_m["linking_rate"] * 0.18
            + coverage_m["entity_coverage"] * 0.23
            + coverage_m["relation_coverage"] * 0.23
            + alignment_rate * 0.09
        ) * 10

        return {
            "overall_quality_score": round(quality_score, 2),
            "confidence_score": round(entity_m["avg_confidence"] * 10, 2),
            "linking_score": round(entity_m["linking_rate"] * 10, 2),
            "coverage_score": round((coverage_m["entity_coverage"] + coverage_m["relation_coverage"]) * 5, 2),
            "alignment_score": round(alignment_rate * 10, 2),
        }

    # ============================================================
    # SUMMARY METRICS
    # ============================================================
    def get_summary_metrics(self) -> QualityMetrics:
        """Get summary quality metrics as a QualityMetrics object."""
        all_metrics = self.compute_all()

        entity_m = all_metrics["entity_metrics"]
        relation_m = all_metrics["relation_metrics"]
        coverage_m = all_metrics["coverage_metrics"]
        quality_m = all_metrics["quality_scores"]

        return QualityMetrics(
            total_entities=entity_m["total_entities"],
            unique_entities=entity_m["unique_entities"],
            total_relations=relation_m["total_relations"],
            entity_coverage=coverage_m["entity_coverage"],
            relation_coverage=coverage_m["relation_coverage"],
            avg_confidence=entity_m["avg_confidence"],
            linking_rate=entity_m["linking_rate"],
            overall_quality_score=quality_m["overall_quality_score"],
        )
