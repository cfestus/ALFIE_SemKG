"""
pipeline.py
-----------
Main pipeline orchestrator for the Semantic Knowledge Graph.
Coordinates all modules and handles batch processing.
"""

import gc
import time
import re
import json
import ast
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set, Tuple
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal, Namespace, RDF

from models import Entity, Relation, ProcessingResult, AnnotationBatch
from config import Config
from extractors.entity_extractor import LLMEntityExtractor
from extractors.llm_relation_extractor import LLMRelationExtractor
from ontology.mapper import OntologyMapper
from utils.helpers import (
    GlobalEntityRegistry,
    CoreferenceResolver,
    safe_process_with_retry,
    batch_iterator,
    cleanup_memory,
    save_json,
    load_json,
    make_uri_safe,
    expand_curie,
    make_relation
)
from reasoning.reasoner import MultiHopReasoner, ConflictDetector
from metrics.quality_metrics import QualityMetricsComputer
from utils.graphdb_client import GraphDBClient
from serializers.rdf_serializer import RDFSerializer
from ontology.add_schema import canonicalize_class_local
from utils.semantic_chunker import SemanticChunker
from validation.shacl_validator import SHACLValidator
from utils.entity_normalizer import EntityNormalizer
from utils.wikidata_linker import WikidataLinker
from utils.entity_consolidator import EntityConsolidator
from utils.canonical_linker import CanonicalLinker
from utils.predicate_precision import PredicatePrecisionLayer


ETD = Namespace(Config.ETD_NS)
logger = logging.getLogger(__name__)

def _make_rel_uri(suffix: str) -> str:
    """Helper to create a deterministic relation URI."""
    return str(ETD[f"rel_{suffix}"])


def make_uri(local_id: str) -> str:
    """Ensure all KG IDs become absolute ETD URIs."""
    return str(ETD[local_id])


def normalize_relation_label(label: str) -> str:
    if not label:
        return ""

    # clean whitespace
    label = label.strip()

    # Convert internal CamelCase â†’ spaced ("causedBy" â†’ "caused By")
    label = re.sub(r'(?<!^)(?=[A-Z])', ' ', label)

    # Split on spaces/underscores
    parts = re.split(r"[ _]+", label)
    parts = [p.lower() for p in parts if p]

    if not parts:
        return ""

    return parts[0] + "".join(x.capitalize() for x in parts[1:])

def _mk_evidence_text(*, text: Optional[str], fallback: str, max_len: int = 500) -> str:
    """
    Always return a non-empty evidence string (traceability safe).
    """
    if text is not None:
        s = str(text).strip()
        if s:
            return s[:max_len]
    return fallback[:max_len]


def make_light_relation(
    source: str,
    rel: str,
    target: str,
    confidence: float = 0.60,
    *,
    discourse_id: Optional[str] = None,
    evidence: Optional[str] = None,
):

    """
    Backwards-compatible wrapper.
    Production polish: deterministic URI via make_relation() (no UUIDs).
    """
    # Ensure URIs are absolute
    if not str(source).startswith("http"):
        source = make_uri(str(source))
    if not str(target).startswith("http"):
        target = make_uri(str(target))

    return make_relation(
        source,
        rel,
        target,
        discourse_id=discourse_id,
        extractor="lightweight",
        context_source="light_relation",
        confidence=float(confidence or 0.0),
        evidence=(evidence or ""),
        is_inferred=False,
    )


class EnhancedSemanticPipeline:
    """Main pipeline for processing documents into knowledge graph."""
    
    def __init__(self, config=None):
        self.config = config if config else Config
        if not getattr(self.config, "USE_LLM_EXTRACTION", True) or getattr(self.config, "USE_HYBRID_EXTRACTION", False):
            raise RuntimeError(
                "LLM-only extraction enforced: USE_LLM_EXTRACTION must be true and "
                "USE_HYBRID_EXTRACTION must be false."
            )
        
        print("\nðŸ”§ Initializing pipeline components...")
        self.global_registry = GlobalEntityRegistry()

        # -------------------------------------------------------
        #  Wikidata linker (REAL SPARQL version)
        # -------------------------------------------------------
        # Use configured SPARQL endpoint if available, else default
        endpoint = getattr(
            self.config,
            "WIKIDATA_SPARQL_ENDPOINT",
            "https://query.wikidata.org/sparql",
        )

        # Initialise Wikidata linker only if enabled
        self.wikidata_linker = None
        if getattr(self.config, "USE_WIKIDATA_LINKING", False):
            self.wikidata_linker = WikidataLinker(
                endpoint=endpoint,
                language="en",
                max_results=5,
                min_score=0.65,
                user_agent="ETD-Hub-KG-Pipeline/1.0 (mailto:your-email@example.com)",
                timeout=15,
            )


        # ðŸ”§ Entity normalizer for canonical labels
        self.entity_normalizer = EntityNormalizer(self.global_registry)
        self.canonical_linker = CanonicalLinker(self.global_registry, output_dir=self.config.OUTPUT_DIR)
        self.predicate_precision = PredicatePrecisionLayer()

        # New LLM-based entity extractor
        self.entity_extractor = LLMEntityExtractor(
            ontology_terms=self.config.ONTOLOGY_TERMS,
            model=self.config.LLM_MODEL
        )

        self.llm_relation_extractor = LLMRelationExtractor(
            #api_key=self.config.load_llm_api_key(),
            #model=config.LLM_MODEL,
            model=self.config.LLM_MODEL,
            max_retries=3,
            relation_confidence_threshold=0.35
            #max_retries=self.config.LLM_MAX_RETRIES
        )   
        
        self.ontology_mapper = OntologyMapper()
        self.coref_resolver = CoreferenceResolver()
        self.reasoner = MultiHopReasoner(min_confidence=0.5)
        self.conflict_detector = ConflictDetector()
        self._discourse_chunk_spans: Dict[str, List[Dict[str, int]]] = {}
        self._provenance_diagnostics: Dict[str, Any] = {
            "missing_span_offsets": 0,
            "derived_span_from_entities": 0,
            "span_out_of_range": 0,
            "cross_chunk_evidence": 0,
            "unknown_chunk_index_count": 0,
            "inferred_relations_excluded_from_gate": 0,
            "non_textual_relations_excluded_from_gate": 0,
            "extracted_relations_checked": 0,
            "total_relations_checked": 0,
            "unknown_chunk_index_rate": 0.0,
            "unknown_chunk_index_gate_threshold": float(getattr(self.config, "MAX_UNKNOWN_CHUNK_INDEX_PCT", 0.02)),
            "strict_mode": bool(getattr(self.config, "STRICT_PROVENANCE_GROUNDING", False)),
            "gate_passed": True,
        }
        
        print("âœ“ Pipeline initialized")

    def _compose_origin_key(self, discourse_id: Any, field: Any = None) -> str:
        """
        Deterministic provenance identity:
        disambiguate reused discourse_id values across sections/types.
        """
        did = str(discourse_id).strip() if discourse_id is not None else ""
        fld = str(field).strip() if field is not None else ""
        if did and fld:
            return f"{fld}|{did}"
        return did

    def _ensure_entity_ontology_grounding(self, entity: Entity) -> None:
        """Ensure a single ontology_uri is always present for an entity."""
        if not isinstance(entity, Entity):
            return

        metadata = getattr(entity, "metadata", None) or {}
        entity.metadata = metadata

        # Canonicalize ETD class local-name early so every pipeline branch converges.
        raw_type = str(getattr(entity, "entity_type", "") or "").strip()
        if raw_type:
            canonical_type = canonicalize_class_local(raw_type)
            if canonical_type != raw_type:
                metadata.setdefault("external_entity_type", raw_type)
                entity.entity_type = canonical_type
                if hasattr(entity, "type"):
                    entity.type = canonical_type

        onto_uri = getattr(entity, "ontology_uri", None) or metadata.get("ontology_uri")
        if isinstance(onto_uri, str):
            onto_uri = onto_uri.strip()
        else:
            onto_uri = None

        if not onto_uri and self.config.USE_ONTOLOGY_MAPPING:
            mapped = self.ontology_mapper.map_entity(entity)
            chosen = None
            if isinstance(mapped, str) and mapped.strip():
                chosen = mapped.strip()
            elif isinstance(mapped, list):
                for candidate in mapped:
                    if isinstance(candidate, str) and candidate.strip():
                        chosen = candidate.strip()
                        break
            if chosen:
                onto_uri = expand_curie(chosen)
                metadata.setdefault("ontology_mapping_source", "entity_type_mapping")

        if not onto_uri:
            onto_uri = expand_curie("prov:Entity")
            metadata.setdefault("ontology_mapping_source", "fallback:prov:Entity")

        entity.ontology_uri = onto_uri
        metadata["ontology_uri"] = onto_uri
        metadata["ontology_classes"] = [onto_uri]

    def _coerce_nonneg_int(self, value: Any) -> Optional[int]:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        return iv if iv >= 0 else None

    def _coerce_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_entity_mention(
        self,
        mention: Any,
        discourse_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(mention, dict):
            return None
        did_raw = discourse_id if discourse_id is not None else mention.get("discourse_id")
        did = str(did_raw).strip() if did_raw is not None else ""
        if not did:
            return None
        start = self._coerce_nonneg_int(mention.get("start_char"))
        end = self._coerce_nonneg_int(mention.get("end_char"))
        if start is None or end is None or end <= start:
            return None
        out: Dict[str, Any] = {
            "discourse_id": did,
            "start_char": int(start),
            "end_char": int(end),
        }
        ci = self._coerce_nonneg_int(mention.get("chunk_index"))
        if ci is not None:
            out["chunk_index"] = int(ci)
        return out

    def _append_entity_mention(
        self,
        entity: Entity,
        *,
        discourse_id: Any,
        chunk_index: Any,
        start_char: Any,
        end_char: Any,
    ) -> None:
        if not isinstance(getattr(entity, "metadata", None), dict):
            entity.metadata = {}
        did = str(discourse_id).strip() if discourse_id is not None else ""
        mention = self._normalize_entity_mention(
            {
                "discourse_id": did,
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
            },
            discourse_id=did,
        )
        if mention is None:
            return
        mentions = entity.metadata.get("mentions")
        if not isinstance(mentions, list):
            mentions = []
        mentions.append(mention)
        dedup: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
        for item in mentions:
            norm = self._normalize_entity_mention(item)
            if norm is None:
                continue
            key = (
                str(norm.get("discourse_id", "")),
                int(norm.get("chunk_index")) if norm.get("chunk_index") is not None else -1,
                int(norm.get("start_char", -1)),
                int(norm.get("end_char", -1)),
            )
            dedup[key] = norm
        entity.metadata["mentions"] = [dedup[k] for k in sorted(dedup.keys())]

    def _build_chunk_spans(
        self,
        text: str,
        chunks: List[str],
        discourse_id: Any,
        origin_key: Optional[str] = None,
    ) -> List[Dict[str, int]]:
        did = str(discourse_id or "").strip()
        spans: List[Dict[str, int]] = []
        native_chunks = SemanticChunker.chunk_with_offsets(str(text or ""))
        for idx, item in enumerate(native_chunks):
            start = self._coerce_nonneg_int(item.get("start_offset"))
            end = self._coerce_nonneg_int(item.get("end_offset"))
            if start is None or end is None or end <= start:
                continue
            spans.append(
                {
                    "discourse_id": did,  # type: ignore[dict-item]
                    "chunk_index": int(idx),
                    "start_offset": int(start),
                    "end_offset": int(end),
                }
            )
        if did:
            self._discourse_chunk_spans[did] = spans
        okey = str(origin_key or "").strip()
        if okey and okey != did:
            self._discourse_chunk_spans[okey] = spans
        return spans

    def _get_chunk_spans_for_discourse(
        self,
        discourse_id: Any,
        origin_key: Optional[str] = None,
    ) -> List[Dict[str, int]]:
        did = str(discourse_id or "").strip()
        okey = str(origin_key or "").strip()
        if not did and not okey:
            return []
        spans = []
        if okey:
            spans = self._discourse_chunk_spans.get(okey, [])
        if not spans and did:
            spans = self._discourse_chunk_spans.get(did, [])
        return [dict(s) for s in spans]

    def _extract_relation_span_from_metadata(self, meta: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        s = (
            self._coerce_nonneg_int(meta.get("char_start"))
            if meta.get("char_start") is not None
            else self._coerce_nonneg_int(meta.get("start_char"))
        )
        if s is None:
            s = self._coerce_nonneg_int(meta.get("start_offset"))
        e = (
            self._coerce_nonneg_int(meta.get("char_end"))
            if meta.get("char_end") is not None
            else self._coerce_nonneg_int(meta.get("end_char"))
        )
        if e is None:
            e = self._coerce_nonneg_int(meta.get("end_offset"))
        if s is None or e is None:
            return None
        if e <= s:
            e = s + 1
        return int(s), int(e)

    def _derive_relation_span_from_entities(
        self,
        relation: Relation,
        uri_to_entity: Dict[str, Entity],
    ) -> Optional[Tuple[int, int]]:
        def _entity_span(ent: Optional[Entity]) -> Optional[Tuple[int, int]]:
            if ent is None:
                return None
            meta = getattr(ent, "metadata", None) or {}
            s = self._coerce_nonneg_int(meta.get("start_char"))
            e = self._coerce_nonneg_int(meta.get("end_char"))
            if s is None and getattr(ent, "start_char", None) is not None:
                s = self._coerce_nonneg_int(getattr(ent, "start_char", None))
            if e is None and getattr(ent, "end_char", None) is not None:
                e = self._coerce_nonneg_int(getattr(ent, "end_char", None))
            if s is None or e is None:
                return None
            if e <= s:
                e = s + 1
            return int(s), int(e)

        src_ent = uri_to_entity.get(str(getattr(relation, "source", "") or ""))
        tgt_ent = uri_to_entity.get(str(getattr(relation, "target", "") or ""))
        src_span = _entity_span(src_ent)
        tgt_span = _entity_span(tgt_ent)
        if src_span and tgt_span:
            return min(src_span[0], tgt_span[0]), max(src_span[1], tgt_span[1])
        if src_span:
            return src_span
        if tgt_span:
            return tgt_span
        return None

    def _entity_span_from_uri(self, uri: Any, discourse_id: Any = None) -> Optional[Tuple[int, int]]:
        ent_uri = str(uri or "").strip()
        if not ent_uri:
            return None
        ent = None
        registry = getattr(self, "global_registry", None)
        if registry is None:
            return None
        store = getattr(registry, "entities", None)
        if isinstance(store, dict):
            ent = store.get(ent_uri)
        if ent is None:
            get_all = getattr(registry, "get_all_entities", None)
            if callable(get_all):
                for cand in get_all():
                    if str(getattr(cand, "uri", "") or "").strip() == ent_uri:
                        ent = cand
                        break
        if ent is None:
            return None

        meta = getattr(ent, "metadata", None) or {}
        did = str(discourse_id or "").strip()
        mentions = meta.get("mentions")
        if isinstance(mentions, list):
            scoped: List[Tuple[int, int]] = []
            for item in mentions:
                norm = self._normalize_entity_mention(item)
                if norm is None:
                    continue
                if did and str(norm.get("discourse_id", "")).strip() != did:
                    continue
                scoped.append((int(norm["start_char"]), int(norm["end_char"])))
            if scoped:
                min_start = min(x[0] for x in scoped)
                max_end = max(x[1] for x in scoped)
                if max_end <= min_start:
                    max_end = min_start + 1
                return int(min_start), int(max_end)

        s = self._coerce_nonneg_int(meta.get("start_char"))
        e = self._coerce_nonneg_int(meta.get("end_char"))
        if s is None and getattr(ent, "start_char", None) is not None:
            s = self._coerce_nonneg_int(getattr(ent, "start_char", None))
        if e is None and getattr(ent, "end_char", None) is not None:
            e = self._coerce_nonneg_int(getattr(ent, "end_char", None))
        if s is None or e is None:
            return None
        if e <= s:
            e = s + 1
        return int(s), int(e)

    def _map_span_to_chunk_segments(
        self,
        *,
        discourse_id: str,
        span_start: int,
        span_end: int,
        chunk_spans: List[Dict[str, int]],
    ) -> List[Dict[str, int]]:
        did = str(discourse_id or "").strip()
        segments: List[Dict[str, int]] = []
        for ch in sorted(chunk_spans, key=lambda x: int(x.get("chunk_index", 0))):
            ch_start = self._coerce_nonneg_int(ch.get("start_offset"))
            ch_end = self._coerce_nonneg_int(ch.get("end_offset"))
            ch_idx = self._coerce_nonneg_int(ch.get("chunk_index"))
            if ch_start is None or ch_end is None or ch_idx is None:
                continue
            if ch_end <= span_start or ch_start >= span_end:
                continue
            seg_start = max(span_start, ch_start)
            seg_end = min(span_end, ch_end)
            if seg_end <= seg_start:
                continue
            segments.append(
                {
                    "discourse_id": did,  # type: ignore[dict-item]
                    "chunk_index": int(ch_idx),
                    "char_start": int(seg_start),
                    "char_end": int(seg_end),
                    "start_offset": int(seg_start),
                    "end_offset": int(seg_end),
                }
            )
        return segments

    def _ground_relation_metadata(
        self,
        meta: Dict[str, Any],
        *,
        discourse_id: Any,
        source_uri: Any = None,
        target_uri: Any = None,
        chunk_spans: Optional[List[Dict[str, int]]] = None,
        counters: Optional[Dict[str, int]] = None,
    ) -> None:
        did = str(discourse_id or meta.get("discourse_id") or meta.get("doc_id") or "").strip()
        if did:
            meta["discourse_id"] = did
        origin_key = str(meta.get("origin_key") or self._compose_origin_key(did, meta.get("field"))).strip()
        if origin_key:
            meta["origin_key"] = origin_key
        spans = (
            chunk_spans
            if chunk_spans is not None
            else self._get_chunk_spans_for_discourse(did, origin_key=origin_key)
        )
        span = self._extract_relation_span_from_metadata(meta)
        method = "none"
        status = "ungrounded"
        chunk_index: Optional[int] = None
        evidence_spans: List[Dict[str, int]] = []
        existing_chunk_index = self._coerce_nonneg_int(meta.get("chunk_index"))

        if span is None:
            src_span = self._entity_span_from_uri(source_uri, did)
            tgt_span = self._entity_span_from_uri(target_uri, did)
            if src_span is not None and tgt_span is not None:
                derived_start = min(int(src_span[0]), int(tgt_span[0]))
                derived_end = max(int(src_span[1]), int(tgt_span[1]))
                if derived_end <= derived_start:
                    derived_end = derived_start + 1
                meta["char_start"] = int(derived_start)
                meta["char_end"] = int(derived_end)
                meta["start_offset"] = int(derived_start)
                meta["end_offset"] = int(derived_end)
                span = (int(derived_start), int(derived_end))
                if counters is not None:
                    counters["derived_span_from_entities"] = int(counters.get("derived_span_from_entities", 0)) + 1

        if span is None:
            if counters is not None:
                counters["missing_span_offsets"] = int(counters.get("missing_span_offsets", 0)) + 1
            meta.setdefault("char_start", -1)
            meta.setdefault("char_end", -1)
            meta.setdefault("start_offset", -1)
            meta.setdefault("end_offset", -1)
            if existing_chunk_index is not None:
                chunk_index = int(existing_chunk_index)
                status = "grounded"
                method = str(meta.get("groundingMethod") or "none")
        else:
            span_start, span_end = span
            evidence_spans = self._map_span_to_chunk_segments(
                discourse_id=did,
                span_start=span_start,
                span_end=span_end,
                chunk_spans=spans,
            )
            if not evidence_spans and counters is not None:
                counters["span_out_of_range"] = int(counters.get("span_out_of_range", 0)) + 1

            if len(evidence_spans) == 1:
                chunk_index = int(evidence_spans[0]["chunk_index"])
                method = "span_map"
                status = "grounded"
                seg = evidence_spans[0]
                meta["char_start"] = int(seg["char_start"])
                meta["char_end"] = int(seg["char_end"])
                meta["start_offset"] = int(seg["start_offset"])
                meta["end_offset"] = int(seg["end_offset"])
            elif len(evidence_spans) > 1:
                method = "span_map"
                status = "grounded"
                meta["evidence_spans"] = evidence_spans
                if counters is not None:
                    counters["cross_chunk_evidence"] = int(counters.get("cross_chunk_evidence", 0)) + 1

        if chunk_index is None and len(spans) == 1:
            only_ci = self._coerce_nonneg_int(spans[0].get("chunk_index"))
            if only_ci is not None:
                chunk_index = int(only_ci)
                method = "single_chunk_discourse"
                status = "grounded"

        if chunk_index is None:
            meta["chunk_index"] = -1
            meta["chunk_index_unknown"] = True
            meta["groundingStatus"] = status
            meta["groundingMethod"] = method
            meta.pop("chunk_id", None)
            source_doc_id = meta.get("source_doc_id") or did
            if source_doc_id:
                meta["source_doc_id"] = str(source_doc_id)
            if counters is not None:
                # Count truly ungrounded unknowns; exclude grounded cross-chunk span_map.
                if not (status == "grounded" and method == "span_map" and bool(meta.get("evidence_spans"))):
                    counters["unknown_chunk_index_count"] = int(counters.get("unknown_chunk_index_count", 0)) + 1
        else:
            meta["chunk_index"] = int(chunk_index)
            meta["chunk_index_unknown"] = False
            meta["groundingStatus"] = status
            meta["groundingMethod"] = method
            source_doc_id = meta.get("source_doc_id") or did
            if source_doc_id:
                source_doc_id = str(source_doc_id)
                meta["source_doc_id"] = source_doc_id
                meta["chunk_id"] = str(meta.get("chunk_id") or f"{source_doc_id}:chunk:{chunk_index}")
            if "evidence_spans" in meta and len(evidence_spans) <= 1:
                meta.pop("evidence_spans", None)

    def _ensure_relation_chunk_provenance(
        self,
        relation: Relation,
        fallback_discourse_id: Any = None,
        chunk_spans: Optional[List[Dict[str, int]]] = None,
        counters: Optional[Dict[str, int]] = None,
    ) -> None:
        meta = getattr(relation, "metadata", None) or {}
        self._ground_relation_metadata(
            meta,
            discourse_id=fallback_discourse_id,
            source_uri=getattr(relation, "source", None),
            target_uri=getattr(relation, "target", None),
            chunk_spans=chunk_spans,
            counters=counters,
        )
        relation.metadata = meta

    def fill_missing_relation_provenance(self, annotations_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic pre-serialization provenance fill pass.
        Only fills chunk provenance when:
          a) span->chunk mapping succeeds
          b) discourse has exactly one chunk
        No discourse-majority or first-chunk guessing is performed.
        """
        sections = ("themes", "questions", "answers", "votes", "experts", "documents")
        counters: Dict[str, int] = {
            "missing_span_offsets": 0,
            "derived_span_from_entities": 0,
            "span_out_of_range": 0,
            "cross_chunk_evidence": 0,
            "unknown_chunk_index_count": 0,
            "inferred_relations_excluded_from_gate": 0,
            "non_textual_relations_excluded_from_gate": 0,
            "extracted_relations_checked": 0,
            "total_relations_checked": 0,
            "total_relations": 0,
            "filled_from_span_map": 0,
            "filled_from_single_chunk_discourse": 0,
        }
        unknown_by_extractor: Dict[str, int] = {}
        unknown_by_relation: Dict[str, int] = {}
        unknown_by_reason: Dict[str, int] = {}

        def _inc_bucket(bucket: Dict[str, int], key: str) -> None:
            bucket[key] = int(bucket.get(key, 0)) + 1

        def _as_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))
            if v is None:
                return False
            s = str(v).strip().lower()
            return s in {"true", "1", "yes", "y"}

        def _fill_one(rel_dict: Dict[str, Any]) -> None:
            counters["total_relations"] = int(counters.get("total_relations", 0)) + 1
            meta = rel_dict.get("metadata", {}) or {}
            rel_dict["metadata"] = meta
            did = meta.get("discourse_id") or meta.get("doc_id")
            origin_key = str(meta.get("origin_key") or self._compose_origin_key(did, meta.get("field"))).strip()
            if origin_key:
                meta["origin_key"] = origin_key
            extractor = str(meta.get("extractor") or "").strip().lower()
            non_textual_extractor = extractor in {"structural", "vote", "lightweight"}
            relation_type = str(rel_dict.get("relation") or meta.get("relation") or "").strip() or "unknown"
            source_uri = str(rel_dict.get("source") or rel_dict.get("source_id") or "").strip()
            target_uri = str(rel_dict.get("target") or rel_dict.get("target_id") or "").strip()
            chunk_spans = self._get_chunk_spans_for_discourse(did, origin_key=origin_key)
            chunk_spans_count = len(chunk_spans)
            relation_span_before = self._extract_relation_span_from_metadata(meta)
            source_has_span = self._entity_span_from_uri(source_uri, did) is not None
            target_has_span = self._entity_span_from_uri(target_uri, did) is not None
            excluded_from_gate = (
                extractor == "reasoner"
                or non_textual_extractor
                or _as_bool(meta.get("isInferred"))
                or _as_bool(meta.get("is_inferred"))
                or _as_bool(rel_dict.get("isInferred"))
                or _as_bool(rel_dict.get("is_inferred"))
            )
            unknown_before = int(counters.get("unknown_chunk_index_count", 0))
            self._ground_relation_metadata(
                meta,
                discourse_id=did,
                source_uri=source_uri,
                target_uri=target_uri,
                chunk_spans=chunk_spans,
                counters=counters,
            )
            unknown_after = int(counters.get("unknown_chunk_index_count", 0))
            if excluded_from_gate:
                if non_textual_extractor:
                    counters["non_textual_relations_excluded_from_gate"] = int(
                        counters.get("non_textual_relations_excluded_from_gate", 0)
                    ) + 1
                else:
                    counters["inferred_relations_excluded_from_gate"] = int(
                        counters.get("inferred_relations_excluded_from_gate", 0)
                    ) + 1
                if unknown_after > unknown_before:
                    counters["unknown_chunk_index_count"] = max(0, unknown_before)
            else:
                counters["extracted_relations_checked"] = int(counters.get("extracted_relations_checked", 0)) + 1
                counters["total_relations_checked"] = int(counters.get("total_relations_checked", 0)) + 1
                was_counted_unknown = unknown_after > unknown_before
                if was_counted_unknown:
                    evidence_spans = meta.get("evidence_spans")
                    evidence_spans_len = len(evidence_spans) if isinstance(evidence_spans, list) else 0
                    method = str(meta.get("groundingMethod") or "none")
                    if relation_span_before is None and (not source_has_span) and (not target_has_span):
                        failure_reason = "A_missing_relation_and_entity_spans"
                    elif method == "span_map" and evidence_spans_len > 0:
                        failure_reason = "C_cross_chunk_span_map_unknown_counted"
                    elif relation_span_before is not None and chunk_spans_count > 0 and evidence_spans_len == 0:
                        failure_reason = "B_span_out_of_range_or_chunk_map_miss"
                    else:
                        failure_reason = "D_other"
                    _inc_bucket(unknown_by_extractor, extractor or "unknown")
                    _inc_bucket(unknown_by_relation, relation_type)
                    _inc_bucket(unknown_by_reason, failure_reason)

            if str(meta.get("groundingMethod")) == "span_map" and self._coerce_nonneg_int(meta.get("chunk_index")) is not None:
                counters["filled_from_span_map"] = int(counters.get("filled_from_span_map", 0)) + 1
            if str(meta.get("groundingMethod")) == "single_chunk_discourse" and self._coerce_nonneg_int(meta.get("chunk_index")) is not None:
                counters["filled_from_single_chunk_discourse"] = int(
                    counters.get("filled_from_single_chunk_discourse", 0)
                ) + 1

        for section in sections:
            block = annotations_dict.get(section, {})
            if not isinstance(block, dict):
                continue
            for _, payload in block.items():
                if not isinstance(payload, dict):
                    continue
                for rel in payload.get("relations", []):
                    if isinstance(rel, dict):
                        _fill_one(rel)

        inferred_list = (annotations_dict.get("metadata", {}) or {}).get("inferred_relations", [])
        if isinstance(inferred_list, list):
            for rel in inferred_list:
                if isinstance(rel, dict):
                    _fill_one(rel)

        total_relations = int(counters.get("total_relations", 0))
        total_relations_checked = int(counters.get("total_relations_checked", 0))
        unknown = int(counters.get("unknown_chunk_index_count", 0))
        unknown_rate = (float(unknown) / float(total_relations_checked)) if total_relations_checked > 0 else 0.0
        unknown_by_extractor_pct = {
            key: {
                "count": int(val),
                "pct_of_checked": round((float(val) / float(total_relations_checked)) * 100.0, 3)
                if total_relations_checked > 0
                else 0.0,
            }
            for key, val in sorted(unknown_by_extractor.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        }
        unknown_by_relation_pct = {
            key: {
                "count": int(val),
                "pct_of_checked": round((float(val) / float(total_relations_checked)) * 100.0, 3)
                if total_relations_checked > 0
                else 0.0,
            }
            for key, val in sorted(unknown_by_relation.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        }
        unknown_by_reason_pct = {
            key: {
                "count": int(val),
                "pct_of_checked": round((float(val) / float(total_relations_checked)) * 100.0, 3)
                if total_relations_checked > 0
                else 0.0,
            }
            for key, val in sorted(unknown_by_reason.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        }
        strict_mode = bool(getattr(self.config, "STRICT_PROVENANCE_GROUNDING", False))
        max_unknown = 0.0 if strict_mode else float(getattr(self.config, "MAX_UNKNOWN_CHUNK_INDEX_PCT", 0.02))
        gate_passed = unknown_rate <= max_unknown
        result = {
            "total_relations": total_relations,
            "total_relations_checked": total_relations_checked,
            "extracted_relations_checked": int(counters.get("extracted_relations_checked", 0)),
            "inferred_relations_excluded_from_gate": int(counters.get("inferred_relations_excluded_from_gate", 0)),
            "non_textual_relations_excluded_from_gate": int(
                counters.get("non_textual_relations_excluded_from_gate", 0)
            ),
            "filled_from_span_map": int(counters.get("filled_from_span_map", 0)),
            "filled_from_single_chunk_discourse": int(counters.get("filled_from_single_chunk_discourse", 0)),
            "missing_span_offsets": int(counters.get("missing_span_offsets", 0)),
            "derived_span_from_entities": int(counters.get("derived_span_from_entities", 0)),
            "span_out_of_range": int(counters.get("span_out_of_range", 0)),
            "cross_chunk_evidence": int(counters.get("cross_chunk_evidence", 0)),
            "unknown_chunk_index_count": unknown,
            "unknown_chunk_index_rate": round(unknown_rate, 6),
            "unknown_non_excluded_by_extractor": unknown_by_extractor_pct,
            "unknown_non_excluded_by_relation": unknown_by_relation_pct,
            "unknown_non_excluded_by_reason": unknown_by_reason_pct,
            "unknown_chunk_index_gate_threshold": max_unknown,
            "strict_mode": strict_mode,
            "gate_passed": gate_passed,
        }
        self._provenance_diagnostics = dict(result)
        print("\n[Provenance grounding breakdown (NON-excluded relations)]")
        print(f"  total_relations_checked: {total_relations_checked}")
        print(f"  unknown_chunk_index_count: {unknown}")
        print(f"  unknown_chunk_index_rate: {unknown_rate:.4%}")
        print(f"  missing_span_offsets: {int(counters.get('missing_span_offsets', 0))}")
        print(f"  derived_span_from_entities: {int(counters.get('derived_span_from_entities', 0))}")
        print(f"  span_out_of_range: {int(counters.get('span_out_of_range', 0))}")
        print(f"  cross_chunk_evidence: {int(counters.get('cross_chunk_evidence', 0))}")
        print("  unknown_by_extractor:", json.dumps(unknown_by_extractor_pct, ensure_ascii=False))
        print("  unknown_by_relation:", json.dumps(unknown_by_relation_pct, ensure_ascii=False))
        print("  unknown_by_reason:", json.dumps(unknown_by_reason_pct, ensure_ascii=False))
        if not gate_passed:
            raise RuntimeError(
                "Provenance grounding gate failed: "
                f"unknown_chunk_index_rate={unknown_rate:.4f} exceeds threshold={max_unknown:.4f}"
            )
        logger.info("Pre-serialization provenance fill: %s", result)
        return result
    
    # ---------------------------------------------------------
    # Entity processing
    # ---------------------------------------------------------
    def _process_entities_batch(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> List[List[Entity]]:
        """
        Process entities for multiple texts in batch.
        Uses:
            - coreference resolution
            - semantic chunking
            - LLM entity extraction per chunk
            - registry deduplication
            - ontology enrichment
            - metadata propagation
        """
        all_entities: List[List[Entity]] = []
    
        for text, metadata in zip(texts, metadata_list):

            # -------------------------------------------------------
            # 1. Resolve coreference
            # -------------------------------------------------------
            resolved_text = self.coref_resolver.resolve(text)

            # -------------------------------------------------------
            # 2. Semantic chunking
            # -------------------------------------------------------
            chunks = SemanticChunker.chunk(resolved_text)
            discourse_id = metadata.get("discourse_id")
            origin_key = metadata.get("origin_key") or self._compose_origin_key(discourse_id, metadata.get("field"))
            chunk_spans = self._build_chunk_spans(
                resolved_text,
                chunks,
                discourse_id,
                origin_key=str(origin_key) if origin_key is not None else None,
            )
            chunk_span_by_idx: Dict[int, Dict[str, int]] = {
                int(span["chunk_index"]): span for span in chunk_spans if "chunk_index" in span
            }

            # -------------------------------------------------------
            # 3. Extract entities FROM CHUNKS (only ONCE)
            # -------------------------------------------------------
            entities: List[Entity] = []

            doc_id = f"{metadata.get('discourse_id')}|{metadata.get('field', 'unknown')}"
            discourse_id_raw = metadata.get("discourse_id")
            discourse_id = str(discourse_id_raw).strip() if discourse_id_raw is not None else None
            
            for idx, chunk in enumerate(chunks):

            # ðŸ”’ SAFETY: Skip tiny chunks that break GPT JSON extraction
                if len(chunk.split()) < 10:
                    continue

                extracted = self.entity_extractor.extract(chunk, doc_id=doc_id)
                
                if extracted:
                    for e in extracted:
                        if not isinstance(getattr(e, "metadata", None), dict):
                            e.metadata = {}

                        e.metadata["chunk_index"] = idx
                        e.metadata["chunk_total"] = len(chunks)
                        if discourse_id:
                            e.metadata["discourse_id"] = discourse_id
                        if origin_key:
                            e.metadata["origin_key"] = str(origin_key)

                        # Prefer extractor-provided span values, then model fields.
                        local_start = self._coerce_nonneg_int(e.metadata.get("start_char"))
                        local_end = self._coerce_nonneg_int(e.metadata.get("end_char"))
                        if local_start is None and getattr(e, "start_char", None) is not None:
                            local_start = self._coerce_nonneg_int(getattr(e, "start_char", None))
                        if local_end is None and getattr(e, "end_char", None) is not None:
                            local_end = self._coerce_nonneg_int(getattr(e, "end_char", None))

                        span = chunk_span_by_idx.get(idx)
                        if span is not None:
                            e.metadata["start_offset"] = int(span.get("start_offset", 0))
                            e.metadata["end_offset"] = int(span.get("end_offset", 0))

                            # Offset local chunk spans by deterministic chunk start.
                            chunk_start = self._coerce_nonneg_int(span.get("start_offset"))
                            abs_start = None
                            abs_end = None
                            if local_start is not None and chunk_start is not None:
                                abs_start = int(local_start + chunk_start)
                            if local_end is not None and chunk_start is not None:
                                abs_end = int(local_end + chunk_start)
                            if abs_start is not None:
                                e.metadata["start_char"] = abs_start
                                e.start_char = abs_start
                            else:
                                e.metadata.setdefault("start_char", -1)
                            if abs_end is not None:
                                e.metadata["end_char"] = abs_end
                                e.end_char = abs_end
                            else:
                                e.metadata.setdefault("end_char", -1)

                            if abs_start is not None and abs_end is not None:
                                self._append_entity_mention(
                                    e,
                                    discourse_id=discourse_id,
                                    chunk_index=idx,
                                    start_char=abs_start,
                                    end_char=abs_end,
                                )
                        else:
                            e.metadata.setdefault("start_char", -1)
                            e.metadata.setdefault("end_char", -1)

                    entities.extend(extracted) 

            # -------------------------------------------------------
            # 4. Normalize + merge with global registry
            # -------------------------------------------------------
            normalized_entities: List[Entity] = []

            for e in entities:
              
                # Skip malformed outputs from LLM extractor
                if not isinstance(e, Entity):
                    continue

                # Backwards compatibility for any older code paths
                if not hasattr(e, "name"):
                    e.name = e.label
                if not hasattr(e, "type"):
                    e.type = e.entity_type

                # ðŸ”¹ NEW: normalize label before registry & enrichment
                #e.label = self.entity_normalizer.normalize_label(e.label)
                #e.name = e.label  # keep name in sync with label
                
                # Normalize entity before registry + enrichment
                self.entity_normalizer.normalize_entity(e)
                # Deterministic canonical URI (type-aware + alias-normalized)
                # Use any ontology hint already present in metadata for stability.
                ontology_hint = getattr(e, "ontology_uri", None) or (e.metadata or {}).get("ontology_uri")
                e.uri = self.entity_normalizer.canonical_uri(
                    label=e.label,
                    entity_type=e.entity_type,
                    ontology_uri=ontology_hint,
                )

                # Deduplicate across chunks & documents
                e = self.global_registry.get_or_create(e)

                # Attach top-level metadata (discourse_id, theme_id, etc.)
                for k, v in metadata.items():
                    if v is not None:
                      e.metadata[k] = v


                # -------------------------------------------------------
                # Ontology mapping (production: ONE expanded IRI in ontology_uri)
                # -------------------------------------------------------
                mapped = None
                if self.config.USE_ONTOLOGY_MAPPING:
                    mapped = self.ontology_mapper.map_entity(e)

                    chosen = None
                    if isinstance(mapped, str) and mapped.strip():
                        chosen = mapped.strip()
                    elif isinstance(mapped, list):
                        for m in mapped:
                            if isinstance(m, str) and m.strip():
                                chosen = m.strip()
                                break

                    # store full mapping for traceability
                    if isinstance(mapped, list) and mapped:
                        e.metadata["ontology_classes_curie"] = [str(x).strip() for x in mapped if str(x).strip()]
                    elif isinstance(mapped, str) and mapped.strip():
                        e.metadata["ontology_classes_curie"] = [mapped.strip()]

                    e.ontology_uri = expand_curie(chosen) if chosen else None

                # --- final guard: ontology_uri must NEVER be list/stringified list ---
                ou = getattr(e, "ontology_uri", None)
                if isinstance(ou, list) and ou:
                    ou = str(ou[0])
                if isinstance(ou, str) and ou.strip().startswith("[") and ou.strip().endswith("]"):
                    try:
                        parsed = ast.literal_eval(ou)
                        if isinstance(parsed, list) and parsed:
                            ou = str(parsed[0])
                    except Exception:
                        pass

                e.ontology_uri = expand_curie(ou.strip()) if isinstance(ou, str) and ou.strip() else None

                # Add ontology alignment to metadata for metrics/serializer
                if e.ontology_uri:
                    e.metadata["ontology_classes"] = [e.ontology_uri]
                else:
                    e.metadata.pop("ontology_classes", None)

                # Coverage guard: always keep one ontology grounding.
                self._ensure_entity_ontology_grounding(e)

                # If ontology mapping changed canonical identity, remint URI deterministically
                # and keep registry references consistent through redirect.
                reminted_uri = self.entity_normalizer.canonical_uri(
                    label=e.label,
                    entity_type=e.entity_type,
                    ontology_uri=e.ontology_uri or (e.metadata or {}).get("ontology_uri"),
                )
                if reminted_uri != e.uri:
                    old_uri = e.uri
                    e.uri = reminted_uri
                    e = self.global_registry.get_or_create(e)
                    if old_uri != reminted_uri and hasattr(self.global_registry, "replace_entity"):
                        self.global_registry.replace_entity(old_uri, reminted_uri)

                # Wikidata linking
                if getattr(self.config, "USE_WIKIDATA_LINKING", False) and self.wikidata_linker:
                    wd = self.wikidata_linker.link(e.label)
                    if wd and "qid" in wd:
                        qid = wd["qid"]                
                        # 1) For quality_metrics (linking_rate)
                        e.metadata["wikidata_uri"] = f"http://www.wikidata.org/entity/{qid}"

                        # 2) For RDF serializer (_add_wikidata_metadata expects 'wikidata'* keys)
                        e.metadata["wikidata"] = wd
                        e.metadata["wikidataLabel"] = wd.get("label")
                        e.metadata["wikidataDescription"] = wd.get("description")
                        e.metadata["wikidataScore"] = wd.get("score")

                        # 3) Optional: keep your original debug blob if you like
                        e.metadata["wikidata_match"] = wd

                normalized_entities.append(e)

            # after the loop
            all_entities.append(normalized_entities)

        return all_entities

    # ---------------------------------------------------------
    # Relation processing
    # ---------------------------------------------------------

    def _process_relations_batch(
        self,
        texts: List[str],
        entities_list: List[List[Entity]],
        discourse_ids: List[str],
        provenance_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[List[Relation]]:
        """Process relations for multiple texts in batch.

        Design choice:
        - Text is stored as a literal on the DiscourseUnit via metadata["text"]
        - No Text node, no hasText triple
        - DiscourseUnit URIs are stable (derived from real discourse_ids)
        """
        resolved_texts = [self.coref_resolver.resolve(text) for text in texts]

        print("ðŸ§  Using LLM-based relation extraction...")
        all_relations = self.llm_relation_extractor.extract_batch(resolved_texts, entities_list, discourse_ids)

        # ------------------------------------------------------------
        # Ensure a DiscourseUnit node exists for each text (typed and queryable),
        # store raw text as a literal via metadata["text"],
        # and standardise provenance key to "discourse_id".
        # ------------------------------------------------------------
        for idx, relations in enumerate(all_relations):
            text = texts[idx]
            resolved_text = resolved_texts[idx] if idx < len(resolved_texts) else text
            uri_to_entity: Dict[str, Entity] = {}
            chunk_label_groups: Dict[int, Dict[str, List[Entity]]] = {}

            def _norm_label(value: Any) -> str:
                s = str(value or "").strip().casefold()
                s = re.sub(r"\s+", " ", s)
                return s

            for ent in entities_list[idx]:
                u = getattr(ent, "uri", None)
                if u:
                    uri_to_entity[u] = ent
                ci = (getattr(ent, "metadata", {}) or {}).get("chunk_index")
                if u and isinstance(ci, int):
                    lbl_key = _norm_label(getattr(ent, "label", None))
                    if lbl_key:
                        chunk_label_groups.setdefault(ci, {}).setdefault(lbl_key, []).append(ent)

            # Deterministic chunk-local URI redirects:
            # if multiple entities share the same normalized label in the same chunk,
            # keep exactly one winner for relation emission in that chunk.
            chunk_uri_redirect: Dict[Tuple[int, str], str] = {}
            for ci, groups in chunk_label_groups.items():
                for _, ents in groups.items():
                    if len(ents) <= 1:
                        continue
                    winner = sorted(
                        ents,
                        key=lambda e: (-float(getattr(e, "confidence", 0.0) or 0.0), str(getattr(e, "uri", ""))),
                    )[0]
                    winner_uri = str(getattr(winner, "uri", ""))
                    for ent in ents:
                        ent_uri = str(getattr(ent, "uri", ""))
                        if ent_uri and winner_uri and ent_uri != winner_uri:
                            chunk_uri_redirect[(ci, ent_uri)] = winner_uri

            # Use stable discourse_id from upstream
            did = discourse_ids[idx] if idx < len(discourse_ids) else None
            if did is None:
                did = f"doc_{idx}"
            did = str(did).strip()
            if not did:
                did = f"doc_{idx}"

            prov_meta = (
                provenance_list[idx]
                if provenance_list is not None and idx < len(provenance_list) and isinstance(provenance_list[idx], dict)
                else {}
            )
            field = prov_meta.get("field") if isinstance(prov_meta, dict) else None
            origin_key = (
                str(prov_meta.get("origin_key")).strip()
                if isinstance(prov_meta, dict) and prov_meta.get("origin_key") is not None
                else self._compose_origin_key(did, field)
            )

            chunk_spans = self._get_chunk_spans_for_discourse(did, origin_key=origin_key)
            if not chunk_spans:
                chunk_spans = self._build_chunk_spans(
                    resolved_text,
                    SemanticChunker.chunk(resolved_text),
                    did,
                    origin_key=origin_key,
                )

            # Stable URI for the discourse container node
            # (prefixing avoids accidental clashes with other entity IDs)
            doc_uri = make_uri(f"discourse_{make_uri_safe(origin_key or did)}")

            # Create a DiscourseUnit entity for the document context
            doc_entity = Entity(
                uri=doc_uri,
                label=f"DiscourseUnit {did}",
                entity_type="DiscourseUnit",
                confidence=1.0,               
                metadata={
                    "discourse_id": did,
                    "origin_key": origin_key,
                    "field": field,
                    "surface_form": "discourse unit",
                    "context_source": "lightweight",
                    "text": text,
                },    
            )

            # Some of your pipeline code expects these aliases:
            if not hasattr(doc_entity, "name") or doc_entity.name is None:
                doc_entity.name = doc_entity.label
            if not hasattr(doc_entity, "type") or doc_entity.type is None:
                doc_entity.type = doc_entity.entity_type
            self._ensure_entity_ontology_grounding(doc_entity)

            # Register the DiscourseUnit entity so it gets rdf:type in TTL
            if hasattr(self.global_registry, "get_or_create"):
                doc_entity = self.global_registry.get_or_create(doc_entity)
                self._ensure_entity_ontology_grounding(doc_entity)
            elif hasattr(self.global_registry, "add_entity"):
                self.global_registry.add_entity(doc_entity)
            elif hasattr(self.global_registry, "add"):
                self.global_registry.add(doc_entity)
            else:
                if hasattr(self.global_registry, "entities"):
                    self.global_registry.entities[doc_uri] = doc_entity

            # 1) DiscourseUnit mentions each entity detected in this text
            mention_seen: Set[str] = set()
            for e in entities_list[idx]:
                ent_uri = str(getattr(e, "uri", ""))
                ci = (getattr(e, "metadata", {}) or {}).get("chunk_index")
                if isinstance(ci, int):
                    ent_uri = chunk_uri_redirect.get((ci, ent_uri), ent_uri)
                if not ent_uri or ent_uri in mention_seen:
                    continue
                mention_seen.add(ent_uri)
                mention_entity = uri_to_entity.get(ent_uri, e)
                evidence_txt = _mk_evidence_text(
                    text=(text[:500] if text else None),
                    fallback=f"Lightweight mention: entity '{mention_entity.label}' mentioned in discourse_id={did}.",
                )
                rel_mentions = make_light_relation(doc_uri, "mentions", ent_uri, discourse_id=did, evidence=evidence_txt)
                rel_mentions.context = evidence_txt                
                
                rel_mentions.metadata.update({
                    "discourse_id": did,
                    "origin_key": origin_key,
                    "context_source": "lightweight_mentions",
                    "field": mention_entity.metadata.get("field"),
                    "theme_id": mention_entity.metadata.get("theme_id"),
                    "question_id": mention_entity.metadata.get("question_id"),
                    "answer_id": mention_entity.metadata.get("answer_id"),
                    "chunk_index": mention_entity.metadata.get("chunk_index"),
                })
                relations.append(rel_mentions)


            # 2) Co-occurrence "relatedTo" relations (chunk-scoped + capped)
            MAX_COOC_ENTS = 30  # production cap
            MAX_COOC_EDGES_PER_CHUNK = 120


            by_chunk = {}
            by_chunk_meta = {}  # store provenance per chunk

            for e in entities_list[idx][:MAX_COOC_ENTS]:
                ci = e.metadata.get("chunk_index")
                if ci is None:
                    continue
                ent_uri = chunk_uri_redirect.get((ci, str(e.uri)), str(e.uri))
                bucket = by_chunk.setdefault(ci, [])
                if ent_uri not in bucket:
                    bucket.append(ent_uri)

                # capture chunk-level provenance once
                if ci not in by_chunk_meta:
                    by_chunk_meta[ci] = {
                        "field": e.metadata.get("field"),
                        "theme_id": e.metadata.get("theme_id"),
                        "question_id": e.metadata.get("question_id"),
                        "answer_id": e.metadata.get("answer_id"),
                    }

            for ci, uris in by_chunk.items():
                prov = by_chunk_meta.get(ci, {})
                edge_count = 0

                for i in range(len(uris)):
                    for j in range(i + 1, len(uris)):
                        evidence_txt = _mk_evidence_text(
                            text=(text[:500] if text else None),
                            fallback=f"Lightweight co-occurrence: entities co-occur in discourse_id={did}, chunk_index={ci}.",
                        )
                        rel_related = make_light_relation(
                            uris[i], "relatedTo", uris[j], confidence=0.35, discourse_id=did, evidence=evidence_txt
                        )
                        rel_related.context = evidence_txt                        
                        
                        rel_related.metadata.update({
                            "discourse_id": did,
                            "origin_key": origin_key,
                            "context_source": "cooccurrence_chunk",
                            "chunk_index": ci,
                            **prov
                        })
                        relations.append(rel_related)

                        edge_count += 1
                        if edge_count >= MAX_COOC_EDGES_PER_CHUNK:
                            break
                    if edge_count >= MAX_COOC_EDGES_PER_CHUNK:
                        break

            # Ensure every extracted relation has strict chunk provenance.
            for rel in relations:
                meta = getattr(rel, "metadata", None) or {}
                if origin_key:
                    meta.setdefault("origin_key", origin_key)
                span = self._extract_relation_span_from_metadata(meta)
                if span is None:
                    derived = self._derive_relation_span_from_entities(rel, uri_to_entity)
                    if derived is not None:
                        meta["char_start"] = int(derived[0])
                        meta["char_end"] = int(derived[1])
                        meta["start_offset"] = int(derived[0])
                        meta["end_offset"] = int(derived[1])
                ci = meta.get("chunk_index")
                if isinstance(ci, int):
                    src = str(getattr(rel, "source", "") or "")
                    tgt = str(getattr(rel, "target", "") or "")
                    if src:
                        new_src = chunk_uri_redirect.get((ci, src))
                        if new_src:
                            rel.source = new_src
                    if tgt:
                        new_tgt = chunk_uri_redirect.get((ci, tgt))
                        if new_tgt:
                            rel.target = new_tgt
                rel.metadata = meta
                self._ensure_relation_chunk_provenance(
                    rel,
                    fallback_discourse_id=did,
                    chunk_spans=chunk_spans,
                )



        # Backward compatibility at extraction stage.
        for relations in all_relations:
            for relation in relations:
                if "discourse_id" not in relation.metadata and "doc_id" in relation.metadata:
                    relation.metadata["discourse_id"] = relation.metadata.get("doc_id")

        return all_relations

    # ---------------------------------------------------------
    # Core structural node registration (Theme/Question/Answer)
    # ---------------------------------------------------------
    def _register_structural_node(
        self,
        *,
        uri: str,
        label: str,
        entity_type: str,
        discourse_id: str,
        context_source: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ensure structural nodes exist in registry so they serialize into TTL."""
        meta = {
            "discourse_id": str(discourse_id),
            "surface_form": str(label),
            "context_source": context_source,
        }
        if extra_meta:
            meta.update(extra_meta)
        if "origin_key" not in meta:
            meta["origin_key"] = self._compose_origin_key(meta.get("discourse_id"), meta.get("field"))

        node = Entity(
            uri=uri,
            label=label,
            entity_type=entity_type,
            confidence=1.0,
            metadata=meta,
        )

        # compatibility aliases used elsewhere in your pipeline
        if not hasattr(node, "name") or node.name is None:
            node.name = node.label
        if not hasattr(node, "type") or node.type is None:
            node.type = node.entity_type
        self._ensure_entity_ontology_grounding(node)

        # register (same pattern you use for DiscourseUnit) :contentReference[oaicite:1]{index=1}
        if hasattr(self.global_registry, "get_or_create"):
            node = self.global_registry.get_or_create(node)
            self._ensure_entity_ontology_grounding(node)
        elif hasattr(self.global_registry, "add_entity"):
            self.global_registry.add_entity(node)
        elif hasattr(self.global_registry, "add"):
            self.global_registry.add(node)
        else:
            if hasattr(self.global_registry, "entities"):
                self.global_registry.entities[uri] = node



    # ---------------------------------------------------------
    # Themes
    # ---------------------------------------------------------
    def process_themes_batch(self, themes: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple themes in batch."""
        texts = []
        metadata_list = []
        
        for theme in themes:
            texts.append(theme.get('description', ''))
            metadata_list.append({
                "domain": theme.get("domain_category", ""),
                "discourse_id": theme.get("id"),
                "field": "theme.description",
                "origin_key": self._compose_origin_key(theme.get("id"), "theme.description"),
                "theme_id": theme.get("id"),
                "question_id": None,
                "answer_id": None,
                "context_source": "dataset_theme",
            })


        entities_list = self._process_entities_batch(texts, metadata_list)

        discourse_ids = [m.get("discourse_id") for m in metadata_list]
        relations_list = self._process_relations_batch(
            texts,
            entities_list,
            discourse_ids,
            provenance_list=metadata_list,
        )
        
        # Attach discourse_id to each relation for this batch
        for i, theme in enumerate(themes):
            theme_id = theme.get('id')

            # âœ… FIX: register Theme node so it serialises (prevents dangling theme_<id>)
            theme_uri = f"{self.config.NAMESPACES['etd']}theme_{theme_id}"
            self._register_structural_node(
                uri=theme_uri,
                label=f"Theme {theme_id}",
                entity_type="Theme",
                discourse_id=str(theme_id),
                context_source="dataset_theme",
                extra_meta={
                    "theme_id": theme_id,
                    "field": "theme",
                    "domain": theme.get("domain_category", ""),
                    "problem_category": theme.get("problem_category", ""),
                },
            )

            for rel in relations_list[i]:
                rel.metadata.setdefault("discourse_id", theme_id)
                rel.metadata.setdefault("field", "theme.description")
                rel.metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(theme_id, "theme.description"),
                )
                self._ensure_relation_chunk_provenance(rel, fallback_discourse_id=theme_id)

        results: List[ProcessingResult] = []
        for i, theme in enumerate(themes):
            problem_category = theme.get('problem_category', '')
            domain = theme.get('domain_category', '')
            
            problem_ontologies = self.ontology_mapper.map_problem_category(problem_category)
            vair_domain = self.ontology_mapper.map_domain(domain)
            
            results.append(ProcessingResult(
                entities=entities_list[i],
                relations=relations_list[i],
                metadata={
                    'text': texts[i],
                    'text_length': len(texts[i]),
                    'num_entities': len(entities_list[i]),
                    'num_relations': len(relations_list[i]),
                    'problem_category': problem_category,
                    'problem_ontologies': problem_ontologies,
                    'domain_category': domain,
                    'vair_domain': vair_domain,
                    'model_category': theme.get('model_category', ''),
                    'extraction_method': 'enhanced_llm_ontology'
                }
            ))
        
        return results

    # ---------------------------------------------------------
    # Questions
    # ---------------------------------------------------------
    def process_questions_batch(self, questions: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple questions in batch."""
        texts = []
        metadata_list = []
        
        for question in questions:
            texts.append(question.get('body', question.get('title', '')))       
            metadata_list.append({
                "domain": None,
                "discourse_id": question.get("id"),
                "field": "question.body",
                "origin_key": self._compose_origin_key(question.get("id"), "question.body"),
                "theme_id": question.get("theme_id"),
                "question_id": question.get("id"),
                "answer_id": None,
                "context_source": "dataset_question",
            })

        
        entities_list = self._process_entities_batch(texts, metadata_list)
        
        discourse_ids = [m.get("discourse_id") for m in metadata_list]
        relations_list = self._process_relations_batch(
            texts,
            entities_list,
            discourse_ids,
            provenance_list=metadata_list,
        )

                
        #Attach discourse_id to each relation in this question
        for i, question in enumerate(questions):
            qid = question.get('id')
            for rel in relations_list[i]:
                rel.metadata.setdefault("discourse_id", qid)
                rel.metadata.setdefault("field", "question.body")
                rel.metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(qid, "question.body"),
                )
                self._ensure_relation_chunk_provenance(rel, fallback_discourse_id=qid)

        results: List[ProcessingResult] = []
        for i, question in enumerate(questions):
            question_id = question.get('id')
            theme_id = question.get('theme_id')
            
            # âœ… FIX: register Question node so it serialises (prevents dangling question_<id>)
            q_uri = f"{self.config.NAMESPACES['etd']}question_{question_id}"
            self._register_structural_node(
                uri=q_uri,
                label=f"Question {question_id}",
                entity_type="Question",
                discourse_id=str(question_id),
                context_source="dataset_question",
                extra_meta={
                    "question_id": question_id,
                    "theme_id": theme_id,
                    "field": "question",
                },
            )         
            
            if theme_id:
                # Structural relation: Question -> Theme
                # âœ… FIX: ensure Theme node exists before linking (prevents dangling theme_<id>)
                t_uri = f"{self.config.NAMESPACES['etd']}theme_{theme_id}"
                self._register_structural_node(
                    uri=t_uri,
                    label=f"Theme {theme_id}",
                    entity_type="Theme",
                    discourse_id=str(theme_id),
                    context_source="dataset_theme",
                    extra_meta={
                        "theme_id": theme_id,
                        "field": "theme",
                    },
                )
                
                source_uri = f"{self.config.NAMESPACES['etd']}question_{question_id}"
                target_uri = f"{self.config.NAMESPACES['etd']}theme_{theme_id}"
                rel_uri = _make_rel_uri(f"question_{question_id}_theme_{theme_id}")
                relations_list[i].append(make_relation(
                    source_uri,
                    "belongsToTheme",
                    target_uri,
                    discourse_id=str(question_id),
                    extractor="structural",
                    context_source="dataset_question",
                    confidence=0.98,
                    evidence=_mk_evidence_text(
                        text=None,
                        fallback=f"Structural link: question_{question_id} belongsToTheme theme_{theme_id} (dataset_question).",
                    ),
                ))
                relations_list[i][-1].uri = rel_uri
                relations_list[i][-1].metadata.setdefault("field", "question.body")
                relations_list[i][-1].metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(str(question_id), "question.body"),
                )
                relations_list[i][-1].metadata.setdefault("source_doc_id", str(question_id))
                
                # keep ontology mapping info
                relations_list[i][-1].metadata["ontology_properties"] = ["aipo:hasTheme"]

            question_type = self._classify_question_type(texts[i])
            has_ethical = any(e.entity_type == 'EthicalConcept' for e in entities_list[i])

            results.append(ProcessingResult(
                entities=entities_list[i],
                relations=relations_list[i],
                metadata={
                    'text': texts[i],
                    'text_length': len(texts[i]),
                    'num_entities': len(entities_list[i]),
                    'num_relations': len(relations_list[i]),
                    'question_type': question_type,
                    'has_ethical_dimension': has_ethical
                }
            ))
        
        return results

    # ---------------------------------------------------------
    # Answers
    # ---------------------------------------------------------
    def process_answers_batch(self, answers: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple answers in batch."""
        texts = []
        metadata_list = []
        
        for answer in answers:
            texts.append(answer.get('description', ''))
            metadata_list.append({
                "domain": None,
                "discourse_id": answer.get("id"),
                "field": "answer.description",
                "origin_key": self._compose_origin_key(answer.get("id"), "answer.description"),
                "theme_id": answer.get("theme_id"),
                "question_id": answer.get("question_id"),
                "answer_id": answer.get("id"),
                "context_source": "dataset_answer",
            })

        entities_list = self._process_entities_batch(texts, metadata_list)
        
        discourse_ids = [m.get("discourse_id") for m in metadata_list]
        relations_list = self._process_relations_batch(
            texts,
            entities_list,
            discourse_ids,
            provenance_list=metadata_list,
        )
        
        # Attach discourse_id to each relation in this answer
        for i, answer in enumerate(answers):
            aid = answer.get('id')
            for rel in relations_list[i]:
                rel.metadata.setdefault("discourse_id", aid)
                rel.metadata.setdefault("field", "answer.description")
                rel.metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(aid, "answer.description"),
                )
                self._ensure_relation_chunk_provenance(rel, fallback_discourse_id=aid)

        results: List[ProcessingResult] = []
        for i, answer in enumerate(answers):
            answer_id = answer.get('id')
            question_id = answer.get('question_id')
            expert_id = answer.get('expert_id')
            
            # âœ… FIX: register Answer node so it serialises (prevents dangling answer_<id>)
            a_uri = f"{self.config.NAMESPACES['etd']}answer_{answer_id}"
            self._register_structural_node(
                uri=a_uri,
                label=f"Answer {answer_id}",
                entity_type="Answer",
                discourse_id=str(answer_id),
                context_source="dataset_answer",
                extra_meta={
                    "answer_id": answer_id,
                    "question_id": question_id,
                    "theme_id": answer.get("theme_id"),
                    "field": "answer",
                },
            )
                           
            if question_id:
                # âœ… FIX: ensure Question node exists before linking (prevents dangling question_<id>)
                q_uri = f"{self.config.NAMESPACES['etd']}question_{question_id}"
                self._register_structural_node(
                    uri=q_uri,
                    label=f"Question {question_id}",
                    entity_type="Question",
                    discourse_id=str(question_id),
                    context_source="dataset_question",
                    extra_meta={
                        "question_id": question_id,
                        "field": "question",
                    },
                )

                source_uri = f"{self.config.NAMESPACES['etd']}answer_{answer_id}"
                target_uri = f"{self.config.NAMESPACES['etd']}question_{question_id}"
                rel_uri = _make_rel_uri(f"answer_{answer_id}_question_{question_id}")
                relations_list[i].append(make_relation(
                    source_uri,
                    "answersQuestion",
                    target_uri,
                    discourse_id=str(answer_id),
                    extractor="structural",
                    context_source="dataset_answer",
                    confidence=0.98,
                    evidence=_mk_evidence_text(
                        text=None,
                        fallback=f"Structural link: answer_{answer_id} answersQuestion question_{question_id} (dataset_answer).",
                    ),
                ))
                relations_list[i][-1].uri = rel_uri
                relations_list[i][-1].metadata.setdefault("field", "answer.description")
                relations_list[i][-1].metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(str(answer_id), "answer.description"),
                )
                relations_list[i][-1].metadata.setdefault("source_doc_id", str(answer_id))
                relations_list[i][-1].metadata["ontology_properties"] = ["sioc:reply_of"]

            if expert_id:
                # âœ… FIX: ensure Expert node exists before linking (prevents dangling expert_<id>)
                e_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
                self._register_structural_node(
                    uri=e_uri,
                    label=f"Expert {expert_id}",
                    entity_type="Expert",
                    discourse_id=str(answer_id),  # discourse unit is the answer
                    context_source="dataset_answer",
                    extra_meta={
                        "expert_id": expert_id,
                        "field": "expert",
                    },
                )

                source_uri = f"{self.config.NAMESPACES['etd']}answer_{answer_id}"
                target_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
                rel_uri = _make_rel_uri(f"answer_{answer_id}_expert_{expert_id}")
                relations_list[i].append(make_relation(
                    source_uri,
                    "authoredBy",
                    target_uri,
                    discourse_id=str(answer_id),
                    extractor="structural",
                    context_source="dataset_answer",
                    confidence=0.96,
                    evidence=_mk_evidence_text(
                        text=None,
                        fallback=f"Structural link: answer_{answer_id} authoredBy expert_{expert_id} (dataset_answer).",
                    ),
                ))
                relations_list[i][-1].uri = rel_uri
                relations_list[i][-1].metadata.setdefault("field", "answer.description")
                relations_list[i][-1].metadata.setdefault(
                    "origin_key",
                    self._compose_origin_key(str(answer_id), "answer.description"),
                )
                relations_list[i][-1].metadata.setdefault("source_doc_id", str(answer_id))
                relations_list[i][-1].metadata["ontology_properties"] = ["prov:wasAttributedTo"]


            results.append(ProcessingResult(
                entities=entities_list[i],
                relations=relations_list[i],
                metadata={
                    'text': texts[i],
                    'text_length': len(texts[i]),
                    'num_entities': len(entities_list[i]),
                    'num_relations': len(relations_list[i])
                }
            ))
        
        return results

    # ---------------------------------------------------------
    # Votes
    # ---------------------------------------------------------
    def process_vote(self, vote_dict: Dict[str, Any]) -> ProcessingResult:
        """Process a vote (expert endorsement)."""
        expert_id = vote_dict.get('expert_id')
        answer_id = vote_dict.get('answer_id')
        vote_raw = vote_dict.get('vote_value', None)

        # normalised text form (existing behaviour)
        vote_value = str(vote_raw).strip().lower() if vote_raw is not None else ''

        # raw integer form for RDF literal (new)
        vote_raw_int = None
        try:
            # handles 1, 0, -1, "1", "0", "-1"
            vote_raw_int = int(vote_raw)
        except (TypeError, ValueError):
            vote_raw_int = None
        
        vote_id = vote_dict.get('id')
        vote_node_id = vote_id if vote_id is not None else f"{expert_id}_{answer_id}"

        # Create Expert entity
        expert_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
        expert_entity = Entity(
            uri=expert_uri,
            label=f"Expert {expert_id}",
            entity_type="Expert",
            confidence=0.95,
            metadata={'airo_type': 'Stakeholder'}
        )
        # backwards compat
        expert_entity.name = expert_entity.label
        expert_entity.type = expert_entity.entity_type
        self._ensure_entity_ontology_grounding(expert_entity)

        expert_entity = self.global_registry.get_or_create(expert_entity)
        self._ensure_entity_ontology_grounding(expert_entity)

        # Create Answer entity
        answer_uri = f"{self.config.NAMESPACES['etd']}answer_{answer_id}"
        answer_entity = Entity(
            uri=answer_uri,
            label=f"Answer {answer_id}",
            entity_type="Answer",
            confidence=0.90,
            metadata={'airo_type': 'Document'}
        )
        answer_entity.name = answer_entity.label
        answer_entity.type = answer_entity.entity_type
        self._ensure_entity_ontology_grounding(answer_entity)

        answer_entity = self.global_registry.get_or_create(answer_entity)
        self._ensure_entity_ontology_grounding(answer_entity)

        # Create Vote entity for structural backbone links
        vote_uri = f"{self.config.NAMESPACES['etd']}vote_{vote_node_id}"
        vote_entity = Entity(
            uri=vote_uri,
            label=f"Vote {vote_node_id}",
            entity_type="Vote",
            confidence=1.0,
            metadata={
                "discourse_id": str(vote_node_id),
                "context_source": "dataset_vote",
                "vote_value": vote_value,
            },
        )
        vote_entity.name = vote_entity.label
        vote_entity.type = vote_entity.entity_type
        self._ensure_entity_ontology_grounding(vote_entity)
        vote_entity = self.global_registry.get_or_create(vote_entity)
        self._ensure_entity_ontology_grounding(vote_entity)

        # Backbone type guards (hard invariant for vote structure)
        if getattr(vote_entity, "entity_type", None) != "Vote":
            raise RuntimeError(f"Backbone invariant failed: vote entity is not Vote ({vote_entity.uri})")
        if getattr(expert_entity, "entity_type", None) not in {"Expert", "Person"}:
            raise RuntimeError(f"Backbone invariant failed: castBy target must be Expert/Person ({expert_entity.uri})")
        if getattr(answer_entity, "entity_type", None) != "Answer":
            raise RuntimeError(f"Backbone invariant failed: receivesVote target must be Answer ({answer_entity.uri})")

        positive_votes = [
            'upvote', 'approve', 'endorse', 'like', 'agree', 'support',
            'recommend', 'positive', '1', 'true', 'yes'
        ]
        negative_votes = [
            'downvote', 'disagree', 'reject', '0', 'oppose', 'criticize',
            'negative', 'false', 'no'
        ]
        neutral_votes = [
            'neutral', 'abstain', 'skip', 'maybe', 'undecided'
        ]

        if vote_value in positive_votes:
            relation_type = 'endorses'
            ontology_properties = ['vair:ApprovalAction', 'prov:Activity']
        elif vote_value in negative_votes:
            relation_type = 'disagreesWith'
            ontology_properties = ['vair:DisagreeAction', 'prov:Activity']
        elif vote_value in neutral_votes:
            relation_type = 'isNeutralOn'
            ontology_properties = ['prov:EntityInfluence']
        else:
            relation_type = 'interactsWith'
            ontology_properties = ['prov:Interaction']

        # Map to SKOS concept URI (new)
        # Assumes you added VoteValueScheme concepts in add_schema.py:
        #   etd:VoteValueSchemeUpvote / Downvote / Neutral
        if relation_type == 'endorses':
            vote_concept_local = "VoteValueSchemeUpvote"
        elif relation_type == 'disagreesWith':
            vote_concept_local = "VoteValueSchemeDownvote"
        elif relation_type == 'isNeutralOn':
            vote_concept_local = "VoteValueSchemeNeutral"
        else:
            vote_concept_local = "VoteValueSchemeNeutral"

        vote_value_concept_uri = f"{self.config.NAMESPACES['etd']}{vote_concept_local}"
        rel_suffix = f"vote_{vote_id}" if vote_id is not None else f"vote_{expert_id}_{answer_id}"
        rel_uri = _make_rel_uri(rel_suffix)
        
        relation = make_relation(
            expert_entity.uri,
            relation_type,
            answer_entity.uri,
            discourse_id=str(vote_id or f"{expert_id}_{answer_id}"),
            extractor="vote",
            context_source="dataset_vote",
            confidence=0.90,
            evidence=_mk_evidence_text(
                text=None,
                fallback=f"Vote record: expert_{expert_id} -> {relation_type} -> answer_{answer_id} (vote_value={vote_value}, raw_vote_value={vote_raw_int}).",
            ),
        )

        relation.uri = rel_uri

        relation.metadata["ontology_properties"] = ontology_properties
        relation.metadata["vote_value"] = vote_value                 # keep (string)
        relation.metadata["raw_vote_value"] = vote_raw_int           # new (int or None)
        relation.metadata["vote_value_concept"] = vote_value_concept_uri  # new (URI string)
        relation.metadata.setdefault("field", "vote.vote_value")
        relation.metadata.setdefault(
            "origin_key",
            self._compose_origin_key(str(vote_node_id), "vote.vote_value"),
        )
        relation.metadata.setdefault("source_doc_id", str(vote_node_id))

        # Deterministic backbone edges
        cast_by_rel = make_relation(
            vote_entity.uri,
            "castBy",
            expert_entity.uri,
            discourse_id=str(vote_node_id),
            extractor="structural",
            context_source="dataset_vote",
            confidence=0.99,
            evidence=_mk_evidence_text(
                text=None,
                fallback=f"Structural link: vote_{vote_node_id} castBy expert_{expert_id}.",
            ),
        )
        cast_by_rel.uri = _make_rel_uri(f"vote_{vote_node_id}_castBy_{expert_id}")
        cast_by_rel.metadata.setdefault("field", "vote.vote_value")
        cast_by_rel.metadata.setdefault(
            "origin_key",
            self._compose_origin_key(str(vote_node_id), "vote.vote_value"),
        )
        cast_by_rel.metadata.setdefault("source_doc_id", str(vote_node_id))

        receives_vote_rel = make_relation(
            vote_entity.uri,
            "receivesVote",
            answer_entity.uri,
            discourse_id=str(vote_node_id),
            extractor="structural",
            context_source="dataset_vote",
            confidence=0.99,
            evidence=_mk_evidence_text(
                text=None,
                fallback=f"Structural link: vote_{vote_node_id} receivesVote answer_{answer_id}.",
            ),
        )
        receives_vote_rel.uri = _make_rel_uri(f"vote_{vote_node_id}_receivesVote_{answer_id}")
        receives_vote_rel.metadata.setdefault("field", "vote.vote_value")
        receives_vote_rel.metadata.setdefault(
            "origin_key",
            self._compose_origin_key(str(vote_node_id), "vote.vote_value"),
        )
        receives_vote_rel.metadata.setdefault("source_doc_id", str(vote_node_id))

        for rel in (relation, cast_by_rel, receives_vote_rel):
            self._ensure_relation_chunk_provenance(rel, fallback_discourse_id=str(vote_node_id))

        return ProcessingResult(
            entities=[expert_entity, answer_entity, vote_entity],
            relations=[relation, cast_by_rel, receives_vote_rel],
            metadata={
                'vote_id': vote_id,
                'expert_id': expert_id,
                'answer_id': answer_id,
                'vote_value': vote_value,
                'raw_vote_value': vote_raw_int,
                'vote_value_concept': vote_value_concept_uri
            }
        )

    # ---------------------------------------------------------
    # Experts
    # ---------------------------------------------------------
    def process_expert(self, expert_dict: Dict[str, Any]) -> ProcessingResult:
        """Process an expert profile."""
        expert_id = expert_dict.get('id')
        user_id = expert_dict.get('user_id')
        expertise = expert_dict.get('area_of_expertise', '')
        bio = expert_dict.get('bio', '')

        expert_uri = f"{self.config.NAMESPACES['etd']}expert_{expert_id}"
        label = expert_dict.get('display_name', f"Expert {expert_id}")
        
        expert_entity = Entity(
            uri=expert_uri,
            label=label,
            entity_type="Expert",
            confidence=0.95,
            metadata={
                'airo_type': 'Stakeholder',
                'ontology_classes': ['foaf:Person', 'prov:Agent'],
                'area_of_expertise': expertise,
                'bio': bio,
                'user_id': user_id
            }
        )

        expert_entity.name = expert_entity.label
        expert_entity.type = expert_entity.entity_type
        self._ensure_entity_ontology_grounding(expert_entity)
        
        expert_entity = self.global_registry.get_or_create(expert_entity)
        self._ensure_entity_ontology_grounding(expert_entity)
        
        return ProcessingResult(
            entities=[expert_entity],
            relations=[],
            metadata={
                'user_id': user_id,
                'expert_id': expert_id,
                'area_of_expertise': expertise
            }
        )

    # ---------------------------------------------------------
    # Documents
    # ---------------------------------------------------------
    def process_document(self, document_dict: Dict[str, Any]) -> ProcessingResult:
        """Process a document (report, file)."""
        discourse_id = document_dict.get('id')
        title = document_dict.get('title', f"Document {discourse_id}")
        description = document_dict.get('description', '')
        url = document_dict.get('url', '')

        doc_uri = f"{self.config.NAMESPACES['etd']}document_{discourse_id}"
        
        document_entity = Entity(
            uri=doc_uri,
            label=title,
            entity_type="Resource",
            confidence=0.90,
            metadata={
                'airo_type': 'Document',
                'ontology_classes': ['bibo:Document', 'foaf:Document'],
                'url': url,
                'description': description
            }
        )

        document_entity.name = document_entity.label
        document_entity.type = document_entity.entity_type
        self._ensure_entity_ontology_grounding(document_entity)
        
        document_entity = self.global_registry.get_or_create(document_entity)
        self._ensure_entity_ontology_grounding(document_entity)

        # For now, we skip URL extraction from description to avoid
        # dependency on the old entity_extractor._extract_urls().
        # This can be re-added later as a small utility.
        relations: List[Relation] = []

        return ProcessingResult(
            entities=[document_entity],
            relations=relations,
            metadata={
                'discourse_id': discourse_id,
                'title': title,
                'text': description or title,
            }
        )

    # ---------------------------------------------------------
    # Question classification
    # ---------------------------------------------------------
    def _classify_question_type(self, text: str) -> str:
        """Classify question type."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['what', 'define', 'definition']):
            return 'definitional'
        elif any(word in text_lower for word in ['how', 'explain', 'method']):
            return 'explanatory'
        elif any(word in text_lower for word in ['why', 'reason', 'cause']):
            return 'causal'
        elif any(word in text_lower for word in ['should', 'ought', 'must']):
            return 'normative'
        return 'general'

    # ---------------------------------------------------------
    # Batch processing orchestration
    # ---------------------------------------------------------
    def process_batch(self, data: Dict[str, Any]) -> AnnotationBatch:
        """Process all documents in batches."""
        annotations = AnnotationBatch()
        batch_size = max(1, int(self.config.BATCH_SIZE))
        
        themes = data.get('themes', data.get('Theme', []))
        if themes:
            print(f"\nâš™ï¸ Processing {len(themes)} themes...")
            theme_batches = (len(themes) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iterator(themes, batch_size), total=theme_batches):
                results = self.process_themes_batch(batch)
                for theme, result in zip(batch, results):
                    annotations.themes[theme.get('id')] = result
                cleanup_memory()
        
        questions = data.get('questions', data.get('Question', []))
        if questions:
            print(f"\nâš™ï¸ Processing {len(questions)} questions...")
            question_batches = (len(questions) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iterator(questions, batch_size), total=question_batches):
                results = self.process_questions_batch(batch)
                for question, result in zip(batch, results):
                    annotations.questions[question.get('id')] = result
                cleanup_memory()
        
        answers = data.get('answers', data.get('Answer', []))
        if answers:
            print(f"\nâš™ï¸ Processing {len(answers)} answers...")
            answer_batches = (len(answers) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iterator(answers, batch_size), total=answer_batches):
                results = self.process_answers_batch(batch)
                for answer, result in zip(batch, results):
                    annotations.answers[answer.get('id')] = result
                cleanup_memory()
        
        votes = data.get('votes', data.get('Vote', []))
        if votes:
            print(f"\nâš™ï¸ Processing {len(votes)} votes...")
            vote_batches = (len(votes) + batch_size - 1) // batch_size
            for batch in tqdm(batch_iterator(votes, batch_size), total=vote_batches):
                for vote in batch:
                    result = safe_process_with_retry(
                        vote,
                        lambda v: self.process_vote(v)
                    )
                    annotations.votes[vote.get('id')] = result
                cleanup_memory()
        
        experts = data.get('experts', data.get('Expert', []))
        if experts:
            print(f"\nâš™ï¸ Processing {len(experts)} experts...")
            for expert in tqdm(experts):
                result = safe_process_with_retry(
                    expert,
                    lambda e: self.process_expert(e)
                )
                annotations.experts[expert.get('id')] = result
        
        documents = data.get('documents', data.get('Document', []))
        if documents:
            print(f"\nâš™ï¸ Processing {len(documents)} documents...")
            for document in tqdm(documents):
                result = safe_process_with_retry(
                    document,
                    lambda d: self.process_document(d)
                )
                annotations.documents[document.get('id')] = result

        # ðŸ”¹ ENTITY CONSOLIDATION (NEW STEP)
        print("\nðŸ”„ Consolidating entities across the entire registry...")

        consolidator = EntityConsolidator(
            self.global_registry,
            self.entity_normalizer,
            output_dir=self.config.OUTPUT_DIR,
        )

        # 1) Pull all entities from the registry (production-safe)
        store = getattr(self.global_registry, "entities", None)

        if isinstance(store, dict):
            all_entities = list(store.values())
        elif isinstance(store, list):
            all_entities = list(store)
        else:
            all_entities = []
            if hasattr(self.global_registry, "get_all_entities"):
                all_entities = list(self.global_registry.get_all_entities())


        # 2) Consolidate (returns a LIST of canonical entities)
        consolidated_entities = consolidator.consolidate(all_entities)

        # 3) Write back into registry (DO NOT replace the registry object)
        if hasattr(self.global_registry, "entities"):
            # Keep registry storage as dict keyed by entity URI (backward compatible)
            self.global_registry.entities = {e.uri: e for e in consolidated_entities if getattr(e, "uri", None)}

            # Optional: rebuild registry indexes if supported
            if hasattr(self.global_registry, "reindex"):
                self.global_registry.reindex()
        elif hasattr(self.global_registry, "set_entities"):
            self.global_registry.set_entities(consolidated_entities)
        else:
            # last resort: keep registry as-is, but avoid crashing
            raise TypeError("GlobalEntityRegistry has no 'entities' or set_entities() API to store consolidated entities.")

        print(f"âœ“ Entity consolidation complete. Total unique entities: {len(consolidated_entities)}")
        self._apply_entity_redirects_to_annotations(annotations, getattr(consolidator, "redirects", {}))

        # 4) Canonical linking across FULL registry (cross-chunk/cross-document)
        print("\n🔗 Running canonical linking across full registry...")
        coverage = self.run_canonical_linking(annotations)
        try:
            save_json(coverage, self.config.OUTPUT_DIR / "canonical_linking_report.json")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to persist canonical_linking_report.json: %s", exc)
        print(
            f"✓ Canonical linking complete. Raw={coverage.get('raw_entities')} "
            f"Canonical={coverage.get('canonical_entities')} Merges={coverage.get('merges_count')}"
        )

        # Ensure ontology grounding coverage for the full canonical registry.
        get_all = getattr(self.global_registry, "get_all_entities", None)
        if callable(get_all):
            for ent in get_all():
                self._ensure_entity_ontology_grounding(ent)


        # 5) Predicate precision + ontology enrichment after canonical linking.
        self.apply_predicate_precision_and_enrichment(annotations)

        return annotations

    def run_canonical_linking(self, annotations: AnnotationBatch) -> Dict[str, Any]:
        pre_link_count = len(self.global_registry.get_all_entities()) if hasattr(self.global_registry, "get_all_entities") else 0
        canonical_entities = self.canonical_linker.run_full_registry()
        self._apply_entity_redirects_to_annotations(annotations, getattr(self.canonical_linker, "redirects", {}))
        for rel_list in self._iter_annotation_relation_lists(annotations):
            self.canonical_linker.rewrite_relation_endpoints(rel_list)
        return self.canonical_linker.get_report(pre_link_count, len(canonical_entities))
    def apply_predicate_precision_and_enrichment(self, annotations: AnnotationBatch) -> None:
        """Apply predicate precision and ontology enrichment after canonical linking."""
        # Use FULL canonical registry for type-aware predicate validation.
        # This preserves structural backbone predicates even when per-document
        # `pr.entities` does not contain reminted canonical endpoints.
        canonical_entities: List[Entity] = []
        get_all = getattr(self.global_registry, "get_all_entities", None)
        if callable(get_all):
            canonical_entities = list(get_all())

        for block in (
            annotations.themes,
            annotations.questions,
            annotations.answers,
            annotations.votes,
            annotations.experts,
            annotations.documents,
        ):
            for pr in block.values():
                discourse_text = None
                if isinstance(pr.metadata, dict):
                    discourse_text = pr.metadata.get("text")
                refined_relations = self.predicate_precision.refine_relations(
                    relations=pr.relations,
                    entities=canonical_entities or pr.entities,
                    discourse_text=discourse_text,
                )
                if getattr(self.config, "STRICT_PREDICATE_INVENTORY", False):
                    for relation in refined_relations:
                        if relation.relation not in self.predicate_precision.allowed:
                            raise RuntimeError(
                                f"Predicate precision violation: '{relation.relation}' is outside allowed inventory"
                            )
                for relation in refined_relations:
                    enrichment = self.ontology_mapper.enrich_relation(relation)
                    if isinstance(enrichment, dict) and enrichment:
                        relation.metadata.update(enrichment)
                    if "discourse_id" not in relation.metadata and "doc_id" in relation.metadata:
                        relation.metadata["discourse_id"] = relation.metadata.get("doc_id")
                pr.relations = refined_relations
    def _iter_annotation_relation_lists(self, annotations: AnnotationBatch) -> List[List[Relation]]:
        relation_lists: List[List[Relation]] = []
        for block in (
            annotations.themes,
            annotations.questions,
            annotations.answers,
            annotations.votes,
            annotations.experts,
            annotations.documents,
        ):
            for pr in block.values():
                relation_lists.append(pr.relations)
        return relation_lists
    def _apply_entity_redirects_to_annotations(self, annotations: AnnotationBatch, redirects: Dict[str, str]) -> None:
        """Rewrite entity and relation URIs to canonical redirects after consolidation."""
        if not redirects:
            return

        def _rewrite_result(pr: ProcessingResult) -> None:
            for ent in pr.entities:
                uri = getattr(ent, "uri", None)
                if uri in redirects:
                    ent.uri = redirects[uri]
            for rel in pr.relations:
                src = getattr(rel, "source", None)
                tgt = getattr(rel, "target", None)
                if src in redirects:
                    rel.source = redirects[src]
                if tgt in redirects:
                    rel.target = redirects[tgt]

        for block in (
            annotations.themes,
            annotations.questions,
            annotations.answers,
            annotations.votes,
            annotations.experts,
            annotations.documents,
        ):
            for pr in block.values():
                _rewrite_result(pr)

    # ---------------------------------------------------------
    # Reasoning & quality
    # ---------------------------------------------------------
    def apply_reasoning(self, annotations: Any) -> List[Relation]:
        """Apply multi-hop reasoning to infer new relations.

        Works with:
          - AnnotationBatch (has get_all_relations)
          - dict (nested structure: {doc_type: {doc_id: {relations:[...]}}})
        """
        if not self.config.USE_REASONING:
            return []

        print("\nðŸ§  Applying multi-hop reasoning...")

        # ---------------------------------------------------------
        # Collect relations (supports both object + dict)
        # ---------------------------------------------------------
        all_relations: List[Union[Relation, Dict[str, Any]]] = []

        # Case A: model/object API
        if hasattr(annotations, "get_all_relations") and callable(getattr(annotations, "get_all_relations")):
            all_relations = annotations.get_all_relations()

        # Case B: dict API (annotations_dict)
        elif isinstance(annotations, dict):
            for doc_type, docs in annotations.items():
                if not isinstance(docs, dict):
                    continue
                for _, doc in docs.items():
                    if isinstance(doc, dict):
                        rels = doc.get("relations", [])
                        if isinstance(rels, list):
                            all_relations.extend(rels)

        else:
            raise TypeError(f"apply_reasoning() expected AnnotationBatch or dict, got {type(annotations)}")

        # Build deterministic discourse priors from extracted relations.
        entity_did_counts: Dict[str, Dict[str, int]] = {}
        entity_origin_counts: Dict[str, Dict[str, int]] = {}
        entity_field_counts: Dict[str, Dict[str, int]] = {}

        def _inc(counter: Dict[Any, int], key: Any) -> None:
            counter[key] = int(counter.get(key, 0)) + 1

        def _best_key(counter: Optional[Dict[Any, int]]) -> Optional[Any]:
            if not counter:
                return None
            return sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0][0]

        for item in all_relations:
            if isinstance(item, Relation):
                src = str(getattr(item, "source", "") or "")
                tgt = str(getattr(item, "target", "") or "")
                meta = getattr(item, "metadata", None) or {}
            elif isinstance(item, dict):
                src = str(item.get("source") or item.get("source_id") or "")
                tgt = str(item.get("target") or item.get("target_id") or "")
                meta = item.get("metadata", {}) or {}
            else:
                continue

            did_raw = meta.get("discourse_id") or meta.get("doc_id")
            did = str(did_raw).strip() if did_raw is not None else ""
            field_raw = meta.get("field")
            field = str(field_raw).strip() if field_raw is not None else ""
            origin_raw = meta.get("origin_key")
            origin = str(origin_raw).strip() if origin_raw is not None else ""
            if not origin:
                origin = self._compose_origin_key(did, field)

            if did:
                for ent_uri in (src, tgt):
                    if ent_uri:
                        ent_counter = entity_did_counts.setdefault(ent_uri, {})
                        _inc(ent_counter, did)
            if origin:
                for ent_uri in (src, tgt):
                    if ent_uri:
                        origin_counter = entity_origin_counts.setdefault(ent_uri, {})
                        _inc(origin_counter, origin)
            if field:
                for ent_uri in (src, tgt):
                    if ent_uri:
                        field_counter = entity_field_counts.setdefault(ent_uri, {})
                        _inc(field_counter, field)

        inferred = self.reasoner.infer_relations(all_relations)
        self.canonical_linker.rewrite_relation_endpoints(inferred)
        for rel in inferred:
            meta = getattr(rel, "metadata", None) or {}
            src = str(getattr(rel, "source", "") or "")
            tgt = str(getattr(rel, "target", "") or "")

            did_raw = meta.get("discourse_id") or meta.get("doc_id") or getattr(rel, "doc_id", None)
            did = str(did_raw).strip() if did_raw is not None else ""
            if not did:
                did = str(_best_key(entity_did_counts.get(src)) or _best_key(entity_did_counts.get(tgt)) or "").strip()

            origin_raw = meta.get("origin_key")
            origin = str(origin_raw).strip() if origin_raw is not None else ""
            if not origin:
                origin = str(
                    _best_key(entity_origin_counts.get(src))
                    or _best_key(entity_origin_counts.get(tgt))
                    or ""
                ).strip()

            field_raw = meta.get("field")
            field = str(field_raw).strip() if field_raw is not None else ""
            if not field:
                field = str(
                    _best_key(entity_field_counts.get(src))
                    or _best_key(entity_field_counts.get(tgt))
                    or ""
                ).strip()
            if not field and origin and "|" in origin:
                field = origin.split("|", 1)[0].strip()
            if not origin:
                origin = self._compose_origin_key(did, field)

            if did:
                meta["discourse_id"] = did
            if field:
                meta["field"] = field
            if origin:
                meta["origin_key"] = origin
            meta.setdefault("extractor", "reasoner")
            meta.setdefault("context_source", "reasoner_inference")
            rel.metadata = meta
            self._ensure_relation_chunk_provenance(
                rel,
                fallback_discourse_id=did or getattr(rel, "doc_id", None),
                chunk_spans=self._get_chunk_spans_for_discourse(did, origin_key=origin),
            )
        print(f"  âœ“ Inferred {len(inferred)} new relations")
        return inferred


    def detect_conflicts(self, annotations: Any) -> List[Dict[str, Any]]:
        """Detect conflicts in the knowledge graph.

        Supports:
          - AnnotationBatch (has to_dict())
          - dict (already serialized annotations)
        """
        if not self.config.USE_CONFLICT_DETECTION:
            return []

        print("\nðŸ” Detecting conflicts...")

        # ---------------------------------------------------------
        # Normalize input to dict
        # ---------------------------------------------------------
        if isinstance(annotations, dict):
            ann_dict = annotations
        elif hasattr(annotations, "to_dict") and callable(getattr(annotations, "to_dict")):
            ann_dict = annotations.to_dict()
        else:
            raise TypeError(
                f"detect_conflicts() expected dict or AnnotationBatch, got {type(annotations)}"
            )

        conflicts = self.conflict_detector.detect(ann_dict)

        if conflicts:
            print(f"  âš ï¸  Found {len(conflicts)} conflicts")
        else:
            print("  âœ“ No conflicts detected")

        return conflicts


    def compute_quality_metrics(self, annotations: Any) -> dict:
        """Compute quality metrics.

        Supports:
          - AnnotationBatch (has to_dict())
          - dict (already serialized annotations)
        """
        print("\nðŸ“Š Computing quality metrics...")

        # ---------------------------------------------------------
        # Normalize input to dict
        # ---------------------------------------------------------
        if isinstance(annotations, dict):
            ann_dict = annotations
        elif hasattr(annotations, "to_dict") and callable(getattr(annotations, "to_dict")):
            ann_dict = annotations.to_dict()
        else:
            raise TypeError(
                f"compute_quality_metrics() expected dict or AnnotationBatch, got {type(annotations)}"
            )

        metrics_computer = QualityMetricsComputer(
            ann_dict,
            self.global_registry
        )

        metrics = metrics_computer.compute_all()
        metrics["provenance_diagnostics"] = dict(getattr(self, "_provenance_diagnostics", {}))
        summary = metrics_computer.get_summary_metrics()

        print(f"\nðŸ“ˆ Quality Metrics:")
        print(f"  Overall Quality Score: {summary.overall_quality_score}/10")
        print(f"  Total Entities: {summary.total_entities}")
        print(f"  Total Relations: {summary.total_relations}")
        print(f"  Entity Coverage: {summary.entity_coverage:.1%}")
        print(f"  Relation Coverage: {summary.relation_coverage:.1%}")

        return metrics












