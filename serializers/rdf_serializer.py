from __future__ import annotations

import ast
import os
import re
import uuid
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, Optional, List, Set, Union, Tuple

from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS, OWL, SKOS, XSD, DCTERMS

from models import Relation
from utils.helpers import canonicalize_local_name, make_uri_safe, safe_hash
from ontology.add_schema import add_schema, canonicalize_class_local
from ontology.property_registry import is_registered_etd_predicate
from ontology.semantic_typing import SemanticTyper

logger = logging.getLogger(__name__)
PROV = Namespace("http://www.w3.org/ns/prov#")


def consolidate_duplicate_reified_relations(graph: Graph, etd_ns: Namespace) -> Dict[str, int]:
    """
    Consolidate duplicate reified ETD relations with identical (subject, predicate, object).

    Deterministic winner selection:
      1) highest confidence
      2) prefer isInferred=false over true
      3) lexicographically smallest relation URI

    Evidence links from losing relations are preserved by attaching them to the winner.
    """
    rel_type = etd_ns.Relation
    p_subject = etd_ns.subject
    p_predicate = etd_ns.predicate
    p_object = etd_ns.object
    p_confidence = etd_ns.confidence
    p_is_inferred = etd_ns.isInferred
    p_has_evidence = etd_ns.hasEvidence
    p_discourse_id = etd_ns.discourse_id
    p_field = etd_ns.field
    p_extracted_from = etd_ns.extractedFrom

    def _single_iri(rel_node: URIRef, pred: URIRef) -> Optional[URIRef]:
        vals = [v for v in graph.objects(rel_node, pred) if isinstance(v, URIRef)]
        if len(vals) != 1:
            return None
        return vals[0]

    def _confidence(rel_node: URIRef) -> float:
        vals = []
        for v in graph.objects(rel_node, p_confidence):
            try:
                vals.append(float(v))
            except Exception:
                continue
        return max(vals) if vals else 0.0

    def _is_inferred(rel_node: URIRef) -> bool:
        vals = list(graph.objects(rel_node, p_is_inferred))
        for v in vals:
            if str(v).strip().lower() in {"true", "1"}:
                return True
        return False

    def _literal_values(node: URIRef, pred: URIRef) -> List[str]:
        out: Set[str] = set()
        for v in graph.objects(node, pred):
            if isinstance(v, Literal):
                s = str(v).strip()
                if s:
                    out.add(s)
        return sorted(out)

    def _iri_values(node: URIRef, pred: URIRef) -> List[str]:
        out: Set[str] = set()
        for v in graph.objects(node, pred):
            if isinstance(v, URIRef):
                out.add(str(v))
        return sorted(out)

    def _provenance_signature(rel_node: URIRef) -> Tuple[str, str, str, str, str, str]:
        rel_dids = _literal_values(rel_node, p_discourse_id)
        rel_fields = _literal_values(rel_node, p_field)
        rel_origins = _iri_values(rel_node, p_extracted_from)

        ev_dids: Set[str] = set()
        ev_fields: Set[str] = set()
        ev_origins: Set[str] = set()
        for ev in graph.objects(rel_node, p_has_evidence):
            if not isinstance(ev, URIRef):
                continue
            ev_dids.update(_literal_values(ev, p_discourse_id))
            ev_fields.update(_literal_values(ev, p_field))
            ev_origins.update(_iri_values(ev, p_extracted_from))

        return (
            "|".join(rel_dids),
            "|".join(rel_fields),
            "|".join(rel_origins),
            "|".join(sorted(ev_dids)),
            "|".join(sorted(ev_fields)),
            "|".join(sorted(ev_origins)),
        )

    groups: Dict[Tuple[str, str, str], List[URIRef]] = {}
    skipped = 0

    for rel in graph.subjects(RDF.type, rel_type):
        if not isinstance(rel, URIRef):
            skipped += 1
            continue

        s = _single_iri(rel, p_subject)
        p = _single_iri(rel, p_predicate)
        o = _single_iri(rel, p_object)
        if s is None or p is None or o is None:
            skipped += 1
            logger.warning("Skipping relation with missing/ambiguous SPO: %s", rel)
            continue

        key = (str(s), str(p), str(o), *_provenance_signature(rel))
        groups.setdefault(key, []).append(rel)

    duplicate_groups = 0
    removed = 0
    evidence_moved = 0

    for _, rel_nodes in groups.items():
        unique_rels = sorted(set(rel_nodes), key=str)
        if len(unique_rels) <= 1:
            continue

        duplicate_groups += 1

        winner = sorted(
            unique_rels,
            key=lambda r: (-_confidence(r), 0 if not _is_inferred(r) else 1, str(r)),
        )[0]

        losers = [r for r in unique_rels if r != winner]
        for loser in losers:
            for ev in list(graph.objects(loser, p_has_evidence)):
                if (winner, p_has_evidence, ev) not in graph:
                    graph.add((winner, p_has_evidence, ev))
                    evidence_moved += 1

            for t in list(graph.triples((loser, None, None))):
                graph.remove(t)
            for t in list(graph.triples((None, None, loser))):
                graph.remove(t)
            removed += 1

    return {
        "duplicate_spo_groups_count": duplicate_groups,
        "relations_removed_count": removed,
        "evidence_links_moved_count": evidence_moved,
        "rels_skipped_missing_spo_count": skipped,
    }


def _uri_localname(node: URIRef) -> str:
    s = str(node)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _first_literal_for_predicate(graph: Graph, subject: URIRef, predicate: URIRef) -> Optional[str]:
    vals = sorted(
        [str(o).strip() for o in graph.objects(subject, predicate) if isinstance(o, Literal) and str(o).strip()],
        key=lambda x: x.lower(),
    )
    return vals[0] if vals else None


def _normalized_label_key(label: str) -> str:
    """
    Normalized key for alias equivalence checks only.
    Keeps display labels untouched and deterministic.
    """
    s = unicodedata.normalize("NFKC", str(label or ""))
    s = s.casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def enforce_single_en_labels(graph: Graph, etd_ns: Namespace) -> Dict[str, int]:
    """
    Guarantee exactly one rdfs:label@en for ETD Entity/Vote/Relation/Evidence/Export/FieldAssertion nodes.
    Existing labels are normalized to a single deterministic @en literal.
    """
    target_types = [
        etd_ns.Entity,
        etd_ns.DiscourseUnit,
        etd_ns.Vote,
        etd_ns.Relation,
        etd_ns.Evidence,
        etd_ns.Export,
        etd_ns.FieldAssertion,
    ]
    nodes: Set[URIRef] = set()
    for t in target_types:
        for n in graph.subjects(RDF.type, t):
            if isinstance(n, URIRef):
                nodes.add(n)

    # One-pass index for reified vote edges to avoid O(V*R) scans.
    reified_cast_by: Dict[URIRef, Set[URIRef]] = {}
    reified_receives: Dict[URIRef, Set[URIRef]] = {}
    for rel in graph.subjects(RDF.type, etd_ns.Relation):
        if not isinstance(rel, URIRef):
            continue
        subj = next((o for o in graph.objects(rel, etd_ns.subject) if isinstance(o, URIRef)), None)
        pred = next((o for o in graph.objects(rel, etd_ns.predicate) if isinstance(o, URIRef)), None)
        obj = next((o for o in graph.objects(rel, etd_ns.object) if isinstance(o, URIRef)), None)
        if subj is None or pred is None or obj is None:
            continue
        if pred == etd_ns.castBy:
            reified_cast_by.setdefault(subj, set()).add(obj)
        elif pred == etd_ns.receivesVote:
            reified_receives.setdefault(subj, set()).add(obj)

    converted = 0
    generated = 0
    deduped = 0

    for node in sorted(nodes, key=str):
        classes = {c for c in graph.objects(node, RDF.type) if isinstance(c, URIRef)}
        is_relation = etd_ns.Relation in classes
        is_evidence = etd_ns.Evidence in classes
        is_vote = etd_ns.Vote in classes
        is_discourse_unit = etd_ns.DiscourseUnit in classes
        is_export = etd_ns.Export in classes
        is_field_assertion = etd_ns.FieldAssertion in classes

        existing = [o for o in graph.objects(node, RDFS.label) if isinstance(o, Literal)]
        en_labels = sorted(
            [str(l).strip() for l in existing if str(l).strip() and str(getattr(l, "language", "")).lower() == "en"],
            key=lambda x: x.lower(),
        )
        any_labels = sorted([str(l).strip() for l in existing if str(l).strip()], key=lambda x: x.lower())

        if len(existing) == 1 and len(en_labels) == 1:
            # Already in canonical shape: exactly one label@en.
            continue

        chosen: Optional[str] = None
        if en_labels:
            chosen = en_labels[0]
            if len(en_labels) > 1 or len(existing) > 1:
                deduped += 1
        elif any_labels:
            chosen = any_labels[0]
            converted += 1
            if len(any_labels) > 1:
                deduped += 1
        else:
            if is_relation:
                s = next((o for o in graph.objects(node, etd_ns.subject) if isinstance(o, URIRef)), None)
                p = next((o for o in graph.objects(node, etd_ns.predicate) if isinstance(o, URIRef)), None)
                o = next((o for o in graph.objects(node, etd_ns.object) if isinstance(o, URIRef)), None)
                pred_l = _uri_localname(p) if p is not None else "unknownPredicate"
                subj_l = _uri_localname(s) if s is not None else "unknownSubject"
                obj_l = _uri_localname(o) if o is not None else "unknownObject"
                chosen = f"Relation: {pred_l} ({subj_l} -> {obj_l})"
            elif is_evidence:
                snippet = _first_literal_for_predicate(graph, node, etd_ns.snippet_id)
                base = snippet or _uri_localname(node)
                if not snippet:
                    source = _first_literal_for_predicate(graph, node, etd_ns.source)
                    if source:
                        base = source
                base = (base or _uri_localname(node)).strip()
                prefix = "Evidence: "
                max_total = 80
                max_base = max_total - len(prefix)
                if max_base < 4:
                    max_base = 4
                if len(base) > max_base:
                    base = base[: max_base - 3] + "..."
                chosen = f"{prefix}{base}"
            elif is_vote:
                cast = next((o for o in graph.objects(node, etd_ns.castBy) if isinstance(o, URIRef)), None)
                recv = next((o for o in graph.objects(node, etd_ns.receivesVote) if isinstance(o, URIRef)), None)
                if cast is None:
                    cands = sorted(reified_cast_by.get(node, set()), key=str)
                    cast = cands[0] if cands else None
                if recv is None:
                    cands = sorted(reified_receives.get(node, set()), key=str)
                    recv = cands[0] if cands else None
                if cast is not None and recv is not None:
                    chosen = f"Vote by {_uri_localname(cast)} for {_uri_localname(recv)}"
                else:
                    chosen = _uri_localname(node)
            elif is_discourse_unit:
                chosen = f"Discourse unit: {_uri_localname(node)}"
            elif is_export:
                ds = _first_literal_for_predicate(graph, node, etd_ns.datasetId)
                run = _first_literal_for_predicate(graph, node, etd_ns.runId)
                if ds and run:
                    chosen = f"Export: {ds} ({run})"
                else:
                    chosen = _uri_localname(node)
            elif is_field_assertion:
                source_section = _first_literal_for_predicate(graph, node, etd_ns.sourceSection)
                field_name = _first_literal_for_predicate(graph, node, etd_ns.fieldName)
                record_id = _first_literal_for_predicate(graph, node, etd_ns.recordId)
                if source_section and field_name:
                    chosen = f"Field assertion: {source_section}.{field_name}"
                elif field_name:
                    chosen = f"Field assertion: {field_name}"
                elif record_id:
                    chosen = f"Field assertion: record {record_id}"
                else:
                    chosen = f"Field assertion: {_uri_localname(node)}"
            else:
                fallback: Optional[str] = None
                for p, o in graph.predicate_objects(node):
                    if not isinstance(p, URIRef) or not isinstance(o, Literal):
                        continue
                    local = _uri_localname(p).lower()
                    if local in {"name", "title"} and str(o).strip():
                        val = str(o).strip()
                        if fallback is None or val.lower() < fallback.lower():
                            fallback = val
                chosen = fallback or _uri_localname(node)
            generated += 1

        assert chosen is not None
        for old in list(graph.objects(node, RDFS.label)):
            graph.remove((node, RDFS.label, old))
        graph.add((node, RDFS.label, Literal(chosen, lang="en")))

    return {
        "nodes_processed": len(nodes),
        "labels_generated": generated,
        "labels_converted_to_en": converted,
        "labels_deduped": deduped,
    }


def prune_isolated_entities(graph: Graph, etd_ns: Namespace) -> Dict[str, int]:
    """
    Remove isolated etd:Entity nodes not referenced by any reified relation
    as etd:subject or etd:object.
    """
    p_subject = etd_ns.subject
    p_object = etd_ns.object
    t_entity = etd_ns.Entity
    t_relation = etd_ns.Relation

    connected: Set[URIRef] = set()
    for rel in graph.subjects(RDF.type, t_relation):
        for node in graph.objects(rel, p_subject):
            if isinstance(node, URIRef):
                connected.add(node)
        for node in graph.objects(rel, p_object):
            if isinstance(node, URIRef):
                connected.add(node)

    isolated: Set[URIRef] = set()
    for ent in graph.subjects(RDF.type, t_entity):
        if isinstance(ent, URIRef) and ent not in connected:
            isolated.add(ent)

    triples_to_remove: Set[Tuple[Node, Node, Node]] = set()
    for ent in isolated:
        for t in graph.triples((ent, None, None)):
            triples_to_remove.add(t)
        for t in graph.triples((None, None, ent)):
            triples_to_remove.add(t)

    for t in triples_to_remove:
        graph.remove(t)

    return {
        "isolated_entities_found": len(isolated),
        "isolated_entities_removed": len(isolated),
        "triples_removed": len(triples_to_remove),
    }

class RDFSerializer:
    """
    Serialize ETD-Hub KG into RDF formats using canonical URIs.
    """

    def __init__(
        self,
        annotations_dict: Dict[str, Any],
        registry: Any = None,
        namespaces: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        # ---- 1) Normalise annotations -----------------------------------------
        if hasattr(annotations_dict, "to_dict"):
            annotations_dict = annotations_dict.to_dict()

        if not isinstance(annotations_dict, dict):
            raise TypeError(
                f"annotations_dict must be a dict, got {type(annotations_dict)}"
            )

        self.annotations: Dict[str, Any] = annotations_dict

        # ---- 2) Registry -------------------------------------------------------
        if registry is None and "registry" in kwargs:
            registry = kwargs["registry"]

        self.registry = registry

        # ---- 3) Namespaces -----------------------------------------------------
        config = kwargs.get("config")
        self.config = config
        if config is None:
            try:
                from config import Config as _RuntimeConfig  # lazy import to avoid hard dependency at module import

                self.config = _RuntimeConfig
            except Exception:
                self.config = None

        prune_override = kwargs.get("prune_isolated_entities")
        if prune_override is None:
            self.prune_isolated_entities = bool(getattr(self.config, "PRUNE_ISOLATED_ENTITIES", True))
        else:
            self.prune_isolated_entities = bool(prune_override)
        self.strict_relation_backbone = bool(getattr(self.config, "STRICT_RELATION_BACKBONE", False))
        self.strict_etd_property_registry = bool(getattr(self.config, "STRICT_ETD_PROPERTY_REGISTRY", True))
        self.materialize_raw_fields = bool(getattr(self.config, "MATERIALIZE_RAW_FIELDS", True))
        self.materialize_sensitive_fields = bool(getattr(self.config, "MATERIALIZE_SENSITIVE_FIELDS", False))
        self.media_base_url = str(getattr(self.config, "MEDIA_BASE_URL", "") or "").strip()
        self._raw_datetime_warning_keys: Set[str] = set()

        if namespaces is None:
            if "namespaces" in kwargs and kwargs["namespaces"] is not None:
                namespaces = kwargs["namespaces"]
            elif config is not None and hasattr(config, "NAMESPACES"):
                namespaces = getattr(config, "NAMESPACES")

        if namespaces is None:
            raise ValueError("RDFSerializer requires a namespaces dict (e.g., Config.NAMESPACES)")

        self.namespaces: Dict[str, str] = namespaces
        # Export metadata must be initialized once after config/prune resolution.
        self._init_export_metadata(kwargs)
        etd_base = str(self.namespaces["etd"])
        self.export_node = URIRef(
            f"{etd_base}export_{safe_hash(self.dataset_id + '|' + self.run_id)}"
        )

        # ---- 4) RDF Graph ------------------------------------------------------
        self.graph: Graph = Graph()

        # Bind namespaces cleanly
        for pfx, uri in self.namespaces.items():
            self.graph.bind(pfx, Namespace(uri))

        # Bind SKOS prefix for readability in TTL output
        self.graph.bind("skos", SKOS)
        # Guarantee canonical PROV namespace binding for provenance triples.
        self.graph.bind("prov", PROV)

        # Initialize Fix 6 LLM Semantic Refiner
        #self.semantic_refiner = SemanticRefiner(self.graph, Namespace(self.namespaces["etd"]))
        #self.semantic_refiner = SemanticRefiner()
        #self.semantic_refiner = None

        # ---- 5) Build entity index ---------------------------------------------
        # Key → entity (label, id, uri)
        self.entity_index: Dict[str, Any] = self._build_entity_index()

        # Track which ETD classes we actually used, for Option A alignment
        self._seen_entity_classes: Set[str] = set()
        self._class_alignment_targets: Dict[str, URIRef] = {}

        # --- Add schema early ---
        #etd_ns = Namespace(self.namespaces["etd"])
        #add_schema(self.graph, etd_ns)

        #add_schema(self.graph, Namespace(self.namespaces["etd"]))

        #self.semantic_typer = SemanticTyper(self.graph, Namespace(self.namespaces["etd"]))
        
        # --- Add schema early (single call) ---
        etd_ns = Namespace(self.namespaces["etd"])
        allowed = add_schema(self.graph, etd_ns)   # add_schema returns set[str]
        self.semantic_typer = SemanticTyper(self.graph, etd_ns, allowed_classes=allowed)



        # ---- 6) Construct RDF graph --------------------------------------------
        self._add_ontology_header()
        self._add_entities()
        self._external_to_internal_index = self._build_external_to_internal_uri_index(etd_ns)
        self._add_alignment_axioms()   # Option A: minimal conceptual ontology alignment
        self._add_relations()
        self._assert_relation_spine_atomicity(etd_ns)
        self.duplicate_relation_consolidation = consolidate_duplicate_reified_relations(self.graph, etd_ns)
        logger.info("Duplicate reified relation consolidation: %s", self.duplicate_relation_consolidation)
        self.relation_evidence_alignment = self._align_relation_evidence_provenance(etd_ns)
        logger.info("Relation/evidence provenance alignment: %s", self.relation_evidence_alignment)
        # Create export metadata node before label enforcement so Export always gets rdfs:label@en.
        self._add_dataset_metadata()
        self.final_relation_provenance = self._finalize_relation_provenance(etd_ns)
        logger.info("Final relation provenance completion: %s", self.final_relation_provenance)
        self.label_enforcement = enforce_single_en_labels(self.graph, etd_ns)
        logger.info("Label enforcement metrics: %s", self.label_enforcement)
        if self.prune_isolated_entities:
            self.isolated_entity_prune_metrics = prune_isolated_entities(self.graph, etd_ns)
        else:
            self.isolated_entity_prune_metrics = {
                "isolated_entities_found": 0,
                "isolated_entities_removed": 0,
                "triples_removed": 0,
            }
        logger.info("Isolated entity prune metrics: %s", self.isolated_entity_prune_metrics)

    def _init_export_metadata(self, kwargs: Dict[str, Any]) -> None:
        """
        Initialize run-scoped export metadata once per serializer instance.
        """
        meta = self.annotations.get("metadata", {}) or {}
        input_path = (
            kwargs.get("input_path")
            or getattr(self.config, "INPUT_JSON", "")
            or meta.get("input_path")
            or "unknown_input"
        )
        dataset_id = kwargs.get("dataset_id") or meta.get("datasetId") or Path(str(input_path)).stem
        run_id = kwargs.get("run_id") or meta.get("runId")
        if not run_id:
            now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"run-{safe_hash(str(dataset_id) + '|' + now)}"

        self.dataset_id = str(dataset_id)
        self.run_id = str(run_id)
        self.generated_at_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.pipeline_version = str(
            kwargs.get("pipeline_version")
            or getattr(self.config, "PIPELINE_VERSION", None)
            or "5.1"
        )
        git_commit = kwargs.get("git_commit") or os.getenv("GIT_COMMIT")
        if not git_commit:
            try:
                import subprocess

                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode("utf-8").strip()
            except Exception:
                git_commit = "unknown"
        self.git_commit = str(git_commit)
        self.shacl_profile_version = str(
            kwargs.get("shacl_profile_version")
            or getattr(self.config, "SHACL_SHAPES_PATH", None)
            or "ontology/shacl_shapes.ttl"
        )
        self.export_mode = f"ai;prune_isolated_entities={str(bool(self.prune_isolated_entities)).lower()}"

    # ------------------------------------------------------------------
    # Internal: entity index + resolution
    # ------------------------------------------------------------------
    def _build_entity_index(self) -> Dict[str, Any]:
        """
        Build a lookup from various keys → entity objects to
        resolve literals / IDs back to canonical entity URIs.

        Keys we store:
          - label.lower()
          - name.lower()
          - str(ent.id) / str(ent.idx) / str(ent.index) if present
          - str(ent.uri)
        """
        index: Dict[str, Any] = {}

        if self.registry is None:
            return index

        get_all = getattr(self.registry, "get_all_entities", None)
        if not callable(get_all):
            return index

        for ent in get_all():
            label = getattr(ent, "label", None) or getattr(ent, "name", None)
            if label:
                key = label.strip()
                if key:
                    index[key.lower()] = ent

            # Potential numeric / string identifiers
            for attr in ("id", "idx", "index"):
                val = getattr(ent, attr, None)
                if val is not None:
                    index[str(val)] = ent

            # Also index by canonical URI string if available
            ent_uri = getattr(ent, "uri", None)
            if ent_uri:
                index[str(ent_uri).strip()] = ent

        return index

    def _resolve_entity_uri_from_value(self, value: Any) -> Optional[URIRef]:
        """
        Try hard to map a value to an existing entity's URIRef:

        1. If it matches an existing entity label/name/id/uri → entity.uri
        2. If it looks like a full HTTP(S) IRI → safe URIRef
        3. If it looks like a CURIE with known prefix → expanded safe URIRef

        Otherwise returns None (caller decides what to do).
        """
        # Already a URIRef
        if isinstance(value, URIRef):
            return value

        if not isinstance(value, str):
            return None

        s = value.strip()
        if not s:
            return None

        # Entity index hit: try as-is, then lowercase (labels)
        ent = self.entity_index.get(s) or self.entity_index.get(s.lower())
        if ent is not None and getattr(ent, "uri", None):
            return _safe_uriref(str(ent.uri), self.namespaces)

        # HTTP(S) IRI
        if s.startswith("http://") or s.startswith("https://"):
            return _safe_uriref(s, self.namespaces)

        # CURIE prefix:local with known prefix
        if ":" in s and " " not in s:
            prefix, local = s.split(":", 1)
            if prefix in self.namespaces:
                iri = self.namespaces[prefix] + local
                return _safe_uriref(iri, self.namespaces)

        return None

    def _is_internal_etd_uri(self, uri: URIRef) -> bool:
        return isinstance(uri, URIRef) and str(uri).startswith(str(self.namespaces["etd"]))

    def _extract_wikidata_qid(self, uri: URIRef) -> Optional[str]:
        s = str(uri).strip()
        if not s:
            return None
        m = re.search(r"(Q\d+)$", s, flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(1).upper()

    def _build_external_to_internal_uri_index(self, etd_ns: Namespace) -> Dict[str, List[URIRef]]:
        """
        Build a deterministic map external-identity-URI -> canonical internal ETD URI candidates.
        """
        index: Dict[str, Set[URIRef]] = {}

        def _add(ext_uri: URIRef, internal_uri: URIRef) -> None:
            if not isinstance(ext_uri, URIRef) or not isinstance(internal_uri, URIRef):
                return
            if not self._is_internal_etd_uri(internal_uri):
                return
            key = str(ext_uri)
            index.setdefault(key, set()).add(internal_uri)
            qid = self._extract_wikidata_qid(ext_uri)
            if qid:
                index.setdefault(f"http://www.wikidata.org/entity/{qid}", set()).add(internal_uri)
                index.setdefault(f"https://www.wikidata.org/entity/{qid}", set()).add(internal_uri)

        for pred in (OWL.sameAs, etd_ns.wikidata):
            for subj, obj in self.graph.subject_objects(pred):
                if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                    _add(obj, subj)

        # Registry-backed fallback mapping for cases where sameAs isn't materialized yet.
        get_all = getattr(self.registry, "get_all_entities", None)
        if callable(get_all):
            for ent in get_all():
                ent_uri = getattr(ent, "uri", None)
                if not ent_uri:
                    continue
                internal = _safe_uriref(str(ent_uri), self.namespaces)
                if not self._is_internal_etd_uri(internal):
                    continue
                meta = getattr(ent, "metadata", None) or {}
                ext_candidates: List[str] = []
                wdu = meta.get("wikidata_uri")
                if isinstance(wdu, str) and wdu.strip():
                    ext_candidates.append(wdu.strip())
                wd = meta.get("wikidata")
                if isinstance(wd, dict):
                    qid = wd.get("qid")
                    if isinstance(qid, str) and qid.strip():
                        ext_candidates.append(f"http://www.wikidata.org/entity/{qid.strip()}")
                        ext_candidates.append(f"https://www.wikidata.org/entity/{qid.strip()}")
                for ext in ext_candidates:
                    _add(_safe_uriref(ext, self.namespaces), internal)

        return {k: sorted(v, key=str) for k, v in index.items()}

    def _canonicalize_relation_endpoint_uri(self, endpoint: Any) -> Any:
        """
        Canonicalize relation endpoints:
        - keep non-URI nodes unchanged
        - keep internal ETD URIs unchanged
        - replace external identity URI with canonical internal ETD URI when mapped
        """
        if not isinstance(endpoint, URIRef):
            return endpoint
        if self._is_internal_etd_uri(endpoint):
            return endpoint

        candidates = self._external_to_internal_index.get(str(endpoint), [])
        if not candidates:
            qid = self._extract_wikidata_qid(endpoint)
            if qid:
                candidates = (
                    self._external_to_internal_index.get(f"http://www.wikidata.org/entity/{qid}", [])
                    or self._external_to_internal_index.get(f"https://www.wikidata.org/entity/{qid}", [])
                )

        if candidates:
            chosen = sorted(candidates, key=str)[0]
            if len(candidates) > 1:
                logger.warning(
                    "Multiple internal mappings for external endpoint %s; chosen=%s candidates=%s",
                    endpoint,
                    chosen,
                    [str(c) for c in sorted(candidates, key=str)],
                )
            return chosen
        return endpoint

    def _ensure_relation_node_identity(
        self,
        rnode: URIRef,
        subj: Any,
        pred_uri: URIRef,
        obj: Any,
        etd_ns: Namespace,
    ) -> URIRef:
        """
        Reified relation backbone is immutable:
        if rnode already carries conflicting SPO, remint deterministically
        (or raise in strict mode).
        """
        existing_s = sorted(set(self.graph.objects(rnode, etd_ns.subject)), key=str)
        existing_p = sorted(set(self.graph.objects(rnode, etd_ns.predicate)), key=str)
        existing_o = sorted(set(self.graph.objects(rnode, etd_ns.object)), key=str)
        if not existing_s and not existing_p and not existing_o:
            return rnode

        same_spo = (
            len(existing_s) == 1 and existing_s[0] == subj and
            len(existing_p) == 1 and existing_p[0] == pred_uri and
            len(existing_o) == 1 and existing_o[0] == obj
        )
        if same_spo:
            return rnode

        msg = (
            f"Conflicting SPO write for relation node {rnode}: "
            f"existing_s={existing_s}, existing_p={existing_p}, existing_o={existing_o}, "
            f"new=({subj}, {pred_uri}, {obj})"
        )
        if self.strict_relation_backbone:
            raise RuntimeError(msg)

        base = self.namespaces["etd"]
        seed = f"{str(rnode)}|{self._spo_key(subj, pred_uri, obj)}"
        salt = 0
        while True:
            suffix = "" if salt == 0 else f"|{salt}"
            candidate = URIRef(f"{base}rel_{safe_hash(seed + suffix)}")
            c_s = sorted(set(self.graph.objects(candidate, etd_ns.subject)), key=str)
            c_p = sorted(set(self.graph.objects(candidate, etd_ns.predicate)), key=str)
            c_o = sorted(set(self.graph.objects(candidate, etd_ns.object)), key=str)
            if not c_s and not c_p and not c_o:
                logger.warning("%s. Reminted relation URI: %s", msg, candidate)
                return candidate
            if (
                len(c_s) == 1 and c_s[0] == subj and
                len(c_p) == 1 and c_p[0] == pred_uri and
                len(c_o) == 1 and c_o[0] == obj
            ):
                return candidate
            salt += 1

    def _assert_relation_spine_atomicity(self, etd_ns: Namespace) -> None:
        """
        Fail-fast structural guard: every reified relation must have exactly one S/P/O.
        """
        violations: List[str] = []
        for rnode in sorted(self.graph.subjects(RDF.type, etd_ns.Relation), key=str):
            if not isinstance(rnode, URIRef):
                continue
            s_count = len(set(self.graph.objects(rnode, etd_ns.subject)))
            p_count = len(set(self.graph.objects(rnode, etd_ns.predicate)))
            o_count = len(set(self.graph.objects(rnode, etd_ns.object)))
            if s_count != 1 or p_count != 1 or o_count != 1:
                violations.append(
                    f"{rnode} subjectCount={s_count} predicateCount={p_count} objectCount={o_count}"
                )
        if violations:
            sample = "; ".join(violations[:10])
            raise RuntimeError(
                f"Relation spine cardinality violation ({len(violations)}): {sample}"
            )

    # ------------------------------------------------------------------
    # Ontology header
    # ------------------------------------------------------------------
    def _add_ontology_header(self) -> None:
        """Ensure a single canonical ontology resource exists."""
        graph = self.graph
        ns = self.namespaces

        etd_ns = Namespace(ns["etd"])

        # Canonical ontology IRI: <ETD_NS>ontology
        onto_uri = etd_ns["ontology"]

        # add_schema() already materialises ontology metadata on etd:ontology.
        # Here we only ensure ontology typing exists if schema injection changes.
        graph.add((onto_uri, RDF.type, OWL.Ontology))


    def _maybe_record_alignment_target(self, class_local: str, external_uri: URIRef) -> None:
        """
        Record that ETD class `class_local` should be aligned to `external_uri`.
        Only records mappings for recognised external ontology namespaces.
        """
        if not isinstance(external_uri, URIRef):
            return

        external_prefixes = {
            "dcat",
            "schema",
            "aipo",
            "relaieo",
            "vair",
            "airo",
            "hudock",
            "fmo",
            "dpv",
        }

        uri_str = str(external_uri)
        for pfx in external_prefixes:
            base = self.namespaces.get(pfx)
            if base and uri_str.startswith(base):
                if class_local not in self._class_alignment_targets:
                    self._class_alignment_targets[class_local] = external_uri
                return

    def _add_alignment_axioms(self) -> None:
        """
        Option A: Minimal conceptual ontology alignment.

        Primary strategy (fixes case-mismatch):
        ---------------------------------------
        For each ETD class where we have seen an individual with an
        `etd:ontology_uri`, we use that URI as the rdfs:subClassOf target:

            etd:<ClassLocal> rdfs:subClassOf <external-class-IRI> .

        Fallback strategy (heuristic, only when no ontology_uri-based
        mapping is available):
        -------------------------------------------------------------
        For each ETD class name matching <Prefix>_<Local> where
        Prefix.lower() is a known external namespace, we add:

            etd:<Prefix_Local> rdfs:subClassOf <prefix>:<Local> .
        """
        etd_ns = Namespace(self.namespaces["etd"])

        # 1) Preferred: use mappings learned from ontology_uri on entities
        for class_local, external_uri in sorted(self._class_alignment_targets.items()):
            etd_class_uri = etd_ns[class_local]
            # subclass to the exact external URI we observed (correct case)
            self.graph.add((etd_class_uri, RDFS.subClassOf, external_uri))
            # explicitly type the ETD proxy as a class
            self.graph.add((etd_class_uri, RDF.type, OWL.Class))
            self._annotate_dynamic_etd_class(etd_class_uri, class_local)

        # 2) Fallback: heuristic from class_local if we have no explicit target
        external_prefixes = {
            "dcat",
            "schema",
            "aipo",
            "relaieo",
            "vair",
            "airo",
            "hudock",
            "fmo",
            "dpv",
        }

        for class_local in sorted(self._seen_entity_classes):
            # Skip classes already covered by explicit ontology_uri mapping
            if class_local in self._class_alignment_targets:
                continue

            if "_" not in class_local:
                continue

            prefix, local = class_local.split("_", 1)
            pfx_lower = prefix.lower()

            if pfx_lower not in external_prefixes:
                continue

            base = self.namespaces.get(pfx_lower)
            if not base:
                continue

            external_class_uri = URIRef(base + local)
            etd_class_uri = etd_ns[class_local]

            self.graph.add((etd_class_uri, RDFS.subClassOf, external_class_uri))
            self.graph.add((etd_class_uri, RDF.type, OWL.Class))
            self._annotate_dynamic_etd_class(etd_class_uri, class_local)

    def _annotate_dynamic_etd_class(self, cls_uri: URIRef, class_local: str) -> None:
        """
        Ensure classes minted during alignment remain self-describing.
        """
        if not any(True for _ in self.graph.objects(cls_uri, RDFS.label)):
            human = re.sub(r"[_\-]+", " ", str(class_local or "").strip())
            human = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", human)
            human = re.sub(r"\s+", " ", human).strip() or "Entity Class"
            label = " ".join(w.capitalize() for w in human.split())
            self.graph.add((cls_uri, RDFS.label, Literal(label, lang="en")))

        if not any(True for _ in self.graph.objects(cls_uri, RDFS.comment)):
            self.graph.add(
                (
                    cls_uri,
                    RDFS.comment,
                    Literal(
                        "ETD proxy class materialised during ontology alignment; used to preserve stable local typing.",
                        lang="en",
                    ),
                )
            )


    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------
    def _add_entities(self) -> None:
        """Serialise entities using canonical entity.uri and ontology classes."""
        if self.registry is None:
            return

        etd_ns = Namespace(self.namespaces["etd"])

        get_all = getattr(self.registry, "get_all_entities", None)
        if not callable(get_all):
            return

        for ent in get_all():
            uri_str = getattr(ent, "uri", None)
            if not uri_str:
                # Fallback: derive from label
                label = getattr(ent, "label", None) or getattr(ent, "name", None) or "Entity"
                local = self._sanitize(label)
                uri_str = f"{self.namespaces['etd']}{local}"

            uri = _safe_uriref(str(uri_str), self.namespaces)

            # FIX 1: ensure every entity is at least typed as etd:Entity
            self.graph.add((uri, RDF.type, etd_ns.Entity))


            # rdf:type based on entity_type
            entity_type = getattr(ent, "entity_type", None) or getattr(ent, "type", None)
            if entity_type:
                local_class = canonicalize_class_local(self._sanitize(str(entity_type)))
                self._seen_entity_classes.add(local_class)
                if local_class in self.semantic_typer.allowed_classes:
                    self.graph.add((uri, RDF.type, etd_ns[local_class]))


            # rdfs:label
            label = getattr(ent, "label", None) or getattr(ent, "name", None)
            if label:
                self.graph.add((uri, RDFS.label, Literal(label, lang="en")))
            meta_for_alias = getattr(ent, "metadata", None) or {}
            canonical_label = str(label or "").strip()
            canonical_key = _normalized_label_key(canonical_label) if canonical_label else ""
            alt_labels: Set[str] = set()
            hidden_labels: Set[str] = set()

            def _collect_alias_values(raw_val: Any, force_hidden: bool = False) -> None:
                if raw_val is None:
                    return
                if isinstance(raw_val, (list, tuple, set)):
                    values = list(raw_val)
                else:
                    values = [raw_val]
                for v in values:
                    s = str(v or "").strip()
                    if not s:
                        continue
                    if canonical_label and s == canonical_label:
                        continue
                    nkey = _normalized_label_key(s)
                    if not nkey:
                        continue
                    if force_hidden or (canonical_key and nkey == canonical_key):
                        hidden_labels.add(s)
                    else:
                        alt_labels.add(s)

            for k in ("aliases", "alias", "alt_labels", "alternative_labels"):
                _collect_alias_values(meta_for_alias.get(k), force_hidden=False)
            for k in ("hidden_labels", "noisy_aliases", "normalized_label", "label_normalized", "surface_form"):
                _collect_alias_values(meta_for_alias.get(k), force_hidden=True)

            for alt in sorted(alt_labels, key=lambda x: x.lower()):
                self.graph.add((uri, SKOS.altLabel, Literal(alt, lang="en")))
            for hidden in sorted(hidden_labels - alt_labels, key=lambda x: x.lower()):
                self.graph.add((uri, SKOS.hiddenLabel, Literal(hidden, lang="en")))
            
                # ----------------------------------------------------
                # Semantic Type Classification
                # ----------------------------------------------------
                try:
                    self.semantic_typer.classify(uri, label.lower().strip())
                except Exception as e:
                    logger.warning("Semantic typing failed for %s: %s", label, e)


                # --- Fix 6: LLM Semantic Refinement ---
                #refined = self.semantic_refiner.refine(uri, label)
                #final_type = refined.get("final_type")
                #if final_type:
                    #self.graph.add((uri, RDF.type, etd_ns[final_type]))


            # confidence (if any)
            conf = getattr(ent, "confidence", None)
            if conf is not None:
                self.graph.add((uri, etd_ns.confidence, self._confidence_literal(conf)))

            # ontology_uri (if mapped directly on entity)
            onto_uri = getattr(ent, "ontology_uri", None)
            metadata = getattr(ent, "metadata", None) or {}
            if not onto_uri:
                onto_uri = metadata.get("ontology_uri")
            if not onto_uri:
                oc = metadata.get("ontology_classes")
                if isinstance(oc, list) and oc:
                    onto_uri = oc[0]
            if onto_uri:
                # ----------------------------------------------------
                # FIX: prevent ontology_uri being a stringified list
                # e.g. "['dcat:Dataset']" -> "dcat:Dataset"
                # ----------------------------------------------------
                if isinstance(onto_uri, list) and onto_uri:
                    onto_uri = onto_uri[0]

                if isinstance(onto_uri, str):
                    s = onto_uri.strip()

                    # stringified list
                    if s.startswith("[") and s.endswith("]"):
                        try:
                            parsed = ast.literal_eval(s)
                            if isinstance(parsed, list) and parsed:
                                onto_uri = parsed[0]
                        except Exception:
                            pass

                    # final string cleanup
                    if isinstance(onto_uri, str):
                        onto_uri = onto_uri.strip()

                onto_ref: Optional[Union[URIRef, Literal]] = self._resolve_entity_uri_from_value(onto_uri)
                onto_str = str(onto_uri).strip() if onto_uri is not None else ""

                # Ensure ontology grounding is materialized whenever ontology_uri exists.
                # Prefer full IRI/CURIE expansion; fallback to a stable ETD class URI.
                if not isinstance(onto_ref, URIRef) and onto_str:
                    if onto_str.startswith("http://") or onto_str.startswith("https://"):
                        onto_ref = _safe_uriref(onto_str, self.namespaces)
                    elif ":" in onto_str and " " not in onto_str:
                        prefix, local = onto_str.split(":", 1)
                        if prefix in self.namespaces:
                            onto_ref = _safe_uriref(self.namespaces[prefix] + local, self.namespaces)
                    if not isinstance(onto_ref, URIRef):
                        onto_ref = URIRef(f"{self.namespaces['etd']}{self._sanitize(onto_str)}")

                if onto_ref is None:
                    onto_ref = Literal(str(onto_uri), datatype=XSD.string)

                self.graph.add((uri, etd_ns.ontology_uri, onto_ref))
                if isinstance(onto_ref, URIRef):
                    # Materialized ontology grounding for SPARQL type queries.
                    self.graph.add((uri, RDF.type, onto_ref))

                # Record alignment target if ontology_uri is a real external class URI
                if entity_type and isinstance(onto_ref, URIRef):
                    local_class = self._sanitize(str(entity_type))
                    self._maybe_record_alignment_target(local_class, onto_ref)
                src = metadata.get("ontology_mapping_source")
                if src:
                    self.graph.add((uri, etd_ns.ontologyMappingSource, Literal(str(src), datatype=XSD.string)))
                mconf = metadata.get("ontology_mapping_confidence")
                if mconf is not None:
                    try:
                        self.graph.add((uri, etd_ns.ontologyConfidence, Literal(float(mconf), datatype=XSD.float)))
                    except Exception:
                        pass


            # Entity-level metadata (including surface_form, context_source, discourse_id, etc.)
            self._add_entity_metadata(uri, metadata, etd_ns)
            raw_fields = metadata.get("raw_fields", metadata)
            kind_hint = str(entity_type or getattr(ent, "entity_type", "") or "").strip()
            self._emit_raw_fields(uri, kind_hint, raw_fields, etd_ns)
            # Wikidata is a special sub-case
            self._add_wikidata_metadata(uri, metadata, etd_ns)

            # ----------------------------------------------------
            # Production: guarantee every entity has an rdfs:label (human-friendly)
            # ----------------------------------------------------
            if not any(self.graph.objects(uri, RDFS.label)):
                meta = getattr(ent, "metadata", None) or {}

                # Prefer meaningful metadata labels if available
                fallback = (
                    (meta.get("surface_form") or "").strip()
                    or (meta.get("canonical_label") or "").strip()
                    or (meta.get("normalized_label") or "").strip()
                )

                if not fallback:
                    # Final fallback: last segment of URI (still better than missing label)
                    try:
                        fallback = str(uri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
                    except Exception:
                        fallback = "Unknown"

                # Clamp for UI safety
                if len(fallback) > 120:
                    fallback = fallback[:120]

                self.graph.add((uri, RDFS.label, Literal(fallback, lang="en")))

    def _coerce_int(self, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)

        s = str(v).strip()
        if not s:
            return None

        # stringified list like "[174, 717]"
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    return int(parsed[0])
            except Exception:
                pass

        # first integer substring
        m = re.search(r"-?\d+", s)
        if m:
            return int(m.group(0))

        return None

    def _confidence_literal(self, value: Any) -> Literal:
        """
        Emit confidence as a strict xsd:float typed literal.
        """
        try:
            fval = float(value)
        except Exception:
            fval = 0.0
        return Literal(float(fval), datatype=XSD.float)

    def _coerce_bool(self, v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(int(v))
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        return None

    def _registered_etd_predicate(self, local_name: str, etd_ns: Namespace) -> Optional[URIRef]:
        """
        Resolve ETD predicate only if it is explicitly registered.
        """
        key = str(local_name or "").strip()
        if not key:
            return None
        if not is_registered_etd_predicate(key):
            msg = f"Unregistered ETD predicate emission blocked: etd:{key}"
            if self.strict_etd_property_registry:
                raise RuntimeError(msg)
            logger.warning(msg)
            return None
        return etd_ns[key]

    def _emit_registered_etd_literal(
        self,
        subject: Union[URIRef, BNode],
        local_name: str,
        value: Any,
        etd_ns: Namespace,
        *,
        datatype: Optional[URIRef] = None,
        lang: Optional[str] = None,
    ) -> None:
        pred = self._registered_etd_predicate(local_name, etd_ns)
        if pred is None:
            return
        if lang is not None:
            self.graph.add((subject, pred, Literal(str(value), lang=lang)))
            return
        if datatype is not None:
            self.graph.add((subject, pred, Literal(value, datatype=datatype)))
            return
        self.graph.add((subject, pred, Literal(value)))

    def _emit_registered_etd_object(
        self,
        subject: Union[URIRef, BNode],
        local_name: str,
        obj: URIRef,
        etd_ns: Namespace,
    ) -> None:
        pred = self._registered_etd_predicate(local_name, etd_ns)
        if pred is None:
            return
        self.graph.add((subject, pred, obj))

    def _pick_single_value(self, val: Any) -> Any:
        """
        Deterministically pick one value:
        - scalar -> scalar
        - list/tuple/set -> lexicographically smallest non-empty string representation
        """
        if isinstance(val, (list, tuple, set)):
            candidates = sorted(
                [str(x).strip() for x in val if str(x).strip()],
                key=lambda x: x.lower(),
            )
            return candidates[0] if candidates else None
        return val

    def _coerce_datetime_literal(self, raw_val: Any, field_key: str) -> Optional[Literal]:
        if raw_val is None:
            return None
        raw = str(raw_val).strip()
        if not raw:
            return None
        candidate = raw
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            norm = dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
            return Literal(norm, datatype=XSD.dateTime)
        except Exception:
            if field_key not in self._raw_datetime_warning_keys:
                logger.warning(
                    "Raw field '%s' datetime parse failed; emitting xsd:string fallback",
                    field_key,
                )
                self._raw_datetime_warning_keys.add(field_key)
            return Literal(raw, datatype=XSD.string)

    def _looks_like_url(self, s: str) -> bool:
        if not s:
            return False
        try:
            p = urlparse(s)
            return p.scheme in {"http", "https"} and bool(p.netloc)
        except Exception:
            return False

    def _emit_raw_field_with_provenance(
        self,
        subject_uri: URIRef,
        predicate_uri: URIRef,
        object_node: Node,
        field_name: str,
        record_id: str,
        section_name: str,
        raw_dict: Dict[str, Any],
        etd_ns: Namespace,
    ) -> None:
        if not self.materialize_raw_fields:
            return
        self.graph.add((subject_uri, predicate_uri, object_node))
        prov_key = f"{str(subject_uri)}|{str(predicate_uri)}|{field_name}|{record_id}"
        field_node = URIRef(f"{self.namespaces['etd']}field_{safe_hash(prov_key)}")

        self.graph.add((field_node, RDF.type, etd_ns.FieldAssertion))
        self.graph.add((subject_uri, etd_ns.hasFieldAssertion, field_node))
        self.graph.add((field_node, etd_ns.fieldName, Literal(str(field_name), datatype=XSD.string)))
        self.graph.add((field_node, etd_ns.recordId, Literal(str(record_id), datatype=XSD.string)))
        self.graph.add((field_node, etd_ns.sourceSection, Literal(str(section_name), datatype=XSD.string)))
        self.graph.add((field_node, etd_ns.assertedPredicate, predicate_uri))
        self.graph.add((field_node, etd_ns.assertedValue, object_node))
        self.graph.add((field_node, etd_ns.datasetId, Literal(self.dataset_id, datatype=XSD.string)))
        self.graph.add((field_node, etd_ns.runId, Literal(self.run_id, datatype=XSD.string)))
        self.graph.add((field_node, PROV.generatedAtTime, Literal(self.generated_at_time, datatype=XSD.dateTime)))
        self.graph.add((field_node, PROV.wasDerivedFrom, self.export_node))

        source_val = raw_dict.get("source") or raw_dict.get("context_source")
        if source_val is not None and str(source_val).strip():
            self.graph.add((field_node, etd_ns.source, Literal(str(source_val).strip(), datatype=XSD.string)))
        if raw_dict.get("discourse_id") is not None:
            self._add_discourse_id_literal(field_node, raw_dict.get("discourse_id"), etd_ns)
        if raw_dict.get("chunk_index") is not None:
            cidx = self._coerce_int(raw_dict.get("chunk_index"))
            if cidx is not None:
                self.graph.add((field_node, etd_ns.chunk_index, Literal(cidx, datatype=XSD.integer)))

    def _emit_raw_fields(
        self,
        node: URIRef,
        kind: str,
        raw_dict: Dict[str, Any],
        etd_ns: Namespace,
    ) -> None:
        if not self.materialize_raw_fields:
            return
        if not isinstance(raw_dict, dict) or not raw_dict:
            return

        allowlist: Dict[str, Dict[str, str]] = {
            "Theme": {
                "created_at": "createdAt",
                "views": "views",
                "expert_id": "expertId",
                "problem_category": "problemCategory",
                "model_category": "modelCategory",
                "domain_category": "domainCategory",
                "name": "name",
                "description": "description",
            },
            "Question": {
                "created_at": "createdAt",
                "views": "views",
                "expert_id": "expertId",
                "theme_id": "themeId",
                "title": "title",
                "body": "body",
            },
            "Answer": {
                "created_at": "createdAt",
                "question_id": "questionId",
                "expert_id": "expertId",
                "parent_id": "parentId",
                "description": "description",
            },
            "Vote": {
                "created_at": "createdAt",
                "updated_at": "updatedAt",
                "expert_id": "expertId",
                "answer_id": "answerId",
                "vote_value": "voteValueRaw",
            },
            "Expert": {
                "user_id": "userId",
                "is_deleted": "isDeleted",
                "area_of_expertise": "areaOfExpertise",
                "date_joined": "dateJoined",
                "profile_picture": "profilePicture",
                "bio": "bio",
            },
        }
        section_map = {
            "Theme": "themes",
            "Question": "questions",
            "Answer": "answers",
            "Vote": "votes",
            "Expert": "experts",
        }
        section_name = section_map.get(kind, "unknown")

        def _record_id_for_kind() -> str:
            candidates_by_kind = {
                "Theme": ["id", "theme_id", "discourse_id"],
                "Question": ["id", "question_id", "discourse_id"],
                "Answer": ["id", "answer_id", "discourse_id"],
                "Vote": ["id", "vote_id", "discourse_id"],
                "Expert": ["id", "expert_id", "user_id", "discourse_id"],
            }
            for key in candidates_by_kind.get(kind, []):
                val = raw_dict.get(key)
                if val is not None and str(val).strip():
                    return str(self._pick_single_value(val)).strip()
            return _uri_localname(node)

        record_id = _record_id_for_kind()

        datetime_fields = {"created_at", "updated_at", "date_joined"}
        int_fields = {
            "views",
            "expert_id",
            "theme_id",
            "question_id",
            "answer_id",
            "parent_id",
            "user_id",
            "vote_value",
        }
        bool_fields = {"is_deleted"}
        string_fields = {
            "problem_category",
            "model_category",
            "domain_category",
            "area_of_expertise",
            "bio",
            "name",
            "title",
            "body",
            "description",
        }

        kind_map = allowlist.get(kind, {})
        for raw_key, pred_local in kind_map.items():
            if raw_key == "bio" and not self.materialize_sensitive_fields:
                continue
            if raw_key not in raw_dict:
                continue

            picked = self._pick_single_value(raw_dict.get(raw_key))
            if picked is None:
                continue
            if isinstance(picked, str) and not picked.strip():
                continue

            pred = self._registered_etd_predicate(pred_local, etd_ns)
            if pred is None:
                continue
            for old in list(self.graph.objects(node, pred)):
                self.graph.remove((node, pred, old))

            if raw_key in datetime_fields:
                lit = self._coerce_datetime_literal(picked, raw_key)
                if lit is not None:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=lit,
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=raw_dict,
                        etd_ns=etd_ns,
                    )
                continue

            if raw_key in int_fields:
                ival = self._coerce_int(picked)
                if ival is not None:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=Literal(ival, datatype=XSD.integer),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=raw_dict,
                        etd_ns=etd_ns,
                    )
                continue

            if raw_key in bool_fields:
                bval = self._coerce_bool(picked)
                if bval is not None:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=Literal(bval, datatype=XSD.boolean),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=raw_dict,
                        etd_ns=etd_ns,
                    )
                continue

            if raw_key == "profile_picture":
                sval = str(picked).strip()
                if not sval:
                    continue
                if self._looks_like_url(sval):
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=URIRef(sval),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=raw_dict,
                        etd_ns=etd_ns,
                    )
                else:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=Literal(sval, datatype=XSD.string),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=raw_dict,
                        etd_ns=etd_ns,
                    )
                    if self.media_base_url:
                        base = self.media_base_url.rstrip("/")
                        rel = sval.lstrip("/")
                        self.graph.add((node, PROV.atLocation, URIRef(f"{base}/{rel}")))
                continue

            if raw_key in string_fields:
                s = str(picked).strip()
                if s:
                    self._emit_raw_field_with_provenance(
                        subject_uri=node,
                        predicate_uri=pred,
                        object_node=Literal(s, datatype=XSD.string),
                        field_name=raw_key,
                        record_id=record_id,
                        section_name=section_name,
                        raw_dict=raw_dict,
                        etd_ns=etd_ns,
                    )

    def _add_discourse_id_literal(self, subject: Union[URIRef, BNode], did: Any, etd_ns: Namespace) -> None:
        """
        Emit etd:discourse_id with robust typing:
        - None/empty -> skip
        - numeric -> xsd:integer
        - non-numeric -> warn and emit xsd:string
        """
        if did is None:
            return
        raw = str(did).strip()
        if not raw:
            return
        num = self._coerce_int(raw)
        if num is not None:
            self.graph.add((subject, etd_ns.discourse_id, Literal(int(num), datatype=XSD.integer)))
            return
        logger.warning(
            "[RDFSerializer] Non-numeric discourse_id '%s'; emitting as xsd:string",
            raw,
        )
        self.graph.add((subject, etd_ns.discourse_id, Literal(raw, datatype=XSD.string)))

    def _resolve_origin_discourse_uri(self, metadata: Optional[Dict[str, Any]]) -> Optional[URIRef]:
        """
        Resolve deterministic DiscourseUnit URI from provenance metadata.
        Uses origin_key when available, else field|discourse_id, else discourse_id.
        """
        if not isinstance(metadata, dict):
            return None
        origin_key = metadata.get("origin_key")
        if origin_key is not None and str(origin_key).strip():
            token = str(origin_key).strip()
        else:
            did_raw = metadata.get("discourse_id") or metadata.get("doc_id")
            did = str(did_raw).strip() if did_raw is not None else ""
            if not did:
                return None
            field = metadata.get("field")
            fld = str(field).strip() if field is not None else ""
            token = f"{fld}|{did}" if fld else did
        safe = make_uri_safe(token)
        if not safe:
            return None
        return URIRef(f"{self.namespaces['etd']}discourse_{safe}")

    _ISO_DT_RE = re.compile(
        r"^\d{4}-\d{2}-\d{2}"                  # date
        r"(?:[T ]\d{2}:\d{2}:\d{2}"            # time
        r"(?:\.\d+)?"                          # optional fractional
        r"(?:Z|[+-]\d{2}:\d{2})?)?$"           # optional TZ
    )

    def _is_iso_datetime(self, s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        return bool(self._ISO_DT_RE.match(s))

    def _add_entity_metadata(
        self,
        ent_uri: URIRef,
        metadata: Dict[str, Any],
        etd_ns: Namespace,
    ) -> None:
        """
        Add core, SHACL-friendly metadata for entities.

        At minimum we support:
          - surface_form      (xsd:string)
          - context_source    (xsd:string)
          - discourse_id      (xsd:string)
          - chunk_index       (xsd:integer)
          - chunk_total       (xsd:integer)
          - start_char        (xsd:integer)
          - end_char          (xsd:integer)
          - ontology_uri      (URIRef if resolvable, else xsd:string)
        """
        if not metadata:
            return

        # Backward compatibility: accept doc_id if discourse_id is missing
        if "discourse_id" not in metadata and "doc_id" in metadata:
            metadata["discourse_id"] = metadata.get("doc_id")

        # String fields
        for key in ("surface_form", "context_source", "text"):
            if key in metadata and metadata[key] is not None:
                if isinstance(metadata[key], (dict, list)):
                    continue  # or flatten intentionally
                val = str(metadata[key]).strip()
                if val:
                    self._emit_registered_etd_literal(ent_uri, key, val, etd_ns, datatype=XSD.string)

        # discourse_id with robust integer typing
        self._add_discourse_id_literal(ent_uri, metadata.get("discourse_id"), etd_ns)


        # Integer-like fields (production-safe: coerce + preserve raw)
        for key in ("chunk_index", "chunk_total", "start_char", "end_char"):
            if key in metadata and metadata[key] is not None:
                raw_val = metadata.get(key)
                num = self._coerce_int(raw_val)

                if num is not None:
                    self._emit_registered_etd_literal(ent_uri, key, num, etd_ns, datatype=XSD.integer)


        # ontology_uri as a proper resource if possible
        if "ontology_uri" in metadata and metadata["ontology_uri"] is not None:
            ou = metadata["ontology_uri"]

            # ----------------------------------------------------
            # FIX: prevent ontology_uri being list or stringified list
            # e.g. ["dcat:Dataset"] or "['dcat:Dataset']"
            # ----------------------------------------------------
            if isinstance(ou, list) and ou:
                ou = ou[0]

            if isinstance(ou, str):
                s = ou.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, list) and parsed:
                            ou = parsed[0]
                    except Exception:
                        pass

            raw = str(ou).strip()
            if raw:
                ref = self._resolve_entity_uri_from_value(raw)
                if ref is None:
                    ref = Literal(raw, datatype=XSD.string)
                self.graph.add((ent_uri, etd_ns.ontology_uri, ref))

    #-----------------------------------------
    # Adding wikidata_metadata
    # ----------------------------------------
    def _add_wikidata_metadata(
        self,
        ent_uri: URIRef,
        metadata: Dict[str, Any],
        etd_ns: Namespace,
    ) -> None:
        """
        Handle Wikidata-related metadata so we don't store raw JSON strings.

        Expected keys (by convention in your pipeline):
        - 'wikidata'             → dict or string with 'qid' and 'label'
        - 'wikidataLabel'        → label string
        - 'wikidataDescription'  → description string
        - 'wikidataScore'        → float
        """
        if not metadata:
            return

        def _replace_value(p: URIRef) -> None:
            for o in list(self.graph.objects(ent_uri, p)):
                self.graph.remove((ent_uri, p, o))

        wrote_label = False
        wrote_desc = False
        wrote_score = False

        wikidata_val = metadata.get("wikidata")
        if wikidata_val:
            if isinstance(wikidata_val, str):
                try:
                    wikidata_val = ast.literal_eval(wikidata_val)
                except Exception:
                    wikidata_val = None

            if isinstance(wikidata_val, dict):
                qid = wikidata_val.get("qid")
                if qid:
                    wd_uri = f"http://www.wikidata.org/entity/{qid}"
                    wd_ref = _safe_uriref(wd_uri, self.namespaces)

                    # owl:sameAs (best practice equivalence link)
                    self.graph.add((ent_uri, OWL.sameAs, wd_ref))

                    # Custom ETD property
                    self.graph.add((ent_uri, etd_ns.wikidata, wd_ref))

                label = wikidata_val.get("label")
                if label:
                    _replace_value(etd_ns.wikidataLabel)
                    self.graph.add(
                        (ent_uri, etd_ns.wikidataLabel, Literal(str(label).strip(), lang="en"))
                    )
                    wrote_label = True

                desc = wikidata_val.get("description")
                if desc:
                    _replace_value(etd_ns.wikidataDescription)
                    self.graph.add(
                        (ent_uri, etd_ns.wikidataDescription, Literal(str(desc).strip(), lang="en"))
                    )
                    wrote_desc = True

                score = wikidata_val.get("score")
                if score is not None:
                    try:
                        sc = float(score)
                    except Exception:
                        sc = 0.0
                    _replace_value(etd_ns.wikidataScore)
                    self.graph.add((ent_uri, etd_ns.wikidataScore, Literal(sc, datatype=XSD.float)))
                    wrote_score = True

        # Direct convenience keys (only if not already written from wikidata dict)
        lbl = metadata.get("wikidataLabel")
        if lbl and not wrote_label:
            _replace_value(etd_ns.wikidataLabel)
            self.graph.add((ent_uri, etd_ns.wikidataLabel, Literal(str(lbl).strip(), lang="en")))

        desc2 = metadata.get("wikidataDescription")
        if desc2 and not wrote_desc:
            _replace_value(etd_ns.wikidataDescription)
            self.graph.add((ent_uri, etd_ns.wikidataDescription, Literal(str(desc2).strip(), lang="en")))

        score2 = metadata.get("wikidataScore")
        if score2 is not None and not wrote_score:
            try:
                sc2 = float(score2)
            except Exception:
                sc2 = 0.0
            _replace_value(etd_ns.wikidataScore)
            self.graph.add((ent_uri, etd_ns.wikidataScore, Literal(sc2, datatype=XSD.float)))

    # ------------------------------------------------------------------
    # Relations (direct + relation node with metadata)
    # ------------------------------------------------------------------
    def _get_all_relations(self) -> Iterable[Relation]:
        """
        Flatten relations from all document groups in annotations
        into a list[Relation].
        """
        rels: List[Relation] = []

        base_etd = self.namespaces.get("etd")
        if not base_etd:
            raise ValueError("Missing 'etd' namespace in RDFSerializer.namespaces")

        def _compose_origin_key(discourse_id: Any, field: Any) -> str:
            did = str(discourse_id).strip() if discourse_id is not None else ""
            fld = str(field).strip() if field is not None else ""
            if did and fld:
                return f"{fld}|{did}"
            return did

        def collect(section_key: str) -> None:
            section = self.annotations.get(section_key, {})
            for _, doc in section.items():
                # Skip non-dict entries
                if not isinstance(doc, dict):
                    continue
                doc_meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}

                for rel_data in doc.get("relations", []):
                    # Skip non-dicts just in case
                    if not isinstance(rel_data, dict):
                        continue

                    src = rel_data.get("source_id") or rel_data.get("source")
                    tgt = rel_data.get("target_id") or rel_data.get("target")
                    pred = rel_data.get("relation")
                    rel_meta = dict(rel_data.get("metadata", {}) or {})
                    did = rel_meta.get("discourse_id")
                    if did is None:
                        did = doc_meta.get("discourse_id")
                    if did is not None:
                        rel_meta.setdefault("discourse_id", did)
                    field = rel_meta.get("field")
                    if field is None:
                        field = doc_meta.get("field")
                    if field is not None and str(field).strip():
                        rel_meta.setdefault("field", str(field).strip())
                    origin_key = rel_meta.get("origin_key")
                    if origin_key is None:
                        origin_key = _compose_origin_key(rel_meta.get("discourse_id"), rel_meta.get("field"))
                    if origin_key:
                        rel_meta.setdefault("origin_key", str(origin_key).strip())

                    raw = f"{src}|{pred}|{tgt}|{did}"
                    uri = rel_data.get("uri") or f"{base_etd}rel_{safe_hash(raw)}"

                    rels.append(
                        Relation(
                            uri=uri,
                            source=src,
                            target=tgt,
                            relation=pred,
                            confidence=rel_data.get("confidence", 0.0),
                            context=rel_data.get("evidence") or rel_data.get("context"),
                            is_inferred=bool(rel_data.get("is_inferred", False)),
                            metadata=rel_meta,
                        )
                    )


        for key in ["themes", "questions", "answers", "votes", "experts"]:
            if key in self.annotations:
                collect(key)

        # Include inferred relations persisted at metadata level.
        # Contract: annotations["metadata"]["inferred_relations"] is list[dict].
        metadata = self.annotations.get("metadata", {})
        if isinstance(metadata, dict):
            inferred = metadata.get("inferred_relations", [])
            if isinstance(inferred, list):
                for rel_data in inferred:
                    if not isinstance(rel_data, dict):
                        continue

                    src = rel_data.get("source_id") or rel_data.get("source")
                    tgt = rel_data.get("target_id") or rel_data.get("target")
                    pred = rel_data.get("relation")
                    rel_meta = rel_data.get("metadata", {}) or {}
                    did = rel_meta.get("discourse_id")
                    field = rel_meta.get("field")
                    if field is not None and str(field).strip():
                        rel_meta["field"] = str(field).strip()
                    if rel_meta.get("origin_key") is None:
                        ok = _compose_origin_key(did, rel_meta.get("field"))
                        if ok:
                            rel_meta["origin_key"] = ok

                    raw = f"{src}|{pred}|{tgt}|{did}|inferred"
                    uri = rel_data.get("uri") or f"{base_etd}rel_{safe_hash(raw)}"

                    rels.append(
                        Relation(
                            uri=uri,
                            source=src,
                            target=tgt,
                            relation=pred,
                            confidence=rel_data.get("confidence", 0.0),
                            context=rel_data.get("evidence") or rel_data.get("context"),
                            is_inferred=True,
                            metadata=rel_meta,
                        )
                    )

        return rels

    def _mint_subject_iri(self, label: str) -> URIRef:
        """Mint a stable ETD IRI for unresolved relation subjects (SHACL requires IRI)."""
        etd_base = self.namespaces["etd"]
        etd_ns = Namespace(etd_base)
        txt = (label or "Unknown").strip() or "Unknown"
        u = URIRef(f"{etd_base}entity_{safe_hash(txt)}")
        # Materialise minimal node
        self.graph.add((u, RDF.type, etd_ns.Entity))
        self.graph.add((u, RDFS.label, Literal(txt, lang="en")))
        return u

    def _resolve_relation_node(
        self,
        endpoint: Any,
        for_subject: bool,
    ) -> Union[URIRef, Literal]:
        """
        Resolve a relation endpoint:

        1. If it's already an RDF node, return as-is (except: literal subjects are converted to minted ETD IRIs).
        2. If it matches a known entity (by id, label, uri) → that entity URIRef.
        3. If it looks like a full IRI or CURIE → URIRef.
        4. Otherwise:
        - for subjects: mint a stable ETD IRI (URIRef) with rdfs:label @en
        - for objects: use an xsd:string literal
        """
        # Already an RDF node
        if isinstance(endpoint, (URIRef, Literal)):
            if isinstance(endpoint, Literal) and for_subject:
                # SHACL requires IRI subjects → mint stable ETD IRI instead of BNode
                return self._mint_subject_iri(str(endpoint))            
            
            return endpoint

        # Map numeric index/id
        if isinstance(endpoint, int):
            ent = self.entity_index.get(str(endpoint))
            if ent is not None and getattr(ent, "uri", None):
                return _safe_uriref(str(ent.uri), self.namespaces)

        # String handling
        if isinstance(endpoint, str):
            s = endpoint.strip()
            if not s:
                if for_subject:
                    return self._mint_subject_iri("Unknown")                
               
                return Literal("", datatype=XSD.string)

            # Exact entity hit: id / uri / label
            ent = (
                self.entity_index.get(s)
                or self.entity_index.get(s.lower())
            )
            if ent is not None and getattr(ent, "uri", None):
                return _safe_uriref(str(ent.uri), self.namespaces)

            # Full IRI
            if s.startswith("http://") or s.startswith("https://"):
                return _safe_uriref(s, self.namespaces)

            # CURIE with known prefix
            if ":" in s and not s.startswith("urn:"):
                prefix, local = s.split(":", 1)
                if prefix in self.namespaces:
                    iri = self.namespaces[prefix] + local
                    return _safe_uriref(iri, self.namespaces)

            # Otherwise fallback
            if for_subject:
                return self._mint_subject_iri(s)            
        
        # Fallback: stringify
        if for_subject:
            return self._mint_subject_iri(str(endpoint))

        # Object fallback: always return a literal (prevents None)
        return Literal(str(endpoint).strip(), datatype=XSD.string)

    def _ensure_typed_node(self, node: Any, etd_ns: Namespace) -> None:
        """
        Graph-closure: ensure any URIRef endpoint used in relations is materialised
        with at least rdf:type etd:Entity. This prevents dangling endpoints.
        """
        if not isinstance(node, URIRef):
            return

        # If node already has any triple, it's materialised enough
        if (node, None, None) in self.graph:
            return

        # Minimal typing (safe default)
        self.graph.add((node, RDF.type, etd_ns.Entity))


    def _map_rating_value_to_skos(self, pred_local: str, obj: Any) -> Optional[URIRef]:
        """
        Map literal rating values used in risk assessments (e.g., 'High') to
        SKOS concept URIs minted in the ETD namespace (e.g., etd:SeverityScaleHigh).

        This assumes add_schema.py creates:
        - etd:SeverityScale, etd:LikelihoodScale, etd:RiskLevelScale (skos:ConceptScheme)
        - etd:SeverityScaleHigh/Medium/Low (skos:Concept), etc.
        """
        if obj is None:
            return None

        # If object is already a URI/BNode, keep it
        if isinstance(obj, (URIRef, BNode)):
            return None

        # Convert Literal -> string
        if isinstance(obj, Literal):
            val = str(obj)
        else:
            val = str(obj)

        val = val.strip()
        if not val:
            return None

        # Normalise common variants
        norm = val.lower()
        normalise = {
            "very high": "VeryHigh",
            "high": "High",
            "medium": "Medium",
            "moderate": "Medium",
            "low": "Low",
            "very low": "VeryLow",
        }
        if norm not in normalise:
            # If you later add numeric scales, handle them here
            return None

        level = normalise[norm]

        scheme_for_pred = {
        "hasSeverity": "SeverityScale",
            "hasLikelihood": "LikelihoodScale",
            "hasRiskLevel": "RiskLevelScale",
        }
        scheme = scheme_for_pred.get(pred_local)
        if not scheme:
            return None

        etd_ns = Namespace(self.namespaces["etd"])
        return etd_ns[f"{scheme}{level}"]


    def _is_vote_relation(self, rel: Relation) -> bool:
        meta = getattr(rel, "metadata", None) or {}
        # Only semantic vote relations should trigger vote-node materialization.
        # Structural backbone relations (castBy / receivesVote) are already
        # emitted directly from relation records and must not be remapped here.
        return meta.get("extractor") == "vote"

    def _label_for_node(self, node: Any) -> str:
        """Best-effort human label for URI/BNode/Literal nodes."""
        if isinstance(node, URIRef):
            existing = next(self.graph.objects(node, RDFS.label), None)
            if isinstance(existing, Literal) and str(existing).strip():
                return str(existing).strip()
            s = str(node)
            tail = s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            return tail or s
        if isinstance(node, Literal):
            return str(node).strip() or "Unknown"
        return str(node).strip() or "Unknown"

    def _spo_key(self, subj: Any, pred_uri: Any, obj: Any) -> str:
        return f"{str(subj)}|{str(pred_uri)}|{str(obj)}"

    def _has_reified_relation_for_spo(self, subj: Any, pred_uri: Any, obj: Any, etd_ns: Namespace) -> bool:
        key = self._spo_key(subj, pred_uri, obj)
        if not hasattr(self, "_reified_spo_keys"):
            self._reified_spo_keys = set()
        if key in self._reified_spo_keys:
            return True

        for rel_node in self.graph.subjects(RDF.type, etd_ns.Relation):
            if (
                (rel_node, etd_ns.subject, subj) in self.graph
                and (rel_node, etd_ns.predicate, pred_uri) in self.graph
                and (rel_node, etd_ns.object, obj) in self.graph
            ):
                self._reified_spo_keys.add(key)
                return True
        return False

    def _emit_reified_backbone_relation(
        self,
        *,
        subj: Any,
        pred_local: str,
        obj: Any,
        etd_ns: Namespace,
        discourse_id: Any = None,
        chunk_index: Any = None,
        field: Optional[str] = None,
        origin_key: Optional[str] = None,
        evidence_text: str,
        source: str = "dataset_vote",
    ) -> None:
        """
        Emit direct triple + deterministic reified Relation node for a backbone edge.
        Deduplicates if an equivalent reified SPO already exists.
        """
        pred_uri = etd_ns[pred_local]
        subj = self._canonicalize_relation_endpoint_uri(subj)
        obj = self._canonicalize_relation_endpoint_uri(obj)

        # Maintain contract: direct triple must always exist.
        self.graph.add((subj, pred_uri, obj))

        if self._has_reified_relation_for_spo(subj, pred_uri, obj, etd_ns):
            return

        rid_raw = f"{str(subj)}|{pred_local}|{str(obj)}|{source}|{discourse_id}"
        rnode = URIRef(f"{self.namespaces['etd']}rel_{safe_hash(rid_raw)}")
        rnode = self._ensure_relation_node_identity(rnode, subj, pred_uri, obj, etd_ns)
        self.graph.add((rnode, RDF.type, etd_ns.Relation))
        self.graph.add((rnode, etd_ns.subject, subj))
        self.graph.add((rnode, etd_ns.predicate, pred_uri))
        self.graph.add((rnode, etd_ns.object, obj))
        self.graph.add((rnode, etd_ns.confidence, Literal(1.0, datatype=XSD.float)))
        self.graph.add((rnode, etd_ns.isInferred, Literal(False, datatype=XSD.boolean)))

        rel_meta = {
            "extractor": "structural",
            "context_source": source,
            "evidence": evidence_text,
        }
        if discourse_id is not None:
            rel_meta["discourse_id"] = discourse_id
        if chunk_index is not None:
            rel_meta["chunk_index"] = chunk_index
        if field is not None and str(field).strip():
            rel_meta["field"] = str(field).strip()
        if origin_key is not None and str(origin_key).strip():
            rel_meta["origin_key"] = str(origin_key).strip()
        if rel_meta.get("discourse_id") is not None:
            rel_meta["source_doc_id"] = str(rel_meta.get("discourse_id"))

        self._add_relation_metadata(rnode, rel_meta, etd_ns)

        rel_obj = Relation(
            uri=str(rnode),
            source=str(subj),
            relation=pred_local,
            target=str(obj),
            confidence=1.0,
            is_inferred=False,
            metadata=rel_meta,
        )
        self._add_evidence_node(rnode, rel_obj, etd_ns)
        self._reified_spo_keys.add(self._spo_key(subj, pred_uri, obj))

    def _relation_confidence_for_spo(self, subj: URIRef, pred_uri: URIRef, obj: URIRef, etd_ns: Namespace) -> float:
        """Best-effort confidence lookup from existing reified relation nodes."""
        best = 0.0
        for rel_node in self.graph.subjects(RDF.type, etd_ns.Relation):
            if (
                (rel_node, etd_ns.subject, subj) in self.graph
                and (rel_node, etd_ns.predicate, pred_uri) in self.graph
                and (rel_node, etd_ns.object, obj) in self.graph
            ):
                for c in self.graph.objects(rel_node, etd_ns.confidence):
                    try:
                        best = max(best, float(c))
                    except Exception:
                        continue
        return best

    def _remove_reified_backbone_spo(self, subj: URIRef, pred_uri: URIRef, obj: URIRef, etd_ns: Namespace) -> None:
        """Remove reified relation nodes (and attached evidence) for a specific SPO."""
        to_remove: List[URIRef] = []
        for rel_node in self.graph.subjects(RDF.type, etd_ns.Relation):
            if (
                (rel_node, etd_ns.subject, subj) in self.graph
                and (rel_node, etd_ns.predicate, pred_uri) in self.graph
                and (rel_node, etd_ns.object, obj) in self.graph
            ):
                to_remove.append(rel_node)

        for rel_node in to_remove:
            evidence_nodes = list(self.graph.objects(rel_node, etd_ns.hasEvidence))
            # Remove all triples where relation node is subject/object
            for t in list(self.graph.triples((rel_node, None, None))):
                self.graph.remove(t)
            for t in list(self.graph.triples((None, None, rel_node))):
                self.graph.remove(t)

            # Remove evidence nodes fully to avoid dangling evidence
            for ev in evidence_nodes:
                for t in list(self.graph.triples((ev, None, None))):
                    self.graph.remove(t)
                for t in list(self.graph.triples((None, None, ev))):
                    self.graph.remove(t)

        # Keep de-dupe cache in sync
        if hasattr(self, "_reified_spo_keys"):
            self._reified_spo_keys.discard(self._spo_key(subj, pred_uri, obj))

    def _enforce_vote_castby_backbone(self, etd_ns: Namespace) -> None:
        """
        Guarantee every Vote has exactly one castBy IRI.
        - Materialise inverse from (expert castVote vote) when needed.
        - Resolve multi-candidate deterministically (highest confidence, then sorted URI).
        - Reify castBy edge 1:1 with deterministic evidence.
        """
        strict = bool(getattr(self.config, "FAIL_ON_SHACL_VIOLATIONS", False))
        missing_cast_by = 0
        cast_by_uri = etd_ns.castBy
        cast_vote_uri = etd_ns.castVote

        vote_nodes = sorted(
            {v for v in self.graph.subjects(RDF.type, etd_ns.Vote) if isinstance(v, URIRef)},
            key=str,
        )
        for vote in vote_nodes:
            candidate_scores: Dict[URIRef, float] = {}

            # Direct candidates from existing castBy triples
            for expert in self.graph.objects(vote, cast_by_uri):
                if isinstance(expert, URIRef):
                    score = self._relation_confidence_for_spo(vote, cast_by_uri, expert, etd_ns)
                    candidate_scores[expert] = max(candidate_scores.get(expert, 0.0), score)

            # Inverse materialization candidates from expert castVote vote
            for expert in self.graph.subjects(cast_vote_uri, vote):
                if isinstance(expert, URIRef):
                    score = self._relation_confidence_for_spo(expert, cast_vote_uri, vote, etd_ns)
                    candidate_scores[expert] = max(candidate_scores.get(expert, 0.0), score)

            if not candidate_scores:
                missing_cast_by += 1
                logger.error("Vote missing castBy candidate: %s", vote)
                if strict:
                    raise RuntimeError(f"Vote data-contract violation: missing castBy for vote {vote}")
                continue

            # Deterministic pick: highest score, then lexicographic URI.
            chosen = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
            if len(candidate_scores) > 1:
                logger.warning(
                    "Multiple castBy candidates for %s; chosen=%s candidates=%s",
                    vote,
                    chosen,
                    [str(u) for u in sorted(candidate_scores.keys(), key=str)],
                )

            # Remove any non-chosen castBy edges + reified equivalents.
            for expert in list(self.graph.objects(vote, cast_by_uri)):
                if isinstance(expert, URIRef) and expert != chosen:
                    self.graph.remove((vote, cast_by_uri, expert))
                    self._remove_reified_backbone_spo(vote, cast_by_uri, expert, etd_ns)

            # Ensure chosen direct triple exists.
            self.graph.add((vote, cast_by_uri, chosen))

            # Reify castBy 1:1 with deterministic evidence, dedup-safe.
            did_candidates: Set[str] = set()
            for v in self.graph.objects(vote, etd_ns.discourse_id):
                raw = str(v).strip()
                if raw:
                    did_candidates.add(raw)
            chosen_did = sorted(did_candidates)[0] if did_candidates else None

            chunk_candidates: Set[int] = set()
            for v in self.graph.objects(vote, etd_ns.chunk_index):
                c = self._coerce_int(v)
                if c is not None and c >= 0:
                    chunk_candidates.add(int(c))
            chosen_chunk = sorted(chunk_candidates)[0] if chunk_candidates else None

            field_candidates: Set[str] = set()
            for v in self.graph.objects(vote, etd_ns.field):
                raw = str(v).strip()
                if raw:
                    field_candidates.add(raw)
            chosen_field = sorted(field_candidates)[0] if field_candidates else "vote.vote_value"
            chosen_origin = (
                f"{chosen_field}|{chosen_did}"
                if (chosen_did and chosen_field)
                else (chosen_did or None)
            )

            vote_lbl = self._label_for_node(vote)
            expert_lbl = self._label_for_node(chosen)
            evidence_text = f"Vote record: {expert_lbl} cast {vote_lbl}"
            self._emit_reified_backbone_relation(
                subj=vote,
                pred_local="castBy",
                obj=chosen,
                etd_ns=etd_ns,
                discourse_id=chosen_did,
                chunk_index=chosen_chunk,
                field=chosen_field,
                origin_key=chosen_origin,
                evidence_text=evidence_text,
                source="dataset_vote",
            )

        if missing_cast_by > 0:
            meta = self.annotations.setdefault("metadata", {})
            diagnostics = meta.setdefault("diagnostics", {})
            diagnostics["missing_castBy_votes"] = missing_cast_by
            logger.warning("Vote castBy diagnostics: missing_castBy_votes=%d", missing_cast_by)

    def _assert_backbone_relation_types(self, rel: Relation, pred_local: str) -> None:
        """Hard invariant: validate backbone relation endpoint types before emit."""
        if pred_local not in {"castBy", "receivesVote"}:
            return

        src_ent = self.entity_index.get(str(rel.source).strip())
        tgt_ent = self.entity_index.get(str(rel.target).strip())
        src_type = str(getattr(src_ent, "entity_type", "") or "")
        tgt_type = str(getattr(tgt_ent, "entity_type", "") or "")

        if pred_local == "castBy":
            if src_type != "Vote":
                raise RuntimeError(
                    f"Backbone invariant failed: castBy subject must be Vote, got {src_type or 'Unknown'} ({rel.source})"
                )
            if tgt_type not in {"Expert", "Person"}:
                raise RuntimeError(
                    f"Backbone invariant failed: castBy object must be Expert/Person, got {tgt_type or 'Unknown'} ({rel.target})"
                )

        if pred_local == "receivesVote":
            if src_type != "Vote":
                raise RuntimeError(
                    f"Backbone invariant failed: receivesVote subject must be Vote, got {src_type or 'Unknown'} ({rel.source})"
                )
            if tgt_type != "Answer":
                raise RuntimeError(
                    f"Backbone invariant failed: receivesVote object must be Answer, got {tgt_type or 'Unknown'} ({rel.target})"
                )


    def _add_relations(self) -> None:
        """
        Emit relations in two forms:

        (1) DIRECT SEMANTIC TRIPLE:
            <subject>  etd:<predicate>  <object>

        (2) RELATION METADATA NODE:
            _:r  a etd:Relation ;
                  etd:subject <subject> ;
                  etd:predicate etd:<predicate> ;
                  etd:object <object> ;
                  etd:confidence xsd:float ;
                  etd:context_source xsd:string ;
                  etd:isInferred xsd:boolean ;
                  etd:discourse_id xsd:string ;
                  ... (other metadata)

          This ensures relations ALWAYS appear in TTL.
          """
        etd_ns = Namespace(self.namespaces["etd"])

        base_etd = self.namespaces.get("etd", "")
        if not base_etd:
            raise ValueError("Missing 'etd' namespace in RDFSerializer.namespaces")

        # Flatten relation objects
        all_rels = self._get_all_relations()
        self._reified_spo_keys = set()

        for rel in all_rels:

            def _maybe_expand_local_to_etd(value: Any) -> Any:
                if not isinstance(value, str) or not base_etd:
                    return value
                s = value.strip()
                if not s:
                    return value

                # If already absolute IRI, keep
                if s.startswith("http://") or s.startswith("https://"):
                    return s

                # If CURIE-like (prefix:local), keep (resolver will expand later)
                if ":" in s and " " not in s:
                    prefix = s.split(":", 1)[0]
                    if prefix in self.namespaces:
                        return s

                # Otherwise treat as ETD-local and expand
                return base_etd + s.lstrip("#/")

            # Use it:
            rel.source = _maybe_expand_local_to_etd(rel.source)
            rel.target = _maybe_expand_local_to_etd(rel.target)


            # -------------------------
            # 1. Determine predicate
            # -------------------------
            raw_pred = rel.relation or "relatedTo"
            pred_local = self._sanitize(str(raw_pred))
            pred_uri = etd_ns[pred_local]   # etd:uses, etd:hasRisk, etc.

            # Enforce strict type contract for structural vote backbone edges.
            self._assert_backbone_relation_types(rel, pred_local)

            # -------------------------
            # 2. Resolve SUBJECT / OBJECT
            # -------------------------
            subj = self._resolve_relation_node(rel.source, for_subject=True)
            obj = self._resolve_relation_node(rel.target, for_subject=False)
            subj = self._canonicalize_relation_endpoint_uri(subj)
            obj = self._canonicalize_relation_endpoint_uri(obj)

            # ✅ FIX: graph closure (prevent dangling endpoints)
            self._ensure_typed_node(subj, etd_ns)
            self._ensure_typed_node(obj, etd_ns)


            # ----------------------------------------------------
            # Phase 3 (SKOS): map rating literals to SKOS Concepts
            # ----------------------------------------------------
            if pred_local in {"hasSeverity", "hasLikelihood", "hasRiskLevel"}:
                mapped = self._map_rating_value_to_skos(pred_local, obj)
                if mapped is not None:
                    obj = mapped


            # -------------------------
            # 3. Emit DIRECT triple
            # -------------------------
            # Only emit if both are RDF nodes (URIRef or BNode)
            if isinstance(subj, (URIRef, BNode)) and isinstance(obj, (URIRef, BNode, Literal)):
                self.graph.add((subj, pred_uri, obj))
            else:
                # Absolute safety: never allow literal subjects
                if not isinstance(subj, (URIRef, BNode)):
                    subj = self._mint_subject_iri(str(subj))

                # Object may be literal or resource
                if not isinstance(obj, (URIRef, BNode, Literal)):
                    obj = Literal(str(obj), datatype=XSD.string)

                self.graph.add((subj, pred_uri, obj))

            # Ensure relation URI is absolute
            if rel.uri and isinstance(rel.uri, str) and not rel.uri.startswith("http"):
                rel.uri = base_etd + rel.uri.lstrip("#/")

            # -------------------------
            # 4. Create RELATION METADATA NODE
            # -------------------------
            rnode = URIRef(rel.uri) if rel.uri else BNode()
            if isinstance(rnode, URIRef):
                rnode = self._ensure_relation_node_identity(rnode, subj, pred_uri, obj, etd_ns)
                rel.uri = str(rnode)
            self.graph.add((rnode, RDF.type, etd_ns.Relation))

            # ----------------------------------------------------
            # VOTE SUPPORT: attach an explicit etd:Vote node + SKOS voteValue
            # ----------------------------------------------------
            meta = getattr(rel, "metadata", None) or {}
            if self._is_vote_relation(rel):
                # Stable Vote URI derived from relation URI (same pattern as evidence node)
                rid = str(rnode)
                vote_uri = URIRef(f"{self.namespaces['etd']}vote_{safe_hash(rid)}")

                # Type it as etd:Vote
                self.graph.add((vote_uri, RDF.type, etd_ns.Vote))

                # Deterministic vote labels for UX/explainability.
                expert_lbl = self._label_for_node(subj)
                answer_lbl = self._label_for_node(obj)
                vote_label = f"Vote by {expert_lbl} for {answer_lbl}"
                self.graph.add((vote_uri, RDFS.label, Literal(vote_label, datatype=XSD.string)))
                self.graph.add((vote_uri, SKOS.prefLabel, Literal(vote_label, datatype=XSD.string)))

                # Link Vote to the participants using auxiliary predicates only.
                # Backbone predicates (castBy / receivesVote) are emitted only
                # from explicit structural relations so they are always reified.
                discourse_id = meta.get("discourse_id")
                chunk_index = meta.get("chunk_index")
                vote_field = str(meta.get("field") or "vote.vote_value").strip()
                if discourse_id is not None:
                    self._add_discourse_id_literal(vote_uri, discourse_id, etd_ns)
                if vote_field:
                    self._emit_registered_etd_literal(vote_uri, "field", vote_field, etd_ns, datatype=XSD.string)
                ci = self._coerce_int(chunk_index)
                if ci is not None and ci >= 0:
                    self.graph.add((vote_uri, etd_ns.chunk_index, Literal(int(ci), datatype=XSD.integer)))
                expert_lbl = self._label_for_node(subj)
                answer_lbl = self._label_for_node(obj)
                vote_lbl = str(vote_uri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
                vote_evidence = f"Vote record: {expert_lbl} cast {vote_lbl} for {answer_lbl}"

                self._emit_reified_backbone_relation(
                    subj=vote_uri,
                    pred_local="receivesVote",
                    obj=obj,
                    etd_ns=etd_ns,
                    discourse_id=discourse_id,
                    chunk_index=chunk_index,
                    field=vote_field,
                    origin_key=meta.get("origin_key"),
                    evidence_text=vote_evidence,
                    source="dataset_vote",
                )

                # Optional convenience links (if you want the inverse materialised)
                self._emit_reified_backbone_relation(
                    subj=subj,
                    pred_local="castVote",
                    obj=vote_uri,
                    etd_ns=etd_ns,
                    discourse_id=discourse_id,
                    chunk_index=chunk_index,
                    field=vote_field,
                    origin_key=meta.get("origin_key"),
                    evidence_text=vote_evidence,
                    source="dataset_vote",
                )
                self.graph.add((obj, etd_ns.hasVote, vote_uri))

                # rawVoteValue literal (xsd:integer)
                raw_int = meta.get("raw_vote_value")
                if raw_int is not None:
                    try:
                        self.graph.add((vote_uri, etd_ns.rawVoteValue, Literal(int(raw_int), datatype=XSD.integer)))
                    except Exception:
                        pass

                # voteValue as SKOS concept (object property)
                concept = meta.get("vote_value_concept")
                if concept:
                    c = str(concept).strip()

                    # 1) Try normal resolution (full IRI / CURIE / indexed entity)
                    concept_ref = self._resolve_entity_uri_from_value(c)

                    # 2) Fallback: ETD-local concept (e.g., "VoteValueSchemeUpvote")
                    if concept_ref is None and ":" not in c and not (
                        c.startswith("http://") or c.startswith("https://")
                    ):
                        concept_ref = URIRef(f"{self.namespaces['etd']}{c.lstrip('#/')}")


                    if isinstance(concept_ref, URIRef):
                        self.graph.add((vote_uri, etd_ns.voteValue, concept_ref))


                # polarity as string (optional, keeps your old normalised value)
                pol = meta.get("vote_value")
                if pol:
                    self.graph.add((vote_uri, etd_ns.polarity, Literal(str(pol), datatype=XSD.string)))

            # link S, P, O
            self.graph.add((rnode, etd_ns.subject, subj))
            self.graph.add((rnode, etd_ns.predicate, pred_uri))
            self.graph.add((rnode, etd_ns.object, obj))
            self._reified_spo_keys.add(self._spo_key(subj, pred_uri, obj))


            # -------------------------
            # 5. Confidence
            # -------------------------
            self.graph.add((rnode, etd_ns.confidence, self._confidence_literal(rel.confidence or 0.0)))

            # -------------------------
            # 7. isInferred flag
            # -------------------------
            self.graph.add(
                (
                    rnode,
                    etd_ns.isInferred,
                    Literal(bool(getattr(rel, "is_inferred", False)), datatype=XSD.boolean),
                )
            )

            # -------------------------
            # 8. Additional metadata
            # -------------------------
            #meta = getattr(rel, "metadata", None) or {}
            self._add_relation_metadata(rnode, meta, etd_ns)
            if bool(getattr(rel, "is_inferred", False)):
                self._add_inferred_provenance(rnode, rel, etd_ns)

            self._add_evidence_node(rnode, rel, etd_ns)

        # Final vote backbone enforcement pass:
        # every Vote must have exactly one castBy IRI and castBy must be reified.
        self._enforce_vote_castby_backbone(etd_ns)


    def _add_evidence_node(self, rnode: URIRef, rel: Relation, etd_ns: Namespace) -> None:
        """
        Create one or more etd:Evidence nodes for a relation and attach provenance + evidence text.
        If rel.metadata["evidence_spans"] is present, emit one Evidence node per span.
        """
        base_meta = dict(getattr(rel, "metadata", None) or {})
        raw_spans = base_meta.get("evidence_spans")
        payloads: List[Tuple[int, Dict[str, Any]]] = []
        if isinstance(raw_spans, list) and raw_spans:
            seen_keys: Set[str] = set()
            for idx, item in enumerate(raw_spans):
                if not isinstance(item, dict):
                    continue
                p = dict(base_meta)
                p.pop("evidence_spans", None)
                p.update(item)
                pidx = self._coerce_int(p.get("chunk_index"))
                ps = self._coerce_int(p.get("char_start", p.get("start_offset")))
                pe = self._coerce_int(p.get("char_end", p.get("end_offset")))
                dedupe_key = f"{pidx}|{ps}|{pe}"
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                payloads.append((idx, p))
        if not payloads:
            payloads = [(0, dict(base_meta))]

        for payload_idx, meta in payloads:
            rid = str(rnode)
            ev_key = "|".join(
                [
                    rid,
                    str(payload_idx),
                    str(meta.get("chunk_index", "")),
                    str(meta.get("char_start", meta.get("start_offset", ""))),
                    str(meta.get("char_end", meta.get("end_offset", ""))),
                ]
            )
            ev_uri = URIRef(f"{self.namespaces['etd']}evidence_{safe_hash(ev_key)}")

            self.graph.add((ev_uri, RDF.type, etd_ns.Evidence))
            self.graph.add((rnode, etd_ns.hasEvidence, ev_uri))
            self._emit_registered_etd_literal(ev_uri, "datasetId", self.dataset_id, etd_ns, datatype=XSD.string)
            self._emit_registered_etd_literal(ev_uri, "runId", self.run_id, etd_ns, datatype=XSD.string)

            # extractor/source
            src = meta.get("extractor") or meta.get("context_source") or "unknown"
            self._emit_registered_etd_literal(ev_uri, "source", str(src), etd_ns, datatype=XSD.string)

            # confidence (required exactly once on Evidence)
            try:
                cval = float(getattr(rel, "confidence", 0.0) or 0.0)
            except Exception:
                cval = 0.0
            cval = max(0.0, min(1.0, cval))
            self._emit_registered_etd_literal(ev_uri, "confidence", float(cval), etd_ns, datatype=XSD.float)

            # discourse_id
            did = meta.get("discourse_id") or getattr(rel, "discourse_id", None) or getattr(rel, "doc_id", None)
            self._add_discourse_id_literal(ev_uri, did, etd_ns)
            origin_meta = dict(meta)
            if did is not None:
                origin_meta.setdefault("discourse_id", did)
            origin_node = self._resolve_origin_discourse_uri(origin_meta)
            if origin_node is not None:
                self.graph.add((origin_node, RDF.type, etd_ns.DiscourseUnit))
                self.graph.add((ev_uri, etd_ns.extractedFrom, origin_node))

            # --- derive locals (do not mutate meta) ---
            chunk_num = self._coerce_int(meta.get("chunk_index"))
            derived_chunk_index = int(chunk_num) if (chunk_num is not None and chunk_num >= 0) else -1
            derived_chunk_unknown = True if derived_chunk_index == -1 else bool(meta.get("chunk_index_unknown", False))

            derived_source_doc_id = None
            sdoc = meta.get("source_doc_id")
            if sdoc is not None and str(sdoc).strip():
                derived_source_doc_id = str(sdoc).strip()
            elif did is not None and str(did).strip():
                derived_source_doc_id = str(did).strip()

            derived_chunk_id = None
            cid = meta.get("chunk_id")
            if cid is not None and str(cid).strip():
                derived_chunk_id = str(cid).strip()
            elif derived_source_doc_id is not None and not derived_chunk_unknown:
                derived_chunk_id = f"{derived_source_doc_id}:chunk:{derived_chunk_index}"

            derived_start_offset = self._coerce_int(meta.get("start_offset", meta.get("start_char")))
            derived_end_offset = self._coerce_int(meta.get("end_offset", meta.get("end_char")))

            derived_span_hash = None
            sh = meta.get("span_hash")
            if sh is not None and str(sh).strip():
                derived_span_hash = str(sh).strip()
            elif derived_source_doc_id is not None and (
                derived_chunk_id
                or not derived_chunk_unknown
                or derived_start_offset is not None
                or derived_end_offset is not None
            ):
                span_key = "|".join(
                    [
                        derived_source_doc_id or "",
                        derived_chunk_id or "",
                        str(derived_chunk_index if derived_chunk_index >= 0 else ""),
                        str(derived_start_offset if derived_start_offset is not None else ""),
                        str(derived_end_offset if derived_end_offset is not None else ""),
                    ]
                )
                derived_span_hash = safe_hash(span_key)

            derived_meta = {
                "source_doc_id": derived_source_doc_id,
                "chunk_id": derived_chunk_id,
                "chunk_index": derived_chunk_index,
                "chunk_index_unknown": derived_chunk_unknown,
                "char_start": derived_start_offset,
                "char_end": derived_end_offset,
                "start_offset": derived_start_offset,
                "end_offset": derived_end_offset,
                "span_hash": derived_span_hash,
            }

            # field + chunking provenance (production-safe)
            for key in (
                "field",
                "source_doc_id",
                "chunk_id",
                "chunk_index",
                "chunk_index_unknown",
                "groundingStatus",
                "groundingMethod",
                "chunk_total",
                "chunk_start",
                "start_char",
                "end_char",
                "char_start",
                "char_end",
                "start_offset",
                "end_offset",
                "span_hash",
                "batch_id",
                "ingestedAt",
            ):
                val = derived_meta.get(key, meta.get(key))
                if val is None:
                    continue

                if key in (
                    "chunk_index",
                    "chunk_total",
                    "chunk_start",
                    "start_char",
                    "end_char",
                    "char_start",
                    "char_end",
                    "start_offset",
                    "end_offset",
                ):
                    num = self._coerce_int(val)
                    if num is not None:
                        self._emit_registered_etd_literal(ev_uri, key, num, etd_ns, datatype=XSD.integer)
                    continue

                if key == "chunk_index_unknown":
                    bval = self._coerce_bool(val)
                    if bval is not None:
                        self._emit_registered_etd_literal(ev_uri, key, bool(bval), etd_ns, datatype=XSD.boolean)
                    continue

                if key == "ingestedAt":
                    raw_str = str(val).strip()
                    if not raw_str:
                        continue

                    if self._is_iso_datetime(raw_str):
                        self._emit_registered_etd_literal(ev_uri, key, raw_str, etd_ns, datatype=XSD.dateTime)
                    else:
                        self._emit_registered_etd_literal(ev_uri, key, raw_str, etd_ns, datatype=XSD.string)
                    continue

                self._emit_registered_etd_literal(ev_uri, key, str(val).strip(), etd_ns, datatype=XSD.string)

            # evidence text (belt-and-braces: always emit evidenceText)
            evidence = (meta.get("evidence") or getattr(rel, "context", None) or "")
            txt = str(evidence).strip()

            if not txt:
                extractor = str(meta.get("extractor") or "unknown").strip()
                source = str(meta.get("field") or meta.get("context_source") or "unknown").strip()
                txt = f"Evidence unavailable (extractor={extractor}, source={source})."

            # clamp for storage/query safety
            if len(txt) > 500:
                txt = txt[:500]

            self.graph.add((ev_uri, etd_ns.evidenceText, Literal(txt, datatype=XSD.string)))

            qh = meta.get("quote_hash")
            qh = str(qh).strip() if (qh is not None and str(qh).strip()) else (safe_hash(txt) if txt else None)
            if qh:
                self._emit_registered_etd_literal(ev_uri, "quote_hash", qh, etd_ns, datatype=XSD.string)

    def _align_relation_evidence_provenance(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Align relation-level provenance to linked evidence without mutating evidence:
        - Relation provenance is stamped only when evidence provides one unambiguous value.
        - chunk_index=-1 evidence values are treated as unknown and excluded from alignment.
        """
        processed = 0
        ambiguous_did = 0
        ambiguous_chunk = 0
        stamped_did = 0
        stamped_chunk = 0

        for rnode in sorted(self.graph.subjects(RDF.type, etd_ns.Relation), key=str):
            if not isinstance(rnode, URIRef):
                continue
            processed += 1

            evidence_nodes = [ev for ev in self.graph.objects(rnode, etd_ns.hasEvidence) if isinstance(ev, URIRef)]
            if not evidence_nodes:
                continue

            ev_dids: Set[str] = set()
            ev_chunks: Set[int] = set()

            for ev in evidence_nodes:
                for did in self.graph.objects(ev, etd_ns.discourse_id):
                    did_raw = str(did).strip()
                    if did_raw:
                        ev_dids.add(did_raw)
                for chunk in self.graph.objects(ev, etd_ns.chunk_index):
                    cnum = self._coerce_int(chunk)
                    if cnum is not None and cnum >= 0:
                        ev_chunks.add(cnum)

            for old in list(self.graph.objects(rnode, etd_ns.discourse_id)):
                self.graph.remove((rnode, etd_ns.discourse_id, old))
            for old in list(self.graph.objects(rnode, etd_ns.chunk_index)):
                self.graph.remove((rnode, etd_ns.chunk_index, old))

            if len(ev_dids) == 1:
                self._add_discourse_id_literal(rnode, sorted(ev_dids)[0], etd_ns)
                stamped_did += 1
            elif len(ev_dids) > 1:
                ambiguous_did += 1

            if len(ev_chunks) == 1:
                self.graph.add((rnode, etd_ns.chunk_index, Literal(next(iter(ev_chunks)), datatype=XSD.integer)))
                stamped_chunk += 1
            elif len(ev_chunks) > 1:
                ambiguous_chunk += 1

        return {
            "relations_processed": processed,
            "relations_stamped_did": stamped_did,
            "relations_stamped_chunk": stamped_chunk,
            "ambiguous_did": ambiguous_did,
            "ambiguous_chunk": ambiguous_chunk,
        }

    def _finalize_relation_provenance(self, etd_ns: Namespace) -> Dict[str, int]:
        """
        Final deterministic provenance completion for relation nodes.
        Runs after all relation generation/enforcement and before serialization.

        Guarantees:
        - every asserted etd:Relation has exactly one etd:chunk_index (xsd:integer)
        - relation-level datasetId/runId are present once when available
        """

        def _int_values(subject: URIRef, predicate: URIRef, *, non_negative_only: bool) -> List[int]:
            vals: Set[int] = set()
            for o in self.graph.objects(subject, predicate):
                n = self._coerce_int(o)
                if n is None:
                    continue
                if non_negative_only and n < 0:
                    continue
                vals.add(int(n))
            return sorted(vals)

        def _string_values(subject: URIRef, predicate: URIRef) -> List[str]:
            vals: Set[str] = set()
            for o in self.graph.objects(subject, predicate):
                s = str(o).strip()
                if s:
                    vals.add(s)
            return sorted(vals)

        # Build discourse -> known chunk index set from existing graph evidence/entity nodes.
        discourse_chunks: Dict[str, Set[int]] = {}
        for node in list(self.graph.subjects(RDF.type, etd_ns.Evidence)) + list(self.graph.subjects(RDF.type, etd_ns.Entity)):
            if not isinstance(node, URIRef):
                continue
            dids = _string_values(node, etd_ns.discourse_id)
            chunks = _int_values(node, etd_ns.chunk_index, non_negative_only=True)
            if len(dids) == 1 and chunks:
                discourse_chunks.setdefault(dids[0], set()).update(chunks)

        metrics: Dict[str, int] = {
            "relations_processed": 0,
            "filled_from_relation": 0,
            "filled_from_evidence": 0,
            "filled_from_source": 0,
            "filled_from_target": 0,
            "filled_from_discourse_singleton": 0,
            "filled_fallback_unknown": 0,
            "dataset_id_set": 0,
            "run_id_set": 0,
        }

        for rnode in sorted(self.graph.subjects(RDF.type, etd_ns.Relation), key=str):
            if not isinstance(rnode, URIRef):
                continue
            metrics["relations_processed"] += 1

            evidence_nodes = [ev for ev in self.graph.objects(rnode, etd_ns.hasEvidence) if isinstance(ev, URIRef)]
            subj = next((o for o in self.graph.objects(rnode, etd_ns.subject) if isinstance(o, URIRef)), None)
            obj = next((o for o in self.graph.objects(rnode, etd_ns.object) if isinstance(o, URIRef)), None)

            chosen_chunk: Optional[int] = None

            # (a) existing non-negative relation chunk_index
            rel_chunks = _int_values(rnode, etd_ns.chunk_index, non_negative_only=True)
            if rel_chunks:
                chosen_chunk = rel_chunks[0]
                metrics["filled_from_relation"] += 1

            # (b) from linked evidence nodes
            if chosen_chunk is None:
                ev_chunks: Set[int] = set()
                for ev in evidence_nodes:
                    for n in _int_values(ev, etd_ns.chunk_index, non_negative_only=True):
                        ev_chunks.add(n)
                if ev_chunks:
                    chosen_chunk = sorted(ev_chunks)[0]
                    metrics["filled_from_evidence"] += 1

            # (c) from source entity chunk_index
            if chosen_chunk is None and isinstance(subj, URIRef):
                src_chunks = _int_values(subj, etd_ns.chunk_index, non_negative_only=True)
                if src_chunks:
                    chosen_chunk = src_chunks[0]
                    metrics["filled_from_source"] += 1

            # (d) from target entity chunk_index
            if chosen_chunk is None and isinstance(obj, URIRef):
                tgt_chunks = _int_values(obj, etd_ns.chunk_index, non_negative_only=True)
                if tgt_chunks:
                    chosen_chunk = tgt_chunks[0]
                    metrics["filled_from_target"] += 1

            # (e) discourse singleton chunk
            if chosen_chunk is None:
                dids: Set[str] = set(_string_values(rnode, etd_ns.discourse_id))
                for ev in evidence_nodes:
                    dids.update(_string_values(ev, etd_ns.discourse_id))
                if len(dids) == 1:
                    did = next(iter(dids))
                    ds_chunks = sorted(discourse_chunks.get(did, set()))
                    if len(ds_chunks) == 1:
                        chosen_chunk = ds_chunks[0]
                        metrics["filled_from_discourse_singleton"] += 1

            # (f) final fallback
            if chosen_chunk is None:
                chosen_chunk = -1
                metrics["filled_fallback_unknown"] += 1

            # Enforce exactly one chunk_index triple, typed xsd:integer.
            for old in list(self.graph.objects(rnode, etd_ns.chunk_index)):
                self.graph.remove((rnode, etd_ns.chunk_index, old))
            self.graph.add((rnode, etd_ns.chunk_index, Literal(int(chosen_chunk), datatype=XSD.integer)))

            # Relation-level datasetId / runId completion.
            rel_ds = _string_values(rnode, etd_ns.datasetId)
            rel_run = _string_values(rnode, etd_ns.runId)
            ev_ds: Set[str] = set()
            ev_run: Set[str] = set()
            for ev in evidence_nodes:
                ev_ds.update(_string_values(ev, etd_ns.datasetId))
                ev_run.update(_string_values(ev, etd_ns.runId))

            chosen_ds = (rel_ds[0] if rel_ds else (sorted(ev_ds)[0] if ev_ds else str(self.dataset_id or "").strip()))
            chosen_run = (rel_run[0] if rel_run else (sorted(ev_run)[0] if ev_run else str(self.run_id or "").strip()))

            for old in list(self.graph.objects(rnode, etd_ns.datasetId)):
                self.graph.remove((rnode, etd_ns.datasetId, old))
            for old in list(self.graph.objects(rnode, etd_ns.runId)):
                self.graph.remove((rnode, etd_ns.runId, old))

            if chosen_ds:
                self.graph.add((rnode, etd_ns.datasetId, Literal(chosen_ds, datatype=XSD.string)))
                metrics["dataset_id_set"] += 1
            if chosen_run:
                self.graph.add((rnode, etd_ns.runId, Literal(chosen_run, datatype=XSD.string)))
                metrics["run_id_set"] += 1

        return metrics

    def _assert_evidence_contract(self) -> None:
        """Raise if any Evidence node misses required audit-grade fields."""
        etd_ns = Namespace(self.namespaces["etd"])
        violations: List[str] = []
        for ev in self.graph.subjects(RDF.type, etd_ns.Evidence):
            if len(list(self.graph.objects(ev, etd_ns.evidenceText))) != 1:
                violations.append(f"{ev} evidenceText!=1")
            if len(list(self.graph.objects(ev, etd_ns.confidence))) != 1:
                violations.append(f"{ev} confidence!=1")
            if len(list(self.graph.objects(ev, etd_ns.chunk_index))) != 1:
                violations.append(f"{ev} chunk_index!=1")
        if violations:
            sample = "; ".join(violations[:10])
            raise RuntimeError(
                f"Evidence contract violation ({len(violations)} issues): {sample}"
            )


    def _add_relation_metadata(self, rnode: BNode, metadata: Dict[str, Any], etd_ns: Namespace) -> None:
        """
        Attach selected provenance metadata for relations:
        - discourse_id
        - surface_form
        - context_source
        - chunk_index, chunk_total
        - start_char, end_char
        """
        if not metadata:
            return

        # Backward compatibility: accept doc_id if discourse_id missing
        if "discourse_id" not in metadata and "doc_id" in metadata:
            metadata["discourse_id"] = metadata.get("doc_id")

        # discourse_id
        self._add_discourse_id_literal(rnode, metadata.get("discourse_id"), etd_ns)
        origin_node = self._resolve_origin_discourse_uri(metadata)
        if origin_node is not None:
            self.graph.add((origin_node, RDF.type, etd_ns.DiscourseUnit))
            self.graph.add((rnode, etd_ns.extractedFrom, origin_node))

        # surface_form, context_source, field
        for key in ("surface_form", "context_source", "groundingStatus", "groundingMethod", "field"):
            if metadata.get(key) is not None:
                val = str(metadata[key]).strip()
                if val:
                    self._emit_registered_etd_literal(rnode, key, val, etd_ns, datatype=XSD.string)


        # Integer-like fields (production-safe: coerce + preserve raw)
        for key in ("chunk_index", "chunk_total", "start_char", "end_char"):
            if metadata.get(key) is not None:
                raw_val = metadata.get(key)
                num = self._coerce_int(raw_val)

                if num is not None:
                    self._emit_registered_etd_literal(rnode, key, num, etd_ns, datatype=XSD.integer)

    def _add_inferred_provenance(self, rnode: Union[URIRef, BNode], rel: Relation, etd_ns: Namespace) -> None:
        """
        Attach audit-grade provenance to inferred relations:
        - Prefer prov:wasDerivedFrom when supporting relation URIs exist.
        - Fallback to etd:inferenceRule.
        """
        meta = getattr(rel, "metadata", None) or {}

        derived_candidates: List[Any] = []
        for key in (
            "supporting_relation_uris",
            "derived_from",
            "was_derived_from",
            "source_relation_uris",
            "support_relations",
        ):
            val = meta.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                derived_candidates.extend(val)
            else:
                derived_candidates.append(val)

        added_derivation = False
        for item in derived_candidates:
            if isinstance(item, dict):
                item = item.get("uri") or item.get("relation_uri") or item.get("id")
            if item is None:
                continue
            raw = str(item).strip()
            if not raw:
                continue
            if raw.startswith("http://") or raw.startswith("https://"):
                ref = _safe_uriref(raw, self.namespaces)
            else:
                ref = _safe_uriref(f"{self.namespaces['etd']}{raw.lstrip('#/')}", self.namespaces)
            self.graph.add((rnode, PROV.wasDerivedFrom, ref))
            added_derivation = True

        if not added_derivation:
            rule = (
                meta.get("inferenceRule")
                or meta.get("inference_rule")
                or meta.get("inference_type")
                or meta.get("rule")
                or "unspecified_inference_rule"
            )
            self.graph.add((rnode, etd_ns.inferenceRule, Literal(str(rule), datatype=XSD.string)))

    # ------------------------------------------------------------------
    # Dataset-level metadata
    # ------------------------------------------------------------------
    def _add_dataset_metadata(self) -> None:
        """
        Add dataset/run/export metadata as RDF triples on a single export node.
        """
        etd_ns = Namespace(self.namespaces["etd"])
        onto_uri = Namespace(self.namespaces["etd"])["ontology"]
        export_node = self.export_node

        meta = self.annotations.get("metadata", {}) or {}

        # Export node for AI traceability/queryability.
        self.graph.add((export_node, RDF.type, etd_ns.Export))
        self.graph.add((export_node, etd_ns.datasetId, Literal(self.dataset_id, datatype=XSD.string)))
        self.graph.add((export_node, etd_ns.runId, Literal(self.run_id, datatype=XSD.string)))
        self.graph.add((export_node, PROV.generatedAtTime, Literal(self.generated_at_time, datatype=XSD.dateTime)))
        self.graph.add((export_node, etd_ns.pipelineVersion, Literal(self.pipeline_version, datatype=XSD.string)))
        self.graph.add((export_node, etd_ns.gitCommit, Literal(self.git_commit, datatype=XSD.string)))
        self.graph.add((export_node, etd_ns.shaclProfileVersion, Literal(self.shacl_profile_version, datatype=XSD.string)))
        self.graph.add((export_node, etd_ns.exportMode, Literal(self.export_mode, datatype=XSD.string)))
        self.graph.add((export_node, PROV.wasDerivedFrom, onto_uri))

        # Avoid duplicate ontology titles; schema layer already sets a default title.
        title = meta.get("title")
        if title:
            self.graph.add((onto_uri, DCTERMS.title, Literal(title, lang="en")))

        desc = meta.get("description")
        if desc:
            self.graph.add(
                (onto_uri, DCTERMS.description, Literal(desc, lang="en"))
            )

        creator = meta.get("creator")
        if creator:
            self.graph.add(
                (onto_uri, DCTERMS.creator, Literal(creator, datatype=XSD.string))
            )

        version = meta.get("version")
        if version:
            self.graph.add(
                (onto_uri, etd_ns.pipelineVersion, Literal(version, datatype=XSD.string))
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _to_rdf_node(self, value: Any) -> Union[Literal, URIRef]:
        """
        Convert Python value → RDF node.

        - Numbers / booleans → typed literals
        - Strings that look like IRIs or CURIEs → safe URIRefs
        - Everything else → xsd:string literal
        """
        if isinstance(value, URIRef):
            return value

        if isinstance(value, bool):
            return Literal(value, datatype=XSD.boolean)

        if isinstance(value, int):
            return Literal(value, datatype=XSD.integer)

        if isinstance(value, float):
            return Literal(value, datatype=XSD.float)

        if isinstance(value, str):
            s = value.strip()

            # HTTP(S) IRI
            if s.startswith("http://") or s.startswith("https://"):
                return _safe_uriref(s, self.namespaces)

            # CURIE with known prefix
            if ":" in s and " " not in s:
                prefix, local = s.split(":", 1)
                if prefix in self.namespaces:
                    iri = self.namespaces[prefix] + local
                    return _safe_uriref(iri, self.namespaces)

            return Literal(s, datatype=XSD.string)

        # Fallback: string literal
        return Literal(str(value), datatype=XSD.string)

    def _sanitize(self, text: str) -> str:
        """Convert arbitrary string into a safe NCName-like local part."""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return "Resource"
        text = re.sub(r"[^\w]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_") or "Resource"

    # ------------------------------------------------------------------
    # Public serialization methods
    # ------------------------------------------------------------------
    def serialize_to_turtle(self, filepath: Path) -> None:
        """Turtle serialization (.ttl)."""
        try:
            self._assert_evidence_contract()
            self.graph.serialize(destination=str(filepath), format="turtle")
            print(f"  ✓ {filepath.name}")
        except Exception as e:
            print(f"  ✗ Turtle serialization failed: {e}")

    def serialize_to_rdf(self, filepath: Path) -> None:
        """RDF/XML serialization (.rdf)."""
        try:
            self._assert_evidence_contract()
            self.graph.serialize(destination=str(filepath), format="xml")
            print(f"  ✓ {filepath.name}")
        except Exception as e:
            print(f"  ✗ RDF/XML serialization failed: {e}")

    def serialize_to_jsonld(self, filepath: Path) -> None:
        """JSON-LD serialization (.jsonld)."""
        try:
            self._assert_evidence_contract()
            self.graph.serialize(destination=str(filepath), format="json-ld")
            print(f"  ✓ {filepath.name}")
        except Exception as e:
            print(f"  ✗ JSON-LD serialization failed: {e}")

    def serialize_to_owl(self, filepath: Path) -> None:
        """OWL/XML serialization (.owl) – for tools expecting ontology-like serialisation."""
        try:
            self._assert_evidence_contract()
            self.graph.serialize(destination=str(filepath), format="xml")
            print(f"  ✓ {filepath.name}")
        except Exception as e:
            print(f"  ✗ OWL serialization failed: {e}")


# ----------------------------------------------------------------------
# Helper: safe URIRef using namespaces + make_uri_safe
# ----------------------------------------------------------------------
def _safe_uriref(iri_or_curie: str, namespaces: Dict[str, str]) -> URIRef:
    """
    Expand CURIEs and produce a safe URIRef, BUT with special handling
    for ETD-local IRIs so entity URIs do not get percent-encoded.
    """
    if not isinstance(iri_or_curie, str):
        iri_or_curie = str(iri_or_curie)

    iri_or_curie = iri_or_curie.strip()

    # Try CURIE expansion first
    if ":" in iri_or_curie and not iri_or_curie.startswith("http"):
        prefix, local = iri_or_curie.split(":", 1)
        base = namespaces.get(prefix)
        if base:
            iri = base + local
        else:
            iri = iri_or_curie
    else:
        iri = iri_or_curie

    # ------------------------------------------------------------
    # 🔥 FIX: Normalise ETD local names BEFORE encoding
    # ------------------------------------------------------------
    etd_base = namespaces.get("etd")
    if etd_base and isinstance(iri, str) and iri.startswith(etd_base):
        tail = iri[len(etd_base):]

        # If tail contains spaces → canonicalize instead of percent-encoding
        if any(ch.isspace() for ch in tail):
            local = canonicalize_local_name(tail)
            return URIRef(etd_base + local)

    # For all other IRIs (Wikidata, DPV, VAIR, URLs), use normal encoding
    return make_uri_safe(iri)



