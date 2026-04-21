"""
ontology/mapper.py
------------------
OntologyMapper for ETD-Hub Semantic KG Pipeline.

Design:
- map_entity() / map_relation() return CURIEs from ontology_mappings.py (e.g., "dpv:GDPR")
- Pipeline (or serializer) may store ontology_uri as expanded IRI if needed
- Any place we create rdflib.URIRef MUST expand CURIEs first (critical fix)

Key fixes vs legacy:
- map_relation() fallback is etd:relatedTo (NOT schema:relatedTo)
- legacy add_relations(...) expands CURIEs before URIRef(...)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from rdflib import URIRef

from models import Entity, Relation
from config import Config
from ontology_mappings import (
    CONCEPT_TO_ONTOLOGY_CLASSES,
    ENTITY_TYPE_TO_ONTOLOGY,
    RELATION_TO_ONTOLOGY_PROPERTY,
    DOMAIN_TO_VAIR,
    PROBLEM_CATEGORY_TO_ONTOLOGY,
    FMO_MAPPINGS,
    ETHICAL_CONCEPT_TO_AIRO_RISK,
    DEMOGRAPHIC_LEXICON,
    RELATION_MAPPINGS,
)

from utils.helpers import expand_curie, safe_uri

ETD = Config.ETD_NS

class OntologyMapper:
    """Maps extracted entities and relations to standard AI ethics ontologies."""

    def __init__(self):
        self.concept_mappings = CONCEPT_TO_ONTOLOGY_CLASSES
        self.entity_type_mappings = ENTITY_TYPE_TO_ONTOLOGY
        self.relation_mappings = RELATION_TO_ONTOLOGY_PROPERTY
        self.domain_mappings = DOMAIN_TO_VAIR
        self.problem_mappings = PROBLEM_CATEGORY_TO_ONTOLOGY

    # ============================================================
    # ENTITY MAPPING
    # ============================================================
    def map_entity(self, entity: Entity) -> List[str]:
        """
        Map entity to ontology classes based on:
        - LLM semantic type (Dataset, Person, BiasType…)
        - lexical concept match in label (bias, transparency…)
        - risk metadata from extractor (airo_risk_type / airo_type)
        Returns: CURIE list (e.g., ["dcat:Dataset", "airo:BiasRisk"])
        """
        ontology_classes: List[str] = []

        # 1) Map by entity semantic type
        ent_type = (entity.entity_type or "").strip()
        if ent_type in self.entity_type_mappings:
            ontology_classes.extend(self.entity_type_mappings[ent_type])

        # 2) Lexical concept match
        label_lower = (entity.label or "").lower()
        for concept, classes in self.concept_mappings.items():
            if concept in label_lower:
                ontology_classes.extend(classes)

        # 3) AIRO risk type (already a CURIE in your mappings)
        if entity.metadata and "airo_risk_type" in entity.metadata:
            ontology_classes.append(entity.metadata["airo_risk_type"])

        # 4) Entity.metadata['airo_type'] -> "airo:<Type>"
        if entity.metadata and "airo_type" in entity.metadata:
            ontology_classes.append(f"airo:{entity.metadata['airo_type']}")

        # Dedup, preserve order
        seen = set()
        unique: List[str] = []
        for c in ontology_classes:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)

        return unique

    # ============================================================
    # RELATION MAPPING
    # ============================================================
    def map_relation(self, relation: Relation) -> List[str]:
        """
        Map relation label (canonical, schema-aligned) to ontology properties.
        Returns: CURIE list (e.g., ["aipo:uses"]).

        IMPORTANT:
        - Fallback must align with your schema: etd:relatedTo
        """
        relation_label = (relation.relation or "").strip()

        # direct mapping
        if relation_label in self.relation_mappings:
            return self.relation_mappings[relation_label]

        # accept already-curied predicates only if they can be expanded
        if ":" in relation_label and " " not in relation_label:
            try:
                _ = expand_curie(relation_label)
                return [relation_label]
            except Exception:
                pass

        # ✅ Correct fallback
        out = ["etd:relatedTo"]  # or mapping result
        seen = set()
        uniq = []
        for x in out:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    # ============================================================
    # DOMAIN MAPPING
    # ============================================================
    def map_domain(self, domain: str) -> Optional[str]:
        """Map text domain to VAIR domain ontology CURIE."""
        if not domain:
            return None
        d = domain.lower().strip()
        for k, v in self.domain_mappings.items():
            if k in d:
                return v
        return None

    def map_problem_category(self, problem_category: str) -> List[str]:
        """
        Map a problem category string to ontology CURIEs.
        Returns a list of CURIEs (may be empty).
        """
        if not problem_category:
            return []

        pc = problem_category.lower().strip()
        out: List[str] = []

        for k, v in self.problem_mappings.items():
            if k in pc:
                # v can be str or list in mappings; normalise to list
                if isinstance(v, list):
                    out.extend(v)
                else:
                    out.append(v)

        # Deduplicate, preserve order
        seen = set()
        uniq = []
        for x in out:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq


    # ============================================================
    # FAIRNESS METRICS
    # ============================================================
    def get_fairness_metrics(self, entity: Entity) -> List[str]:
        """Detect fairness metric CURIEs via keyword triggers."""
        label = (entity.label or "").lower()
        hits: List[str] = []
        for trigger, metric_curie in FMO_MAPPINGS.items():
            if trigger in label:
                hits.append(metric_curie)
        # Dedup
        seen = set()
        out = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out

    # ============================================================
    # COMPLIANCE REQUIREMENTS (simple heuristic)
    # ============================================================
    def get_compliance_requirements(self, entity: Entity, domain: str = "") -> List[str]:
        """
        Returns CURIEs of compliance requirements / domains.
        (Lightweight heuristic; keep as-is for MVP)
        """
        requirements: List[str] = []
        domain_lower = domain.lower() if domain else ""

        if "health" in domain_lower or "medical" in domain_lower:
            requirements.append("vair:HealthcareDomain")
        if "justice" in domain_lower or "legal" in domain_lower:
            requirements.append("vair:JusticeDomain")

        # Dedup
        seen = set()
        out = []
        for r in requirements:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    # ============================================================
    # ENTITY ENRICHMENT WRAPPER
    # ============================================================
    def enrich_entity(self, entity: Entity, domain: str = "") -> Dict[str, Any]:
        """Return combined ontology enrichment for a single entity."""
        enrichment: Dict[str, Any] = {
            "ontology_classes": self.map_entity(entity),
            "fairness_metrics": self.get_fairness_metrics(entity),
            "compliance_requirements": self.get_compliance_requirements(entity, domain),
        }

        if domain:
            vair_dom = self.map_domain(domain)
            if vair_dom:
                enrichment["vair_domain"] = vair_dom

        return enrichment

    # ============================================================
    # RELATION ENRICHMENT
    # ============================================================
    def enrich_relation(self, relation: Relation) -> Dict[str, Any]:
        return {"ontology_properties": self.map_relation(relation)}


# ============================================================
# LEGACY: Keyword-driven graph enrichment
# Used in older parts of the pipeline (still supported).
#
# CRITICAL FIX:
# - Expand CURIEs before URIRef(...)
# - Use safe_uri for ETD-local nodes (demographics)
# ============================================================
def add_relations(resource_uri: str, extracted_terms: List[str], kg_graph) -> None:
    """
    Keyword → Ontology predicate mapper (legacy helper).
    """
    subj = safe_uri(resource_uri)

    for term in extracted_terms:
        key = (term or "").lower().strip().replace("-", " ")
        if not key:
            continue

        # DOMAIN → VAIR
        for domain_keyword, vair_curie in DOMAIN_TO_VAIR.items():
            if domain_keyword in key:
                kg_graph.add((
                    subj,
                    URIRef(expand_curie(RELATION_MAPPINGS["domain"])),
                    URIRef(expand_curie(vair_curie)),
                ))
                break

        # ETHICAL CONCEPT → AIRO/AIPO
        if key in ETHICAL_CONCEPT_TO_AIRO_RISK:
            kg_graph.add((
                subj,
                URIRef(expand_curie(RELATION_MAPPINGS["ethicalConcept"])),
                URIRef(expand_curie(ETHICAL_CONCEPT_TO_AIRO_RISK[key])),
            ))
            continue

        # DEMOGRAPHIC → ETD
        if key in DEMOGRAPHIC_LEXICON:
            pop_curie = DEMOGRAPHIC_LEXICON[key]  # typically "etd:Race_Black" etc.
            kg_graph.add((
                subj,
                URIRef(expand_curie(RELATION_MAPPINGS["demographic"])),
                URIRef(expand_curie(pop_curie)),
            ))
            continue

        # FAIRNESS METRIC → FMO
        matched_metric = False
        for trigger, metric_curie in FMO_MAPPINGS.items():
            if trigger in key:
                kg_graph.add((
                    subj,
                    URIRef(expand_curie(RELATION_MAPPINGS["fairnessMetric"])),
                    URIRef(expand_curie(metric_curie)),
                ))
                matched_metric = True
                break
        if matched_metric:
            continue

        # CONCEPT → GENERIC RELATION CLASS
        if key in CONCEPT_TO_ONTOLOGY_CLASSES:
            for class_curie in CONCEPT_TO_ONTOLOGY_CLASSES[key]:
                # pick predicate bucket (your mapping keys are camelCase)
                if class_curie.startswith("dpv:"):
                    pred_curie = RELATION_MAPPINGS["privacyConcept"]
                elif class_curie.startswith(("aipo:", "relaieo:", "hudock:")):
                    pred_curie = RELATION_MAPPINGS["governanceOrRegulation"]
                elif class_curie.startswith(("prov:", "foaf:", "org:")):
                    pred_curie = RELATION_MAPPINGS["stakeholder"]
                else:
                    pred_curie = RELATION_MAPPINGS["activityOrProcess"]

                kg_graph.add((
                    subj,
                    URIRef(expand_curie(pred_curie)),
                    URIRef(expand_curie(class_curie)),
                ))
