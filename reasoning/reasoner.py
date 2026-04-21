"""
reasoner.py (Production, schema-aligned)
---------------------------------------
Guarantees:
- Normalises predicates via utils.predicate_map.normalize_predicate
- NEVER mints non-schema predicates (whitelist from CANONICAL_REL_PREDICATES values)
- Deterministic ETD relation URIs for streaming stability
- Inference uses ONLY ETD schema predicates (incl. new governance predicates)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, Union

from config import Config
from models import Relation
from utils.helpers import safe_hash
from utils.predicate_map import normalize_predicate, CANONICAL_REL_PREDICATES

logger = logging.getLogger(__name__)


@dataclass
class ReasonerStats:
    num_input_relations: int = 0
    num_base_after_filter: int = 0
    num_iterations: int = 0
    num_inferred_total: int = 0
    num_inferred_unique: int = 0
    per_rule_counts: Dict[str, int] = field(default_factory=dict)

    def inc(self, rule: str, n: int) -> None:
        self.per_rule_counts[rule] = self.per_rule_counts.get(rule, 0) + n


class MultiHopReasoner:
    """
    Schema-aligned multi-hop reasoner operating on Relation objects.
    """

    def __init__(self, min_confidence: float = 0.5, max_iterations: int = 3, debug: bool = False) -> None:
        self.min_confidence = float(min_confidence)
        self.max_iterations = int(max_iterations)
        self.debug = bool(debug)

        self.base_etd = Config.NAMESPACES["etd"]

        # ✅ Allowed predicates are EXACTLY the canonical schema property names
        self.allowed_predicates: Set[str] = set(CANONICAL_REL_PREDICATES.values())

        # ---- Reasoning configuration (all must be schema predicates) ----
        self.transitive_relations: Set[str] = {
            "relatedTo",
            "uses",
            "trainedOn",
            "containsSubject",
            "partOfDomain",
            "includes",
        }

        self.symmetric_relations: Set[str] = {
            "relatedTo",
            "associatedWith",
        }

        # Composition rules: (p1, p2) -> (p3, cap_conf)
        self.composition_rules: List[Tuple[Tuple[str, str], str, float]] = [
            # Training/data → use
            (("trainedOn", "containsSubject"), "uses", 0.70),
            (("trainedOn", "hasRisk"), "exhibitsRisk", 0.75),

            # Domain propagation
            (("containsSubject", "partOfDomain"), "partOfDomain", 0.70),

            # Causality closure
            (("causes", "causes"), "causes", 0.70),
            (("causes", "affects"), "affects", 0.65),
            (("affects", "affects"), "affects", 0.60),

            # Governance closure (lightweight, safe)
            (("violates", "relatedTo"), "violates", 0.55),
        ]

        # Stance buckets (schema predicates only)
        self.pos_stance = {"supports", "endorses", "compliesWith"}
        self.neg_stance = {"criticizes", "disagreesWith", "opposes", "violates"}

        self.last_stats: Optional[ReasonerStats] = None

        if self.debug:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                logger.addHandler(h)

    RULE_IDS: Dict[str, str] = {
        "transitive": "transitive_closure:v1",
        "composition": "multihop_reasoning:v1",
        "symmetric": "multihop_reasoning:v1",
        "polarity": "schema_bridge:v1",
        "schema_risk": "schema_bridge:v1",
        "schema_group_risk": "schema_bridge:v1",
        "schema_domain": "schema_bridge:v1",
        "schema_mitigation": "schema_bridge:v1",
        "schema_entity_mitigation": "schema_bridge:v1",
        "schema_mentions_risk": "schema_bridge:v1",
        "schema_instrument_risk": "schema_bridge:v1",
        "schema_affectedBy_risk": "schema_bridge:v1",
    }

    def _rule_identifier(self, rule: str) -> str:
        return self.RULE_IDS.get(rule, "multihop_reasoning:v1")

    # --------------------------
    # Public API
    # --------------------------

    def infer_relations(self, relations: List[Union[Relation, Dict[str, Any]]]) -> List[Relation]:
        stats = ReasonerStats(num_input_relations=len(relations))

        normalized_relations: List[Relation] = []
        for idx, rel in enumerate(relations):
            if isinstance(rel, Relation):
                normalized_relations.append(rel)
                continue
            if isinstance(rel, dict):
                try:
                    normalized_relations.append(Relation.from_dict(rel))
                except Exception as exc:
                    raise TypeError(f"infer_relations() invalid relation dict at index {idx}: {exc}") from exc
                continue
            raise TypeError(
                f"infer_relations() expected Relation or dict at index {idx}, got {type(rel)}"
            )

        base: List[Relation] = []
        for r in normalized_relations:
            pred = normalize_predicate(getattr(r, "relation", "") or "")
            if not pred or pred not in self.allowed_predicates:
                continue

            conf = float(getattr(r, "confidence", 0.0) or 0.0)
            if conf < self.min_confidence:
                continue

            r.relation = pred  # normalise in-place
            base.append(r)

        stats.num_base_after_filter = len(base)
        known: Set[Tuple[str, str, str]] = {(r.source, r.relation, r.target) for r in base}

        inferred_all: List[Relation] = []

        for it in range(1, self.max_iterations + 1):
            stats.num_iterations = it

            effective = base + inferred_all
            graph_by_source, by_pred = self._index(effective)

            newly: List[Relation] = []
            t = self._infer_transitive(graph_by_source, known, it)
            stats.inc("transitive", len(t))
            newly.extend(t)

            c = self._infer_composition(graph_by_source, known, it)
            stats.inc("composition", len(c))
            newly.extend(c)

            s = self._infer_symmetric(effective, known, it)
            stats.inc("symmetric", len(s))
            newly.extend(s)

            p = self._infer_polarity(graph_by_source, known, it)
            stats.inc("polarity", len(p))
            newly.extend(p)

            sch = self._infer_schema_rules(by_pred, known, it)
            stats.inc("schema", len(sch))
            newly.extend(sch)

            # Dedup + accept
            accepted: List[Relation] = []
            for r in newly:
                k = (r.source, r.relation, r.target)
                if k in known:
                    continue
                known.add(k)
                accepted.append(r)

            inferred_all.extend(accepted)
            if not accepted:
                break

        out = self._dedup_vs_base(inferred_all, base)
        stats.num_inferred_total = len(inferred_all)
        stats.num_inferred_unique = len(out)
        self.last_stats = stats
        return out

    # --------------------------
    # Indexing
    # --------------------------

    def _index(
        self, relations: Iterable[Relation]
    ) -> Tuple[DefaultDict[str, List[Tuple[str, str, float]]], DefaultDict[str, List[Relation]]]:
        g: DefaultDict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        byp: DefaultDict[str, List[Relation]] = defaultdict(list)

        for r in relations:
            pred = getattr(r, "relation", None)
            if not pred or pred not in self.allowed_predicates:
                continue

            conf = float(getattr(r, "confidence", 0.0) or 0.0)
            # do NOT re-filter by min_confidence here; caller already filtered
            g[r.source].append((pred, r.target, conf))
            byp[pred].append(r)

        return g, byp

    # --------------------------
    # Deterministic minting
    # --------------------------

    def _mint(self, s: str, p: str, o: str, it: int, rule: str) -> str:
        return f"{self.base_etd}rel_{safe_hash(f'{s}|{p}|{o}|it={it}|rule={rule}')}"  # stable

    def _make(self, s: str, p: str, o: str, conf: float, it: int, rule: str, meta: Optional[dict] = None) -> Optional[Relation]:
        p2 = normalize_predicate(p or "")
        if not p2 or p2 not in self.allowed_predicates:
            return None
        conf2 = float(conf or 0.0)
        if conf2 < self.min_confidence:
            return None

        m = dict(meta or {})
        m.update({"inference_type": rule, "iteration": it})
        rid = self._rule_identifier(rule)
        m["inference_rule"] = rid
        m["inferenceRule"] = rid
        doc_id = None
        if meta and isinstance(meta, dict):
            doc_id = meta.get("discourse_id") or meta.get("doc_id")
        
        return Relation(
            uri=self._mint(s, p2, o, it, rule),
            source=s,
            relation=p2,
            target=o,
            confidence=conf2,
            is_inferred=True,
            doc_id=doc_id,
            metadata=m,
        )

    # --------------------------
    # Rules
    # --------------------------

    def _infer_transitive(self, g: Dict[str, List[Tuple[str, str, float]]], known: Set[Tuple[str, str, str]], it: int) -> List[Relation]:
        out: List[Relation] = []
        for s, edges in g.items():
            for p1, mid, c1 in edges:
                if p1 not in self.transitive_relations or mid not in g:
                    continue
                for p2, o, c2 in g[mid]:
                    if p1 != p2 or s == o:
                        continue
                    if (s, p1, o) in known:
                        continue
                    r = self._make(s, p1, o, min(c1, c2), it, "transitive", {"intermediate": mid})
                    if r:
                        out.append(r)
        return out

    def _infer_composition(self, g: Dict[str, List[Tuple[str, str, float]]], known: Set[Tuple[str, str, str]], it: int) -> List[Relation]:
        out: List[Relation] = []
        for s, edges in g.items():
            for p1, mid, c1 in edges:
                if mid not in g:
                    continue
                for p2, o, c2 in g[mid]:
                    if s == o:
                        continue
                    for (a, b), newp, cap in self.composition_rules:
                        if p1 == a and p2 == b:
                            if (s, newp, o) in known:
                                continue
                            r = self._make(s, newp, o, min(c1, c2, cap), it, "composition", {"intermediate": mid, "rule": f"{a}∘{b}->{newp}"})
                            if r:
                                out.append(r)
        return out

    def _infer_symmetric(self, rels: Iterable[Relation], known: Set[Tuple[str, str, str]], it: int) -> List[Relation]:
        out: List[Relation] = []
        for r in rels:
            p = normalize_predicate(getattr(r, "relation", "") or "")
            if p not in self.symmetric_relations:
                continue
            c = float(getattr(r, "confidence", 0.0) or 0.0)
            if c < self.min_confidence:
                continue
            if (r.target, p, r.source) in known:
                continue
            rr = self._make(r.target, p, r.source, c, it, "symmetric")
            if rr:
                out.append(rr)
        return out

    def _infer_polarity(self, g: Dict[str, List[Tuple[str, str, float]]], known: Set[Tuple[str, str, str]], it: int) -> List[Relation]:
        out: List[Relation] = []
        for s, edges in g.items():
            flags: DefaultDict[str, Dict[str, bool]] = defaultdict(lambda: {"pos": False, "neg": False})
            for p, o, c in edges:
                if c < self.min_confidence:
                    continue
                if p in self.pos_stance:
                    flags[o]["pos"] = True
                if p in self.neg_stance:
                    flags[o]["neg"] = True

            for o, f in flags.items():
                if f["pos"] and f["neg"] and (s, "disagreesWith", o) not in known:
                    r = self._make(s, "disagreesWith", o, 0.60, it, "polarity")
                    if r:
                        out.append(r)
        return out

    def _infer_schema_rules(self, by_pred: DefaultDict[str, List[Relation]], known: Set[Tuple[str, str, str]], it: int) -> List[Relation]:
        out: List[Relation] = []

        trained_on = by_pred.get("trainedOn", [])
        has_risk = by_pred.get("hasRisk", [])
        exhibits_risk = by_pred.get("exhibitsRisk", [])
        particularly_affects = by_pred.get("particularlyAffects", [])
        contains_subject = by_pred.get("containsSubject", [])
        part_of_domain = by_pred.get("partOfDomain", [])
        mitigates = by_pred.get("mitigates", [])
        mitigated_by = by_pred.get("mitigatedBy", [])
        mentions = by_pred.get("mentions", [])
        references_instrument = by_pred.get("referencesInstrument", [])
        affected_by = by_pred.get("affectedBy", [])

        # trainedOn(D, X) & hasRisk(X, R) -> exhibitsRisk(D, R)
        for tr in trained_on:
            for hr in has_risk:
                if tr.target != hr.source:
                    continue
                if (tr.source, "exhibitsRisk", hr.target) in known:
                    continue
                meta = {"doc_id": getattr(tr, "doc_id", None) or getattr(hr, "doc_id", None)}
                r = self._make(
                    tr.source,
                    "exhibitsRisk",
                    hr.target,
                    min(float(tr.confidence), float(hr.confidence), 0.80),
                    it,
                    "schema_risk",
                    meta,
                )                
                if r:
                    out.append(r)

        # exhibitsRisk(E, R) & particularlyAffects(R, G) -> affectedBy(G, R)
        for er in exhibits_risk:
            for pa in particularly_affects:
                if er.target != pa.source:
                    continue
                if (pa.target, "affectedBy", er.target) in known:
                    continue
                r = self._make(pa.target, "affectedBy", er.target, min(float(er.confidence), float(pa.confidence), 0.80), it, "schema_group_risk")
                if r:
                    out.append(r)

        # containsSubject(D, E) & partOfDomain(D, Dom) -> partOfDomain(E, Dom)
        for cs in contains_subject:
            for pod in part_of_domain:
                if cs.source != pod.source:
                    continue
                if (cs.target, "partOfDomain", pod.target) in known:
                    continue
                r = self._make(cs.target, "partOfDomain", pod.target, min(float(cs.confidence), float(pod.confidence), 0.80), it, "schema_domain")
                if r:
                    out.append(r)

        # hasRisk(X, R) & mitigates(X, M) -> mitigatedBy(R, M)
        for hr in has_risk:
            for mi in mitigates:
                if hr.source != mi.source:
                    continue
                if (hr.target, "mitigatedBy", mi.target) in known:
                    continue
                r = self._make(hr.target, "mitigatedBy", mi.target, min(float(hr.confidence), float(mi.confidence), 0.75), it, "schema_mitigation")
                if r:
                    out.append(r)

        # hasRisk(X, R) & mitigatedBy(R, M) -> mitigatedBy(X, M)
        for hr in has_risk:
            for mb in mitigated_by:
                if hr.target != mb.source:
                    continue
                if (hr.source, "mitigatedBy", mb.target) in known:
                    continue
                r = self._make(hr.source, "mitigatedBy", mb.target, min(float(hr.confidence), float(mb.confidence), 0.70), it, "schema_entity_mitigation")
                if r:
                    out.append(r)

        # mentions(A, X) & hasRisk(X, R) -> hasRisk(A, R)
        for m in mentions:
            for hr in has_risk:
                if m.target != hr.source:
                    continue
                if (m.source, "hasRisk", hr.target) in known:
                    continue
                r = self._make(
                    m.source, "hasRisk", hr.target,
                    min(float(m.confidence), float(hr.confidence), 0.70),
                    it, "schema_mentions_risk",
                    {
                    "via": "mentions",
                    "intermediate": hr.source,
                    "doc_id": getattr(m, "doc_id", None) or getattr(hr, "doc_id", None),
                    }  
                )
                if r:
                    out.append(r)

        # referencesInstrument(A, I) & hasRisk(I, R) -> hasRisk(A, R)
        for ref in references_instrument:
            for hr in has_risk:
                if ref.target != hr.source:
                    continue
                if (ref.source, "hasRisk", hr.target) in known:
                    continue
                r = self._make(
                    ref.source, "hasRisk", hr.target,
                    min(float(ref.confidence), float(hr.confidence), 0.75),
                    it, "schema_instrument_risk",
                    {
                        "via": "referencesInstrument",
                        "intermediate": hr.source,
                        "doc_id": getattr(ref, "doc_id", None) or getattr(hr, "doc_id", None),
                    }     
                )
                if r:
                    out.append(r)

        # affectedBy(A, X) & hasRisk(X, R) -> hasRisk(A, R)  (careful, but useful)
        for ab in affected_by:
            for hr in has_risk:
                if ab.target != hr.source:
                    continue
                if (ab.source, "hasRisk", hr.target) in known:
                    continue
                r = self._make(
                    ab.source, "hasRisk", hr.target,
                    min(float(ab.confidence), float(hr.confidence), 0.65),
                    it, "schema_affectedBy_risk",
                    {
                        "via": "affectedBy",
                        "intermediate": hr.source,
                        "doc_id": getattr(ab, "doc_id", None) or getattr(hr, "doc_id", None),
                    }        
                )
                if r:
                    out.append(r)

        return out

    # --------------------------
    # Dedup
    # --------------------------

    def _dedup_vs_base(self, inferred: Iterable[Relation], base: Iterable[Relation]) -> List[Relation]:
        base_keys = {(r.source, r.relation, r.target) for r in base}
        seen: Set[Tuple[str, str, str]] = set()
        out: List[Relation] = []
        for r in inferred:
            k = (r.source, r.relation, r.target)
            if k in base_keys or k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out


class ConflictDetector:
    """
    Schema-aligned contradiction detection (no new predicates minted).
    """

    def __init__(self) -> None:
        self.contradictions: List[Tuple[str, str]] = [
            ("compliesWith", "violates"),
            ("supports", "opposes"),
            ("supports", "criticizes"),
            ("endorses", "disagreesWith"),
            ("mitigates", "exhibitsRisk"),
        ]

    def _iter_all_relations(self, annotations: Dict[str, Any]) -> Iterable[Any]:
        """
        Iterate all relation payloads across nested annotation structure:
        - themes/*/relations
        - questions/*/relations
        - answers/*/relations
        - votes/*/relations
        - metadata.inferred_relations
        Also supports legacy top-level "relations" list if present.
        """
        if not isinstance(annotations, dict):
            return

        # Legacy flat list support.
        top_level = annotations.get("relations", [])
        if isinstance(top_level, list):
            for rel in top_level:
                yield rel

        for section_key in ("themes", "questions", "answers", "votes"):
            section = annotations.get(section_key, {})
            if not isinstance(section, dict):
                continue
            for _, doc in section.items():
                if not isinstance(doc, dict):
                    continue
                rels = doc.get("relations", [])
                if not isinstance(rels, list):
                    continue
                for rel in rels:
                    yield rel

        metadata = annotations.get("metadata", {})
        if isinstance(metadata, dict):
            inferred = metadata.get("inferred_relations", [])
            if isinstance(inferred, list):
                for rel in inferred:
                    yield rel

    def detect(self, annotations: Dict[str, Any]) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        pairs: DefaultDict[Tuple[str, str], Set[str]] = defaultdict(set)

        for rel in self._iter_all_relations(annotations):
            if isinstance(rel, dict):
                s = rel.get("source")
                o = rel.get("target")
                p = normalize_predicate(rel.get("relation", "") or "")
            else:
                s = getattr(rel, "source", None)
                o = getattr(rel, "target", None)
                p = normalize_predicate(getattr(rel, "relation", "") or "")

            if not s or not o or not p:
                continue
            pairs[(s, o)].add(p)

        for (s, o), preds in pairs.items():
            for a, b in self.contradictions:
                if a in preds and b in preds:
                    conflicts.append(
                        {"source": s, "target": o, "conflict_type": "contradictory_relations", "relations": [a, b], "severity": "high"}
                    )

        return conflicts
