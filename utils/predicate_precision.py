"""
Deterministic predicate precision layer.

Runs after extraction/canonicalization and before reasoning/RDF.
Returns NEW Relation instances (no in-place mutation).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from models import Entity, Relation
from utils.predicate_map import normalize_predicate


@dataclass(frozen=True)
class PredicateSpec:
    domain_types: Set[str]
    range_types: Set[str]
    min_confidence: float


def _to_coarse_type(raw: Any) -> str:
    s = str(raw or "").strip()
    return s if s else "Entity"


def _infer_type_from_uri(uri: str) -> str:
    s = str(uri or "").lower()
    if "#expert_" in s or "/expert_" in s:
        return "Expert"
    if "/expert/" in s:
        return "Expert"
    if "#answer_" in s or "/answer_" in s:
        return "Answer"
    if "/answer/" in s:
        return "Answer"
    if "#question_" in s or "/question_" in s:
        return "Question"
    if "/question/" in s:
        return "Question"
    if "#theme_" in s or "/theme_" in s:
        return "Theme"
    if "/theme/" in s:
        return "Theme"
    if "#vote_" in s or "/vote_" in s:
        return "Vote"
    if "/vote/" in s:
        return "Vote"
    if "#document_" in s or "/document_" in s:
        return "Document"
    if "/document/" in s:
        return "Document"
    if "#discourse_" in s or "/discourse_" in s:
        return "DiscourseUnit"
    return "Entity"


class PredicatePrecisionLayer:
    def __init__(self) -> None:
        any_t = {"*"}
        policy_targets = {"Standard", "Law", "Regulation", "PolicyFramework", "LegalRequirement", "RegulatoryReference"}
        dataset_targets = {"Dataset", "DataSource", "Corpus"}

        self.specs: Dict[str, PredicateSpec] = {
            "relatedTo": PredicateSpec(any_t, any_t, 0.30),
            "mentions": PredicateSpec({"Theme", "Question", "Answer", "Document", "DiscourseUnit"}, any_t, 0.20),
            "uses": PredicateSpec(any_t, any_t, 0.45),
            "trainedOn": PredicateSpec(any_t, dataset_targets, 0.50),
            "hasRisk": PredicateSpec(any_t, any_t, 0.50),
            "exhibitsRisk": PredicateSpec(any_t, any_t, 0.50),
            "causes": PredicateSpec(any_t, any_t, 0.55),
            "affects": PredicateSpec(any_t, any_t, 0.50),
            "mitigatedBy": PredicateSpec(any_t, any_t, 0.55),
            "mitigates": PredicateSpec(any_t, any_t, 0.55),
            "supports": PredicateSpec(any_t, any_t, 0.50),
            "criticizes": PredicateSpec(any_t, any_t, 0.50),
            "endorses": PredicateSpec(any_t, any_t, 0.50),
            "disagreesWith": PredicateSpec(any_t, any_t, 0.50),
            "opposes": PredicateSpec(any_t, any_t, 0.50),
            "compliesWith": PredicateSpec(any_t, policy_targets, 0.55),
            "violates": PredicateSpec(any_t, policy_targets, 0.55),
            "requiresComplianceWith": PredicateSpec(any_t, policy_targets, 0.55),
            "requires": PredicateSpec(any_t, policy_targets, 0.50),
            "referencesInstrument": PredicateSpec(any_t, policy_targets, 0.45),
            "referencesSource": PredicateSpec(any_t, any_t, 0.45),
            "cites": PredicateSpec(any_t, any_t, 0.45),
            "answersQuestion": PredicateSpec({"Answer"}, {"Question"}, 0.70),
            "authoredBy": PredicateSpec({"Answer"}, {"Expert"}, 0.70),
            "belongsToTheme": PredicateSpec({"Question"}, {"Theme"}, 0.70),
            "castBy": PredicateSpec({"Vote"}, {"Expert"}, 0.70),
            "receivesVote": PredicateSpec({"Vote"}, {"Answer"}, 0.70),
        }
        self.allowed: Set[str] = set(self.specs.keys())

        # NOTE: "requires" intentionally NOT mapped to compliesWith.
        self.synonyms: Dict[str, str] = {
            "criticise": "criticizes",
            "criticises": "criticizes",
            "criticised": "criticizes",
            "publishedIn": "referencesSource",
            "accordingTo": "cites",
            "accordingto": "cites",
            "mitigate": "mitigatedBy",
            "mitigates": "mitigatedBy",
            "requirescompliancewith": "requiresComplianceWith",
            "requires compliance with": "requiresComplianceWith",
            "hasTheme": "belongsToTheme",
            "hastheme": "belongsToTheme",
            "votesForAnswer": "receivesVote",
            "votesforanswer": "receivesVote",
        }

        # upgrades apply only when type-guards pass
        self.upgrade_rules: List[Tuple[re.Pattern[str], str, str]] = [
            (re.compile(r"\b(violates?|non[- ]?compliant|breach)\b", re.I), "violates", "rule:violates_cue"),
            (re.compile(r"\b(mitigat(?:e|es|ed|ion)|reduces?|address(?:es|ed))\b", re.I), "mitigatedBy", "rule:mitigation_cue"),
            (re.compile(r"\b(caus(?:e|es|ed)|leads? to|results? in|drives?)\b", re.I), "causes", "rule:causal_cue"),
            (re.compile(r"\b(requires compliance with|must comply with)\b", re.I), "requiresComplianceWith", "rule:requires_compliance_cue"),
            (re.compile(r"\b(requires?|must)\b", re.I), "requires", "rule:requires_cue"),
            (re.compile(r"\b(trained on|training data|fine[- ]?tuned on)\b", re.I), "trainedOn", "rule:trained_on_cue"),
            (re.compile(r"\b(uses?|utili[sz]es?|applies?)\b", re.I), "uses", "rule:use_cue"),
            (re.compile(r"\b(raises concerns?|criticiz(?:e|es|ed)|criticis(?:e|es|ed)|opposes?)\b", re.I), "criticizes", "rule:critical_stance_cue"),
            (re.compile(r"\b(supports?|endorses?|recommends?)\b", re.I), "supports", "rule:support_stance_cue"),
            (re.compile(r"\b(according to|as per)\b", re.I), "cites", "rule:citation_cue"),
            (re.compile(r"\b(published in|report|paper|study)\b", re.I), "referencesSource", "rule:source_cue"),
            (re.compile(r"\b(risk|bias|harm)\b", re.I), "hasRisk", "rule:risk_cue"),
        ]

    def refine_relations(
        self,
        relations: List[Relation],
        entities: Iterable[Entity],
        discourse_text: Optional[str] = None,
    ) -> List[Relation]:
        type_by_uri: Dict[str, str] = {}
        for e in entities:
            u = str(getattr(e, "uri", "") or "")
            if not u:
                continue
            type_by_uri[u] = _to_coarse_type(getattr(e, "entity_type", "Entity"))

        strong_pairs: Set[Tuple[str, str]] = set()
        for r in relations:
            p = self._normalize_candidate(str(getattr(r, "relation", "")))
            if p not in {"relatedTo", "mentions"}:
                strong_pairs.add((str(r.source), str(r.target)))
                strong_pairs.add((str(r.target), str(r.source)))

        out: List[Relation] = []
        for rel in relations:
            src = str(rel.source)
            tgt = str(rel.target)
            src_t = type_by_uri.get(src) or _infer_type_from_uri(src)
            tgt_t = type_by_uri.get(tgt) or _infer_type_from_uri(tgt)
            meta = copy.deepcopy(getattr(rel, "metadata", {}) or {})

            raw_pred = str(getattr(rel, "relation", "") or "")
            pred = self._normalize_candidate(raw_pred)
            if pred not in self.allowed:
                self._log(meta, raw_pred, "relatedTo", "downgrade:unknown_predicate")
                pred = "relatedTo"

            # Type validation BEFORE evidence upgrade.
            if not self._type_valid(pred, src_t, tgt_t):
                downgraded = "mentions" if pred == "mentions" else "relatedTo"
                self._log(meta, pred, downgraded, f"type_guard:{src_t}->{tgt_t}")
                pred = downgraded

            evidence = str(meta.get("evidence") or getattr(rel, "context", None) or discourse_text or "")
            upgraded_pred, upgrade_rule = self._candidate_upgrade(evidence)
            if pred in {"relatedTo", "mentions"} and upgraded_pred and upgraded_pred in self.allowed:
                # Upgrade only when upgraded predicate passes type checks.
                if self._type_valid(upgraded_pred, src_t, tgt_t):
                    self._log(meta, pred, upgraded_pred, f"upgrade:{upgrade_rule}")
                    pred = upgraded_pred

            # drop weak co-occurrence if pair already has stronger relation
            context_source = str(meta.get("context_source", ""))
            if pred == "relatedTo" and context_source == "cooccurrence_chunk" and (src, tgt) in strong_pairs:
                self._log(meta, pred, pred, "drop:cooccurrence_shadowed")
                continue

            conf = float(getattr(rel, "confidence", 0.0) or 0.0)
            conf2 = self._adjust_confidence(conf, pred, meta)

            out.append(
                Relation(
                    uri=getattr(rel, "uri", ""),
                    source=src,
                    relation=pred,
                    target=tgt,
                    confidence=conf2,
                    ontology_uri=getattr(rel, "ontology_uri", None),
                    is_inferred=bool(getattr(rel, "is_inferred", False)),
                    context=getattr(rel, "context", None),
                    doc_id=getattr(rel, "doc_id", None),
                    metadata=meta,
                )
            )
        return out

    def _normalize_candidate(self, raw_pred: str) -> str:
        base = normalize_predicate(raw_pred)
        if base in self.synonyms:
            return self.synonyms[base]
        key = raw_pred.strip()
        return self.synonyms.get(key, self.synonyms.get(key.lower(), base))

    def _type_valid(self, pred: str, src_t: str, tgt_t: str) -> bool:
        spec = self.specs.get(pred)
        if spec is None:
            return False
        return self._match(spec.domain_types, src_t) and self._match(spec.range_types, tgt_t)

    @staticmethod
    def _match(allowed: Set[str], actual: str) -> bool:
        return "*" in allowed or actual in allowed

    def _candidate_upgrade(self, evidence: str) -> Tuple[Optional[str], Optional[str]]:
        if not evidence:
            return None, None
        for pattern, pred, rule in self.upgrade_rules:
            if pattern.search(evidence):
                return pred, rule
        return None, None

    def _adjust_confidence(self, conf: float, pred: str, metadata: Dict[str, Any]) -> float:
        """Small bounded adjustments only; no hard floors."""
        c = max(0.0, min(1.0, float(conf)))
        logs = metadata.get("predicate_precision_logs") or []
        if any(str(x.get("rule", "")).startswith("upgrade:") for x in logs if isinstance(x, dict)):
            c = min(1.0, c + 0.05)
        if any(str(x.get("rule", "")).startswith("type_guard:") for x in logs if isinstance(x, dict)):
            c = max(0.0, c - 0.07)
        return c

    @staticmethod
    def _log(meta: Dict[str, Any], inp: str, out: str, rule: str) -> None:
        logs = meta.get("predicate_precision_logs")
        if not isinstance(logs, list):
            logs = []
        logs.append({"input": inp, "output": out, "rule": rule})
        meta["predicate_precision_logs"] = logs
