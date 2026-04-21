"""Entity normalization and deterministic canonical linking primitives."""

import hashlib
import re
import unicodedata
from typing import Any, Optional

from config import Config
from utils.helpers import is_acronym
from utils.predicate_map import CANONICAL_REL_PREDICATES


def canonicalize_predicate(pred: Any) -> str:
    if pred is None:
        return "relatedTo"
    key = str(pred).strip().lower()
    if not key:
        return "relatedTo"
    return CANONICAL_REL_PREDICATES.get(key, "relatedTo")


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\*\*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_match_key(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


KNOWN_ACRONYMS = {
    "AI", "ML", "LLM", "NLP", "CV", "RL",
    "GDPR", "OECD", "ISO", "NIST", "IEEE", "IEC",
    "API", "UK", "US", "EU", "UNESCO", "HMDA", "COMPAS",
}

ALIAS_CANONICAL_LABELS = {
    "hmda": "Home Mortgage Disclosure Act",
    "home mortgage disclosure act": "Home Mortgage Disclosure Act",
    "unesco": "UNESCO AI Ethics Recommendation",
    "unesco ai recommendation": "UNESCO AI Ethics Recommendation",
    "unesco ai ethics recommendation": "UNESCO AI Ethics Recommendation",
    "unesco recommendation": "UNESCO AI Ethics Recommendation",
    "oecd ai principles": "OECD AI Principles",
    "oecd ai principle": "OECD AI Principles",
    "oecd principles": "OECD AI Principles",
    "apple inc": "Apple Inc",
    "apple inc.": "Apple Inc",
    "apple inc typo": "Apple Inc",
    "apple incs": "Apple Inc",
    "compas": "COMPAS",
    "correctional offender management profiling for alternative sanctions": "COMPAS",
    "mimic cxr": "MIMIC-CXR",
    "mimic-cxr": "MIMIC-CXR",
    "mimic cxr dataset": "MIMIC-CXR",
}

DO_NOT_MERGE_TERMS = {"ai", "ml", "data", "model", "system"}


TYPE_ALIASES = {
    "AiSystem": "AISystem",
    "Aisystem": "AISystem",
    "Personaldata": "PersonalData",
    "Ai": "AI",
    "Ml": "ML",
    "Organization": "Organisation",
}


def _preserve_acronyms_in_phrase(original: str, titled: str) -> str:
    orig_tokens = re.split(r"(\W+)", original)
    titled_tokens = re.split(r"(\W+)", titled)
    out = []
    for o_tok, t_tok in zip(orig_tokens, titled_tokens):
        if not o_tok or re.fullmatch(r"\W+", o_tok):
            out.append(t_tok)
            continue
        if o_tok.isupper() or o_tok.upper() in KNOWN_ACRONYMS:
            out.append(o_tok.upper())
        else:
            out.append(t_tok)
    return "".join(out) if len(out) == len(titled_tokens) else titled


def _normalize_casing(surface: str) -> str:
    s = surface.strip()
    if is_acronym(s):
        return s.upper()
    if re.search(r"[A-Z].*[a-z]|[a-z].*[A-Z]", s):
        return s
    titled = s.title()
    return _preserve_acronyms_in_phrase(s, titled)


def _normalize_entity_type(entity_type: Any) -> str:
    if entity_type is None:
        return "Entity"
    et = str(entity_type).strip()
    if not et:
        return "Entity"
    if ":" in et and " " not in et:
        _, local = et.split(":", 1)
        et = local or et
    et = et.replace("_", " ").replace("-", " ")
    et = " ".join(et.split()).strip()
    norm = et.title().replace(" ", "")
    return TYPE_ALIASES.get(norm, norm)


def _canonical_name(label: str) -> str:
    label = label.strip()
    label = re.sub(r"[^\w]+", "_", label)
    label = re.sub(r"_+", "_", label)
    return label.lower().strip("_")


class EntityNormalizer:
    def __init__(self, registry=None):
        self.registry = registry

    def match_key(self, label: str) -> str:
        return _norm_match_key(label or "")

    def should_avoid_fuzzy_merge(self, label: str) -> bool:
        return self.match_key(label) in DO_NOT_MERGE_TERMS

    def normalize_label(self, surface_form: Any) -> str:
        cleaned = _clean_text(surface_form)
        key = self.match_key(cleaned)
        if key in ALIAS_CANONICAL_LABELS:
            return ALIAS_CANONICAL_LABELS[key]
        return _normalize_casing(cleaned)

    def normalize_type(self, entity_type: str) -> str:
        return _normalize_entity_type(entity_type)

    def normalize_predicate(self, pred: str) -> str:
        return canonicalize_predicate(pred)

    def normalize_metadata(self, metadata: dict) -> dict:
        if not metadata:
            return metadata
        cleaned = {}
        for key, val in metadata.items():
            if isinstance(val, str):
                cleaned[key] = _clean_text(val)
            elif isinstance(val, list):
                cleaned[key] = [_clean_text(v) if isinstance(v, str) else v for v in val]
            else:
                cleaned[key] = val
        return cleaned

    def canonical_name(self, label: str) -> str:
        return _canonical_name(label)

    def canonical_uri(self, *, label: str, entity_type: str, ontology_uri: Optional[str] = None) -> str:
        norm_label = self.normalize_label(label)
        norm_type = self.normalize_type(entity_type or "Entity")
        slug = self.canonical_name(norm_label) or "entity"
        ont = (ontology_uri or "").strip().lower()
        key = f"{self.match_key(norm_label)}|{norm_type}|{ont}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        return f"{Config.NAMESPACES['etd']}{norm_type.lower()}_{slug}_{h}"

    def normalize_entity(self, entity) -> None:
        if entity is None:
            return

        original_label = getattr(entity, "label", "")
        entity.label = self.normalize_label(original_label)

        raw_type = getattr(entity, "entity_type", None)
        entity.metadata = entity.metadata or {}
        if isinstance(raw_type, str) and ":" in raw_type and " " not in raw_type:
            entity.metadata.setdefault("external_entity_type", raw_type)

        entity.entity_type = self.normalize_type(raw_type)
        entity.name = self.canonical_name(entity.label)

        aliases = entity.metadata.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        orig = str(original_label or "").strip()
        if orig and orig != entity.label and orig not in aliases:
            aliases.append(orig)
        if aliases:
            entity.metadata["aliases"] = aliases

        entity.metadata = self.normalize_metadata(entity.metadata)
