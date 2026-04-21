"""
utils/helpers.py
----------------
Modernised utilities for the ETD-Hub Semantic Knowledge Graph Pipeline.
Fully compatible with:
- new Entity(uri, label, entity_type)
- new LLM-based entity extractor
- ontology-driven metadata
- canonical URI-based deduplication
"""

import gc
import time
import hashlib
import json
import logging
import requests
import re
import traceback
from typing import Callable, Any, List, Optional, Dict
from pathlib import Path
from rdflib import URIRef
from urllib.parse import urlsplit, urlunsplit, quote
from models import Relation  
from models import Entity
from models import ProcessingResult
from config import Config
from utils.wikidata_linker import WikidataLinker

logger = logging.getLogger(__name__)

# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_get(dictionary: dict, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(dictionary, dict):
            dictionary = dictionary.get(key, default)
        else:
            return default
    return dictionary


def normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def generate_hash(text: str) -> str:
    """Generate a short MD5 hash for text."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def batch_iterator(items: List[Any], batch_size: int):
    """Yield fixed-size batches."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def cleanup_memory():
    """Force GC + clear GPU."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def validate_input_json(data: dict) -> bool:
    """Ensure minimal input structure exists."""
    # Accept both modern plural keys and legacy singular keys.
    expected_aliases = {
        "themes": ["themes", "Theme"],
        "questions": ["questions", "Question"],
        "answers": ["answers", "Answer"],
    }
    for canonical, aliases in expected_aliases.items():
        if not any(alias in data for alias in aliases):
            logger.warning("Input JSON key missing: %s", canonical)
    return True


# ============================================================
# RETRY WRAPPER
# ============================================================

def safe_process_with_retry(
    item: Any,
    process_func: Callable,
    max_retries: int = None
) -> ProcessingResult:
    """Perform processing with retries and always return ProcessingResult."""
    max_retries = max_retries or Config.MAX_RETRIES

    def _coerce_processing_result(result: Any) -> ProcessingResult:
        if isinstance(result, ProcessingResult):
            return result

        if isinstance(result, dict):
            entities_raw = result.get("entities", []) or []
            relations_raw = result.get("relations", []) or []
            metadata_raw = result.get("metadata", {}) or {}
            errors_raw = result.get("errors", []) or []

            entities: List[Entity] = []
            for e in entities_raw:
                if isinstance(e, Entity):
                    entities.append(e)
                elif isinstance(e, dict):
                    try:
                        entities.append(Entity.from_dict(e))
                    except Exception:
                        continue

            relations: List[Relation] = []
            for r in relations_raw:
                if isinstance(r, Relation):
                    relations.append(r)
                elif isinstance(r, dict):
                    try:
                        relations.append(Relation.from_dict(r))
                    except Exception:
                        continue

            metadata = metadata_raw if isinstance(metadata_raw, dict) else {"raw_metadata": metadata_raw}
            errors = errors_raw if isinstance(errors_raw, list) else [str(errors_raw)]

            return ProcessingResult(
                entities=entities,
                relations=relations,
                metadata=metadata,
                errors=errors,
            )

        raise TypeError(f"Expected ProcessingResult or dict, got {type(result)}")

    last_exc: Optional[Exception] = None
    last_tb: str = ""
    for attempt in range(max_retries):
        try:
            result = process_func(item)
            return _coerce_processing_result(result)
        except Exception as e:
            last_exc = e
            last_tb = traceback.format_exc()
            logger.warning(
                "safe_process_with_retry failed on attempt %d/%d: %s",
                attempt + 1,
                max_retries,
                str(e)[:120],
            )

            if attempt == max_retries - 1:
                break

            time.sleep(2 ** attempt)

    return ProcessingResult(
        entities=[],
        relations=[],
        metadata={
            "error": str(last_exc)[:200] if last_exc else "Max retries exceeded",
            "exception_type": type(last_exc).__name__ if last_exc else "UnknownError",
            "retries": int(max_retries),
            "traceback": last_tb,
            "processing_failed": True,
            "item_id": item.get("id", "unknown") if isinstance(item, dict) else getattr(item, "id", "unknown"),
        },
        errors=[str(last_exc)] if last_exc else ["Max retries exceeded"],
    )


# ============================================================
# GLOBAL ENTITY REGISTRY (CRITICAL COMPONENT)
# ============================================================

class GlobalEntityRegistry:
    """
    Canonical entity registry across entire pipeline.
    NEW RULES:
    - Deduplicate entities by **entity.uri** (canonical)
    - Preserve entity.label + entity.entity_type
    - Merge metadata and keep highest confidence
    """

    def __init__(self):
        # Map canonical URI → Entity instance
        self.entities: Dict[str, Entity] = {}

    def _normalize_mention(self, mention: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(mention, dict):
            return None
        did_raw = mention.get("discourse_id")
        did = str(did_raw).strip() if did_raw is not None else ""
        if not did:
            return None
        try:
            start = int(mention.get("start_char"))
            end = int(mention.get("end_char"))
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start:
            return None
        out: Dict[str, Any] = {
            "discourse_id": did,
            "start_char": int(start),
            "end_char": int(end),
        }
        try:
            ci = int(mention.get("chunk_index"))
            if ci >= 0:
                out["chunk_index"] = int(ci)
        except (TypeError, ValueError):
            pass
        return out

    def _merge_mentions(self, existing_meta: Dict[str, Any], incoming_meta: Dict[str, Any]) -> None:
        existing_mentions = existing_meta.get("mentions")
        incoming_mentions = incoming_meta.get("mentions")
        merged: List[Dict[str, Any]] = []
        for bucket in (existing_mentions, incoming_mentions):
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                norm = self._normalize_mention(item)
                if norm is not None:
                    merged.append(norm)
        if not merged:
            return
        dedup: Dict[tuple, Dict[str, Any]] = {}
        for m in merged:
            key = (
                str(m.get("discourse_id", "")),
                int(m.get("chunk_index")) if m.get("chunk_index") is not None else -1,
                int(m.get("start_char", -1)),
                int(m.get("end_char", -1)),
            )
            dedup[key] = m
        existing_meta["mentions"] = [dedup[k] for k in sorted(dedup.keys())]

    # -----------------------------------------
    # Registration
    # -----------------------------------------
    def get_or_create(self, entity: Entity) -> Entity:
        """
        Deduplicate using the canonical entity URI.
        Merge confidence + metadata when entity already exists.
        """
        uri = entity.uri

        if uri in self.entities:
            existing = self.entities[uri]

            # Higher confidence wins
            if entity.confidence > existing.confidence:
                existing.confidence = entity.confidence

            # Production-safe metadata merge: merge dicts, extend lists, else overwrite
            self._merge_mentions(existing.metadata, entity.metadata or {})
            for k, v in (entity.metadata or {}).items():
                if k == "mentions":
                    continue
                if k not in existing.metadata:
                    existing.metadata[k] = v
                    continue

                cur = existing.metadata.get(k)
                if isinstance(cur, dict) and isinstance(v, dict):
                    merged = dict(cur)
                    merged.update(v)
                    existing.metadata[k] = merged
                elif isinstance(cur, list) and isinstance(v, list):
                    # extend while keeping order + uniqueness
                    seen = set()
                    out = []
                    for item in cur + v:
                        marker = repr(item)
                        if marker not in seen:
                            seen.add(marker)
                            out.append(item)
                    existing.metadata[k] = out
                else:
                    existing.metadata[k] = v

            # Keep label if longer / more informative
            if len(entity.label) > len(existing.label):
                existing.label = entity.label

            # Keep entity_type if new one seems better (same logic)
            if entity.entity_type != existing.entity_type:
                existing.entity_type = entity.entity_type

            # Keep entity.name / entity.type for compatibility
            if hasattr(entity, "name"):
                existing.name = entity.name
            if hasattr(entity, "type"):
                existing.type = entity.type

            return existing

        # NEW entity
        self.entities[uri] = entity

        # backwards compatibility:
        if not hasattr(entity, "name"):
            entity.name = entity.label
        if not hasattr(entity, "type"):
            entity.type = entity.entity_type

        return entity

    # -----------------------------------------
    # Retrieval
    # -----------------------------------------
    def get_all_entities(self) -> List[Entity]:
        """
        Return all Entity objects regardless of whether self.entities is a dict or list.
        Production-safe for streaming refactors.
        """
        store = getattr(self, "entities", None)

        if isinstance(store, dict):
            return list(store.values())

        if isinstance(store, list):
            return list(store)

        return []


    def get_count(self) -> int:
        return len(self.entities)


    def replace_entity(self, old_uri: str, new_uri: str):
        """
        Redirect all references from old_uri → new_uri.
        Merge metadata, confidence, type, label.
        """
        if old_uri == new_uri:
            return

        if old_uri not in self.entities:
            return

        old_ent = self.entities[old_uri]

        if new_uri not in self.entities:
            # If canonical doesn't yet exist, promote old entity to canonical
            self.entities[new_uri] = old_ent
            del self.entities[old_uri]
            return

        # Merge into canonical
        canon = self.entities[new_uri]

        # Keep best confidence
        if old_ent.confidence > canon.confidence:
            canon.confidence = old_ent.confidence

        # Choose richer label
        if len(old_ent.label) > len(canon.label):
            canon.label = old_ent.label

        # Merge metadata
        self._merge_mentions(canon.metadata, old_ent.metadata or {})
        for k, v in (old_ent.metadata or {}).items():
            if k == "mentions":
                continue
            canon.metadata[k] = v

        # Merge entity_type if different
        if old_ent.entity_type != canon.entity_type:
            canon.entity_type = old_ent.entity_type

        # Remove old
        del self.entities[old_uri]


# ============================================================
# COREFERENCE RESOLVER (placeholder)
# ============================================================

class CoreferenceResolver:
    """Optional text coreference resolver (disabled by default)."""

    def __init__(self):
        self.enabled = False

    def resolve(self, text: str) -> str:
        return text


# ============================================================
# FILE UTILITIES
# ============================================================

def create_output_directory(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", output_dir)


def save_json(data: dict, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON: %s", filepath.name)


def load_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# URI SANITISATION
# ============================================================


def canonicalize_local_name(label: str) -> str:
    """
    Convert labels into RDF-safe camelCase local names.
    Removes punctuation and spaces without percent-encoding.
    """
    if not label:
        return "unnamed"

    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", label)
    parts = cleaned.strip().split()  # split on spaces

    if not parts:
        return "unnamed"

    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])

def make_uri_safe(iri: str) -> URIRef:
    """Percent-encode illegal characters for RDF stores."""
    s = str(iri).strip()
    parts = urlsplit(s)
    path = quote(parts.path, safe="/:@-._~")
    query = quote(parts.query, safe="/:@-._~&=?")
    fragment = quote(parts.fragment, safe="/:@-._~")
    return URIRef(urlunsplit((parts.scheme, parts.netloc, path, query, fragment)))

def safe_uri(iri: str) -> URIRef:
    """
    Convert arbitrary strings into a valid ETD URIRef.
    Uses Config.NAMESPACES['etd'] as base IRI unless input is already absolute.
    """
   
    if iri is None:
        raise ValueError("safe_uri() received None")

    iri = str(iri).strip()

    # If already an absolute HTTP IRI, just percent-encode safely
    if iri.startswith("http://") or iri.startswith("https://"):
        return make_uri_safe(iri)

    # Otherwise treat as an ETD-local identifier
    base = Config.NAMESPACES["etd"]

    # Replace unsafe characters before percent-encoding
    cleaned = re.sub(r"[^A-Za-z0-9\-_]", "_", iri)  # strict safe fragment

    return make_uri_safe(base + cleaned)


def expand_curie(curie: str) -> str:
    """
    Expand CURIE like 'prov:wasAttributedTo' → full URI.
    Falls back to CURIE as-is if prefix unknown.
    """
    if ":" in curie:
        prefix, local = curie.split(":", 1)
        base = Config.NAMESPACES.get(prefix)
        if base:
            return f"{base}{local}"
    return curie

def is_acronym(text: str) -> bool:
    """
    Returns True if the string is an acronym:
    - All uppercase letters (e.g., “AI”, “NLP”, “FBI”)
    - Or uppercase with digits (e.g., “GPT4”, “COVID19”)
    """
    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) <= 1:
        return False

    # No spaces, contains only A–Z or digits
    return text.isupper() and text.replace("-", "").replace("_", "").isalnum()


def safe_hash(value: str) -> str:
    """
    Generate a stable, URL-safe hash for KG identifiers.
    """
    if value is None:
        value = ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def make_relation(
    source_uri: str,
    predicate: str,
    target_uri: str,
    *,
    discourse_id: Optional[str] = None,
    evidence: str = "",
    extractor: str = "pipeline",
    confidence: float = 0.0,
    context_source: str = "unknown",
    is_inferred: bool = False,
) -> Relation:
    """
    Production-grade relation builder:
    - stable URI hash
    - guaranteed metadata provenance
    """
    raw_key = f"{discourse_id}|{source_uri}|{predicate}|{target_uri}"
    rel_hash = safe_hash(raw_key)
    uri = f"{Config.NAMESPACES['etd']}rel_{rel_hash}"

    meta = {
        "discourse_id": discourse_id,
        "extractor": extractor,
        "context_source": context_source,
        "evidence": evidence or "",
    }

    return Relation(
        uri=uri,
        source=source_uri,
        relation=predicate,
        target=target_uri,
        confidence=float(confidence or 0.0),
        is_inferred=bool(is_inferred),
        metadata=meta,
    )

