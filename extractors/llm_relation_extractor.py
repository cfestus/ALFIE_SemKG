"""
LLM-based Relation Extraction (OpenAI JSON Mode Version)
ETD-Hub - Semantic Knowledge Graph Pipeline

This extractor:
- Uses the native OpenAI client
- Forces strict JSON output
- Produces clean Relation objects
"""

import json
import time
import hashlib
import random
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from utils.predicate_map import normalize_predicate
from config import Config
from models import Relation
import logging

logger = logging.getLogger(__name__)


def safe_hash(value: str) -> str:
    """Stable short hash for relation identity."""
    return hashlib.sha256(value.encode()).hexdigest()[:15]


class LLMRelationExtractor:

    def __init__(
        self,
        model: str = None,
        max_retries: int = 3,
        relation_confidence_threshold: float = 0.35
    ):
        self.max_retries = max_retries
        self.model = model or Config.LLM_MODEL
        self.relation_confidence_threshold = relation_confidence_threshold
        self.retry_base_delay = 0.5
        self.retry_max_delay = 4.0

        # Direct OpenAI client
        Config.load_llm_api_key()
        self.client = OpenAI(api_key=getattr(Config, "LLM_API_KEY", None))

        # System prompt
        self.system_prompt = """
You are an expert relation extraction model.
You MUST output ONLY valid JSON in the following format:

{
  "relations": [
    {
      "source_id": "<entity_id>",
      "relation": "<relation_label>",
      "target_id": "<entity_id>",
      "confidence": 0.92,
      "evidence_quote": "<verbatim quote from the text>",
      "evidence": "<short quote/span from the text>"
    }
  ]
}

Rules:
- source_id/target_id MUST be taken from the provided Entities list.
- Output at most 20 relations. If more exist, return the top 20 most salient.
- If you cannot fit the full JSON, return fewer relations (never truncate JSON).
- Only extract relations explicitly stated or clearly implied.
- evidence_quote MUST be a verbatim substring copied from the provided text.
- Use canonical relation labels from this schema set:
  [uses, hasRisk, exhibitsRisk, affectedBy, causes, mitigates, mitigatedBy, endorses, opposes,
   referencesInstrument, partOfDomain, containsSubject, belongsToTheme, answersQuestion, authoredBy, mentions, relatedTo].
- IMPORTANT: Avoid "relatedTo" unless no other label fits. Prefer the most specific predicate.
- confidence MUST be in [0,1].
"""

    # =====================================================================
    # Public API
    # =====================================================================

    def extract_batch(self, texts: List[str], entities_per_doc: List[List[Any]], discourse_ids: Optional[List[str]] = None) -> List[List[Relation]]:
        all_relations: List[List[Relation]] = []
        for i, (text, ents) in enumerate(zip(texts, entities_per_doc)):
            did = discourse_ids[i] if discourse_ids and i < len(discourse_ids) else None
            relations = self.extract(text, ents, discourse_id=did)
            all_relations.append(relations)
        return all_relations

    def extract(self, text: str, entities: List[Any], discourse_id: Optional[str] = None) -> List[Relation]:
        raw = self._call_with_retries(text, entities)

        # ✅ Allowed IDs = ONLY the entity URIs we passed to the model
        allowed_ids = {
            getattr(e, "uri", None)
            for e in entities
            if getattr(e, "uri", None)
        }

        cleaned = self._clean_relations(raw, allowed_ids=allowed_ids)
        converted = self._convert_to_relations(cleaned, discourse_id=discourse_id, text=text, entities=entities)
        return converted


    # =====================================================================
    # Retry wrapper
    # =====================================================================
    def _call_with_retries(self, text: str, entities: List[Any]) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_model(text, entities)
            except Exception as e:
                last_error = e
                retryable = self._is_transient_error(e)
                logger.warning(
                    f"[Relation Extraction] attempt {attempt}/{self.max_retries} failed "
                    f"(retryable={retryable}): {e}"
                )
                if not retryable:
                    logger.error("[Relation Extraction] non-transient failure; not retrying.")
                    raise RuntimeError(f"LLM relation extraction failed (non-transient): {e}") from e
                if attempt == self.max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(f"[Relation Extraction] retrying in {delay:.2f}s")
                time.sleep(delay)
        if last_error is not None:
            logger.error(f"[Relation Extraction] final failure after {self.max_retries} attempts: {last_error}")
            raise RuntimeError(
                f"LLM relation extraction failed after {self.max_retries} attempts: {last_error}"
            ) from last_error
        raise RuntimeError("LLM relation extraction failed with unknown error")

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, min(0.25, delay * 0.25))
        return delay + jitter

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            code = getattr(exc, "status_code", None)
            return code == 429 or (isinstance(code, int) and 500 <= code <= 599)
        return False

    # =====================================================================
    # DIRECT OpenAI JSON-mode call
    # =====================================================================
    def _parse_relations_json(self, content: Any) -> Dict[str, Any]:
        """
        Robust JSON parsing for LLM outputs.
        Always returns a dict with a 'relations' key.
        """
        fallback: Dict[str, Any] = {"relations": []}

        def _validate_payload(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, dict):
                if not isinstance(obj.get("relations"), list):
                    obj["relations"] = []
                return obj
            return fallback

        if not isinstance(content, str):
            logger.error("LLM returned non-string content: %r", type(content))
            return fallback

        raw = content.strip()
        if not raw:
            logger.error("LLM returned empty content")
            return fallback

        # 1) Strict JSON parse
        try:
            return _validate_payload(json.loads(raw))
        except Exception:
            pass

        # 2) Slice from first '{' to last '}'.
        # 3) If no closing '}', append one.
        # 4) If relations array is unterminated, append ']}'.
        repaired_candidates: List[str] = []
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1:
            if last != -1 and last >= first:
                sliced = raw[first:last + 1]
            else:
                sliced = raw[first:] + "}"
            repaired_candidates.append(sliced)

            rel_key_pos = sliced.find('"relations"')
            rel_open = sliced.find("[", rel_key_pos) if rel_key_pos != -1 else -1
            rel_close = sliced.rfind("]")
            if rel_open != -1 and rel_close < rel_open:
                repaired_candidates.append(sliced.rstrip() + "]}")

        for cand in repaired_candidates:
            try:
                return _validate_payload(json.loads(cand))
            except Exception:
                pass

        # 5) Lightweight repair (single quotes + trailing commas)
        candidates = [raw]
        candidates.extend(repaired_candidates)
        for cand in candidates:
            repaired = cand.replace("'", '"')
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            try:
                return _validate_payload(json.loads(repaired))
            except Exception:
                continue

        # 6) Final fallback: log and return empty
        logger.error("Failed to parse LLM JSON output. Raw content: %s", raw[:4000])
        return fallback

    def _call_model(self, text: str, entities: List[Any]) -> List[Dict[str, Any]]:

        entity_list = [
        {"id": getattr(e, "uri", None), "label": getattr(e, "label", None), "type": getattr(e, "entity_type", None)}
        for e in entities
        if getattr(e, "uri", None) and getattr(e, "label", None)
        ]

        entity_list = entity_list[:80]

        prompt = f"""
Extract all relations from the following text.

Entities (use ONLY these ids in source_id/target_id):
{json.dumps(entity_list, indent=2)}

Text:
{text}

Return only valid JSON.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"}  # STRICT JSON MODE
        )

        content = response.choices[0].message.content

        parsed = self._parse_relations_json(content)
        rels = parsed.get("relations", [])
        if not isinstance(rels, list):
            rels = []

        clean = [x for x in rels if isinstance(x, dict)]
        return clean

    # =====================================================================
    # Filter + clean relations
    # =====================================================================
    def _clean_relations(
        self,
        rels: List[Dict[str, Any]],
        *,
        allowed_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        allowed_ids = allowed_ids or set()

        for r in rels:
            try:
                conf = float(r.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0            
            if conf < self.relation_confidence_threshold:
                continue

            sid = r.get("source_id")
            tid = r.get("target_id")
            if not sid or not tid:
                continue

            # keep your URL sanity check
            if not str(sid).startswith("http") or not str(tid).startswith("http"):
                continue

            # ✅ PRODUCTION FIX: enforce that IDs were provided in entity list
            if allowed_ids:
                if sid not in allowed_ids or tid not in allowed_ids:
                    continue

            out.append(r)

        return out


    # =====================================================================
    # Convert JSON relations to Relation objects
    # =====================================================================
    @staticmethod
    def _all_occurrences(text: str, quote: str) -> List[Tuple[int, int]]:
        occurrences: List[Tuple[int, int]] = []
        if not text or not quote:
            return occurrences
        start = 0
        while True:
            idx = text.find(quote, start)
            if idx == -1:
                break
            occurrences.append((idx, idx + len(quote)))
            start = idx + 1
        return occurrences

    @staticmethod
    def _coerce_nonneg_int(value: Any) -> Optional[int]:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        return iv if iv >= 0 else None

    def _mention_midpoint_for_entity(self, entity: Any, discourse_id: Optional[str]) -> Optional[float]:
        meta = getattr(entity, "metadata", None) or {}
        did = str(discourse_id or "").strip()
        mentions = meta.get("mentions")
        if isinstance(mentions, list):
            scoped: List[Tuple[int, int]] = []
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                m_did = str(m.get("discourse_id", "") or "").strip()
                if did and m_did != did:
                    continue
                s = self._coerce_nonneg_int(m.get("start_char"))
                e = self._coerce_nonneg_int(m.get("end_char"))
                if s is None or e is None or e <= s:
                    continue
                scoped.append((int(s), int(e)))
            if scoped:
                min_s = min(s for s, _ in scoped)
                max_e = max(e for _, e in scoped)
                return float(min_s + max_e) / 2.0
        s = self._coerce_nonneg_int(meta.get("start_char"))
        e = self._coerce_nonneg_int(meta.get("end_char"))
        if s is not None and e is not None and e > s:
            return float(s + e) / 2.0
        return None

    def _pair_midpoint(
        self,
        source_uri: str,
        target_uri: str,
        entities: List[Any],
        discourse_id: Optional[str],
    ) -> Optional[float]:
        by_uri: Dict[str, Any] = {}
        for e in entities:
            u = str(getattr(e, "uri", "") or "").strip()
            if u:
                by_uri[u] = e
        src_ent = by_uri.get(str(source_uri or "").strip())
        tgt_ent = by_uri.get(str(target_uri or "").strip())
        if src_ent is None or tgt_ent is None:
            return None
        src_mid = self._mention_midpoint_for_entity(src_ent, discourse_id)
        tgt_mid = self._mention_midpoint_for_entity(tgt_ent, discourse_id)
        if src_mid is None or tgt_mid is None:
            return None
        return (float(src_mid) + float(tgt_mid)) / 2.0

    def _select_quote_span(
        self,
        text: str,
        evidence_quote: str,
        source_uri: str,
        target_uri: str,
        entities: List[Any],
        discourse_id: Optional[str],
    ) -> Optional[Tuple[int, int]]:
        quote = str(evidence_quote or "").strip()
        if not quote:
            return None
        occurrences = self._all_occurrences(text, quote)
        if not occurrences:
            return None
        if len(occurrences) == 1:
            return occurrences[0]
        # Duplicate-safe deterministic tie break:
        # use pair midpoint if both source/target mention spans exist in this discourse.
        pair_mid = self._pair_midpoint(source_uri, target_uri, entities, discourse_id)
        if pair_mid is None:
            return sorted(occurrences, key=lambda x: x[0])[0]
        ranked = sorted(
            occurrences,
            key=lambda span: (abs(((span[0] + span[1]) / 2.0) - pair_mid), span[0]),
        )
        return ranked[0]

    def _convert_to_relations(
        self,
        rels: List[Dict[str, Any]],
        discourse_id: Optional[str] = None,
        text: str = "",
        entities: Optional[List[Any]] = None,
    ) -> List[Relation]:
        results: List[Relation] = []
        ent_list = entities or []
        for r in rels:
            source = r.get("source_id")
            rel_label = normalize_predicate(r.get("relation", ""))
        
            target = r.get("target_id")
            try:
                confidence = float(r.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            # 🔒 Safety checks (avoid bad edges)
            if not source or not target or not rel_label:
                continue
            if not str(source).startswith("http") or not str(target).startswith("http"):
                continue

            # Production: suppress low-value generic edges
            if rel_label == "relatedTo" and confidence < 0.80:
                continue

            # Build a stable, streaming-safe URI
            raw_key = f"{discourse_id}|{source}|{rel_label}|{target}"
            rel_hash = safe_hash(raw_key)
            uri = f"{Config.NAMESPACES['etd']}rel_{rel_hash}"
            evidence_quote_raw = str(r.get("evidence_quote") or "").strip()
            evidence = (r.get("evidence") or evidence_quote_raw).strip()
            chosen_span = self._select_quote_span(
                text=text or "",
                evidence_quote=evidence_quote_raw,
                source_uri=str(source),
                target_uri=str(target),
                entities=ent_list,
                discourse_id=discourse_id,
            )
            if chosen_span is None:
                char_start = -1
                char_end = -1
                grounding_status = "ungrounded"
                grounding_method = "none"
            else:
                char_start = int(chosen_span[0])
                char_end = int(chosen_span[1])
                grounding_status = "grounded"
                grounding_method = "evidence_quote_span"
            if len(evidence) > 500:
                evidence = evidence[:500]
            evidence_quote = evidence_quote_raw
            if len(evidence_quote) > 500:
                evidence_quote = evidence_quote[:500]

            metadata = {
                "evidence": evidence,
                "evidence_quote": evidence_quote,
                "context_source": "llm_json",
                "discourse_id": discourse_id,
                "extractor": "llm_relation_extractor",
                "char_start": int(char_start),
                "char_end": int(char_end),
                "start_offset": int(char_start),
                "end_offset": int(char_end),
                "groundingStatus": grounding_status,
                "groundingMethod": grounding_method,
            }

            relation_obj = Relation(
                uri=uri,
                source=source,
                relation=rel_label,
                target=target,
                confidence=confidence,
                is_inferred=False,
                metadata=metadata,
            )

            results.append(relation_obj)

        return results
