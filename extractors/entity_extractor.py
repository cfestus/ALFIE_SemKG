"""
LLM-based Entity Extraction (OpenAI JSON Mode Version)
ETD-Hub - Semantic Knowledge Graph Pipeline
"""

import json
import time
import hashlib
import random
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

from config import Config
from models import Entity
import logging

logger = logging.getLogger(__name__)


def safe_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:15]


class LLMEntityExtractor:

    def __init__(
        self,
        ontology_terms: Dict[str, str],
        model: str = None,
        max_retries: int = 3,
        min_confidence: float = 0.35
    ):
        self.ontology_terms = ontology_terms
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        self.model = model or Config.LLM_MODEL
        self.retry_base_delay = 0.5
        self.retry_max_delay = 4.0

        # Use the same key loading strategy as the rest of the pipeline
        Config.load_llm_api_key()
        self.client = OpenAI(api_key=getattr(Config, "LLM_API_KEY", None))

        self.system_prompt = """
You are an expert AI entity extractor. You MUST return ONLY valid JSON.

Output Example:
{
  "entities": [
    {
      "surface_form": "COMPAS",
      "entity_type": "Dataset",
      "ontology_uri": "aipo:Dataset",
      "confidence": 0.92
    }
  ]
}
"""

    # =============================================================
    # Public API
    # =============================================================
    def extract(self, text: str, doc_id: Optional[str] = None) -> List[Entity]:
        raw = self._call_with_retries(text)
        cleaned = self._clean_entities(raw)
        converted = self._convert_to_entities(cleaned, text, doc_id)
        deduped = self._deduplicate(converted)
        return deduped

    # =============================================================
    # Retry wrapper
    # =============================================================
    def _call_with_retries(self, text: str) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_model(text)
            except Exception as e:
                last_error = e
                retryable = self._is_transient_error(e)
                logger.warning(
                    f"[LLM Entity Extraction] attempt {attempt}/{self.max_retries} failed "
                    f"(retryable={retryable}): {e}"
                )
                if not retryable:
                    logger.error("[LLM Entity Extraction] non-transient failure; not retrying.")
                    raise RuntimeError(f"LLM entity extraction failed (non-transient): {e}") from e
                if attempt == self.max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(f"[LLM Entity Extraction] retrying in {delay:.2f}s")
                time.sleep(delay)
        if last_error is not None:
            logger.error(f"[LLM Entity Extraction] final failure after {self.max_retries} attempts: {last_error}")
            raise RuntimeError(
                f"LLM entity extraction failed after {self.max_retries} attempts: {last_error}"
            ) from last_error
        raise RuntimeError("LLM entity extraction failed with unknown error")

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

    # =============================================================
    # Direct OpenAI JSON-mode call
    # =============================================================
    def _call_model(self, text: str) -> List[Dict[str, Any]]:

        prompt = f"""
Extract all entities from the text.

Ontology classes (JSON):
{json.dumps(self.ontology_terms, indent=2)}

Text:
{text}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"}   # STRICT JSON MODE
        )

        content = response.choices[0].message.content

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON returned: {content}")

        if not isinstance(data, dict):
            return []

        ents = data.get("entities", [])
        if not isinstance(ents, list):
            return []

        # ensure list[dict]
        clean = [x for x in ents if isinstance(x, dict)]
        return clean

    # =============================================================
    # Confidence + ontology cleaning
    # =============================================================
    def _clean_entities(self, ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for e in ents:   
            raw_conf = e.get("confidence", 0.0)
            try:
                conf = float(raw_conf) if raw_conf is not None and str(raw_conf).strip() != "" else 0.0
            except Exception:
                conf = 0.0            

            conf = max(0.0, min(1.0, conf))
            
            if conf < self.min_confidence:
                continue
            out.append(e)
        return out

    # =============================================================
    # Convert JSON entity objects to Entity model
    # =============================================================
    def _convert_to_entities(
        self,
        ents: List[Dict[str, Any]],
        text: str,
        doc_id: Optional[str]
    ) -> List[Entity]:
        output: List[Entity] = []
        lower_text = text.lower()

        for e in ents:
            surface = e.get("surface_form", "")
            surface = surface.strip() if isinstance(surface, str) else ""
            if not surface:
                continue  # avoid empty needle causing infinite/huge matches

            etype = e.get("entity_type", "Entity") or "Entity"
            ont = e.get("ontology_uri", "") or ""
            conf = float(e.get("confidence", 0.0) or 0.0)

            # --- locate ALL spans in this chunk/text ---
            spans: List[Tuple[int, int]] = []
            needle = surface.lower()
            pos = 0
            while True:
                start = lower_text.find(needle, pos)
                if start == -1:
                    break
                spans.append((start, start + len(surface)))
                pos = start + len(surface)

            # choose best span (first match) but store all
            start_char, end_char = spans[0] if spans else (None, None)
            
            # --- entity URI (stable identity) ---            
            norm_surface = " ".join(surface.lower().split())
            identity_key = f"{norm_surface}|{etype}|{ont}"            
            
            uri = f"{Config.NAMESPACES['etd']}{safe_hash(identity_key)}"

            # --- metadata ---
            metadata = {
                "doc_id": doc_id,
                "source": "llm",
                "surface_form": surface,
                "start_char": start_char,
                "end_char": end_char,
                "spans": spans,          
                "ontology_uri": ont,
            }

            ent_obj = Entity(
                uri=uri,
                label=surface,
                entity_type=etype,
                confidence=conf,
                metadata=metadata
            )

            output.append(ent_obj)

        return output

    # =============================================================
    # Deduplicate entities by label
    # =============================================================
# =============================================================
# Deduplicate entities (strong key, confidence-aware)
# =============================================================
    def _deduplicate(self, ents: List[Entity]) -> List[Entity]:
        seen = {}

        for e in ents:
            if e is None:
                continue

            label = (e.label or "").strip().lower()
            etype = getattr(e, "entity_type", "Entity") or "Entity"

            # ontology_uri may live on entity or in metadata
            ont = ""
            if hasattr(e, "metadata") and e.metadata:
                ont = e.metadata.get("ontology_uri") or ""
            if not ont:
                ont = getattr(e, "ontology_uri", "") or ""

            key = (label, etype, str(ont))

            # keep the highest-confidence instance
            if key not in seen:
                seen[key] = e
            else:
                prev = seen[key]
                prev_conf = float(getattr(prev, "confidence", 0.0) or 0.0)
                curr_conf = float(getattr(e, "confidence", 0.0) or 0.0)
                if curr_conf > prev_conf:
                    seen[key] = e

        return list(seen.values())

