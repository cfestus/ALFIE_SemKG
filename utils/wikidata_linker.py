"""
utils/wikidata_linker.py
------------------------
Real Wikidata linker using the Wikidata wbsearchentities API.

- Uses the lightweight search API instead of heavy SPARQL + mwapi.
- Returns best-matching Q-ID, label, description, and similarity score.
- Includes rate limiting, retries, and an in-memory cache.
- Designed to be called from the ETD-Hub semantic KG pipeline:

    from utils.wikidata_linker import WikidataLinker

    linker = WikidataLinker()
    result = linker.link("CheXpert Dataset")
    # -> {"qid": "Qxxxx", "label": "...", "description": "...", "score": 0.91}
"""

from __future__ import annotations

import logging
import re
import time
import random
from typing import Any, Dict, List, Optional

import requests
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class WikidataLinker:
    """
    Link entity labels to Wikidata items using the public Wikidata API.

    Public method:
        link(label: str) -> Optional[Dict[str, Any]]
    """

    API_URL = "https://www.wikidata.org/w/api.php"

    def __init__(
        self,
        endpoint: str = "https://query.wikidata.org/sparql",  # kept for backwards compat; unused
        language: str = "en",
        max_results: int = 5,
        min_score: float = 0.65,
        user_agent: str = "ETD-Hub-KG-Pipeline/1.0 (mailto:your-email@example.com)",
        timeout: int = 15,
        max_retries: int = 3,
        rate_limit_delay: float = 0.25,
    ):
        """
        :param endpoint: (ignored, kept for backwards compatibility with SPARQL-based version)
        :param language: Preferred language for labels/descriptions.
        :param max_results: Max results to fetch from Wikidata search.
        :param min_score: Minimum fuzzy similarity score to accept a match.
        :param user_agent: User-Agent header (Wikidata requires a descriptive UA).
        :param timeout: HTTP timeout in seconds.
        :param max_retries: Number of retries on transient HTTP errors.
        :param rate_limit_delay: Delay (in seconds) between requests to avoid throttling.
        """
        self.language = language
        self.max_results = max_results
        self.min_score = min_score
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay

        # Simple in-memory cache: normalized label -> result dict or None
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def link(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Link a label to the best Wikidata item.

        :param label: Entity label (ideally already normalized).
        :return: dict with {qid, label, description, score, source} or None if no good match.
        """
        if not isinstance(label, str):
            return None

        query_label = label.strip()
        if not query_label:
            return None

        key = self._sanitize_search_label(query_label).lower()
        if key in self._cache:
            return self._cache[key]

        # Basic rate limiting to avoid hammering the public API
        time.sleep(self.rate_limit_delay)

        try:
            candidates = self._search_api(query_label, limit=self.max_results)
        except Exception as e:
            logger.warning(f"[WikidataLinker] search error for '{label}': {e}")
            self._cache[key] = None
            return None

        if not candidates:
            self._cache[key] = None
            return None

        best = self._pick_best_candidate(query_label, candidates)
        if best and best.get("score", 0.0) >= self.min_score:
            best["source"] = "wikidata-api"
            self._cache[key] = best
            return best

        self._cache[key] = None
        return None

    # ------------------------------------------------------------------
    # Internal: call Wikidata search API
    # ------------------------------------------------------------------
    def _search_api(self, label: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a text search on Wikidata using the wbsearchentities API.

        Returns a list of candidates:
            [{"qid": "Q42", "label": "Douglas Adams", "description": "..."}]
        """
        sanitized = self._sanitize_search_label(label)

        params = {
            "action": "wbsearchentities",
            "search": sanitized,
            "language": self.language,
            "uselang": self.language,
            "type": "item",
            "format": "json",
            "limit": str(int(limit)),
        }

        headers = {
            "User-Agent": self.user_agent,
        }

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    self.API_URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    search_results = data.get("search", [])
                    candidates: List[Dict[str, Any]] = []

                    for item in search_results:
                        qid = item.get("id")
                        cand_label = item.get("label", "") or ""
                        cand_desc = item.get("description", "") or ""

                        if not qid:
                            continue

                        candidates.append(
                            {
                                "qid": qid,
                                "label": cand_label,
                                "description": cand_desc,
                            }
                        )
                    return candidates

                # Retry only transient status codes
                if self._is_transient_status(resp.status_code):
                    delay = self._retry_delay(attempt, resp.headers.get("Retry-After"))
                    logger.warning(
                        f"[WikidataLinker] transient HTTP {resp.status_code} for '{label}' "
                        f"(attempt {attempt}/{self.max_retries}); retrying in {delay:.2f}s"
                    )
                    last_exc = Exception(f"HTTP {resp.status_code}")
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        continue
                else:
                    logger.error(
                        f"[WikidataLinker] non-transient HTTP {resp.status_code} for '{label}'; not retrying"
                    )
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
            except requests.RequestException as e:
                last_exc = e
                delay = self._retry_delay(attempt, None)
                logger.warning(
                    f"[WikidataLinker] request error for '{label}' "
                    f"(attempt {attempt}/{self.max_retries}); retrying in {delay:.2f}s: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    continue

        # If we exhausted retries and still failed
        if last_exc:
            logger.error(
                f"[WikidataLinker] final failure for '{label}' after {self.max_retries} attempts: {last_exc}"
            )
            raise last_exc
        return []

    @staticmethod
    def _is_transient_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= int(status_code) <= 599

    def _retry_delay(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return float(retry_after)
            except Exception:
                pass
        base = min(4.0, 0.5 * (2 ** (attempt - 1)))
        jitter = random.uniform(0.0, min(0.25, base * 0.25))
        return max(self.rate_limit_delay, base + jitter)

    # ------------------------------------------------------------------
    # Internal: ranking
    # ------------------------------------------------------------------
    def _pick_best_candidate(
        self,
        query_label: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Rank candidates by fuzzy similarity between query label and candidate label.

        :param query_label: Original label from the entity.
        :param candidates: List of {qid, label, description}.
        """
        if not candidates:
            return None

        q_norm = query_label.strip().lower()
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        if len(q_norm) < 4:
            # For very short labels, only accept exact match (case-insensitive)
            for c in candidates:
                if (c.get("label", "") or "").strip().lower() == q_norm:
                    c["score"] = 1.0
                    return c
            return None

        for c in candidates:
            cand_label = c.get("label", "") or ""
            c_norm = cand_label.strip().lower()
            if not c_norm:
                continue

            score = SequenceMatcher(None, q_norm, c_norm).ratio()

            # Small bonus for exact label match ignoring case
            if q_norm == c_norm:
                score += 0.15

            if score > best_score:
                best_score = score
                best = c

        if best is not None:
            best["score"] = best_score

        return best

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_search_label(label: str) -> str:
        """
        Escape quotes and condense whitespace for safe use in URL params.
        """
        s = label.strip()
        s = s.replace('"', " ")
        s = re.sub(r"\s+", " ", s)
        return s
