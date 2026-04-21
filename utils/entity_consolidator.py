import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set, Union

from config import Config
from utils.entity_normalizer import EntityNormalizer


class EntityConsolidator:
    _STRUCTURAL_LABEL_RE = re.compile(
        r"^(theme|question|answer|vote|expert|document|discourseunit)\s*[_-]?\s*\d+$",
        flags=re.IGNORECASE,
    )

    def __init__(self, registry, normalizer: Optional[EntityNormalizer] = None, output_dir: Optional[Union[Path, str]] = None):
        self.registry = registry
        self.normalizer = normalizer or EntityNormalizer(registry=registry)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(getattr(Config, "OUTPUT_DIR", Path("output")))
        self.redirects: Dict[str, str] = {}
        self.merge_events: List[Dict[str, Any]] = []

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

    def _merge_mentions(self, canonical_meta: Dict[str, Any], source_meta: Dict[str, Any]) -> None:
        merged: List[Dict[str, Any]] = []
        for bucket in (canonical_meta.get("mentions"), source_meta.get("mentions")):
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                norm = self._normalize_mention(item)
                if norm is not None:
                    merged.append(norm)
        if not merged:
            return
        dedup: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
        for m in merged:
            key = (
                str(m.get("discourse_id", "")),
                int(m.get("chunk_index")) if m.get("chunk_index") is not None else -1,
                int(m.get("start_char", -1)),
                int(m.get("end_char", -1)),
            )
            dedup[key] = m
        canonical_meta["mentions"] = [dedup[k] for k in sorted(dedup.keys())]

    def consolidate(self, entities: List[Any]) -> List[Any]:
        if not entities:
            return []

        groups: Dict[Tuple[str, str, str], List[Any]] = defaultdict(list)
        alias_map: Dict[str, str] = {}

        for e in entities:
            if e is None:
                continue

            label = getattr(e, "label", "") or ""
            norm_label = self.normalizer.normalize_label(label)
            etype = self.normalizer.normalize_type(getattr(e, "entity_type", "Entity") or "Entity")

            ont = ""
            if hasattr(e, "metadata") and e.metadata:
                ont = e.metadata.get("ontology_uri") or ""
            if not ont:
                ont = getattr(e, "ontology_uri", "") or ""

            key = (self.normalizer.match_key(norm_label), etype, str(ont).strip())
            groups[key].append(e)
            alias_map[label] = norm_label

        consolidated: List[Any] = []

        for _, group in groups.items():
            canonical = self._pick_canonical(group)
            for other in group:
                if other is canonical:
                    continue
                self._merge_entity_into_canonical(canonical, other, reason="exact_key", score=1.0)
            consolidated.append(canonical)

        # Conservative fuzzy pass within type buckets only
        by_type: Dict[str, List[Any]] = defaultdict(list)
        for e in consolidated:
            by_type[getattr(e, "entity_type", "Entity")].append(e)

        final_entities: List[Any] = []
        consumed: Set[str] = set()

        for etype, ents in by_type.items():
            for i, a in enumerate(ents):
                a_uri = str(getattr(a, "uri", ""))
                if a_uri in consumed:
                    continue

                for j in range(i + 1, len(ents)):
                    b = ents[j]
                    b_uri = str(getattr(b, "uri", ""))
                    if b_uri in consumed:
                        continue
                    if self._should_merge_fuzzy(a, b):
                        score = self._fuzzy_score(getattr(a, "label", ""), getattr(b, "label", ""))
                        self._merge_entity_into_canonical(a, b, reason="fuzzy", score=score)
                        consumed.add(b_uri)

                final_entities.append(a)
                consumed.add(a_uri)

        self._write_artifacts(alias_map)
        return final_entities

    def _pick_canonical(self, group: List[Any]) -> Any:
        return max(group, key=lambda x: float(getattr(x, "confidence", 0.0) or 0.0))

    def _fuzzy_score(self, a: str, b: str) -> float:
        return SequenceMatcher(None, self.normalizer.match_key(a), self.normalizer.match_key(b)).ratio()

    def _is_structural_id_like(self, entity: Any) -> bool:
        etype = str(getattr(entity, "entity_type", "") or "")
        if etype in {"Theme", "Question", "Answer", "Vote", "Expert", "Document", "DiscourseUnit"}:
            return True
        label = str(getattr(entity, "label", "") or "").strip()
        if label and self._STRUCTURAL_LABEL_RE.match(label):
            return True
        uri = str(getattr(entity, "uri", "") or "").strip().lower()
        if any(tok in uri for tok in ("/theme/", "/question/", "/answer/", "/vote/", "/expert/", "discourse_")):
            return True
        return False

    def _should_merge_fuzzy(self, a: Any, b: Any) -> bool:
        la = getattr(a, "label", "") or ""
        lb = getattr(b, "label", "") or ""
        if self._is_structural_id_like(a) or self._is_structural_id_like(b):
            return False
        if self.normalizer.should_avoid_fuzzy_merge(la) or self.normalizer.should_avoid_fuzzy_merge(lb):
            return False
        if getattr(a, "entity_type", None) != getattr(b, "entity_type", None):
            return False

        ka = self.normalizer.match_key(la)
        kb = self.normalizer.match_key(lb)
        if not ka or not kb:
            return False

        # strict threshold; allow acronym/full-form containment pairs
        score = self._fuzzy_score(la, lb)
        if score >= 0.93:
            return True
        if ka in kb or kb in ka:
            return len(ka) >= 4 and len(kb) >= 4
        return False

    def _merge_entity_into_canonical(self, canonical: Any, source: Any, *, reason: str, score: float) -> None:
        canonical.metadata = canonical.metadata or {}
        source.metadata = source.metadata or {}
        self._merge_mentions(canonical.metadata, source.metadata)

        # aliases
        aliases = canonical.metadata.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        src_label = getattr(source, "label", None)
        if src_label and src_label not in aliases and src_label != getattr(canonical, "label", ""):
            aliases.append(src_label)
        if aliases:
            canonical.metadata["aliases"] = aliases

        if not getattr(canonical, "ontology_uri", None) and getattr(source, "ontology_uri", None):
            canonical.ontology_uri = source.ontology_uri

        for k, v in source.metadata.items():
            if k == "wikidata":
                continue
            if k == "mentions":
                continue
            if k not in canonical.metadata:
                canonical.metadata[k] = v
                continue
            existing = canonical.metadata[k]
            if not isinstance(existing, list):
                existing = [existing]
            incoming = v if isinstance(v, list) else [v]
            seen = set()
            merged = []
            for item in existing + incoming:
                marker = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
                if marker not in seen:
                    seen.add(marker)
                    merged.append(item)
            canonical.metadata[k] = merged

        c_conf = float(getattr(canonical, "confidence", 0.0) or 0.0)
        s_conf = float(getattr(source, "confidence", 0.0) or 0.0)
        if s_conf > c_conf:
            canonical.confidence = s_conf

        old_uri = str(getattr(source, "uri", ""))
        new_uri = str(getattr(canonical, "uri", ""))
        if old_uri and new_uri and old_uri != new_uri:
            self.redirects[old_uri] = new_uri
            if hasattr(self.registry, "replace_entity"):
                self.registry.replace_entity(old_uri, new_uri)

        self.merge_events.append({
            "from_uri": old_uri,
            "to_uri": new_uri,
            "from_label": src_label,
            "to_label": getattr(canonical, "label", ""),
            "reason": reason,
            "score": round(float(score), 4),
            "entity_type": getattr(canonical, "entity_type", "Entity"),
        })

    def _write_artifacts(self, alias_map: Dict[str, str]) -> None:
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        alias_path = out_dir / "alias_map.json"
        alias_path.write_text(json.dumps(dict(sorted(alias_map.items())), ensure_ascii=False, indent=2), encoding="utf-8")

        merge_path = out_dir / "merge_log.jsonl"
        with merge_path.open("w", encoding="utf-8") as f:
            for ev in sorted(
                self.merge_events,
                key=lambda x: (
                    str(x.get("to_uri", "")),
                    str(x.get("from_uri", "")),
                    str(x.get("reason", "")),
                    str(x.get("from_label", "")),
                ),
            ):
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
