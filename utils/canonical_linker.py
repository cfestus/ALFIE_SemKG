"""Deterministic cross-chunk canonical entity linker."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from config import Config


@dataclass
class LinkDecision:
    input_label: str
    norm_label: str
    chosen_canonical_uri: str
    rule: str
    scores: Dict[str, float]
    source_ids: List[str]


class CanonicalLinker:
    DO_NOT_MERGE: Set[str] = {"ai", "ml", "data", "model", "system"}
    STRUCTURAL_TYPES: Set[str] = {"Theme", "Question", "Answer", "Vote", "Expert", "Document", "DiscourseUnit"}
    ORG_SUFFIXES: Set[str] = {"inc", "ltd", "corp", "company", "co"}
    CLASS_NS_PREFIXES: Tuple[str, ...] = (
        Config.NAMESPACES.get("aipo", ""),
        Config.NAMESPACES.get("dpv", ""),
        Config.NAMESPACES.get("vair", ""),
        Config.NAMESPACES.get("airo", ""),
        Config.NAMESPACES.get("fmo", ""),
        Config.NAMESPACES.get("hudock", ""),
        Config.NAMESPACES.get("relaieo", ""),
        Config.NAMESPACES.get("dcat", ""),
        Config.NAMESPACES.get("schema", ""),
        Config.NAMESPACES.get("org", ""),
        Config.NAMESPACES.get("foaf", ""),
    )

    ALIAS_TABLE: Dict[str, str] = {
        "hmda": "Home Mortgage Disclosure Act",
        "home mortgage disclosure act": "Home Mortgage Disclosure Act",
        "gdpr": "General Data Protection Regulation",
        "general data protection regulation": "General Data Protection Regulation",
        "eu ai act": "Artificial Intelligence Act",
        "artificial intelligence act": "Artificial Intelligence Act",
        "unesco ai ethics recommendation": "UNESCO Recommendation on the Ethics of Artificial Intelligence",
        "unesco recommendation on the ethics of artificial intelligence": "UNESCO Recommendation on the Ethics of Artificial Intelligence",
        "oecd ai principles": "OECD Principles on Artificial Intelligence",
        "oecd principles on artificial intelligence": "OECD Principles on Artificial Intelligence",
        "compas": "COMPAS",
        "correctional offender management profiling for alternative sanctions": "COMPAS",
        "mimic-cxr": "MIMIC-CXR",
        "mimic cxr": "MIMIC-CXR",
    }

    def __init__(self, registry: Any, output_dir: Optional[Union[Path, str]] = None) -> None:
        self.registry = registry
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.alias_index: Dict[str, str] = {}
        self.alias_index_raw: Dict[str, str] = {}
        self.redirects: Dict[str, str] = {}
        self.decisions: List[LinkDecision] = []

    def run_full_registry(self) -> List[Any]:
        entities = self._get_registry_entities()
        linked = self.link_entities(entities)
        if hasattr(self.registry, "entities"):
            self.registry.entities = {e.uri: e for e in linked if getattr(e, "uri", None)}
        elif hasattr(self.registry, "set_entities"):
            self.registry.set_entities(linked)
        return linked

    def link_entities(self, entities: List[Any]) -> List[Any]:
        if not entities:
            self._persist()
            return []

        ordered_entities = sorted(
            [e for e in entities if e is not None],
            key=self._entity_sort_key,
        )

        # Pass 1: absolute identity by normalized external ID (type-agnostic).
        # This prevents same external entity from leaking into multiple canonical URIs
        # due to coarse-type/path differences.
        external_groups: Dict[str, List[Any]] = defaultdict(list)
        remainder: List[Any] = []
        for e in ordered_entities:
            ext_id = self._external_id(e)
            if ext_id:
                external_groups[ext_id].append(e)
            else:
                remainder.append(e)

        canonicals: List[Any] = []
        for ext_id in sorted(external_groups.keys(), key=lambda x: str(x)):
            members = sorted(external_groups[ext_id], key=self._entity_sort_key)
            c = self._pick_canonical(members)
            ctype = self._coarse_type(c)
            minted = self._mint_canonical_uri(ctype, f"ext:{ext_id}")
            old_can_uri = str(getattr(c, "uri", "") or "")
            c.uri = minted
            if old_can_uri and old_can_uri != minted:
                self.redirects[old_can_uri] = minted
            for s in members:
                self._register_alias(s, c, "external_id_exact", {"score": 1.0})
                if s is c:
                    continue
                self._merge(c, s, "external_id_exact", {"score": 1.0})
            canonicals.append(c)

        # Pass 2: typed exact-key grouping for entities without external IDs.
        groups: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for e in remainder:
            ctype = self._coarse_type(e)
            key = self._normalized_match_label(getattr(e, "label", "") or "")
            groups[(ctype, key)].append(e)

        for (ctype, key) in sorted(groups.keys(), key=lambda x: (str(x[0]), str(x[1]))):
            members = sorted(groups[(ctype, key)], key=self._entity_sort_key)
            c = self._pick_canonical(members)
            minted = self._mint_canonical_uri(ctype, key)
            old_can_uri = str(getattr(c, "uri", "") or "")
            c.uri = minted
            if old_can_uri and old_can_uri != minted:
                self.redirects[old_can_uri] = minted
            for s in members:
                self._register_alias(s, c, "exact_key", {"score": 1.0})
                if s is c:
                    continue
                self._merge(c, s, "exact_key", {"score": 1.0})
            canonicals.append(c)

        canonicals = self._fuzzy_cluster_and_merge(canonicals)
        self._persist()
        return canonicals

    def rewrite_relation_endpoints(self, all_relations: Iterable[Any]) -> None:
        if not self.redirects:
            return
        for rel in all_relations:
            src = getattr(rel, "source", None)
            tgt = getattr(rel, "target", None)
            if src in self.redirects:
                rel.source = self._resolve_redirect(src)
            if tgt in self.redirects:
                rel.target = self._resolve_redirect(tgt)

    def _resolve_redirect(self, uri: str) -> str:
        seen: Set[str] = set()
        cur = uri
        while cur in self.redirects and cur not in seen:
            seen.add(cur)
            nxt = self.redirects[cur]
            if not nxt or nxt == cur:
                break
            cur = nxt
        return cur

    def get_report(self, raw_entities: int, canonical_entities: int) -> Dict[str, Any]:
        rows = []
        for e in self._get_registry_entities():
            aliases = (getattr(e, "metadata", {}) or {}).get("aliases", [])
            rows.append({"uri": str(getattr(e, "uri", "")), "alias_count": len(aliases) if isinstance(aliases, list) else 0})
        rows.sort(key=lambda x: (-int(x["alias_count"]), str(x["uri"])))
        return {
            "raw_entities": int(raw_entities),
            "canonical_entities": int(canonical_entities),
            "merges_count": max(0, int(raw_entities) - int(canonical_entities)),
            "top_entities_by_alias_count": rows[:20],
        }

    def _fuzzy_cluster_and_merge(self, entities: List[Any]) -> List[Any]:
        # Union-find clustering per coarse type with capped stabilization loop.
        by_type: Dict[str, List[Any]] = defaultdict(list)
        for e in entities:
            by_type[self._coarse_type(e)].append(e)

        merged_out: List[Any] = []
        for ctype in sorted(by_type.keys(), key=lambda x: str(x)):
            typed = sorted(by_type[ctype], key=self._entity_sort_key)
            index = {str(getattr(e, "uri", "")): i for i, e in enumerate(typed)}
            parent = list(range(len(typed)))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            # bucketed candidate generation (no global O(n^2)).
            buckets: Dict[str, List[int]] = defaultdict(list)
            for i, e in enumerate(typed):
                toks = sorted(list(self._tokenize_for_match(getattr(e, "label", "") or "")))
                if not toks:
                    buckets["_empty"].append(i)
                else:
                    for t in toks[:2]:
                        buckets[t].append(i)

            seen_pairs: Set[Tuple[int, int]] = set()
            for _bucket_key in sorted(buckets.keys(), key=lambda x: str(x)):
                idxs = sorted(set(buckets[_bucket_key]))
                for a_pos in range(len(idxs)):
                    for b_pos in range(a_pos + 1, len(idxs)):
                        ia, ib = idxs[a_pos], idxs[b_pos]
                        p = (min(ia, ib), max(ia, ib))
                        if p in seen_pairs:
                            continue
                        seen_pairs.add(p)
                        a = typed[ia]
                        b = typed[ib]
                        if not self._can_fuzzy_merge(a, b, ctype):
                            continue
                        jacc, lev = self._similarity_scores(str(getattr(a, "label", "")), str(getattr(b, "label", "")))
                        score = (jacc + lev) / 2.0
                        if score >= 0.92:
                            union(ia, ib)

            clusters: Dict[int, List[Any]] = defaultdict(list)
            for i, e in enumerate(typed):
                clusters[find(i)].append(e)

            for root in sorted(clusters.keys()):
                members = sorted(clusters[root], key=self._entity_sort_key)
                c = self._pick_canonical(members)
                key = self._external_id(c) or self._normalized_match_label(getattr(c, "label", "") or "")
                minted = self._mint_canonical_uri(ctype, key)
                old = str(getattr(c, "uri", "") or "")
                c.uri = minted
                if old and old != minted:
                    self.redirects[old] = minted
                for s in members:
                    self._register_alias(s, c, "fuzzy_cluster", {"score": 0.92})
                    if s is c:
                        continue
                    self._merge(c, s, "fuzzy_cluster", {"score": 0.92})
                merged_out.append(c)

        return merged_out

    def _mint_canonical_uri(self, coarse_type: str, key: str) -> str:
        base = Config.NAMESPACES["etd"].rstrip("#/")
        ctype = self._slug(coarse_type) or "entity"
        label_slug = self._slug(key) or "item"
        h = hashlib.sha1(f"{coarse_type}|{key}".encode("utf-8")).hexdigest()[:8]
        return f"{base}/entity/{ctype}/{label_slug}-{h}"

    def _pick_canonical(self, group: List[Any]) -> Any:
        if len(group) == 1:
            return group[0]
        return max(
            group,
            key=lambda x: (
                float(getattr(x, "confidence", 0.0) or 0.0),
                len(str(getattr(x, "label", "") or "")),
                str(getattr(x, "label", "") or ""),
                str(getattr(x, "uri", "") or ""),
            ),
        )

    @staticmethod
    def _entity_sort_key(e: Any) -> Tuple[str, str, str]:
        return (
            str(getattr(e, "uri", "") or ""),
            str(getattr(e, "entity_type", "") or ""),
            str(getattr(e, "label", "") or ""),
        )

    def _register_alias(self, source: Any, canonical: Any, rule: str, scores: Dict[str, float]) -> None:
        raw_label = str(getattr(source, "label", "") or "")
        norm_label = self._normalized_match_label(raw_label)
        can_uri = str(getattr(canonical, "uri", "") or "")
        if norm_label:
            self.alias_index[norm_label] = can_uri
        if raw_label:
            self.alias_index_raw[raw_label] = can_uri
        self.decisions.append(
            LinkDecision(
                input_label=raw_label,
                norm_label=norm_label,
                chosen_canonical_uri=can_uri,
                rule=rule,
                scores={k: float(v) for k, v in scores.items()},
                source_ids=[str(getattr(source, "uri", "") or ""), can_uri],
            )
        )

    def _merge(self, canonical: Any, source: Any, rule: str, scores: Dict[str, float]) -> None:
        canonical.metadata = canonical.metadata or {}
        source.metadata = source.metadata or {}

        aliases = canonical.metadata.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        for lbl in (getattr(canonical, "label", None), getattr(source, "label", None)):
            if lbl and lbl not in aliases:
                aliases.append(lbl)
        canonical.metadata["aliases"] = aliases

        source_ids = canonical.metadata.get("source_ids")
        if not isinstance(source_ids, list):
            source_ids = []
        for sid in (str(getattr(source, "uri", "") or ""), str(getattr(canonical, "uri", "") or "")):
            if sid and sid not in source_ids:
                source_ids.append(sid)
        canonical.metadata["source_ids"] = source_ids

        self._safe_merge_metadata(canonical.metadata, source.metadata)

        if getattr(source, "ontology_uri", None) and not getattr(canonical, "ontology_uri", None):
            canonical.ontology_uri = source.ontology_uri

        if float(getattr(source, "confidence", 0.0) or 0.0) > float(getattr(canonical, "confidence", 0.0) or 0.0):
            canonical.confidence = source.confidence
            if getattr(source, "label", None):
                canonical.label = source.label

        old_uri = str(getattr(source, "uri", "") or "")
        new_uri = str(getattr(canonical, "uri", "") or "")
        if old_uri and new_uri and old_uri != new_uri:
            self.redirects[old_uri] = new_uri
            if hasattr(self.registry, "replace_entity"):
                self.registry.replace_entity(old_uri, new_uri)

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

    def _merge_mentions(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        merged: List[Dict[str, Any]] = []
        for bucket in (target.get("mentions"), source.get("mentions")):
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
        target["mentions"] = [dedup[k] for k in sorted(dedup.keys())]

    def _safe_merge_metadata(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        self._merge_mentions(target, source or {})
        for k, v in (source or {}).items():
            if k == "mentions":
                continue
            if v is None:
                continue
            if k not in target or target[k] is None:
                target[k] = v
                continue
            tv = target[k]
            if isinstance(tv, dict) and isinstance(v, dict):
                self._safe_merge_metadata(tv, v)
            elif isinstance(tv, list):
                incoming = v if isinstance(v, list) else [v]
                for item in incoming:
                    if item not in tv:
                        tv.append(item)
            elif tv != v:
                target[k] = [tv, v] if not isinstance(tv, list) else tv + ([v] if v not in tv else [])

    def _can_fuzzy_merge(self, a: Any, b: Any, coarse_type: str) -> bool:
        if self._coarse_type(a) != coarse_type or self._coarse_type(b) != coarse_type:
            return False
        if coarse_type in self.STRUCTURAL_TYPES:
            return False
        la = self._normalized_match_label(getattr(a, "label", "") or "")
        lb = self._normalized_match_label(getattr(b, "label", "") or "")
        if not la or not lb:
            return False
        if re.fullmatch(r"(theme|question|answer|vote|expert|document|discourseunit)\s*[_-]?\s*\d+", la):
            return False
        if re.fullmatch(r"(theme|question|answer|vote|expert|document|discourseunit)\s*[_-]?\s*\d+", lb):
            return False
        if la in self.DO_NOT_MERGE or lb in self.DO_NOT_MERGE:
            return False
        if len(la) < 3 or len(lb) < 3:
            return False

        a_ext = self._external_id(a)
        b_ext = self._external_id(b)
        if a_ext and b_ext and a_ext != b_ext:
            return False

        a_ont = str(getattr(a, "ontology_uri", "") or (getattr(a, "metadata", {}) or {}).get("ontology_uri") or "")
        b_ont = str(getattr(b, "ontology_uri", "") or (getattr(b, "metadata", {}) or {}).get("ontology_uri") or "")
        if a_ont and b_ont and a_ont != b_ont:
            return False
        return True

    def _external_id(self, e: Any) -> str:
        meta = getattr(e, "metadata", {}) or {}
        candidates: List[Any] = [
            meta.get("external_id"),
            meta.get("wikidata_uri"),
            meta.get("doi"),
            meta.get("url"),
            meta.get("qid"),
            meta.get("wikidata_id"),
        ]

        wd_obj = meta.get("wikidata")
        if isinstance(wd_obj, dict):
            candidates.append(wd_obj.get("qid"))
            candidates.append(wd_obj.get("id"))
            candidates.append(wd_obj.get("uri"))

        wd_match = meta.get("wikidata_match")
        if isinstance(wd_match, dict):
            candidates.append(wd_match.get("qid"))
            candidates.append(wd_match.get("id"))
            candidates.append(wd_match.get("uri"))

        for v in candidates:
            norm = self._normalize_external_id(v)
            if norm:
                return norm
        return ""

    def _normalize_external_id(self, value: Any) -> str:
        if value is None:
            return ""
        s = str(value).strip()
        if not s:
            return ""

        # Wikidata normalization to a stable token.
        m = re.search(r"(?:^|/|:)(Q\d+)$", s, flags=re.IGNORECASE)
        if m:
            return f"wikidata:{m.group(1).upper()}"

        if "wikidata.org/entity/" in s.lower() or "wikidata.org/wiki/" in s.lower():
            tail = s.rstrip("/").split("/")[-1]
            if re.fullmatch(r"Q\d+", tail, flags=re.IGNORECASE):
                return f"wikidata:{tail.upper()}"

        # DOI normalization.
        if s.lower().startswith("doi:"):
            return f"doi:{s[4:].strip().lower()}"
        if "doi.org/" in s.lower():
            doi_tail = s.split("doi.org/", 1)[-1].strip().lower()
            if doi_tail:
                return f"doi:{doi_tail}"

        # URL normalization.
        if s.startswith("http://") or s.startswith("https://"):
            return self._normalize_url(s)

        return s.casefold()

    def _coarse_type(self, e: Any) -> str:
        ont = str(getattr(e, "ontology_uri", "") or "").strip()
        if ont and self._is_class_uri(ont):
            local = ont.split("#")[-1].split("/")[-1]
            if local:
                return local
        et = str(getattr(e, "entity_type", "Entity") or "Entity").strip()
        return et if et else "Entity"

    def _is_class_uri(self, uri: str) -> bool:
        if not uri:
            return False
        for ns in self.CLASS_NS_PREFIXES:
            if ns and uri.startswith(ns):
                return True
        return False

    def _normalized_match_label(self, label: str) -> str:
        s = self._normalize_text(label)
        if s in self.ALIAS_TABLE:
            s = self._normalize_text(self.ALIAS_TABLE[s])
        return s

    def _normalize_text(self, text: str) -> str:
        s = unicodedata.normalize("NFKC", str(text or ""))
        s = s.strip()
        if not s:
            return ""
        if s.startswith("http://") or s.startswith("https://"):
            s = self._normalize_url(s)
        s = s.casefold()
        s = re.sub(r"[\u2013\u2014]", "-", s)
        s = re.sub(r"[^\w\s:/.-]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _normalize_url(self, raw_url: str) -> str:
        p = urlparse(raw_url.strip())
        scheme = p.scheme.lower() or "https"
        netloc = p.netloc.lower()
        path = re.sub(r"/+$", "", p.path or "")
        query_pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if not k.lower().startswith("utm_")]
        query = urlencode(query_pairs)
        return urlunparse((scheme, netloc, path, "", query, ""))

    def _tokenize_for_match(self, label: str) -> Set[str]:
        s = self._normalized_match_label(label)
        toks = [t for t in re.split(r"[\s_/:-]+", s) if t]
        if self._looks_organization_label(label):
            toks = [t for t in toks if t not in self.ORG_SUFFIXES]
        return set(toks)

    def _looks_organization_label(self, label: str) -> bool:
        s = self._normalize_text(label)
        return any(re.search(rf"\b{re.escape(suf)}\.?$", s) for suf in self.ORG_SUFFIXES)

    def _similarity_scores(self, a: str, b: str) -> Tuple[float, float]:
        ta = self._tokenize_for_match(a)
        tb = self._tokenize_for_match(b)
        if not ta or not tb:
            return 0.0, 0.0
        jaccard = len(ta.intersection(tb)) / len(ta.union(tb))
        lev = SequenceMatcher(None, self._normalized_match_label(a), self._normalized_match_label(b)).ratio()
        return jaccard, lev

    def _slug(self, s: str) -> str:
        t = self._normalize_text(s)
        t = re.sub(r"[^a-z0-9]+", "-", t)
        t = re.sub(r"-+", "-", t).strip("-")
        return t

    def _persist(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        alias_payload = {
            "normalized": dict(sorted(self.alias_index.items())),
            "raw": dict(sorted(self.alias_index_raw.items())),
        }
        (self.output_dir / "alias_map.json").write_text(
            json.dumps(alias_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (self.output_dir / "canonical_linking_log.jsonl").open("w", encoding="utf-8") as f:
            ordered_decisions = sorted(
                self.decisions,
                key=lambda d: (
                    d.chosen_canonical_uri,
                    d.norm_label,
                    d.input_label,
                    d.rule,
                    "|".join(d.source_ids),
                ),
            )
            for d in ordered_decisions:
                f.write(
                    json.dumps(
                        {
                            "input_label": d.input_label,
                            "norm_label": d.norm_label,
                            "chosen_canonical_uri": d.chosen_canonical_uri,
                            "rule": d.rule,
                            "scores": d.scores,
                            "source_ids": d.source_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _get_registry_entities(self) -> List[Any]:
        get_all = getattr(self.registry, "get_all_entities", None)
        if callable(get_all):
            return list(get_all())
        store = getattr(self.registry, "entities", None)
        if isinstance(store, dict):
            return list(store.values())
        if isinstance(store, list):
            return list(store)
        return []


