"""
utils/semantic_chunker.py
-------------------------
Semantic Chunker for ETD-Hub Knowledge Graph Pipeline.

This module splits long text (themes, questions, answers) into
semantically meaningful segments that improve LLM entity and
relation extraction accuracy.

Chunking Strategy:
    1. Header-based segmentation (Bias / Dataset / Governance / Risk)
    2. Semantic keyword-based segmentation
    3. Sentence-boundary chunking (250–450 token chunks)
    4. Fallback: token-limited chunks

Output:
    List[str] = list of text chunks
"""

import re
from typing import Any, List, Dict, Tuple
import tiktoken


class SemanticChunker:

    # ============================================================
    # 1. Canonical topic headers (very common in your JSON dataset)
    # ============================================================
    TOPIC_HEADERS = [
        r"ethical challenge[s]?",
        r"fairness",
        r"bias",
        r"dataset",
        r"data[- ]set",
        r"risk",
        r"governance",
        r"privacy",
        r"transparency",
        r"explainability",
        r"accountability",
        r"context",
        r"methodology",
        r"problem",
        r"motivation",
        r"conclusion",
        r"summary",
    ]

    # ============================================================
    # 2. Keyword-based soft boundaries
    # ============================================================
    SEMANTIC_CUES = [
        "in this dataset",
        "the main bias",
        "the key risk",
        "the primary risk",
        "ethical concern",
        "fairness concern",
        "according to",
        "research shows",
        "for example",
        "case study",
        "in practice",
        "in reality",
        "evaluation",
        "metric",
        "governance",
        "policy",
        "regulation",
        "human oversight",
        "decision-making",
    ]

    # ============================================================
    # Tokenizer for GPT-4o-mini
    # ============================================================
    try:
        ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        ENCODER = None

    @staticmethod
    def count_tokens(text: str) -> int:
        """Approximate token count for GPT models."""
        if SemanticChunker.ENCODER:
            return len(SemanticChunker.ENCODER.encode(text))
        # fallback approximation
        return max(1, int(len(text.split()) * 1.3))

    # ============================================================
    # MAIN ENTRY: chunk()
    # ============================================================
    @classmethod
    def chunk(cls, text: str) -> List[str]:
        """
        Master chunking function.
        Returns a list of semantically meaningful segments.
        """
        return [x["text"] for x in cls.chunk_with_offsets(text)]

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
        s = int(max(0, start))
        e = int(max(s, end))
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        return s, e

    @classmethod
    def _chunk_by_topic_headers_with_offsets(cls, text: str) -> List[Tuple[int, int]]:
        header_patterns = [fr"(?:^|\n)\s*({header})\s*:?" for header in cls.TOPIC_HEADERS]
        pattern = r"(?im)" + "|".join(header_patterns)
        matches = list(re.finditer(pattern, text))
        if not matches:
            return [(0, len(text))]

        spans: List[Tuple[int, int]] = []
        prev = 0
        for m in matches:
            start = m.start()
            if start > 0 and text[start] == "\n":
                start += 1
            if start - prev > 30:
                spans.append((prev, start))
            prev = start
        spans.append((prev, len(text)))
        return spans

    @classmethod
    def _split_segment_by_tokens_with_offsets(cls, text: str, seg_start: int, seg_end: int) -> List[Tuple[int, int]]:
        seg_text = text[seg_start:seg_end]
        if not seg_text.strip():
            return []

        MAX_TOK = 450
        if cls.count_tokens(seg_text.strip()) <= MAX_TOK:
            s, e = cls._trim_span(text, seg_start, seg_end)
            return [(s, e)] if e > s else []

        sentence_pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)\s*", flags=re.S)
        sentence_spans: List[Tuple[int, int]] = []
        for m in sentence_pattern.finditer(seg_text):
            s = seg_start + m.start()
            e = seg_start + m.end()
            ts, te = cls._trim_span(text, s, e)
            if te > ts:
                sentence_spans.append((ts, te))

        if not sentence_spans:
            ts, te = cls._trim_span(text, seg_start, seg_end)
            return [(ts, te)] if te > ts else []

        out: List[Tuple[int, int]] = []
        cur_start: int = sentence_spans[0][0]
        cur_end: int = sentence_spans[0][1]

        for s, e in sentence_spans[1:]:
            candidate = text[cur_start:e]
            if cls.count_tokens(candidate) <= MAX_TOK:
                cur_end = e
            else:
                out.append((cur_start, cur_end))
                if cls.count_tokens(text[s:e]) > MAX_TOK:
                    words = list(re.finditer(r"\S+\s*", text[s:e]))
                    if not words:
                        out.append((s, e))
                        cur_start, cur_end = s, e
                        continue
                    approx_words = max(1, int(MAX_TOK / 1.3))
                    i = 0
                    while i < len(words):
                        ws = s + words[i].start()
                        j = min(len(words), i + approx_words) - 1
                        we = s + words[j].end()
                        ts, te = cls._trim_span(text, ws, we)
                        if te > ts:
                            out.append((ts, te))
                        i = j + 1
                    cur_start, cur_end = s, e
                else:
                    cur_start, cur_end = s, e
        out.append((cur_start, cur_end))
        return out

    @classmethod
    def _merge_small_chunks_with_offsets(
        cls,
        text: str,
        spans: List[Tuple[int, int]],
        min_tok: int = 200,
        max_tok: int = 450,
    ) -> List[Tuple[int, int]]:
        merged: List[Tuple[int, int]] = []
        if not spans:
            return merged
        cur_s, cur_e = spans[0]
        for s, e in spans[1:]:
            candidate_s = cur_s
            candidate_e = e
            cur_text = text[cur_s:cur_e]
            candidate_text = text[candidate_s:candidate_e]
            if cls.count_tokens(candidate_text) <= max_tok and cls.count_tokens(cur_text) < min_tok:
                cur_e = candidate_e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    @classmethod
    def chunk_with_offsets(cls, text: str) -> List[Dict[str, Any]]:
        if text is None:
            return []
        source_text = str(text)
        if not source_text.strip():
            return []

        if len(source_text.strip()) < 80:
            s, e = cls._trim_span(source_text, 0, len(source_text))
            return [{"text": source_text[s:e], "start_offset": int(s), "end_offset": int(e), "chunk_index": 0}] if e > s else []

        header_spans = cls._chunk_by_topic_headers_with_offsets(source_text)
        token_spans: List[Tuple[int, int]] = []
        for hs, he in header_spans:
            token_spans.extend(cls._split_segment_by_tokens_with_offsets(source_text, hs, he))
        token_spans = cls._merge_small_chunks_with_offsets(source_text, token_spans, min_tok=200, max_tok=450)

        out: List[Dict[str, int]] = []
        for idx, (s, e) in enumerate(token_spans):
            ts, te = cls._trim_span(source_text, s, e)
            if te <= ts:
                continue
            out.append(
                {
                    "text": source_text[ts:te],
                    "start_offset": int(ts),
                    "end_offset": int(te),
                    "chunk_index": int(idx),
                }
            )
        return out


    # ============================================================
    # STEP 1 — HEADER-BASED CHUNKING
    # ============================================================
    @classmethod
    def _chunk_by_topic_headers(cls, text: str) -> List[str]:
        """
        Splits text on logical topic sections such as:
        "Ethical Challenges", "Dataset", "Fairness", "Risk".
        """

        # Move (?i) to the front of the full expression, not each fragment
        # Treat headers as headings only when they appear at line starts or after a newline,
        # optionally followed by ":" (reduces splitting inside normal sentences).
        header_patterns = [fr"(?:^|\n)\s*({header})\s*:?" for header in cls.TOPIC_HEADERS]
        pattern = r"(?im)" + "|".join(header_patterns)

        matches = list(re.finditer(pattern, text))
        if not matches:
            return [text]

        chunks = []
        prev = 0

        for m in matches:
            start = m.start()
            # keep newline boundary clean
            if start > 0 and text[start] == "\n":
                start += 1            
            
            if start - prev > 30:  # avoid tiny gaps
                chunks.append(text[prev:start])
            prev = start

        chunks.append(text[prev:])
        
        return chunks



    # ============================================================
    # STEP 2 — SOFT SEMANTIC CUE SPLITTING
    # ============================================================
    @classmethod
    def _chunk_by_semantic_cues(cls, segments: List[str]) -> List[str]:
        """
         Further splits segments based on soft semantic cues
        (e.g., 'ethical concern', 'in this dataset', 'evaluation').
        Uses case-insensitive splitting WITHOUT inline regex flags.
        """
        new_segments = []

        # Build alternation pattern WITHOUT (?i)
        cue_pattern = "|".join([re.escape(cue) for cue in cls.SEMANTIC_CUES])

        for segment in segments:
            # Apply case-insensitive splitting using flags
            splits = re.split(cue_pattern, segment, flags=re.IGNORECASE)
            if len(splits) > 1:
                new_segments.extend(splits)
            else:
                new_segments.append(segment)

        return new_segments


    # ============================================================
    # STEP 3 — TOKEN-SAFE CHUNKING
    # ============================================================
    @classmethod
    def _split_by_tokens(cls, segments: List[str]) -> List[str]:
        """
        Ensures chunks are between 200–450 tokens.
        Preserves sentence boundaries.
        """
        MIN_TOK = 200
        MAX_TOK = 450

        final_chunks = []

        for seg in segments:
            seg = seg.strip()
            tok = cls.count_tokens(seg)

            if tok <= MAX_TOK:
                final_chunks.append(seg)
                continue

            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", seg)
            buffer = ""

            for sent in sentences:
                if not sent.strip():
                    continue

                new_buffer = buffer + " " + sent if buffer else sent
                if cls.count_tokens(new_buffer) > MAX_TOK:
                    # flush and reset
                    if buffer:
                        final_chunks.append(buffer)
                        buffer = sent
                    else:
                        # extremely long sentence; hard-split by tokens
                        if cls.ENCODER:
                            toks = cls.ENCODER.encode(sent)
                            for i in range(0, len(toks), MAX_TOK):
                                piece = cls.ENCODER.decode(toks[i:i + MAX_TOK]).strip()
                                if piece:
                                    final_chunks.append(piece)
                        else:
                            # fallback: split by words
                            words = sent.split()
                            approx_words = max(1, int(MAX_TOK / 1.3))
                            for i in range(0, len(words), approx_words):
                                piece = " ".join(words[i:i + approx_words]).strip()
                                if piece:
                                    final_chunks.append(piece)
                        buffer = ""
                else:
                    buffer = new_buffer

            if buffer:
                final_chunks.append(buffer)

        return final_chunks


    @classmethod
    def _merge_small_chunks(cls, chunks: List[str], min_tok: int = 200, max_tok: int = 450) -> List[str]:
        """Merge adjacent small chunks to reach min_tok without exceeding max_tok."""
        merged: List[str] = []
        buf = ""

        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue

            if not buf:
                buf = ch
                continue

            # try merge
            candidate = buf + "\n" + ch
            if cls.count_tokens(candidate) <= max_tok and cls.count_tokens(buf) < min_tok:
                buf = candidate
            else:
                merged.append(buf)
                buf = ch

        if buf:
            merged.append(buf)

        return merged
