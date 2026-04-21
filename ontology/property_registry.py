from __future__ import annotations

from typing import Dict, List

from rdflib.namespace import RDF, RDFS, XSD


# Authoritative ETD predicate registry for serializer-emitted predicates.
# Serializer code must not emit ETD predicates from dynamic keys unless the local
# name appears in this registry.
ETD_PROPERTY_REGISTRY: Dict[str, Dict[str, object]] = {
    # --- Export/build metadata (operational metadata, broad domain by design) ---
    "datasetId": {
        "kind": "datatype",
        "label": "dataset identifier",
        "comment": "Identifier of the source dataset used for this export/run.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "runId": {
        "kind": "datatype",
        "label": "run identifier",
        "comment": "Identifier of the pipeline execution run.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "pipelineVersion": {
        "kind": "datatype",
        "label": "pipeline version",
        "comment": "Version identifier of the pipeline code/config used for export.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "gitCommit": {
        "kind": "datatype",
        "label": "git commit",
        "comment": "Source code commit hash associated with the export run.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "shaclProfileVersion": {
        "kind": "datatype",
        "label": "SHACL profile version",
        "comment": "Version or identifier of SHACL profile used for validation.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "exportMode": {
        "kind": "datatype",
        "label": "export mode",
        "comment": "Export mode identifier (for example: ai).",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    # --- Reified relation metadata ---
    "confidence": {
        "kind": "datatype",
        "label": "confidence",
        "comment": "Confidence score for an entity/relation/evidence assertion.",
        "domains": ["Entity"],
        "range": XSD.float,
    },
    "isInferred": {
        "kind": "datatype",
        "label": "is inferred",
        "comment": "True when relation is inferred rather than directly extracted/structural.",
        "domains": ["Relation"],
        "range": XSD.boolean,
    },
    "inferenceRule": {
        "kind": "datatype",
        "label": "inference rule",
        "comment": "Deterministic identifier of the inference rule/path that produced a relation.",
        "domains": ["Relation"],
        "range": XSD.string,
    },
    # --- Raw field materialisation provenance ---
    "hasFieldAssertion": {
        "kind": "object",
        "label": "has field assertion",
        "comment": "Links a subject node to a field-level provenance assertion.",
        "domains": ["Entity"],
        "range": None,
    },
    "fieldName": {
        "kind": "datatype",
        "label": "field name",
        "comment": "Original source field name from raw dataset JSON.",
        "domains": ["FieldAssertion"],
        "range": XSD.string,
    },
    "recordId": {
        "kind": "datatype",
        "label": "record identifier",
        "comment": "Original source record identifier from raw dataset JSON.",
        "domains": ["FieldAssertion"],
        "range": XSD.string,
    },
    "sourceSection": {
        "kind": "datatype",
        "label": "source section",
        "comment": "Source section name (themes/questions/answers/votes/experts).",
        "domains": ["FieldAssertion"],
        "range": XSD.string,
    },
    "assertedPredicate": {
        "kind": "object",
        "label": "asserted predicate",
        "comment": "Predicate used by the corresponding materialised raw-field triple.",
        "domains": ["FieldAssertion"],
        "range": RDF.Property,
    },
    "assertedValue": {
        "kind": "rdf_property",
        "label": "asserted value",
        "comment": "Object used by the corresponding materialised raw-field triple.",
        "domains": ["FieldAssertion"],
        "range": RDFS.Resource,
    },
    # --- Evidence / grounding / offsets ---
    "source": {
        "kind": "datatype",
        "label": "source",
        "comment": "Extractor/provenance source identifier literal attached to relation/evidence metadata.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "field": {
        "kind": "datatype",
        "label": "field",
        "comment": "Source field identifier (for example theme.description).",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "discourse_id": {
        "kind": "datatype",
        "label": "discourse identifier",
        "comment": "Discourse unit identifier associated with provenance.",
        "domains": ["Resource"],
        "range": RDFS.Literal,
    },
    "chunk_index": {
        "kind": "datatype",
        "label": "chunk index",
        "comment": "Chunk index of source span; -1 denotes unknown.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "chunk_index_unknown": {
        "kind": "datatype",
        "label": "chunk index unknown",
        "comment": "Flag indicating unknown chunk index provenance.",
        "domains": ["Resource"],
        "range": XSD.boolean,
    },
    "chunk_total": {
        "kind": "datatype",
        "label": "chunk total",
        "comment": "Total number of chunks in the source discourse.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "chunk_start": {
        "kind": "datatype",
        "label": "chunk start",
        "comment": "Chunk start character offset.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "start_char": {
        "kind": "datatype",
        "label": "start character",
        "comment": "Start character offset.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "end_char": {
        "kind": "datatype",
        "label": "end character",
        "comment": "End character offset.",
        "domains": ["Resource"],
        "range": XSD.integer,
    },
    "char_start": {
        "kind": "datatype",
        "label": "character start",
        "comment": "Start character offset for evidence span.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "char_end": {
        "kind": "datatype",
        "label": "character end",
        "comment": "End character offset for evidence span.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "start_offset": {
        "kind": "datatype",
        "label": "start offset",
        "comment": "Normalised start offset for evidence span.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "end_offset": {
        "kind": "datatype",
        "label": "end offset",
        "comment": "Normalised end offset for evidence span.",
        "domains": ["Evidence"],
        "range": XSD.integer,
    },
    "source_doc_id": {
        "kind": "datatype",
        "label": "source document id",
        "comment": "Identifier of the source discourse/document.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "chunk_id": {
        "kind": "datatype",
        "label": "chunk id",
        "comment": "Stable chunk identifier.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "quote_hash": {
        "kind": "datatype",
        "label": "quote hash",
        "comment": "Deterministic hash of stored evidence text.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "span_hash": {
        "kind": "datatype",
        "label": "span hash",
        "comment": "Deterministic hash of source/chunk/span provenance key.",
        "domains": ["Evidence"],
        "range": XSD.string,
    },
    "groundingStatus": {
        "kind": "datatype",
        "label": "grounding status",
        "comment": "Grounding status marker.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "groundingMethod": {
        "kind": "datatype",
        "label": "grounding method",
        "comment": "Grounding method marker.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "batch_id": {
        "kind": "datatype",
        "label": "batch id",
        "comment": "Processing batch identifier.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "ingestedAt": {
        "kind": "datatype",
        "label": "ingested at",
        "comment": "Ingestion timestamp.",
        "domains": ["Resource"],
        "range": XSD.dateTime,
    },
    "surface_form": {
        "kind": "datatype",
        "label": "surface form",
        "comment": "Surface lexical form.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "context_source": {
        "kind": "datatype",
        "label": "context source",
        "comment": "Extractor/source context.",
        "domains": ["Resource"],
        "range": XSD.string,
    },
    "text": {
        "kind": "datatype",
        "label": "text",
        "comment": "Text literal carried in metadata.",
        "domains": ["Entity"],
        "range": XSD.string,
    },
    # --- Raw materialised source fields ---
    "createdAt": {"kind": "datatype", "label": "created at", "comment": "Creation timestamp in source data.", "domains": ["Entity"], "range": XSD.dateTime},
    "updatedAt": {"kind": "datatype", "label": "updated at", "comment": "Update timestamp in source data.", "domains": ["Entity"], "range": XSD.dateTime},
    "views": {"kind": "datatype", "label": "views", "comment": "View count in source data.", "domains": ["Entity"], "range": XSD.integer},
    "isDeleted": {"kind": "datatype", "label": "is deleted", "comment": "Deletion marker in source data.", "domains": ["Entity"], "range": XSD.boolean},
    "dateJoined": {"kind": "datatype", "label": "date joined", "comment": "Joined timestamp in source profile data.", "domains": ["Entity"], "range": XSD.dateTime},
    "areaOfExpertise": {"kind": "datatype", "label": "area of expertise", "comment": "Declared area of expertise.", "domains": ["Entity"], "range": XSD.string},
    "bio": {"kind": "datatype", "label": "bio", "comment": "Biographical text.", "domains": ["Entity"], "range": XSD.string},
    "profilePicture": {"kind": "rdf_property", "label": "profile picture", "comment": "Profile picture IRI or literal path.", "domains": ["Entity"], "range": RDFS.Resource},
    "modelCategory": {"kind": "datatype", "label": "model category", "comment": "Model category from source data.", "domains": ["Entity"], "range": XSD.string},
    "domainCategory": {"kind": "datatype", "label": "domain category", "comment": "Domain category from source data.", "domains": ["Entity"], "range": XSD.string},
    "problemCategory": {"kind": "datatype", "label": "problem category", "comment": "Problem category from source data.", "domains": ["Entity"], "range": XSD.string},
    "name": {"kind": "datatype", "label": "name", "comment": "Name from source data.", "domains": ["Entity"], "range": XSD.string},
    "title": {"kind": "datatype", "label": "title", "comment": "Title from source data.", "domains": ["Entity"], "range": XSD.string},
    "body": {"kind": "datatype", "label": "body", "comment": "Body text from source data.", "domains": ["Entity"], "range": XSD.string},
    "description": {"kind": "datatype", "label": "description", "comment": "Description text from source data.", "domains": ["Entity"], "range": XSD.string},
    "userId": {"kind": "datatype", "label": "user identifier", "comment": "User identifier from source data.", "domains": ["Person"], "range": XSD.integer},
    "expertId": {"kind": "datatype", "label": "expert identifier", "comment": "Expert identifier from source data.", "domains": ["Entity"], "range": XSD.integer},
    "themeId": {"kind": "datatype", "label": "theme identifier", "comment": "Theme foreign key from source data.", "domains": ["Entity"], "range": XSD.integer},
    "questionId": {"kind": "datatype", "label": "question identifier", "comment": "Question foreign key from source data.", "domains": ["Entity"], "range": XSD.integer},
    "answerId": {"kind": "datatype", "label": "answer identifier", "comment": "Answer foreign key from source data.", "domains": ["Entity"], "range": XSD.integer},
    "parentId": {"kind": "datatype", "label": "parent identifier", "comment": "Parent answer identifier from source data.", "domains": ["Entity"], "range": XSD.integer},
    "voteValueRaw": {"kind": "datatype", "label": "vote value raw", "comment": "Raw vote value from source data.", "domains": ["Entity"], "range": XSD.integer},
}


# Optional governance metadata used by ontology quality checks.
_LIFECYCLE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "createdAt": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "updatedAt": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "views": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "isDeleted": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "dateJoined": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "areaOfExpertise": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
    "bio": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when sensitive-field materialisation is enabled.",
    },
    "profilePicture": {
        "lifecycle": "conditional",
        "lifecycle_note": "Materialised only when raw source fields are enabled and present.",
    },
}

for _local_name, _spec in ETD_PROPERTY_REGISTRY.items():
    _spec.setdefault("lifecycle", "active")
    _spec.setdefault("lifecycle_note", "")
    _override = _LIFECYCLE_OVERRIDES.get(_local_name)
    if _override:
        _spec.update(_override)


def is_registered_etd_predicate(local_name: str) -> bool:
    return str(local_name) in ETD_PROPERTY_REGISTRY


def get_registered_etd_predicate(local_name: str) -> Dict[str, object]:
    return ETD_PROPERTY_REGISTRY[str(local_name)]


def registered_etd_predicates() -> List[str]:
    return sorted(ETD_PROPERTY_REGISTRY.keys())
