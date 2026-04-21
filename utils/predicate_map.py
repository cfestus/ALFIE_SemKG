import re

# ---------------------------------------------------------
# Canonical predicate mapping (relation normalization)
# Keys: messy variants (LLM output, spacing, case, synonyms)
# Values: MUST be one of the object property names in add_schema.py
# ---------------------------------------------------------
CANONICAL_REL_PREDICATES = {
    # -------------------------
    # Generic / fallback
    # -------------------------
    "relatedto": "relatedTo",
    "related_to": "relatedTo",
    "related to": "relatedTo",
    "relatesto": "relatedTo",
    "relates_to": "relatedTo",
    "relates to": "relatedTo",
    "isrelatedto": "relatedTo",
    "is related to": "relatedTo",
    "linkedto": "relatedTo",
    "linked to": "relatedTo",
    "connectsto": "relatedTo",
    "connects to": "relatedTo",

    "associatedwith": "associatedWith",
    "associated with": "associatedWith",
    "associated_to": "associatedWith",
    "associatedto": "associatedWith",

    # -------------------------
    # Discourse / structure
    # -------------------------
    "mentions": "mentions",
    "mention": "mentions",

    "answersquestion": "answersQuestion",
    "answers question": "answersQuestion",

    "authoredby": "authoredBy",
    "authored by": "authoredBy",
    "writtenby": "authoredBy",
    "written by": "authoredBy",

    "repliesto": "repliesTo",
    "replies to": "repliesTo",
    "replyof": "repliesTo",
    "reply of": "repliesTo",

    "extractedfrom": "extractedFrom",
    "extracted from": "extractedFrom",
    "derivedfrom": "extractedFrom",
    "derived from": "extractedFrom",

    "belongstotheme": "belongsToTheme",
    "belongs to theme": "belongsToTheme",
    "belongsto": "belongsToTheme",
    "hastheme": "belongsToTheme",
    "has theme": "belongsToTheme",

    # Votes / interaction
    "hasvote": "hasVote",
    "has vote": "hasVote",
    "receivesvote": "receivesVote",
    "receives vote": "receivesVote",
    "votesforanswer": "receivesVote",
    "votes for answer": "receivesVote",
    "castby": "castBy",
    "cast by": "castBy",
    "castvote": "castVote",
    "cast vote": "castVote",

    # -------------------------
    # Stance / opinion
    # -------------------------
    "supports": "supports",
    "support": "supports",
    "endorses": "endorses",
    "endorse": "endorses",
    "approves": "endorses",
    "approve": "endorses",
    "criticizes": "criticizes",
    "criticise": "criticizes",
    "critiques": "criticizes",
    "disagreeswith": "disagreesWith",
    "disagrees with": "disagreesWith",
    #"opposes": "disagreesWith",
    "oppose": "disagreesWith",

    # LLM “free text” stance-ish verbs -> collapse to schema
    "discriminatesagainst": "criticizes",
    "discriminates against": "criticizes",
    "reinforces": "relatedTo",

    # -------------------------
    # Use / training / containment / domain
    # -------------------------
    "uses": "uses",
    "use": "uses",
    "utilises": "uses",
    "utilizes": "uses",
    "employs": "uses",
    "applies": "uses",

    "trainedon": "trainedOn",
    "trained on": "trainedOn",
    "trains on": "trainedOn",
    "trainingdata": "trainedOn",
    "training data": "trainedOn",

    "contains": "containsSubject",
    "containssubject": "containsSubject",
    "contains subject": "containsSubject",
    "hassubject": "containsSubject",
    "has subject": "containsSubject",

    "partofdomain": "partOfDomain",
    "part of domain": "partOfDomain",
    "indomain": "partOfDomain",
    "in domain": "partOfDomain",
    "partof": "partOfDomain",
    "part of": "partOfDomain",
    "occursin": "partOfDomain",
    "occurs in": "partOfDomain",
    "usedin": "partOfDomain",
    "used in": "partOfDomain",

    # These “generic process” verbs should not become new predicates
    "provides": "relatedTo",
    "developed": "relatedTo",

    # -------------------------
    # Risk relations
    # -------------------------
    "hasrisk": "hasRisk",
    "has risk": "hasRisk",
    "posesrisk": "hasRisk",
    "poses risk": "hasRisk",
    "introducesrisk": "hasRisk",
    "introduces risk": "hasRisk",

    "exhibitsrisk": "exhibitsRisk",
    "exhibits risk": "exhibitsRisk",
    "showsrisk": "exhibitsRisk",
    "shows risk": "exhibitsRisk",

    "particularlyaffects": "particularlyAffects",
    "particularly affects": "particularlyAffects",
    "disproportionatelyaffects": "particularlyAffects",
    "disproportionately affects": "particularlyAffects",

    # -------------------------
    # Causality / impact
    # -------------------------
    "causes": "causes",
    "cause": "causes",
    "leads to": "causes",
    "leadsto": "causes",
    "results in": "causes",
    "resultsin": "causes",
    "brings about": "causes",
    "creates": "causes",
    "triggers": "causes",
    "induces": "causes",

    "affects": "affects",
    "affect": "affects",
    "impacts": "affects",
    "impact": "affects",
    "influences": "affects",
    "influence": "affects",
    "harms": "affects",
    "harm": "affects",

    "mitigatedby": "mitigatedBy",
    "mitigated by": "mitigatedBy",
    "mitigatedBy": "mitigatedBy",
    "mitigate": "mitigatedBy",
    "reducedby": "mitigatedBy",
    "reduced by": "mitigatedBy",
    "addressedby": "mitigatedBy",
    "addressed by": "mitigatedBy",

    # Backward-compat
    "causedby": "causedBy",
    "caused by": "causedBy",
    "affectedby": "affectedBy",
    "affected by": "affectedBy",
    "chargedhigherfares": "affectedBy",
    "charged higher fares": "affectedBy",
    "paysmorethan": "affectedBy",
    "pays more than": "affectedBy",

    # -------------------------
    # Risk assessment bundle
    # -------------------------
    "assessesrisk": "assessesRisk",
    "assesses risk": "assessesRisk",
    "assessedby": "assessedBy",
    "assessed by": "assessedBy",
    "usesevidence": "usesEvidence",
    "uses evidence": "usesEvidence",
    "hasseverity": "hasSeverity",
    "has severity": "hasSeverity",
    "haslikelihood": "hasLikelihood",
    "has likelihood": "hasLikelihood",
    "hasrisklevel": "hasRiskLevel",
    "has risk level": "hasRiskLevel",

    # Evidence-ish free text -> collapse to schema
    "documentedby": "usesEvidence",
    "documented by": "usesEvidence",
    "revealed": "usesEvidence",
    "found": "usesEvidence",

    # -------------------------
    # Governance instrument references
    # -------------------------
    "hasregulatoryreference": "hasRegulatoryReference",
    "has regulatory reference": "hasRegulatoryReference",
    "referencesinstrument": "referencesInstrument",
    "references instrument": "referencesInstrument",
    "references governance instrument": "referencesInstrument",
    "references law": "referencesInstrument",
    "references standard": "referencesInstrument",
    "referencessource": "referencesSource",
    "references source": "referencesSource",
    "cites": "cites",
    "cite": "cites",
    "accordingto": "cites",
    "according to": "cites",
    "publishedin": "referencesSource",
    "published in": "referencesSource",

    # ---------------------------------------------------------
    # Map non-schema / noisy LLM predicates to canonical schema predicates
    # ---------------------------------------------------------
    "similarto": "relatedTo",
    "similar to": "relatedTo",
    "overlapswith": "relatedTo",
    "overlaps with": "relatedTo",

    "agreeswith": "supports",
    "agrees with": "supports",

    "indirectlycontradicts": "disagreesWith",
    "indirectly contradicts": "disagreesWith",

    "locatedin": "relatedTo",
    "located in": "relatedTo",

    "isdatasetfor": "trainedOn",
    "is dataset for": "trainedOn",

    # Governance reasoning (new schema predicates)
    "complieswith": "compliesWith",
    "complies with": "compliesWith",
    "requirescompliancewith": "requiresComplianceWith",
    "requires compliance with": "requiresComplianceWith",
    "requires": "requires",
    "violates": "violates",
    "mitigates": "mitigates",
    "opposes": "opposes",

    # ---------------------------------------------------------
    # Structural / containment relations (schema-aligned)
    # ---------------------------------------------------------
    "includes": "includes",
    "include": "includes",
    "includedin": "includes",
    "included in": "includes",

}


def normalize_predicate(pred: str) -> str:
    """
    Production rule:
    - return ONLY schema-approved property names (values of CANONICAL_REL_PREDICATES)
    - never return novel predicates (prevents ontology explosion in streaming)
    """
    if not pred:
        return "relatedTo"

    key = pred.strip().lower()

    # normalize separators (handles "chargedHigherFares", "charged_higher_fares", etc.)
    key = re.sub(r"[\s\-]+", " ", key)   # unify whitespace/dashes
    key = key.replace("_", " ")          # unify underscores
    key = key.strip()

    # try exact
    if key in CANONICAL_REL_PREDICATES:
        return CANONICAL_REL_PREDICATES[key]

    # try compact key (remove spaces)
    compact = key.replace(" ", "")
    if compact in CANONICAL_REL_PREDICATES:
        return CANONICAL_REL_PREDICATES[compact]

    # final production fallback: DO NOT mint new predicates
    return "relatedTo"
