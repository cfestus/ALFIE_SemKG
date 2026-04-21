from typing import Dict, List

# ============================================================
# ORIGINAL FULL ONTOLOGY MAPPINGS (required by OntologyMapper)
# ============================================================

CONCEPT_TO_ONTOLOGY_CLASSES: Dict[str, List[str]] = {
    "bias": ["airo:BiasRisk"],
    "racial bias": ["airo:RacialBiasRisk"],
    "historical bias": ["fmo:HistoricalBias"],
    "algorithmic bias": ["fmo:Bias"],
    "fairness": ["fmo:Fairness"],
    "transparency": ["aipo:Transparency"],
    "explainability": ["aipo:Explainability"],
    "privacy": ["dpv:Privacy"],
    "personal data": ["dpv:PersonalData"],
    "consent": ["dpv:Consent"],
    "accountability": ["aipo:Accountability"],
    "risk": ["airo:Risk"],
    "safety": ["airo:SafetyRisk"],
    "governance": ["aipo:AIGovernance"],
    "regulation": ["vair:Regulation"],
    "audit": ["aipo:AssessmentProcedure"],
}

ENTITY_TYPE_TO_ONTOLOGY: Dict[str, List[str]] = {
    "Dataset": ["dcat:Dataset"],
    "AISystem": ["aipo:AISystem"],
    "Model": ["aipo:AISystem"],
    "Person": ["foaf:Person"],
    "Expert": ["foaf:Person"],
    "Organization": ["org:Organization"],
    "EthicalConcept": ["aipo:EthicalConcept"],
    "BiasType": ["fmo:Bias"],
    "Risk": ["airo:Risk"],
    "Regulation": ["vair:Regulation"],
    "Theme": ["skos:Concept"],
    "Question": ["schema:Question"],
    "Answer": ["schema:Answer"],
    "Vote": ["prov:Activity"],
    "Document": ["foaf:Document"],
    "Resource": ["foaf:Document"],
    "DiscourseUnit": ["prov:Entity"],
    "PolicyFramework": ["vair:Regulation"],
    "Law": ["vair:Regulation"],
    "Standard": ["vair:Regulation"],
    "Corpus": ["dcat:Dataset"],
    "DataSource": ["dcat:Dataset"],
}

RELATION_TO_ONTOLOGY_PROPERTY: Dict[str, List[str]] = {
    "uses": ["aipo:uses"],
    "trainedOn": ["aipo:trainedOn"],
    "developedBy": ["prov:wasAttributedTo"],
    "evaluatedBy": ["aipo:evaluatedBy"],
    "affects": ["aipo:affects"],
    "exhibitsRisk": ["airo:hasRisk"],
    "violates": ["aipo:violates"],
    "compliesWith": ["aipo:compliesWith"],
    "requiresComplianceWith": ["vair:hasRequirement"],
    "requires": ["vair:hasRequirement"],
    "mitigates": ["aipo:mitigates"],
    "referencesSource": ["dcterms:source"],
    "cites": ["dcterms:references"],
    "contains": ["dcat:contains"],
}

DOMAIN_TO_VAIR: Dict[str, str] = {
    "health": "vair:HealthcareDomain",
    "medical": "vair:HealthcareDomain",
    "justice": "vair:JusticeDomain",
    "criminal": "vair:JusticeDomain",
    "finance": "vair:FinanceDomain",
    "education": "vair:EducationDomain",
    "employment": "vair:EmploymentDomain",
    "transport": "vair:TransportationDomain",
}

PROBLEM_CATEGORY_TO_ONTOLOGY: Dict[str, List[str]] = {
    "bias": ["airo:BiasRisk"],
    "privacy": ["dpv:Privacy"],
    "fairness": ["fmo:Fairness"],
    "transparency": ["aipo:Transparency"],
}

FMO_MAPPINGS: Dict[str, str] = {
    "equalized odds": "fmo:EqualizedOdds",
    "equal opportunity": "fmo:EqualOpportunity",
    "demographic parity": "fmo:DemographicParity",
    "predictive parity": "fmo:PredictiveParity",
    "calibration": "fmo:Calibration",
}

ETHICAL_CONCEPT_TO_AIRO_RISK: Dict[str, str] = {
    "bias": "airo:BiasRisk",
    "racial bias": "airo:RacialBiasRisk",
    "discrimination": "airo:DiscriminationRisk",
    "privacy breach": "dpv:DataBreach",
}

DEMOGRAPHIC_LEXICON: Dict[str, str] = {
    "black": "etd:Race_Black",
    "white": "etd:Race_White",
    "asian": "etd:Race_Asian",
    "women": "etd:Gender_Female",
    "men": "etd:Gender_Male",
}

RELATION_MAPPINGS: Dict[str, str] = {
    "domain": "vair:hasDomain",
    "ethicalConcept": "aipo:hasConcept",
    "demographic": "etd:hasPopulation",
    "fairnessMetric": "fmo:hasMetric",
    "privacyConcept": "dpv:hasConcept",
    "governanceOrRegulation": "vair:hasRequirement",
    "activityOrProcess": "aipo:involves",
    "stakeholder": "prov:wasAttributedTo",
}

# ============================================================
# NEW: ONTOLOGY_TERMS FOR LLM ENTITY EXTRACTOR
# ============================================================
ONTOLOGY_TERMS: Dict[str, str] = {
    "bias": "airo:BiasRisk",
    "racial bias": "airo:RacialBiasRisk",
    "algorithmic bias": "airo:BiasRisk",
    "fairness": "fmo:Fairness",
    "transparency": "aipo:Transparency",
    "privacy": "dpv:Privacy",
    "consent": "dpv:Consent",
    "safety": "airo:SafetyRisk",
    "governance": "aipo:AIGovernance",
    "audit": "aipo:AssessmentProcedure",
    "machine learning": "aipo:MachineLearning",
    "deep learning": "aipo:DeepLearning",
    "dataset": "dcat:Dataset",
    "model": "aipo:AISystem",
    "ai system": "aipo:AISystem",
    "compas": "dcat:Dataset",
    "mimic-cxr": "dcat:Dataset",
    "equalized odds": "fmo:EqualizedOdds",
    "equal opportunity": "fmo:EqualOpportunity",
    "demographic parity": "fmo:DemographicParity",
    "high-risk": "vair:HighRiskAI",
    "gdpr": "dpv:GDPR",
    "personal data": "dpv:PersonalData",
    "criminal justice": "vair:JusticeDomain",
    "healthcare": "vair:HealthcareDomain",
    "employment": "vair:EmploymentDomain",
}
