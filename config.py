"""
config.py
---------
Configuration management for the Semantic Knowledge Graph Pipeline.
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional
try:
    from dotenv import load_dotenv
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - optional dependency in constrained runtimes
    def load_dotenv(*args, **kwargs):
        return False
    def dotenv_values(*args, **kwargs):
        return {}
from ontology_mappings import ONTOLOGY_TERMS

# Canonical project namespaces (single source of truth)
ETD_NS: str = os.getenv("ETD_NS", "https://alfie-project.eu/etd-hub#").strip()
ONTO_NS: str = os.getenv("ONTO_NS", "https://alfie-project.eu/ontology#").strip()

def _load_environment() -> None:
    """
    Load environment variables once.
    - If DOTENV_PATH (or ENV_FILE) is set, use that path.
    - Otherwise use default dotenv discovery.
    """
    env_path = os.getenv("DOTENV_PATH") or os.getenv("ENV_FILE")
    if env_path and str(env_path).strip():
        load_dotenv(dotenv_path=str(env_path).strip(), override=False)
        return

    # Default discovery from current working directory / parents.
    load_dotenv(override=False)

    # If key still missing, try deterministic repo-local .env discovery
    # (supports layouts like key/.env without hardcoding a fixed path).
    if not os.getenv("OPENAI_API_KEY"):
        candidates = sorted(Path(".").glob("**/.env"), key=lambda p: str(p).lower())
        for candidate in candidates:
            load_dotenv(dotenv_path=str(candidate), override=False)
            if os.getenv("OPENAI_API_KEY"):
                break


_load_environment()

class Config:
    """Configuration manager for the pipeline."""
    
    INPUT_JSON: str = "data/ai_ethics_datasets.json"
    OUTPUT_PREFIX: str = "etd_hub_enhanced_kg"
    OUTPUT_DIR: Path = Path("output")
    ETD_NS: str = ETD_NS
    ONTO_NS: str = ONTO_NS
    
    BATCH_SIZE: int = 100
    MAX_RETRIES: int = 3
    DEBUG: bool = False
    
    # BERT NER Settings
    #BERT_NER_MODEL: str = "tner/deberta-v3-large-ontonotes5"
        
    # === Relation Extraction (LLM-only mode) ===
    USE_LLM_EXTRACTION: bool = True
    USE_HYBRID_EXTRACTION: bool = False
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_API_KEY = None
    LLM_MAX_RETRIES: int = 3
    LLM_CONFIDENCE_THRESHOLD: float = 0.45
    LLM_BATCH_SIZE: int = 10  # Process N documents at a time with LLM
    
    USE_LLM = USE_LLM_EXTRACTION
    
    # --- Ontology terms for LLM grounding ---
    ONTOLOGY_TERMS = ONTOLOGY_TERMS     # ← ← ← ADD THIS
    
    # Relation Extraction Settings
    #REBEL_MODEL: str = "Babelscape/rebel-large"
    #USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    #USE_ONNX: bool = False
    
    WIKIDATA_SPARQL_ENDPOINT = os.getenv(
        "WIKIDATA_SPARQL_ENDPOINT",
        "https://query.wikidata.org/sparql",
    )
    
    MAX_TEXT_LENGTH: int = 512
    MAX_ENTITIES_PER_DOC: int = 200
    MAX_RELATIONS_PER_DOC: int = 500
    
    MIN_ENTITY_CONFIDENCE: float = 0.6
    MIN_RELATION_CONFIDENCE: float = 0.65
    MIN_WIKIDATA_CONFIDENCE: float = 0.75
    
    USE_COREFERENCE: bool = False
    USE_REASONING: bool = True
    USE_CONFLICT_DETECTION: bool = True
    USE_ANALYTICS: bool = False
    USE_GRAPHDB_UPLOAD: bool = os.getenv("USE_GRAPHDB_UPLOAD", "false").lower() == "true"
    STRICT_PREDICATE_INVENTORY: bool = os.getenv("STRICT_PREDICATE_INVENTORY", "false").lower() == "true"
    STRICT_RELATION_BACKBONE: bool = os.getenv("STRICT_RELATION_BACKBONE", "false").lower() == "true"
    STRICT_ETD_PROPERTY_REGISTRY: bool = os.getenv("STRICT_ETD_PROPERTY_REGISTRY", "true").lower() == "true"
    FAIL_ON_SHACL_VIOLATIONS: bool = os.getenv("FAIL_ON_SHACL_VIOLATIONS", "false").lower() == "true"
    FAIL_ON_UPLOAD_FAILURE: bool = os.getenv("FAIL_ON_UPLOAD_FAILURE", "false").lower() == "true"
    PRUNE_ISOLATED_ENTITIES: bool = os.getenv("PRUNE_ISOLATED_ENTITIES", "true").lower() == "true"
    MATERIALIZE_RAW_FIELDS: bool = os.getenv("MATERIALIZE_RAW_FIELDS", "true").lower() == "true"
    MATERIALIZE_SENSITIVE_FIELDS: bool = os.getenv("MATERIALIZE_SENSITIVE_FIELDS", "false").lower() == "true"
    MEDIA_BASE_URL: str = os.getenv("MEDIA_BASE_URL", "")
    MAX_UNKNOWN_CHUNK_INDEX_PCT: float = float(os.getenv("MAX_UNKNOWN_CHUNK_INDEX_PCT", "0.02"))
    STRICT_PROVENANCE_GROUNDING: bool = os.getenv("STRICT_PROVENANCE_GROUNDING", "true").lower() == "true"
    MAX_SAME_CHUNK_COLLISIONS: int = int(os.getenv("MAX_SAME_CHUNK_COLLISIONS", "0"))
    MAX_CROSS_CHUNK_COLLISIONS: int = int(os.getenv("MAX_CROSS_CHUNK_COLLISIONS", "0"))
    
    USE_ONTOLOGY_MAPPING = True
    USE_WIKIDATA_LINKING: bool = os.getenv("USE_WIKIDATA_LINKING", "true").lower() == "true"

    # === GraphDB Integration ===
    GRAPHDB_SERVER_URL: str = os.getenv("GRAPHDB_SERVER_URL", "")
    GRAPHDB_REPO_NAME: str = os.getenv("GRAPHDB_REPO_NAME", "")
    GRAPHDB_USERNAME: str = os.getenv("GRAPHDB_USERNAME", "")
    GRAPHDB_PASSWORD: str = os.getenv("GRAPHDB_PASSWORD", "")
    GRAPHDB_USE_AUTH: bool = os.getenv("GRAPHDB_USE_AUTH", "true").lower() == "true"

    
    NAMESPACES: Dict[str, str] = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "dc": "http://purl.org/dc/elements/1.1/",
        "dcterms": "http://purl.org/dc/terms/",
        "foaf": "http://xmlns.com/foaf/0.1/",
        "org": "http://www.w3.org/ns/org#",
        "sioc": "http://rdfs.org/sioc/ns#",
        "prov": "http://www.w3.org/ns/prov#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "dcat": "http://www.w3.org/ns/dcat#",
        "schema": "http://schema.org/",
        "aipo": "https://w3id.org/aipo#",
        "relaieo": "http://www.ontology.audit4sg.org/RelAIEO#",
        "vair": "https://w3id.org/vair#",
        "airo": "https://w3id.org/airo#",
        "hudock": "http://www.semanticweb.org/rhudock/ontologies/2023/6/ai-risk-compliance-ontology#",
        "fmo": "http://purl.org/fairness-metrics-ontology/",
        "dpv": "https://w3id.org/dpv#",
        "bibo": "http://purl.org/ontology/bibo/",
        "odrl": "http://www.w3.org/ns/odrl/2/",
        "oa": "http://www.w3.org/ns/oa#",
        "cnt": "http://www.w3.org/2011/content#",
        "etd": ETD_NS,
        "onto": ONTO_NS,
    }
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> None:
        """Load configuration from JSON file."""
        if not config_path.exists():
            return
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        for key, value in config_data.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
            else:
                print(f"⚠️ Unknown config key ignored: {key}")

    @classmethod
    def load_llm_api_key(cls):
        """Load OpenAI API key securely from .env or environment variable."""
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv("LLM_API_KEY")
        if isinstance(api_key, str):
            api_key = api_key.strip().strip('"').strip("'")
            if not api_key:
                api_key = None

        # Last-resort deterministic lookup from discovered .env files
        # if process environment did not get populated.
        if not api_key:
            candidates = sorted(Path(".").glob("**/.env"), key=lambda p: str(p).lower())
            for candidate in candidates:
                values = dotenv_values(candidate)
                val = values.get("OPENAI_API_KEY") or values.get("LLM_API_KEY")
                if isinstance(val, str):
                    val = val.strip().strip('"').strip("'")
                if val:
                    api_key = val
                    break

        if not api_key:
            try:
                from google.colab import userdata  # type: ignore
                api_key = userdata.get('OPENAI_API_KEY')
            except Exception:
                api_key = None

        if not api_key and cls.USE_LLM_EXTRACTION:
            print("⚠️ Warning: LLM extraction enabled but no API key found!")
            print("   Set OPENAI_API_KEY in your .env file or environment variables.")

        cls.LLM_API_KEY = api_key
        return api_key


    @classmethod
    def save_to_file(cls, config_path: Path) -> None:
        """Save current configuration to JSON file."""
        config_data = {
            key.lower(): getattr(cls, key)
            for key in dir(cls)
            if key.isupper() and not key.startswith('_')
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def get_output_paths(cls) -> Dict[str, Path]:
        """Get all output file paths."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        return {
            'annotations': cls.OUTPUT_DIR / f"{cls.OUTPUT_PREFIX}_annotations.json",
            'metrics': cls.OUTPUT_DIR / f"{cls.OUTPUT_PREFIX}_quality_metrics.json",
            'conflicts': cls.OUTPUT_DIR / f"{cls.OUTPUT_PREFIX}_conflicts.json",
            'analytics': cls.OUTPUT_DIR / f"{cls.OUTPUT_PREFIX}_analytics.json",
            'ttl': cls.OUTPUT_DIR / f"{cls.OUTPUT_PREFIX}.ttl",
            'rdf': cls.OUTPUT_DIR / f"{cls.OUTPUT_PREFIX}.rdf",
            'jsonld': cls.OUTPUT_DIR / f"{cls.OUTPUT_PREFIX}.jsonld",
            'owl': cls.OUTPUT_DIR / f"{cls.OUTPUT_PREFIX}.owl",
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        errors = []
        
        if not Path(cls.INPUT_JSON).exists():
            errors.append(f"Input file not found: {cls.INPUT_JSON}")
        
        if cls.BATCH_SIZE < 1:
            errors.append("BATCH_SIZE must be at least 1")
        
        if not 0 <= cls.MIN_ENTITY_CONFIDENCE <= 1:
            errors.append("MIN_ENTITY_CONFIDENCE must be between 0 and 1")
        
        if not 0 <= cls.MIN_RELATION_CONFIDENCE <= 1:
            errors.append("MIN_RELATION_CONFIDENCE must be between 0 and 1")
        
        if cls.USE_LLM_EXTRACTION is not True:
            errors.append("LLM-only enforcement violation: USE_LLM_EXTRACTION must be true")
        if cls.USE_HYBRID_EXTRACTION is not False:
            errors.append("LLM-only enforcement violation: USE_HYBRID_EXTRACTION must be false")

        api_key = cls.load_llm_api_key()
        if not api_key:
            errors.append("LLM-only extraction requires OPENAI_API_KEY")

        if cls.USE_WIKIDATA_LINKING and not cls.WIKIDATA_SPARQL_ENDPOINT:
            errors.append("USE_WIKIDATA_LINKING=true but WIKIDATA_SPARQL_ENDPOINT is missing")

        if cls.USE_GRAPHDB_UPLOAD:
            required_graphdb = {
                "GRAPHDB_SERVER_URL": cls.GRAPHDB_SERVER_URL,
                "GRAPHDB_REPO_NAME": cls.GRAPHDB_REPO_NAME,
            }
            if cls.GRAPHDB_USE_AUTH:
                required_graphdb["GRAPHDB_USERNAME"] = cls.GRAPHDB_USERNAME
                required_graphdb["GRAPHDB_PASSWORD"] = cls.GRAPHDB_PASSWORD
            missing = [k for k, v in required_graphdb.items() if not str(v).strip()]
            if missing:
                errors.append(
                    "USE_GRAPHDB_UPLOAD=true but missing required GraphDB env vars: "
                    + ", ".join(missing)
                )
        
        if errors:
            for error in errors:
                print(f"❌ Configuration Error: {error}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        print("\n" + "="*70)
        print("  Configuration Settings")
        print("="*70)
        print(f"  Input: {cls.INPUT_JSON}")
        print(f"  Output Directory: {cls.OUTPUT_DIR}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")

        # Models actually used
        print(f"  LLM Entity Extraction Model: {cls.LLM_MODEL}")
        print(f"  LLM Relation Extraction Model: {cls.LLM_MODEL}")

        print(f"  Wikidata Linking: {cls.USE_WIKIDATA_LINKING}")
        print(f"  GraphDB Upload: {cls.USE_GRAPHDB_UPLOAD}")
        print(f"  Ontology Mapping: {cls.USE_ONTOLOGY_MAPPING}")
        print(f"  Reasoning: {cls.USE_REASONING}")
        print(f"  Analytics: {cls.USE_ANALYTICS}")
        print(f"  Prune Isolated Entities: {cls.PRUNE_ISOLATED_ENTITIES}")

        print("="*70 + "\n")
