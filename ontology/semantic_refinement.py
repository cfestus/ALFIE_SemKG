from __future__ import annotations
import re
from typing import Dict, Optional
from rdflib import URIRef
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class SemanticRefiner:
    """
    Fix 6.3 — LangChain-enabled semantic refinement.

    Includes:
      - label normalisation
      - rule-based fallback (Fix 6.2)
      - LangChain-based refinement when enabled
      - Consistent return format { "final_type": str | None }
    """

    def __init__(self, use_llm: bool = True, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        
        self.use_llm = use_llm
        self._cache: Dict[str, Optional[str]] = {}
        
        if self.use_llm:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
            # Structured output
            response_schemas = [
                ResponseSchema(
                    name="final_type",
                    type="string",
                    description="The best ontology-aligned class for this entity."
                )
            ]

            self.parser = StructuredOutputParser.from_response_schemas(response_schemas)
            self.format_instructions = self.parser.get_format_instructions()

            self.prompt = ChatPromptTemplate.from_template("""
You are an expert in AI governance ontologies (OECD, EU AI Act, NIST RMF, DPV, VAIR).

Given this label: "{label}"

Infer ONE best entity type from the following ontology concepts:
- Dataset
- Model
- AISystem
- PopulationGroup
- ProtectedGroup
- Risk
- EthicalPrinciple
- Process
- Evaluation
- Action
- Stakeholder
- Organisation

Return ONLY JSON:
{format_instructions}
""")

    # -------------------------
    # Utility functions
    # -------------------------
    def _normalise(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip().lower()


    def _rule_based_fallback(self, normalised: str) -> str:
        """
        Deterministic fallback when LLM refinement is unavailable or fails.

        Production-grade behaviour:
        - Prefer the most specific class when possible (e.g., Bias vs Harm vs Risk).
        - Return only values that exist in the ALLOWED set in refine().
        """
        s = (normalised or "").strip().lower()

        # 1) Dataset / data artefacts
        if any(w in s for w in ("dataset", "data set", "csv", "records", "corpus", "benchmark")):
            return "Dataset"

        # 2) Standards / governance
        if any(w in s for w in ("iso", "iec", "nist", "ieee", "standard", "regulation", "policy", "guideline", "framework")):
            return "Standard"

        # 3) Processes / evaluation / actions
        if any(w in s for w in ("audit", "assessment", "evaluation", "testing", "validation", "verification", "monitoring", "measurement")):
            return "Evaluation"
        if any(w in s for w in ("mitigation", "remediation", "intervention", "enforcement", "governance", "oversight", "control")):
            return "Action"
        if any(w in s for w in ("process", "procedure", "workflow", "pipeline", "review", "documentation")):
            return "Process"

        # 4) Ethics principles (normative ideals)
        if any(w in s for w in ("transparency", "explainability", "accountability", "privacy", "fairness", "justice", "beneficence", "non-maleficence")):
            return "EthicalPrinciple"

        # 5) Risk family (prefer specific)
        # Bias first (more specific than generic Risk)
        if any(w in s for w in ("bias", "biased", "discrimination", "discriminatory", "unfair", "fairness")):
            return "Risk"  # keep as Risk if your ALLOWED set doesn't include Bias
            # If you add "Bias" to ALLOWED later, change to: return "Bias"

        # Harm next (more specific than generic Risk)
        if any(w in s for w in ("harm", "unsafe", "injury", "damage", "abuse", "violence", "toxicity", "harassment")):
            return "Risk"  # keep as Risk if your ALLOWED set doesn't include Harm
            # If you add "Harm" to ALLOWED later, change to: return "Harm"

        # Generic risk last
        if any(w in s for w in ("risk", "threat", "vulnerability", "attack", "misuse", "failure", "error", "liability")):
            return "Risk"

        # 6) Systems / models
        if any(w in s for w in ("model", "algorithm", "classifier", "llm", "neural", "system", "ai system", "ml system")):
            return "AISystem"

        # 7) Stakeholders / organisations / groups
        if any(w in s for w in ("company", "organisation", "organization", "firm", "agency", "authority", "university", "institution", "ngo")):
            return "Organisation"
        if any(w in s for w in ("user", "developer", "operator", "provider", "regulator", "auditor", "stakeholder")):
            return "Stakeholder"
        if any(w in s for w in ("population", "community", "citizens", "patients", "students", "workers", "children", "women", "men")):
            return "PopulationGroup"
        if any(w in s for w in ("protected", "minority", "disabled", "disability", "race", "ethnicity", "religion", "gender")):
            return "ProtectedGroup"

        # Default safe type
        return "Risk"

    # -------------------------
    # PUBLIC API (required by RDFSerializer)
    # -------------------------
    def refine(self, uri: URIRef, label: str) -> dict:
        """
        Required entrypoint used by rdf_serializer.

        Returns:
            { "final_type": <string or None> }
        """
        if not isinstance(label, str):
            return {"final_type": None}

        normalised = self._normalise(label)

        if normalised in self._cache:
            return {"final_type": self._cache[normalised]}

        # 1 — Try LLM refinement (if enabled)
        if self.use_llm:
            try:
                formatted = self.prompt.format_messages(
                    label=label,
                    format_instructions=self.format_instructions
                )
           
                response = self.llm.invoke(formatted)
                parsed = self.parser.parse(getattr(response, "content", str(response)))

                ALLOWED = {
                    "Dataset", "Model", "AISystem", "PopulationGroup", "ProtectedGroup",
                    "Risk", "EthicalPrinciple", "Process", "Evaluation", "Action",
                    "Stakeholder", "Organisation", "Bias", "Harm",
                }

                ft = parsed.get("final_type")
                if ft in ALLOWED:
                    self._cache[normalised] = ft
                    return {"final_type": ft}            
            
            except Exception:
                # soft fail — do not break the pipeline
                pass

        # 2 — Rule-based fallback
        ft2 = self._rule_based_fallback(normalised)
        self._cache[normalised] = ft2
        return {"final_type": ft2}
