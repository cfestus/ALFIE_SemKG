# semantic_typing.py
import re
from typing import Optional, Set
from rdflib.namespace import RDF

class SemanticTyper:
    def __init__(self, graph, etd_ns, allowed_classes: Optional[Set[str]] = None):
        self.graph = graph
        self.etd = etd_ns
        self.allowed_classes = allowed_classes  # None means "no filtering"
    
    def _add_type(self, uri, class_name: str) -> bool:
        """
        Only mint rdf:type etd:<class_name> if class_name is allowed/declared.
        Returns True if added, False if blocked.
        """
        if self.allowed_classes and class_name not in self.allowed_classes:
            print(f"⚠️ [SemanticTyper] Type blocked by allowed_classes: {class_name} for {uri}")
            return False
        self.graph.add((uri, RDF.type, self.etd[class_name]))
        return True

    # --- Core public method -----------------------------
    def classify(self, uri, label_text):
        """
        Apply all semantic typing heuristics (Fix 5 + Fix 5.5).
        label_text must be lowercase string.
        """

        # Normalize once
        label_text = label_text.lower().strip()

        # ------------------------------------------------------------
        # 1. Bias / Harm / Impact
        # ------------------------------------------------------------
        if any(w in label_text for w in [
            "bias", "harm", "disparate", "inequ", "inequal",
            "unfair", "risk", "discrimination", "justice",
            "impact"
        ]):
            self._add_type(uri, "Bias")
            self._add_type(uri, "Risk")
            self._add_type(uri, "Harm")
            self._add_type(uri, "Impact")
            #no return  # stop here, primary class detected

        # ------------------------------------------------------------
        # 2. Dataset / DataElement / DataIssue
        # ------------------------------------------------------------
        if any(w in label_text for w in [
            "dataset", "data set", "records", "csv",
            "311", "census", "demographics"
        ]):
            if any(kw in label_text for kw in ["missing", "skew", "drift", "imbalance"]):
                self._add_type(uri, "DataIssue")
            elif any(kw in label_text for kw in ["history", "attribute", "field"]):
                self._add_type(uri, "DataElement")
            else:
                self._add_type(uri, "Dataset")
            # no return: allow additional typing rules to apply

        # ------------------------------------------------------------
        # 3. System / Component (precision-safe token matching)
        # ------------------------------------------------------------

        system_terms = [
            "model", "algorithm", "system",
            "classifier", "framework", "pipeline"
        ]

        # Word-boundary matching for short tokens
        short_tokens = ["ai", "ml"]

        # Match longer substrings normally
        matched = any(term in label_text for term in system_terms)

        # Match short tokens safely using word boundaries
        for token in short_tokens:
            if re.search(rf"\b{re.escape(token)}\b", label_text):
                matched = True
                break

        if matched:
            self._add_type(uri, "AISystem")
            self._add_type(uri, "SystemComponent")

        # ------------------------------------------------------------
        # 4. Ethical Principles
        # ------------------------------------------------------------
        ethical_terms = [
            "transparency", "explainability", "interpretability",
            "fairness", "equity", "equal", "non-discrimination",
            "accountability", "responsibility", "liability",
            "privacy", "confidentiality", "data protection",
            "security", "robustness", "reliability", "safety",
            "trust", "trustworthy", "human oversight",
            "autonomy", "dignity", "human rights",
            "justice", "inclusion", "inclusiveness",
            "well-being", "beneficence", "non-maleficence",
            "sustainability", "environmental", "green",
            "bias mitigation", "harm mitigation", "risk mitigation",
            "ethical", "ethics", "principle", "value"
        ]

        if any(term in label_text for term in ethical_terms):
            self._add_type(uri, "EthicalPrinciple")
            # no return: allow additional typing rules to apply

        # ------------------------------------------------------------
        # 5. Standards
        # ------------------------------------------------------------
        if any(w in label_text for w in [
            "nist", "iso", "ieee", "unesco", "standard", "oecd"
        ]):
            self._add_type(uri, "Standard")
            # no return: allow additional typing rules to apply

        # ------------------------------------------------------------
        # 6. Processes / Evaluation / Action
        # ------------------------------------------------------------
        if any(w in label_text for w in [
            "audit", "assessment", "evaluation", "review",
            "analysis", "process"
        ]):
            self._add_type(uri, "Process")
            self._add_type(uri, "Evaluation")
            self._add_type(uri, "Action")
            # no return: allow additional typing rules to apply

        # ------------------------------------------------------------
        # 7. Stakeholder logic (Fix 5.5 – final)
        # ------------------------------------------------------------
        self._apply_stakeholder_rules(uri, label_text)

    # --- Stakeholder-specific logic (Fix 5.5) -----------
    def _apply_stakeholder_rules(self, uri, label_text):
        """
        FIX 5.5 — Final Stakeholder Expansion & Future-Proofing.
        Strict classification:
        - Only classify when the entire label equals a stakeholder term.
        - Prevents false positives like "racial injustice against black people".
        """

        text = label_text.strip()

        # ---------------------------------------------------------
        # General population groups (non-protected)
        # ---------------------------------------------------------
        population_groups = [
            "group", "community", "population",
            "students", "learners", "children",
            "boys", "girls",
            "immigrants", "migrants",
            "patients", "borrowers",
            "defendants", "applicants",
            "customers", "users", "consumers",
            "minority", "minorities"
        ]

        # ---------------------------------------------------------
        # Protected demographic groups
        # ---------------------------------------------------------
        protected_terms = [
            # Race & ethnicity
            "black", "blacks",
            "white", "whites",
            "asian", "asians",
            "latino", "latinos",
            "hispanic", "hispanics",
            "indigenous", "indigenous peoples",

            # Gender & orientation
            "women", "men",
            "lgbt", "lgbtq", "lgbtq+",
            "queer",
            "trans", "transgender",

            # Disability
            "disabled", "persons with disabilities",

            # Religion
            "muslim", "christian", "jewish"
        ]

        # Case 1: Label equals general population group
        if text in population_groups:
            self._add_type(uri, "Stakeholder")
            self._add_type(uri, "PopulationGroup")
            return

        # Case 2: Label equals protected group
        if text in protected_terms:
            self._add_type(uri, "Stakeholder")
            self._add_type(uri, "PopulationGroup")
            self._add_type(uri, "ProtectedGroup")
            return

        # Case 3: label contains but does NOT equal → ignore (avoid false positives)
        return
