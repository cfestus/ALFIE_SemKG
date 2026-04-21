"""
validation/shacl_validator.py
-----------------------------
SHACL validation for the ETD-Hub Knowledge Graph.

Uses pySHACL to validate an rdflib.Graph against ontology/shacl_shapes.ttl.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

from rdflib import Graph
from pyshacl import validate
from rdflib.namespace import SH
from rdflib.term import URIRef

class SHACLValidator:
    """Run SHACL validation over an RDF graph."""

    def __init__(
        self,
        shapes_path: Union[Path, str] = Path("ontology/shacl_shapes.ttl"),
        report_path: Optional[Union[Path, str]] = None,
        report_graph_path: Optional[Union[Path, str]] = None,
        output_dir: Optional[Union[Path, str]] = None,
        inference: Optional[str] = None,  # None, "rdfs", "owlrl"
        allow_infos: bool = True,
        allow_warnings: bool = True,
        meta_shacl: bool = True,
        advanced: bool = True,
        debug: bool = False,
        ):
        
        self.shapes_path = Path(shapes_path)
        out_dir = Path(output_dir) if output_dir is not None else Path("output")
        self.report_path = Path(report_path) if report_path is not None else out_dir / "shacl_report.txt"
        self.report_graph_path = Path(report_graph_path) if report_graph_path else out_dir / "shacl_report.ttl"
        self.inference = inference
        self.allow_infos = allow_infos
        self.allow_warnings = allow_warnings
        self.meta_shacl = meta_shacl
        self.advanced = advanced
        self.debug = debug

    def validate_graph(self, data_graph: Graph) -> Dict[str, Any]:
        """
        Validate the given RDF graph against the SHACL shapes.

        Returns a dict:
            {
              "conforms": bool,
              "results_text": str,
              "results_graph": rdflib.Graph
            }
        """
        if not self.shapes_path.exists():
            raise FileNotFoundError(f"SHACL shapes file not found: {self.shapes_path}")

        shacl_graph = Graph()
        try:
            shacl_graph.parse(self.shapes_path.as_posix(), format="turtle")
        except Exception as e:
            raise ValueError(f"Failed to parse SHACL shapes TTL at {self.shapes_path}: {e}") from e

        conforms, results_graph, results_text = validate(
            data_graph=data_graph,
            shacl_graph=shacl_graph,
            inference=self.inference,          # was "rdfs"
            abort_on_first=False,
            allow_infos=self.allow_infos,
            allow_warnings=self.allow_warnings,
            meta_shacl=self.meta_shacl,
            advanced=self.advanced,
            debug=self.debug,
            )

        # Ensure output directory exists and persist report text
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(results_text or "", encoding="utf-8")

        # Optionally persist the results graph (useful for SPARQL over SHACL report)
        if self.report_graph_path:
            self.report_graph_path.parent.mkdir(parents=True, exist_ok=True)
            results_graph.serialize(destination=self.report_graph_path.as_posix(), format="turtle")

        def _count(sev: URIRef) -> int:
            return sum(1 for _ in results_graph.triples((None, SH.resultSeverity, sev)))

        counts = {
            "violations": _count(SH.Violation),
            "warnings": _count(SH.Warning),
            "infos": _count(SH.Info),
        }

        return {
            "success": True,
            "conforms": bool(conforms),
            "results_text": results_text,
            "results_graph": results_graph,
            "counts": counts,
            "errors": [] if bool(conforms) else [results_text or "SHACL validation failed"],
            "report_path": str(self.report_path),
            "report_graph_path": str(self.report_graph_path) if self.report_graph_path else None,
        }
