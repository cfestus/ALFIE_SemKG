#!/usr/bin/env python3
"""
main.py
-------
Main entry point for the Semantic Knowledge Graph Pipeline.

Usage:
    python main.py [--config config.json] [--input data.json] [--output output_dir]
"""
import argparse
import sys
import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

class SHACLValidationError(RuntimeError):
    pass


class GraphDBUploadError(RuntimeError):
    pass


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced Semantic Knowledge Graph Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory path'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Disable reasoning'
    )
    parser.add_argument(
        '--no-wikidata',
        action='store_true',
        help='Disable Wikidata linking'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM-only extraction (requires OpenAI API key)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
        default='gpt-4o-mini',
        help='LLM model to use for extraction'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    print("=" * 70)
    print("  Semantic Knowledge Graph Pipeline")
    print("  Version 5.1")
    print("=" * 70)

    args = parse_arguments()

    # Lazy imports: keep --help usable even when optional runtime deps are missing.
    from pipeline import EnhancedSemanticPipeline
    from serializers.rdf_serializer import RDFSerializer
    from validation.shacl_validator import SHACLValidator
    from utils.graphdb_client import GraphDBClient
    from utils.helpers import (
        create_output_directory,
        save_json,
        load_json,
        validate_input_json,
    )

    # ---------------------------------------------------------
    # Load / override configuration
    # ---------------------------------------------------------
    if args.config:
        Config.load_from_file(Path(args.config))

    if args.input:
        Config.INPUT_JSON = args.input
    if args.output:
        Config.OUTPUT_DIR = Path(args.output)
    if args.debug:
        Config.DEBUG = True
    if args.no_reasoning:
        Config.USE_REASONING = False
    if args.no_wikidata:
        Config.USE_WIKIDATA_LINKING = False
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size

    # Extraction mode: we now use LLM-only entity extraction
    # The --use-llm flag is kept for future extension, but we default to LLM anyway.
    Config.USE_LLM_EXTRACTION = True
    Config.USE_HYBRID_EXTRACTION = False
    if hasattr(Config, "USE_BERT_EXTRACTION"):
        Config.USE_BERT_EXTRACTION = False
    print("LLM-only extraction enforced")
    if not Config.USE_LLM_EXTRACTION or Config.USE_HYBRID_EXTRACTION:
        raise RuntimeError(
            "LLM-only extraction enforcement failed: USE_LLM_EXTRACTION must be true and "
            "USE_HYBRID_EXTRACTION must be false."
        )

    # ---------------------------------------------------------
    # Validate configuration
    # ---------------------------------------------------------
    if not Config.validate():
        print("\n❌ Configuration validation failed")
        sys.exit(1)

    Config.print_config()

    # ---------------------------------------------------------
    # IO setup
    # ---------------------------------------------------------
    create_output_directory(Config.OUTPUT_DIR)

    print(f"\n📂 Loading data from {Config.INPUT_JSON}.")
    input_path = Path(Config.INPUT_JSON)

    if not input_path.exists():
        print(f"❌ File not found: {Config.INPUT_JSON}")
        sys.exit(1)

    try:
        data = load_json(input_path)
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        sys.exit(1)

    try:
        validate_input_json(data)
    except Exception as e:
        print(f"❌ Input JSON validation failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Initialise pipeline
    # ---------------------------------------------------------
    try:
        pipeline = EnhancedSemanticPipeline(Config)
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        logger.exception("Pipeline initialization failed")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  Processing Documents")
    print("=" * 70)

    # ---------------------------------------------------------
    # Process documents
    # ---------------------------------------------------------
    try:
        annotations = pipeline.process_batch(data)
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
        logger.exception("Document processing failed")
        sys.exit(1)

    # ---------------------------------------------------------
    # Normalise annotations to a dict for downstream stability
    # ---------------------------------------------------------
    if isinstance(annotations, dict):
        annotations_dict = annotations
    elif hasattr(annotations, "to_dict"):
        annotations_dict = annotations.to_dict()
    else:
        raise TypeError(f"Unsupported annotations type: {type(annotations)}")

    # ---------------------------------------------------------
    # Reasoning (before RDF build)
    # ---------------------------------------------------------
    inferred_relations = pipeline.apply_reasoning(annotations_dict)
    if inferred_relations:
        annotations_dict.setdefault("metadata", {})
        annotations_dict["metadata"]["inferred_relations"] = [r.to_dict() for r in inferred_relations]

    # ---------------------------------------------------------
    # Conflict detection
    # ---------------------------------------------------------
    try:
        conflicts = pipeline.detect_conflicts(annotations_dict)
    except Exception as e:
        print(f"⚠️  Conflict detection failed: {e}")
        logger.exception("Conflict detection failed")
        conflicts = []

    # ---------------------------------------------------------
    # Save outputs and serialize RDF
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Saving Outputs")
    print("=" * 70)

    output_paths = Config.get_output_paths()

    try:
        provenance_fill_fn = getattr(pipeline, "fill_missing_relation_provenance", None)
        if callable(provenance_fill_fn):
            provenance_fill = provenance_fill_fn(annotations_dict)
            print(
                "\nProvenance fill pass: "
                f"filled_span_map={provenance_fill.get('filled_from_span_map', 0)}, "
                f"filled_single_chunk={provenance_fill.get('filled_from_single_chunk_discourse', 0)}, "
                f"cross_chunk={provenance_fill.get('cross_chunk_evidence', 0)}, "
                f"unknown={provenance_fill.get('unknown_chunk_index_count', 0)}"
            )
        print("\n🧱 Building RDF graph (single pass).")
        serializer = RDFSerializer(
            annotations_dict,
            pipeline.global_registry,
            Config.NAMESPACES
        )
        g = serializer.graph

        print("\n🔎 Running SHACL validation.")
        validator = SHACLValidator(
            output_dir=Config.OUTPUT_DIR,
        )
        shacl_result = validator.validate_graph(g)
        if not shacl_result.get("conforms", False):
            msg = (
                "SHACL validation failed. "
                f"Report: {shacl_result.get('report_path')}"
            )
            if Config.FAIL_ON_SHACL_VIOLATIONS:
                raise SHACLValidationError(msg)
            print(f"⚠️ {msg}")
        else:
            print("✅ SHACL validation passed.")

        if Config.USE_GRAPHDB_UPLOAD:
            graphdb = GraphDBClient()
            conn = graphdb.check_connection()
            if not conn.get("success"):
                msg = f"GraphDB connection failed: {conn.get('error')}"
                if Config.FAIL_ON_UPLOAD_FAILURE:
                    raise GraphDBUploadError(msg)
                print(f"⚠️ {msg}")
            else:
                up = graphdb.upload_rdf(g)
                if not up.get("success"):
                    msg = f"GraphDB upload failed: {up.get('error')}"
                    if Config.FAIL_ON_UPLOAD_FAILURE:
                        raise GraphDBUploadError(msg)
                    print(f"⚠️ {msg}")
                else:
                    print("✅ GraphDB upload completed.")

        print("\n💾 Serializing to RDF formats.")
        serializer.serialize_to_turtle(output_paths['ttl'])
        serializer.serialize_to_rdf(output_paths['rdf'])
        serializer.serialize_to_jsonld(output_paths['jsonld'])
        serializer.serialize_to_owl(output_paths['owl'])
        print("✓ RDF serialization complete")

        print("\n💾 Saving annotations.")
        save_json(annotations_dict, output_paths["annotations"])

        # ---------------------------------------------------------
        # Quality metrics (post RDF/SHACL/upload)
        # ---------------------------------------------------------
        quality_metrics = pipeline.compute_quality_metrics(annotations_dict)

        print("\n💾 Saving quality metrics.")
        save_json(quality_metrics, output_paths['metrics'])

        if conflicts:
            print("\n💾 Saving conflicts.")
            save_json({'conflicts': conflicts}, output_paths['conflicts'])

    except (SHACLValidationError, GraphDBUploadError):
        logger.exception("Fatal pipeline validation/upload failure")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error saving outputs: {e}")
        logger.exception("Output persistence failed")
        sys.exit(1)


if __name__ == '__main__':
    main()


