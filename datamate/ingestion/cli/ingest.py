"""CLI entry point for loading business data into DuckDB and building a catalog."""

from datamate.ingestion.processing.pipeline import IngestionPipeline

if __name__ == "__main__":
    raise SystemExit(IngestionPipeline.main())
