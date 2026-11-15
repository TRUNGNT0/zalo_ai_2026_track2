"""Compatibility entry point; prefer app.ingestion.cli.ingest."""

from datamate.ingestion.cli.ingest import IngestionPipeline

if __name__ == "__main__":
    raise SystemExit(IngestionPipeline.main())
