"""Compatibility entry point; prefer app.ingestion.cli.build_manifests."""

from datamate.ingestion.cli.build_manifests import BuildManifestsCommand

if __name__ == "__main__":
    raise SystemExit(BuildManifestsCommand.main())
