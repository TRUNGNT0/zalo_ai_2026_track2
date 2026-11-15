"""CLI entry point; implementation lives in app.manifests.commands."""

from datamate.manifests.commands import AuditManifestsCommand

if __name__ == "__main__":
    raise SystemExit(AuditManifestsCommand.main())
