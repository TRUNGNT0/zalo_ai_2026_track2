"""CLI entry point; implementation lives in app.manifests.commands."""

from datamate.manifests.commands import BuildManifestsCommand

if __name__ == "__main__":
    raise SystemExit(BuildManifestsCommand.main())
