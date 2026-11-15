"""Regenerate database guides from existing exports without ingestion or LLM calls."""
import argparse
from pathlib import Path
from . import DatabaseGuide
from ..artifacts import ArtifactPolicy


class DocumentationCommand:
    @staticmethod
    def main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument('--database', type=Path)
        source.add_argument('--out-dir', type=Path)
        parser.add_argument('--mode', choices=('dev', 'production'), default='dev')
        args = parser.parse_args(argv)
        databases = [args.database] if args.database else sorted(args.out_dir.glob('*/data.duckdb'))
        if not databases:
            parser.error('No databases found')
        failed = False
        for database in databases:
            try:
                print(DatabaseGuide.write_guide(database), flush=True)
                if args.mode == 'production':
                    ArtifactPolicy.clean_reports(database.parent)
            except Exception as exc:
                failed = True
                print(f'FAILED {database}: {exc}', flush=True)
        if args.out_dir and args.mode == 'production':
            ArtifactPolicy.clean_reports(args.out_dir, split=True)
        return int(failed)

if __name__ == '__main__':
    raise SystemExit(DocumentationCommand.main())
