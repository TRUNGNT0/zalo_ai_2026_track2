"""Output policy. Only explicitly named generated reports may be removed."""
from pathlib import Path

BUSINESS_REPORTS = ('REPORT.md', 'report.json', 'ORGANIZATION.md', 'organization.json',
                    'ORGANIZATION_BUILD.md', 'organization-build.json',
                    'ORGANIZATION_MODEL.md', 'organization-model.json')
SPLIT_REPORTS = ('summary.json', 'REVIEW.md', 'review.json', 'organization-summary.json')


class ArtifactPolicy:
    @staticmethod
    def validate_mode(mode):
        if mode not in ('dev', 'production'):
            raise ValueError(f'Unknown output mode: {mode}')

    @staticmethod
    def clean_reports(directory, *, split=False):
        """Remove known generated reports when switching an output directory to production."""
        for name in SPLIT_REPORTS if split else BUSINESS_REPORTS:
            (Path(directory) / name).unlink(missing_ok=True)
