"""Transactional ordering for dependent stage-2 and stage-3 views."""
from ..raw import RawData
from .build import OrganizationBuilder
from .model import OrganizationModelBuilder

class OrganizationWorkflow:
    @staticmethod
    def organize_connection(con, analysis):
        # Models can reference merged stage-2 views. Remove our models first, rebuild
        # both layers, then check dependencies (including user views) once complete.
        OrganizationModelBuilder.cleanup_models(con)
        analysis['stage2'] = OrganizationBuilder.build_connection(con, analysis, validate_catalog=False)
        analysis['stage3'] = OrganizationModelBuilder.build_models(con, analysis)
        for view in OrganizationBuilder.catalog(con):
            con.execute('SELECT * FROM semantic.' + RawData.quote_identifier(view) + ' LIMIT 0')
