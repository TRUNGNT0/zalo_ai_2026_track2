"""Only fully checked relationships may authorize generated joins."""

from typing import Any, Iterable

from datamate.ingestion.sources.adapters.common import TabularLayout

class RelationshipService:
    """Build, verify and expose relationships safe for catalog joins."""

    @staticmethod
    def from_declarations(
        declarations: Iterable[dict[str, Any]],
        tables: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        tables_by_source = {
            (table["source_files"][0], table["source_objects"][0]): table
            for table in tables
        }
        relationships: list[dict[str, Any]] = []
        unresolved: list[dict[str, str]] = []

        for declaration in declarations:
            source_key = (declaration["source_file"], declaration["from_table"])
            target_key = (declaration["source_file"], declaration["to_table"])
            source_table = tables_by_source.get(source_key)
            target_table = tables_by_source.get(target_key)
            if source_table is None or target_table is None:
                unresolved.append(
                    {
                        "topic": (
                            f"relationship:{declaration['source_file']}:"
                            f"{declaration['from_table']}:{declaration['to_table']}"
                        ),
                        "reason": "A declared SQLite foreign key references a table "
                        "that was not ingested.",
                        "needed_evidence": "Include the referenced SQLite table in the "
                        "ingestion source.",
                    }
                )
                continue

            source_columns = RelationshipService._physical_columns(
                source_table, declaration["from_columns"]
            )
            target_columns = RelationshipService._physical_columns(
                target_table, declaration["to_columns"]
            )
            if source_columns is None or target_columns is None:
                unresolved.append(
                    {
                        "topic": (
                            f"relationship:{declaration['source_file']}:"
                            f"{declaration['from_table']}:{declaration['to_table']}"
                        ),
                        "reason": "A declared SQLite foreign key column could not be "
                        "mapped to an ingested DuckDB column.",
                        "needed_evidence": "Review the source and target key columns.",
                    }
                )
                continue

            relationships.append(
                {
                    "relationship_id": (
                        f"sqlite_fk:{declaration['source_file']}:"
                        f"{declaration['from_table']}:{declaration['constraint_id']}"
                    ),
                    "from_table": source_table["physical_name"],
                    "from_columns": source_columns,
                    "to_table": target_table["physical_name"],
                    "to_columns": target_columns,
                    "status": "declared_not_data_verified",
                    "source": "sqlite_foreign_key",
                    "source_file": declaration["source_file"],
                }
            )
        return relationships, unresolved

    @staticmethod
    def _physical_columns(
        table: dict[str, Any],
        source_names: list[str],
    ) -> list[str] | None:
        source_to_physical = {
            column["source_name"]: column["physical_name"]
            for column in table["columns"]
        }
        if any(source_name not in source_to_physical for source_name in source_names):
            return None
        return [source_to_physical[source_name] for source_name in source_names]

    @staticmethod
    def from_manifest(connection, declarations, tables):
        """Check complete ingested data; never promote inferred joins to trusted joins."""
        from datamate.ingestion.sources.adapters.common import TabularLayout
        from datamate.manifests.validation import ManifestValidator

        by_source = {(t['source_files'][0], t['source_objects'][0]): t for t in tables}
        relationships, issues = [], []
        for declaration in declarations:
            endpoints = []
            for endpoint in (declaration['from'], declaration['to']):
                table = by_source[(endpoint['source_file'], endpoint['source_object'])]
                columns = RelationshipService._physical_columns(table, endpoint['columns'])
                endpoints.append((table['physical_name'], columns))
            (source, source_cols), (target, target_cols) = endpoints
            source_nulls, source_duplicates = RelationshipService._key_counts(connection, source, source_cols)
            target_nulls, target_duplicates = RelationshipService._key_counts(connection, target, target_cols)
            condition = ' AND '.join(f's.{TabularLayout.quote_identifier(a)} = t.{TabularLayout.quote_identifier(b)}' for a, b in zip(source_cols, target_cols))
            nonnull = ' AND '.join(f's.{TabularLayout.quote_identifier(c)} IS NOT NULL' for c in source_cols)
            orphans = connection.execute(f'SELECT COUNT(*) FROM {TabularLayout.quote_identifier(source)} s WHERE {nonnull} AND NOT EXISTS (SELECT 1 FROM {TabularLayout.quote_identifier(target)} t WHERE {condition})').fetchone()[0]
            valid = not (target_duplicates or target_nulls or orphans) and (
                declaration['cardinality'] != 'one_to_one' or not source_duplicates)
            status = ('candidate' if declaration['source'] == 'inferred_candidate' else
                      'verified' if valid else 'failed_validation')
            relationships.append({'relationship_id': declaration['relationship_id'],
                                  'from_table': source, 'from_columns': source_cols,
                                  'to_table': target, 'to_columns': target_cols,
                                  'source_file': declaration['from']['source_file'],
                                  'target_source_file': declaration['to']['source_file'],
                                  'source': declaration['source'], 'status': status,
                                  'cardinality': declaration['cardinality'],
                                  'source_refs': declaration['source_refs'],
                                  'validation': {'scope': 'full_data', 'source_null_rows': source_nulls,
                                      'target_null_rows': target_nulls, 'source_duplicate_keys': source_duplicates,
                                      'target_duplicate_keys': target_duplicates, 'orphan_rows': orphans}})
            if status != 'verified':
                issues.append(ManifestValidator.issue(f'relationship:{declaration["relationship_id"]}',
                                    f'Relationship is {status}; do not use as a trusted join.',
                                    'Review semantic evidence, key scope and full-data validation counts.'))
        return relationships, issues

    @staticmethod
    def validate_join(catalog, relationship_id, *, aggregate_from=None):
        matches = [r for r in catalog['relationships'] if r['relationship_id'] == relationship_id]
        if len(matches) != 1:
            return {'allowed': False, 'reason': 'Unknown or ambiguous relationship ID.'}
        relation = matches[0]
        if relation['status'] != 'verified':
            return {'allowed': False, 'reason': 'Relationship lacks semantic evidence or full-data validation.'}
        # A many-to-one join preserves source rows; reversing it can multiply facts.
        if aggregate_from == relation['to_table'] and relation.get('cardinality') != 'one_to_one':
            return {'allowed': False, 'reason': 'Pre-aggregate the many side before aggregating target measures.'}
        return {'allowed': True, 'reason': 'Verified relationship; preserve the declared grain.'}

    @staticmethod
    def verified(catalog):
        return [relationship for relationship in catalog['relationships']
                if relationship['status'] == 'verified']

    @staticmethod
    def _key_counts(connection, table, columns):
        nonnull = ' AND '.join(f'{TabularLayout.quote_identifier(c)} IS NOT NULL' for c in columns)
        grouped = ', '.join(TabularLayout.quote_identifier(c) for c in columns)
        nulls = connection.execute(f'SELECT COUNT(*) FROM {TabularLayout.quote_identifier(table)} WHERE NOT ({nonnull})').fetchone()[0]
        duplicates = connection.execute(f'SELECT COUNT(*) FROM (SELECT {grouped} FROM {TabularLayout.quote_identifier(table)} WHERE {nonnull} GROUP BY {grouped} HAVING COUNT(*) > 1)').fetchone()[0]
        return nulls, duplicates
