from typing import Any, Iterable

from datamate.ingestion.processing.normalization import IdentifierNormalizer
from datamate.manifests.validation import ManifestValidator

class CatalogBuilder:
    """Bind documented manifest semantics to physical catalog objects."""

    @staticmethod
    def _manifest_matches_file(manifest_table: dict[str, Any], source_file: str) -> bool:
        return source_file in manifest_table.get("source_files", [])

    @staticmethod
    def choose_table(
        manifest: dict[str, Any] | None,
        source_file: str,
        source_object: str,
        source_headers: tuple[str | None, ...],
    ) -> tuple[dict[str, Any] | None, dict[str, str], list[dict[str, Any]]]:
        if manifest is None:
            return None, {}, []

        if manifest.get("schema_version") == "2.0":
            candidates = [t for t in manifest["tables"]
                          if (t["source_file"], t["source_object"]) == (source_file, source_object)]
            if len(candidates) != 1:
                return None, {}, candidates
            table = candidates[0]
            return table, {c["source_name"]: c["description"] for c in table["columns"]
                           if c["status"] == "documented"}, []

        normalized_headers = {
            header.strip().casefold()
            for header in source_headers
            if header is not None and header.strip()
        }
        candidates: list[tuple[int, dict[str, Any], dict[str, str]]] = []
        for table in manifest.get("tables", []):
            if not CatalogBuilder._manifest_matches_file(table, source_file):
                continue
            column_mappings: dict[str, str] = {}
            object_mapping = False
            for key, description in table.get("columns", {}).items():
                prefix, separator, column_name = key.partition(".")
                if separator:
                    if prefix.casefold() == source_object.casefold():
                        column_mappings[column_name] = description
                elif key.casefold() == source_object.casefold():
                    object_mapping = True
                else:
                    column_mappings[key] = description
            overlap = len(
                normalized_headers
                & {value.strip().casefold() for value in column_mappings if value.strip()}
            )
            score = overlap * 100 + (1 if object_mapping else 0)
            candidates.append((score, table, column_mappings))

        if not candidates:
            return None, {}, []
        best_score = max(score for score, _, _ in candidates)
        best = [
            (table, mappings)
            for score, table, mappings in candidates
            if score == best_score
        ]
        if best_score == 0 or len(best) != 1:
            return None, {}, [table for table, _ in best]
        return best[0][0], best[0][1], []

    @staticmethod
    def build_table(
        *,
        source_file: str,
        source_object: str,
        source_headers: tuple[str | None, ...],
        physical_table_name: str,
        header_row_number: int | None,
        header_status: str,
        row_count: int,
        column_profiles: list[dict[str, Any]],
        manifest: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        manifest_table, semantic_columns, ambiguous_candidates = CatalogBuilder.choose_table(
            manifest, source_file, source_object, source_headers
        )
        source_refs = (
            manifest_table.get("source_refs", []) if manifest_table is not None else []
        )
        metadata = {c["source_name"]: c for c in manifest_table["columns"]} if (
            manifest_table is not None and manifest is not None and manifest.get("schema_version") == "2.0"
        ) else {}
        physical_columns = IdentifierNormalizer.unique(source_headers)
        columns: list[dict[str, Any]] = []
        issues: list[dict[str, str]] = []

        if header_status not in {"manifest_header_match", "sqlite_schema", "parquet_schema"}:
            issues.append(
                {
                    "topic": f"header:{source_file}:{source_object}",
                    "reason": f"Header row was selected using {header_status}.",
                    "needed_evidence": (
                        "Confirm the source header row and provide it as explicit "
                        "adapter configuration if this selection is incorrect."
                    ),
                }
            )
        if ambiguous_candidates:
            issues.append(
                {
                    "topic": f"manifest_mapping:{source_file}:{source_object}",
                    "reason": "Multiple manifest table mappings matched this source file "
                    "equally, or none matched the observed headers.",
                    "needed_evidence": "Clarify the table role for this source object.",
                }
            )
        elif manifest is not None and manifest_table is None:
            issues.append(
                {
                    "topic": f"manifest_mapping:{source_file}:{source_object}",
                    "reason": "No business manifest table could be matched to the "
                    "observed source object.",
                    "needed_evidence": "Add the source file and its documented column "
                    "meanings to this business manifest.",
                }
            )

        for index, (source_name, physical_name, profile) in enumerate(
            zip(source_headers, physical_columns, column_profiles, strict=True)
        ):
            description = (
                semantic_columns.get(source_name)
                if source_name is not None
                else None
            )
            details = metadata.get(source_name, {})
            column_refs = details.get("source_refs", source_refs)
            if description is not None:
                semantic_name = details.get("semantic_name") or IdentifierNormalizer.normalize(description)
                mapping_status = "documented_header_match"
            else:
                semantic_name = None
                mapping_status = "unmapped"
                if source_name is None:
                    issues.append(
                        {
                            "topic": f"column:{source_file}:{source_object}:{index + 1}",
                            "reason": "The source header cell is empty.",
                            "needed_evidence": "Provide the correct source column name.",
                        }
                    )
                elif manifest is not None:
                    issues.append(
                        {
                            "topic": f"column_mapping:{source_file}:{source_object}:{source_name}",
                            "reason": "This observed source column has no exact mapping "
                            "in the business manifest.",
                            "needed_evidence": "Document the business meaning of this "
                            "column in the business README or manifest.",
                        }
                    )

            columns.append(
                {
                    **({"semantic_status": details["status"], "unit_basis": details["unit_basis"],
                        "time_role": details["time_role"], "aggregation": details["aggregation"]} if details else {}),
                    "source_name": source_name,
                    "physical_name": physical_name,
                    "semantic_name": semantic_name,
                    "description": description,
                    "duckdb_type": profile["duckdb_type"],
                    "source_type": profile["source_type"],
                    "type_status": "profiled",
                    "aliases": IdentifierNormalizer.aliases(source_name, description),
                    "profile": {
                        "row_count": row_count,
                        "null_count": profile["null_count"],
                    },
                    "source_mappings": [
                        {
                            "source_file": source_file,
                            "source_object": source_object,
                            "source_name": source_name,
                            "mapping_status": mapping_status,
                            "source_refs": column_refs if description is not None else [],
                        }
                    ],
                }
            )

        return (
            {
                **({"grain": manifest_table["grain"]} if metadata else {}),
                "table_id": f"{source_file}::{source_object}",
                "physical_name": physical_table_name,
                "table_role": (
                    manifest_table.get("table_role")
                    if manifest_table is not None
                    else None
                ),
                "source_files": [source_file],
                "source_objects": [source_object],
                "header": {
                    "status": header_status,
                    "source_row_number": header_row_number,
                },
                "row_count": row_count,
                "columns": columns,
                "source_refs": source_refs,
            },
            issues,
        )

    @staticmethod
    def build(
        *,
        business_id: str,
        database_file: str,
        tables: Iterable[dict[str, Any]],
        relationships: Iterable[dict[str, Any]],
        unresolved: Iterable[dict[str, str]],
        manifest: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "catalog_version": "1.0",
            "business_id": business_id,
            "database_file": database_file,
            "tables": list(tables),
            "relationships": list(relationships),
            "facts": manifest.get("facts", []) if manifest is not None else [],
            "unresolved": ManifestValidator.merge_issues(list(unresolved), manifest.get("unresolved", []) if manifest else []),
            "rule_refs": manifest.get("rule_refs", []) if manifest else [],
        }
