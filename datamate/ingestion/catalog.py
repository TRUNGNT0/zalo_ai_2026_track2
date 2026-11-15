"""Publish the existing SQL catalog contract over the single layered database."""
from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from datamate.catalog.builder import CatalogBuilder
from datamate.catalog.relationships import RelationshipService
from datamate.ingestion.processing.normalization import IdentifierNormalizer
from datamate.manifests.validation import ManifestValidator
from datamate.rules.rulebook import Rulebook
from .raw import RawData
PIPELINE_VERSION = "layered-catalog-v2"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"

class CatalogService:
    @staticmethod
    def load_manifest(path, business_id):
        if path is None:
            return None
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if manifest.get("business_id") != business_id:
            raise ValueError("Manifest business_id does not match input directory")
        schema = json.loads((SCHEMA_DIR / "business_manifest.schema.json").read_text())
        Draft202012Validator(schema).validate(manifest)
        return manifest

    @staticmethod
    def install_catalog(con, business, database_file, organization):
        """Called inside the database transaction; aliases never copy or clean data."""
        tables, unresolved, declarations = [], [], []
        used = {"calendar"} if business.business_id != "_shared" else set()
        by_logical = {}
        for logical, db, lineage, source_json, columns_json, _ in con.execute("SELECT * FROM meta.tables ORDER BY logical_name").fetchall():
            table = business.tables[logical]
            source, metadata = json.loads(source_json), json.loads(columns_json)
            source_file = source.get("relative_file", Path(source["file"]).name)
            source_object = source.get("source_object", source.get("sheet", source.get("table", Path(source_file).stem)))
            stem = str(Path(source_file).with_suffix(""))
            base = IdentifierNormalizer.normalize(stem + ("_" + source_object if source_object != Path(source_file).stem else ""))
            name, suffix = base, 2
            while name.casefold() in used:
                name, suffix = f"{base}_{suffix}", suffix + 1
            used.add(name.casefold())
            physical = IdentifierNormalizer.unique([c["name"] for c in metadata])
            columns_map = dict(zip((c["name"] for c in metadata), physical))
            types = {r[0]: r[1] for r in con.execute(f"DESCRIBE typed.{RawData.quote_identifier(db)}").fetchall()}
            expressions = [f"{RawData.quote_identifier(c['name'])} AS {RawData.quote_identifier(p)}" for c, p in zip(metadata, physical)]
            # Empty sheets still have a lineage column in storage.
            con.execute(f"CREATE VIEW main.{RawData.quote_identifier(name)} AS SELECT " + (", ".join(expressions) or RawData.quote_identifier(lineage)) + f" FROM typed.{RawData.quote_identifier(db)}")
            header_number, header_status = RawData.header_info(table.raw)
            source_types = source.get("source_types", [])
            profiles = [{"duckdb_type": types[c["name"]], "source_type": source_types[i] if i < len(source_types) else None,
                         "null_count": int(table.typed[c["name"]].isna().sum())} for i, c in enumerate(metadata)]
            entry, issues = CatalogBuilder.build_table(
                source_file=source_file, source_object=source_object,
                source_headers=tuple(c["name"] for c in metadata), physical_table_name=name,
                header_row_number=header_number, header_status=header_status, row_count=len(table.typed),
                column_profiles=profiles, manifest=business.manifest)
            parse_counts = Counter(i.get("column") for i in table.issues if i["code"] == "parse_failed")
            quantity_columns = {
                issue["column"]: next(
                    name for name in issue["derived_columns"] if name.startswith(issue["column"] + "__quantity")
                )
                for issue in table.issues if issue["code"] == "quantity_split"
            }
            for col, meta in zip(entry["columns"], metadata):
                source_column = meta.get("derived_from", meta["name"])
                if source_column in quantity_columns:
                    failed_column = quantity_columns[source_column]
                    failed_count = parse_counts[source_column] if meta["name"] == failed_column else 0
                else:
                    failed_count = parse_counts[source_column] if not meta.get("derived_from") else 0
                col["profile"]["parse_failed_count"] = failed_count
                col["description"] = col["description"] or meta.get("description")
                col["conversion"] = {k: meta.get(k) for k in ("type", "decimal", "date_format", "excel_epoch", "unit", "derived_from")}
            entry["lineage"] = {"typed_table": f"typed.{db}", "raw_table": f"raw.{db}", "column": lineage,
                                "source_columns": columns_map}
            entry["issue_counts"] = dict(Counter(i["code"] for i in table.issues))
            tables.append(entry)
            by_logical[logical] = entry
            unresolved.extend(issues)
            for code, count in entry["issue_counts"].items():
                if code != "header_detected":
                    unresolved.append(ManifestValidator.issue(f"ingestion:{logical}:{code}", f"{count} {code} issue(s); inspect meta.issues and Raw before aggregation."))
            declarations.extend(source.get("relationships", []))
        if business.manifest:
            observed = {"objects": [{"source_file": t["source_files"][0], "source_object": t["source_objects"][0],
                                     "header_row_number": t["header"]["source_row_number"], "header_status": t["header"]["status"],
                                     "columns": [c for c in t["columns"] if not c["conversion"]["derived_from"]]} for t in tables]}
            validation = ManifestValidator.validate(business.manifest, observed)
            if validation["errors"]:
                raise ValueError("Invalid manifest source bindings: " + "; ".join(validation["errors"]))
            unresolved.extend(validation["unresolved"])
        else:
            unresolved.append(ManifestValidator.issue("business_manifest", "No manifest supplied; documented semantic mappings are unavailable."))
        relationships, issues = RelationshipService.from_declarations(declarations, tables)
        unresolved.extend(issues)
        if business.manifest and business.manifest["schema_version"] == "2.0":
            documented, issues = RelationshipService.from_manifest(con, business.manifest["relationships"], tables)
            relationships.extend(documented)
            unresolved.extend(issues)
        # A failed parse must not disappear as a nullable key and authorize a join.
        by_name = {t["physical_name"]: t for t in tables}
        for rel in relationships:
            if rel["status"] == "verified" and any(
                c["profile"]["parse_failed_count"] for side in ("from", "to")
                for c in by_name[rel[side + "_table"]]["columns"] if c["physical_name"] in rel[side + "_columns"]
            ):
                rel["status"] = "failed_validation"
                unresolved.append(ManifestValidator.issue("relationship:" + rel["relationship_id"], "Join keys contain conversion failures."))
        for problem in business.issues:
            unresolved.append(ManifestValidator.issue("ingestion:" + str(problem.get("file", "business")) + ":" + problem["code"], json.dumps(problem, ensure_ascii=False)))
        catalog = CatalogBuilder.build(business_id=business.business_id, database_file=database_file,
                                       tables=tables, relationships=relationships, unresolved=unresolved, manifest=business.manifest)
        catalog["pipeline_version"] = PIPELINE_VERSION
        catalog["input_fingerprint"] = business.input_fingerprint
        catalog["quality"] = Rulebook.assess_capabilities(catalog)
        CatalogService.update_organization(con, catalog, organization)
        schema = json.loads((SCHEMA_DIR / "catalog.schema.json").read_text())
        Draft202012Validator(schema).validate(catalog)
        con.execute("CREATE TABLE meta.catalog (catalog_json VARCHAR, manifest_json VARCHAR)")
        con.execute("INSERT INTO meta.catalog VALUES (?, ?)", [json.dumps(catalog, ensure_ascii=False), json.dumps(business.manifest, ensure_ascii=False)])
        return catalog

    @staticmethod
    def read_catalog(database):
        import duckdb
        with duckdb.connect(str(database), read_only=True) as con:
            return json.loads(con.execute("SELECT catalog_json FROM meta.catalog").fetchone()[0])

    @staticmethod
    def update_organization(con, catalog, organization):
        by_db = {t["lineage"]["typed_table"].removeprefix("typed."): t for t in catalog["tables"]}
        by_logical = {logical: by_db[db] for logical, db in con.execute("SELECT logical_name, db_name FROM meta.tables").fetchall()}
        accepted = []
        for relation in organization["relationships"]:
            if relation["status"] != "accepted":
                continue
            r = dict(relation)
            for side, old in (("from", "source"), ("to", "target")):
                t = by_logical[r[old + "_table"]]
                r[side + "_table"] = t["physical_name"]
                r[side + "_columns"] = [t["lineage"]["source_columns"][c] for c in r[old + "_columns"]]
            accepted.append(r)
        catalog["organization"] = {"policy_version": organization["policy_version"], "relationships": accepted,
                                   "policy": "Accepted only on this snapshot; not a source-declared constraint. Parent measures may repeat on joins."}
        catalog["semantic_views"] = []
        identities = {r[0] for r in con.execute("SELECT db_name FROM meta.tables").fetchall()}
        for (name,) in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='semantic' ORDER BY table_name").fetchall():
            if name in identities:
                continue
            columns = [{"name": r[0], "duckdb_type": r[1]} for r in con.execute(f"DESCRIBE semantic.{RawData.quote_identifier(name)}").fetchall()]
            catalog["semantic_views"].append({"schema": "semantic", "name": name, "columns": columns})
        catalog["organization"]["transformations"] = organization["stage2"]["transformations"]
        catalog["organization"]["models"] = organization["stage3"]["transformations"]

    @staticmethod
    def refresh_catalog(con, organization):
        """Keep catalog and semantic schema in the same organization transaction."""
        if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='catalog'").fetchone()[0]:
            return None
        catalog = json.loads(con.execute("SELECT catalog_json FROM meta.catalog").fetchone()[0])
        CatalogService.update_organization(con, catalog, organization)
        con.execute("UPDATE meta.catalog SET catalog_json=?", [json.dumps(catalog, ensure_ascii=False)])
        return catalog
