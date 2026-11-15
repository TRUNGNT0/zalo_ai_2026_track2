"""Stage 1: exact SQL profiles, bounded key search and evidence-based relationships.

Names/descriptions are evidence, not proof of business semantics. Accepted means
usable under this versioned policy on this snapshot, never a source-declared FK.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from datamate.text import colnorm, norm
from ..raw import RawData

POLICY_VERSION = "organization-v1"
MAX_KEY_COLUMNS = 10
MAX_KEY_WIDTH = 3
# Deliberately specific concepts. A generic 'id' or matching values is insufficient.
CONCEPTS = {
    "order": {"ma_don", "ma_don_hang", "ma_dh", "so_hd", "ma_hoa_don", "so_hoa_don", "order_id", "order_code"},
    "customer": {"ma_khach", "ma_khach_hang", "ma_kh", "so_kh", "customer_id", "customer_code"},
    "product": {"ma_hang", "ma_san_pham", "ma_sp", "id_sp", "sku", "product_id", "product_code"},
    "payment": {"ma_phieu_thu", "ma_thanh_toan", "payment_id", "receipt_id"},
    "branch": {"ma_chi_nhanh", "branch_id", "branch_code"},
    "line": {"thu_tu", "so_thu_tu", "stt", "line_number", "line_no", "line_id"},
}
DESCRIPTION_CONCEPTS = {
    "order": ("ma don", "ma hoa don"), "customer": ("ma khach",),
    "product": ("ma hang", "ma san pham"), "payment": ("ma phieu thu", "ma thanh toan"),
    "branch": ("ma chi nhanh",), "line": ("so thu tu dong", "thu tu dong"),
}

class OrganizationAnalyzer:
    @staticmethod
    def concept(column):
        name = colnorm(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", column["name"]))
        found = {c for c, aliases in CONCEPTS.items() if name in aliases}
        description = norm(column.get("description") or "")
        # Only primary description; notes about another entity are not identity evidence.
        description = re.split(r"[;:(,]", description)[0]
        for c, prefixes in DESCRIPTION_CONCEPTS.items():
            if any(description == prefix or description.startswith(prefix + " ") for prefix in prefixes):
                found.add(c)
        return next(iter(found)) if len(found) == 1 else None

    @staticmethod
    def table_concept(logical):
        """A small, explicit vocabulary; unknown table roles stay unresolved."""
        name = colnorm(logical)
        aliases = {
            "line": ("order_items", "order_lines", "items", "chi_tiet_don", "chi_tiet_hoa_don"),
            "order": ("orders", "order", "don_hang", "hoa_don"),
            "customer": ("customers", "customer", "khach_hang"),
            "product": ("products", "product", "san_pham"),
            "payment": ("payments", "payment", "thanh_toan", "phieu_thu"),
            "branch": ("branches", "branch", "chi_nhanh"),
        }
        # More specific names (order_items) must win before order.
        for domain, names in aliases.items():
            if any(name == alias or name.startswith(alias + "_") for alias in names):
                return domain
        return None

    @staticmethod
    def stable_id(kind, value):
        return kind + "_" + hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]

    @staticmethod
    def present(column):
        return f"({RawData.quote_identifier(column)} IS NOT NULL AND length(trim(CAST({RawData.quote_identifier(column)} AS VARCHAR))) > 0)"

    @staticmethod
    def analyze_connection(con):
        """Analyze a connected export in its current transaction. No mutation."""
        tables = {}
        parse_counts = Counter()
        for logical, column, count in con.execute("""
            SELECT logical_name, json_extract_string(details_json, '$.column'), count(*)
            FROM meta.issues WHERE code='parse_failed' GROUP BY 1,2
        """).fetchall():
            parse_counts[logical, column] = count
        profiles, keys, relations, checks = [], [], [], []
        for logical, db, columns_json in con.execute(
                "SELECT logical_name, db_name, columns_json FROM meta.tables ORDER BY logical_name").fetchall():
            columns = json.loads(columns_json)
            target = "typed." + RawData.quote_identifier(db)
            # Aggregate whole columns, never sampled, and exclude technical lineage.
            expressions = ["count(*)"]
            for c in columns:
                name = RawData.quote_identifier(c["name"])
                expressions += [f"count(*) FILTER (WHERE {name} IS NULL)",
                                f"count(*) FILTER (WHERE {name} IS NOT NULL AND length(trim(CAST({name} AS VARCHAR)))=0)",
                                f"count(DISTINCT {name})",
                                f"count(*) FILTER (WHERE regexp_full_match(CAST({name} AS VARCHAR), '[0-9]+'))"]
            stats = con.execute("SELECT " + ",".join(expressions) + " FROM " + target).fetchone()
            n = stats[0]
            structural = con.execute(f"""SELECT count(*) FROM raw.{RawData.quote_identifier(db)} WHERE
                flags_json LIKE '%repeated_header%' OR flags_json LIKE '%possible_total%'
                OR flags_json LIKE '%blank_row%'""").fetchone()[0]
            ps = {}
            for i, c in enumerate(columns):
                nulls, blanks, distinct, digits = stats[1+i*4:5+i*4]
                domain = OrganizationAnalyzer.concept(c)
                p = {"table": logical, "column": c["name"], "type": c["type"], "rows": n,
                     "null_count": nulls, "blank_count": blanks, "distinct_non_null": distinct,
                     "duplicate_non_null_rows": n-nulls-distinct, "digit_only_count": digits,
                     "parse_failed_count": parse_counts[logical, c["name"]],
                     "concept": domain, "description": c.get("description"),
                     "identifier_hint": bool(c.get("identifier")), "derived": "derived_from" in c}
                profiles.append(p)
                ps[c["name"]] = p
            tables[logical] = {"db": db, "rows": n, "profiles": ps, "structural_rows": structural, "table_concept": OrganizationAnalyzer.table_concept(logical)}
            identifier_columns = [p["column"] for p in ps.values() if not p["derived"] and
                                  (p["identifier_hint"] or p["concept"])]
            selected = sorted(identifier_columns, key=lambda name: (
                -(ps[name]["distinct_non_null"] / max(n, 1)), name))[:MAX_KEY_COLUMNS]
            checks.append({"subject": logical, "check": "key_search_complete_within_limits",
                           "passed": len(identifier_columns) <= MAX_KEY_COLUMNS,
                           "details": {"eligible_columns": len(identifier_columns), "searched_columns": selected,
                                       "max_width": MAX_KEY_WIDTH, "structural_rows": structural}})
            valid_subsets = []
            for width in range(1, min(MAX_KEY_WIDTH, len(selected))+1):
                for combo in itertools.combinations(selected, width):
                    if any(set(k).issubset(combo) for k in valid_subsets):
                        continue  # A nonminimal superkey adds no identity evidence.
                    valid_condition = " AND ".join(OrganizationAnalyzer.present(c) for c in combo)
                    complete = con.execute(f"SELECT count(*) FROM {target} WHERE {valid_condition}").fetchone()[0]
                    distinct = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT " + ",".join(map(RawData.quote_identifier, combo)) +
                                           f" FROM {target} WHERE {valid_condition})").fetchone()[0]
                    valid = n > 0 and complete == n and distinct == n
                    clean = not structural and all(not ps[c]["parse_failed_count"] for c in combo)
                    domains = [ps[c]["concept"] for c in combo]
                    semantics = (all(d is not None for d in domains) and len(set(domains)) == len(domains)
                                 and (width == 1 and domains[0] != "line" or
                                      width > 1 and bool({"line", "branch"} & set(domains))))
                    status = "accepted" if valid and clean and semantics else "candidate" if valid else "rejected"
                    key = {"id": OrganizationAnalyzer.stable_id("key", [logical, combo]), "table": logical, "columns": list(combo),
                           "status": status, "rows": n, "complete_rows": complete,
                           "distinct_complete_keys": distinct, "duplicate_complete_rows": complete-distinct,
                           "concepts": [ps[c]["concept"] for c in combo], "minimal": valid,
                           "reason": "policy_checks_passed" if status == "accepted" else
                                     "identity_or_quality_unresolved" if valid else "empty_null_or_duplicate",
                           "source_declared": False, "primary_key_selected": False,
                           "table_concept": OrganizationAnalyzer.table_concept(logical)}
                    checks.extend([
                        {"subject": key["id"], "check": "nonempty_table", "passed": n > 0, "details": {"rows": n}},
                        {"subject": key["id"], "check": "key_complete", "passed": complete == n,
                         "details": {"rows": n, "complete_rows": complete}},
                        {"subject": key["id"], "check": "key_unique", "passed": distinct == complete,
                         "details": {"complete_rows": complete, "distinct_keys": distinct}},
                    ])
                    keys.append(key)
                    if valid:
                        valid_subsets.append(combo)

        # Compare only compatible semantic domains; never compare every pair of arbitrary columns.
        for key in keys:
            if key["status"] not in ("accepted", "candidate") or any(c is None for c in key["concepts"]):
                continue
            parent = tables[key["table"]]
            if len(set(key["concepts"])) != len(key["concepts"]):
                continue
            for child_name, child in tables.items():
                if child_name == key["table"]:
                    continue  # Self-references require explicit role/scope evidence, not name guessing.
                choices = [[p["column"] for p in child["profiles"].values()
                            if p["concept"] == domain and not p["derived"]] for domain in key["concepts"]]
                if not all(choices):
                    continue
                for source in itertools.product(*choices):
                    source_ps = [child["profiles"][c] for c in source]
                    target_ps = [parent["profiles"][c] for c in key["columns"]]
                    compatible = all(a["type"] == b["type"] for a, b in zip(source_ps, target_ps))
                    relation = {"id": OrganizationAnalyzer.stable_id("rel", [child_name, source, key["table"], key["columns"]]),
                                "source_table": child_name, "source_columns": list(source),
                                "target_table": key["table"], "target_columns": key["columns"],
                                "concepts": key["concepts"], "source_declared": False,
                                "status": "rejected", "reason": "incompatible_types",
                                "target_role_compatible": (parent["table_concept"] in key["concepts"]
                                    if parent["table_concept"] is not None else None)}
                    if compatible:
                        # Parent key uniqueness was checked exactly. A LEFT JOIN is therefore bounded.
                        condition = " AND ".join(f"c.{RawData.quote_identifier(a)}=p.{RawData.quote_identifier(b)}" for a, b in zip(source, key["columns"]))
                        complete = " AND ".join(f"(c.{RawData.quote_identifier(c)} IS NOT NULL AND length(trim(CAST(c.{RawData.quote_identifier(c)} AS VARCHAR)))>0)"
                                                for c in source)
                        count, full, matched, joined = con.execute(f"""SELECT count(*),
                            count(*) FILTER (WHERE {complete}),
                            count(*) FILTER (WHERE {complete} AND p.{RawData.quote_identifier(key['columns'][0])} IS NOT NULL),
                            count(*) FROM typed.{RawData.quote_identifier(child['db'])} c
                            LEFT JOIN typed.{RawData.quote_identifier(parent['db'])} p ON {condition}""").fetchone()
                        distinct_source = con.execute("SELECT count(*) FROM (SELECT DISTINCT " +
                            ",".join(map(RawData.quote_identifier, source)) + f" FROM typed.{RawData.quote_identifier(child['db'])} WHERE " +
                            " AND ".join(OrganizationAnalyzer.present(c) for c in source) + ")").fetchone()[0]
                        all_null = con.execute(f"SELECT count(*) FROM typed.{RawData.quote_identifier(child['db'])} WHERE " +
                                               " AND ".join(f"{RawData.quote_identifier(c)} IS NULL" for c in source)).fetchone()[0]
                        quality = not child["structural_rows"] and all(not p["parse_failed_count"] for p in source_ps)
                        eligible = full > 0 and matched == full and joined == child["rows"] and quality and key["status"] == "accepted" and relation["target_role_compatible"] is True
                        relation.update({"status": "accepted" if eligible else "candidate",
                            "reason": "policy_checks_passed" if eligible else "coverage_or_quality_unresolved",
                            "source_rows": child["rows"], "complete_reference_rows": full,
                            "matched_rows": matched, "unmatched_rows": full-matched,
                            "incomplete_reference_rows": child["rows"]-full,
                            "all_null_reference_rows": all_null,
                            "partial_or_blank_reference_rows": child["rows"]-full-all_null,
                            "observed_nullable": full < child["rows"],
                            "coverage": matched/full if full else None,
                            "join_rows": joined, "join_extra_rows": joined-child["rows"],
                            "cardinality_basis": "observed_snapshot_only",
                            "relationship_type": "one_to_one" if full and distinct_source == full else "many_to_one",
                            "target_key_id": key["id"]})
                        # Partial composite references / whitespace references are not inferred as valid nulls.
                        if relation["partial_or_blank_reference_rows"] and relation["status"] == "accepted":
                            relation.update(status="candidate", reason="partial_or_blank_reference")
                        checks.extend([
                            {"subject": relation["id"], "check": "complete_references_covered",
                             "passed": full > 0 and full == matched,
                             "details": {"complete_rows": full, "matched_rows": matched}},
                            {"subject": relation["id"], "check": "reference_parse_clean",
                             "passed": all(not p["parse_failed_count"] for p in source_ps),
                             "details": {"parse_failed_count": sum(p["parse_failed_count"] for p in source_ps)}},
                        ])
                        checks.append({"subject": relation["id"], "check": "left_join_preserves_source_rows",
                                       "passed": joined == child["rows"],
                                       "details": {"before": child["rows"], "after": joined}})
                    relations.append(relation)
        # Competing destinations for the same role are ambiguous even if both cover 100%.
        competing = defaultdict(list)
        for r in relations:
            if r.get("coverage") == 1 and r["target_role_compatible"] is not False:
                competing[r["source_table"], tuple(r["source_columns"])].append(r)
        for group in competing.values():
            if len(group) > 1:
                for r in group:
                    if r["status"] == "accepted":
                        r.update(status="candidate", reason="ambiguous_target")
        return {"policy_version": POLICY_VERSION, "profiles": profiles, "keys": keys,
                "relationships": relations, "validations": checks,
                "limits": {"key_columns": MAX_KEY_COLUMNS, "key_width": MAX_KEY_WIDTH,
                           "same_table_relationships": "not_inferred", "value_comparison": "exact_no_normalization"},
                "policy": "Accepted is snapshot-local heuristic evidence, not a declared constraint. No merges, splits or joins are materialized."}

    @staticmethod
    def persist_analysis(con, result):
        """Replace only stage-1 metadata; caller owns transaction."""
        specs = {
            "column_profiles": ("logical_name VARCHAR, column_name VARCHAR, details_json VARCHAR",
                                [(p["table"], p["column"], json.dumps(p, ensure_ascii=False)) for p in result["profiles"]]),
            "key_candidates": ("key_id VARCHAR, logical_name VARCHAR, columns_json VARCHAR, status VARCHAR, details_json VARCHAR",
                               [(k["id"], k["table"], json.dumps(k["columns"], ensure_ascii=False), k["status"], json.dumps(k, ensure_ascii=False)) for k in result["keys"]]),
            "relationships": ("relationship_id VARCHAR, source_table VARCHAR, target_table VARCHAR, status VARCHAR, details_json VARCHAR",
                              [(r["id"], r["source_table"], r["target_table"], r["status"], json.dumps(r, ensure_ascii=False)) for r in result["relationships"]]),
            "validation_results": ("subject VARCHAR, check_name VARCHAR, passed BOOLEAN, details_json VARCHAR",
                                   [(v["subject"], v["check"], v["passed"], json.dumps(v["details"], ensure_ascii=False)) for v in result["validations"]]),
            "organization": ("policy_version VARCHAR, details_json VARCHAR",
                             [(POLICY_VERSION, json.dumps({k: result[k] for k in ("policy", "limits")}, ensure_ascii=False))]),
        }
        for table, (schema, rows) in specs.items():
            con.execute(f"CREATE OR REPLACE TABLE meta.{table} ({schema})")
            if rows:
                con.executemany(f"INSERT INTO meta.{table} VALUES ({','.join('?' for _ in rows[0])})", rows)

    @staticmethod
    def write_analysis_report(result, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "organization.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        safe = lambda value: str(value).replace("|", "/").replace("\n", " ")
        lines = ["# Phân tích cấu trúc và quan hệ — đợt 1", "", f"Policy: `{POLICY_VERSION}`.",
                 "", "Đợt 1 không thay đổi Raw/Typed. View đợt 2 và kiểm chứng: ORGANIZATION_BUILD.md / organization-build.json.",
                 "Accepted là kết luận theo quy tắc trên snapshot hiện tại, không phải ràng buộc nguồn hay bảo đảm nghiệp vụ tuyệt đối.",
                 "", "## Thống kê", "", f"- Cột: {len(result['profiles'])}; ứng viên khóa đã kiểm tra: {len(result['keys'])}.",
                 f"- Khóa: {dict(Counter(k['status'] for k in result['keys']))}.",
                 f"- Quan hệ: {dict(Counter(r['status'] for r in result['relationships']))}.",
                 "", "## Khóa không bị bác bỏ", "", "| Bảng | Cột | Trạng thái | Lý do |", "|---|---|---|---|"]
        lines += [f"| {safe(k['table'])} | {safe(', '.join(k['columns']))} | {k['status']} | {k['reason']} |"
                  for k in result["keys"] if k["status"] != "rejected"]
        lines += ["", "## Quan hệ", "", "| Nguồn → đích | Trạng thái | Khớp / tham chiếu đầy đủ | Lý do |", "|---|---|---:|---|"]
        lines += [f"| {safe(r['source_table'])}({safe(', '.join(r['source_columns']))}) → {safe(r['target_table'])}({safe(', '.join(r['target_columns']))}) | {r['status']} | {r.get('matched_rows', 0)} / {r.get('complete_reference_rows', 0)} | {r['reason']} |" for r in result["relationships"]]
        lines += ["", "## Giới hạn", "", "- Chỉ tìm khóa tối thiểu trong tối đa 10 cột có dấu hiệu định danh, tổ hợp tối đa 3 cột.",
                  "- Quan hệ chỉ sinh khi có cùng concept cụ thể; không suy từ trùng giá trị đơn thuần. Không tự nối trong cùng bảng.",
                  "- So mã chính xác, không strip/casefold/ép kiểu để tạo liên kết. Null và chuỗi trắng được thống kê riêng.",
                  "- Tên/mô tả có thể sai; dữ liệu khớp không chứng minh phạm vi khóa ở dữ liệu tương lai.",
                  "- Không suy luận chính sách cho phép null hoặc ràng buộc bắt buộc từ snapshot.",
                  "- Chi tiết toàn bộ hồ sơ, đề xuất bị bác bỏ và kiểm tra: organization.json / các bảng meta."]
        (directory / "ORGANIZATION.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    @staticmethod
    def analyze_database(database, *, mode="dev"):
        """Analyze an existing export atomically without rebuilding business data."""
        import duckdb
        from ..artifacts import ArtifactPolicy
        ArtifactPolicy.validate_mode(mode)
        database = Path(database)
        if not database.is_file():
            raise FileNotFoundError(database)
        with duckdb.connect(str(database)) as con:
            con.execute("BEGIN TRANSACTION")
            try:
                result = OrganizationAnalyzer.analyze_connection(con)
                OrganizationAnalyzer.persist_analysis(con, result)
                from .workflow import OrganizationWorkflow
                OrganizationWorkflow.organize_connection(con, result)
                from ..catalog import CatalogService
                catalog = CatalogService.refresh_catalog(con, result)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        from ..documentation import DatabaseGuide
        DatabaseGuide.write_guide(database)
        if catalog is not None:
            from datamate.storage.json_store import JsonStore
            JsonStore.write(catalog, database.parent / "catalog.json")
        if mode == "dev":
            OrganizationAnalyzer.write_analysis_report(result, database.parent)
            from .build import OrganizationBuilder
            OrganizationBuilder.write_build_report(result["stage2"], database.parent)
            from .model import OrganizationModelBuilder
            OrganizationModelBuilder.write_model_report(result["stage3"], database.parent)
        else:
            ArtifactPolicy.clean_reports(database.parent)
        return result
