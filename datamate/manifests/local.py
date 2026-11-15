"""Generate source-sized manifest batches with Qwen and validate against source evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from datamate.llm.base import GenerationError
from datamate.llm.prompts import BUSINESS_MANIFEST_SYSTEM_PROMPT
from datamate.manifests.service import BusinessManifestService, SCHEMA_FILE
from datamate.manifests.validation import ManifestValidator
from datamate.storage.json_store import JsonStore
from datamate.text import norm


BATCH_SYSTEM_PROMPT = BUSINESS_MANIFEST_SYSTEM_PROMPT + (
    "\n\nCHẾ ĐỘ BATCH (ghi đè quy tắc 'một entry cho từng source object' ở trên):\n"
    "- Đầu ra `tables` bắt buộc chỉ chứa ĐÚNG MỘT entry tương ứng với `target_source` và CHỈ chứa các cột được cung cấp trong batch này.\n"
    "- Mảng `source_index` chỉ dùng để tham chiếu tên cột khi suy luận quan hệ hoặc đối chiếu, TUYỆT ĐỐI KHÔNG tạo entry trong `tables` cho các source object khác ngoài `target_source`.\n"
    "- Tuân thủ nghiêm ngặt schema: minItems=1, maxItems=1 cho mảng `tables`."
)


class LocalManifestGenerator:
    VERSION = "source-batches-v3"
    COLUMNS_PER_BATCH = 12

    def __init__(self, llm):
        self.llm = llm
        self.schema = json.loads(SCHEMA_FILE.with_name("business_manifest.v2.schema.json").read_text())

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def canonicalize_metadata(manifest):
        """Derive redundant fields from documented semantics, recording every correction.

        Call only after source/evidence validation. Do not infer semantic names,
        promote unresolved columns, change units, or fill unknown/null metadata.
        """
        result, corrections = deepcopy(manifest), []
        for table in result["tables"]:
            for column in table["columns"]:
                if column["status"] != "documented":
                    continue
                semantic, updates = column["semantic_name"], {}
                if semantic in ManifestValidator.COLUMN_CONTRACTS:
                    expected = ManifestValidator.COLUMN_CONTRACTS[semantic][1]
                    if column["aggregation"] not in (expected, "unknown"):
                        updates["aggregation"] = expected
                if semantic in ManifestValidator.TIME_ROLES and column["time_role"] not in (None, semantic):
                    updates["time_role"] = semantic
                for field, value in updates.items():
                    corrections.append({"source_file": table["source_file"], "source_object": table["source_object"],
                                        "source_name": column["source_name"], "semantic_name": semantic,
                                        "field": field, "before": column[field], "after": value,
                                        "reason": "Derived from documented semantic_name contract"})
                    column[field] = value
        return result, corrections

    @staticmethod
    def restrict_columns(part, expected):
        """Giữ đúng các cột của batch (bỏ cột thuộc batch khác, cột trùng, cột lạ). Trả về (part, danh sách cột bị bỏ)."""
        wanted, seen, kept, dropped = set(expected), set(), [], []
        table = part["tables"][0]
        for column in table["columns"]:
            name = column["source_name"]
            if name in wanted and name not in seen:
                seen.add(name)
                kept.append(column)
            else:
                dropped.append(name)
        table["columns"] = kept
        return part, dropped

    def fit_evidence(self, prompt, schema):
        """Select complete evidence sections for a column batch, never truncate source/schema."""
        system = self.llm.json_system(BATCH_SYSTEM_PROMPT, schema)
        def fits():
            return self.llm.fits(system, json.dumps(prompt, ensure_ascii=False), self.llm.manifest_tokens + 512)
        if fits():
            return prompt
        source = prompt["target_source"]
        words = set(norm(source["source_file"] + " " + source["source_object"] + " " +
                         " ".join(c["source_name"] for c in source["columns"])).split())
        prompt["omitted_evidence"] = []
        prompt["instruction"] += " Evidence was selected for this batch. Unprovided meanings must stay unknown."
        candidates = [(field, section) for field in ("readme_sections", "rulebook_sections") for section in prompt[field]]
        candidates.sort(key=lambda item: len(words & set(norm(item[1]["text"]).split())))
        while candidates and not fits():
            field, section = candidates.pop(0)
            prompt[field].remove(section)
            prompt["omitted_evidence"].append(section["evidence_id"])
            prompt["evidence_ids"] = [ref for ref in prompt["evidence_ids"] if ref != section["evidence_id"]]
        # Large source indexes are optional for this batch's column mapping.
        if not fits():
            prompt["source_index"] = [s for s in prompt["source_index"] if
                                      (s["source_file"], s["source_object"]) == (source["source_file"], source["source_object"])]
            prompt["evidence_ids"] = [f"profile:{source['source_file']}::{source['source_object']}"]
        return prompt  # LocalLLMService rejects any remaining overflow before generation.

    def generate(self, business_dir, rulebook, output_dir):
        directory, output = Path(business_dir), Path(output_dir)
        data = BusinessManifestService.build_input(directory, rulebook, self.schema)
        evidence = BusinessManifestService.build_evidence(data)
        JsonStore.write(evidence, output / "evidence" / f"{directory.name}.json")
        signature = self.digest([self.VERSION, BATCH_SYSTEM_PROMPT, self.llm.tag(), self.schema, evidence])
        path = output / f"{directory.name}.json"
        metadata = output / "provenance" / f"{directory.name}.json"
        if path.exists() and metadata.exists() and json.loads(metadata.read_text()).get("signature") == signature:
            manifest = json.loads(path.read_text())
            self.validate(manifest, data)
            if json.loads(metadata.read_text()).get("manifest_sha256") == self.digest(manifest):
                return path

        parts, batches = [], []
        for source in data["source_inventory"]["objects"]:
            columns = [c for c in source["columns"] if c["source_name"]]
            for start in range(0, max(1, len(columns)), self.COLUMNS_PER_BATCH):
                batches.append({**source, "columns": columns[start:start + self.COLUMNS_PER_BATCH]})
        merged_tables = {}
        for batch_number, source in enumerate(batches, 1):
            print(f"[manifest] {directory.name} batch {batch_number}/{len(batches)}: "
                  f"{source['source_file']} / {source['source_object']}", flush=True)
            schema = deepcopy(self.schema)
            schema["properties"]["business_id"]["const"] = directory.name
            table = schema["properties"]["tables"]
            table.update(minItems=1, maxItems=1)
            unresolved = schema["properties"]["unresolved"]
            unresolved["maxItems"] = 4
            for field in ("topic", "reason", "needed_evidence"):
                unresolved["items"]["properties"][field]["maxLength"] = 160
            for key in ("source_file", "source_object", "header_row_number"):
                table["items"]["properties"][key]["const"] = source[key]
            identity = (source["source_file"], source["source_object"])
            if identity in merged_tables:
                for key in ("table_role", "grain"):
                    table["items"]["properties"][key]["const"] = merged_tables[identity][key]
            # Keep all object identities/column names for documented cross-table relations.
            batch_names = [c["source_name"] for c in source["columns"] if c["source_name"]]
            inventory = []
            for s in data["source_inventory"]["objects"]:
                entry = {"source_file": s["source_file"], "source_object": s["source_object"]}
                names = [c["source_name"] for c in s["columns"]]
                if (s["source_file"], s["source_object"]) == identity:   # object đang là target: tách cột batch này / batch khác
                    entry["columns"] = batch_names
                    entry["columns_handled_in_other_batches"] = [n for n in names if n not in set(batch_names)]
                else:
                    entry["columns"] = names
                inventory.append(entry)
            prompt = {"business_id": directory.name, "target_source": source, "source_index": inventory,
                      "readme_sections": data["readme_sections"], "rulebook_sections": data["rulebook_sections"],
                      "evidence_ids": data["_evidence_ids"],
                      "column_contracts": {
                          "documented_semantics": {name: {"unit_basis": unit, "aggregation": aggregation}
                                                   for name, (unit, aggregation) in ManifestValidator.COLUMN_CONTRACTS.items()},
                          "time_role_matches_semantic_name": sorted(ManifestValidator.TIME_ROLES),
                          "note": "Keep genuinely unknown metadata unknown/null. Do not guess semantic_name or units to satisfy these rules."},
                      "instruction": f"Map exactly the {len(batch_names)} columns of target_source.columns. Include ALL of them and no others: "
                                     "columns listed under columns_handled_in_other_batches must NOT appear in tables[0].columns. "
                                     "Keep undocumented semantics unknown. "
                                     "Include only facts/relationships supported by supplied evidence. "
                                     "The source profile evidence id is profile:<source_file>::<source_object>. "
                                     "Keep `unresolved` to at most 4 items under 160 characters; merge related gaps; do not list a column there if it already has status=unresolved."}
            prompt = self.fit_evidence(prompt, schema)
            key = self.digest([signature, prompt, schema])
            cache = output / "batches" / directory.name / f"{key}.json"
            part = json.loads(cache.read_text()) if cache.exists() else None
            error = ""
            for attempt in range(3):
                try:
                    if part is None:
                        request = json.dumps(prompt, ensure_ascii=False) + error
                        part = self.llm.json(BATCH_SYSTEM_PROMPT, request, schema, task="manifest")
                    Draft202012Validator(schema).validate(part)
                    expected = [c["source_name"] for c in source["columns"] if c["source_name"]]
                    part, dropped = self.restrict_columns(part, expected)
                    if dropped:
                        print(f"[manifest] {directory.name} batch {batch_number}: bỏ {len(dropped)} cột ngoài batch/trùng: {dropped}", flush=True)
                    bindings = ManifestValidator.validate(part, data["source_inventory"], data["_evidence_ids"])
                    if bindings["errors"]:
                        raise ValueError("; ".join(bindings["errors"]))
                    part, corrections = self.canonicalize_metadata(part)
                    self.validate(part, data, complete=False)
                    names = {c["source_name"] for c in part["tables"][0]["columns"]}
                    missing = [n for n in expected if n not in names]
                    if missing:
                        raise ValueError(f"Target source is missing columns {missing}; unexpected columns ignored: {dropped}. "
                                         f"Return exactly the {len(expected)} columns of target_source.columns.")
                    break
                except GenerationError as exc:
                    JsonStore.write({"error": str(exc), "manifest": getattr(exc, "last_raw", None),
                                     "raw_outputs": getattr(exc, "raw_outputs", [])},
                                    output / "reports" / "failures" / directory.name / f"{key}.json")
                    raise
                except (ValueError, ValidationError) as exc:
                    if attempt == 2:
                        JsonStore.write({"error": str(exc), "manifest": part},
                                        output / "reports" / "failures" / directory.name / f"{key}.json")
                        raise ValueError(f"Invalid manifest for {directory.name}/{source['source_object']}: {exc}") from exc
                    previous = [{k: c.get(k) for k in ("source_name", "semantic_name", "unit_basis", "time_role", "aggregation")}
                                for t in part.get("tables", []) for c in t.get("columns", [])]
                    error = "\nCorrect the source mapping and return complete JSON. " + json.dumps(
                        {"validation_error": str(exc)[:1000], "previous_columns": previous}, ensure_ascii=False)
                    part = None
            JsonStore.write(part, cache)
            if corrections:
                JsonStore.write({"corrections": corrections},
                                output / "reports" / "normalization" / directory.name / f"{key}.json")
                print(f"[manifest] normalized {len(corrections)} dependent metadata fields; audit saved", flush=True)
            if prompt.get("omitted_evidence"):
                part["unresolved"] = ManifestValidator.merge_issues(part["unresolved"], [ManifestValidator.issue(
                    f"context:{source['source_file']}:{source['source_object']}",
                    "Evidence sections omitted to fit local context: " + ", ".join(prompt["omitted_evidence"]),
                    "Review omitted sections before promoting unknown mappings to documented semantics.")])
            parts.append(part)
            mapped = part["tables"][0]
            if identity not in merged_tables:
                merged_tables[identity] = deepcopy(mapped)
            else:
                merged_tables[identity]["columns"].extend(mapped["columns"])
                merged_tables[identity]["source_refs"] = sorted(set(merged_tables[identity]["source_refs"]) | set(mapped["source_refs"]))

        if not parts:
            raise ValueError(f"No source objects in {directory}")
        manifest = {"schema_version": "2.0", "business_id": directory.name,
                    "tables": list(merged_tables.values()),
                    "facts": [], "relationships": [], "rule_refs": sorted({r for p in parts for r in p["rule_refs"]}),
                    "unresolved": ManifestValidator.merge_issues(*(p["unresolved"] for p in parts))}
        for field, id_key in (("facts", "fact_id"), ("relationships", "relationship_id")):
            entries = {}
            for part in parts:
                for entry in part[field]:
                    content = {k: v for k, v in entry.items() if k != id_key}
                    key = self.digest(content)[:24]
                    entries[key] = {**content, id_key: f"local_{key}"}
            manifest[field] = list(entries.values())
        report = self.validate(manifest, data)
        manifest["unresolved"] = ManifestValidator.merge_issues(manifest["unresolved"], report["unresolved"])
        JsonStore.write(report, output / "reports" / f"{directory.name}.json")
        JsonStore.write(manifest, path)
        JsonStore.write({"signature": signature, "model": self.llm.tag(),
                         "manifest_sha256": self.digest(manifest)}, metadata)
        return path

    def validate(self, manifest, data, *, complete=True):
        Draft202012Validator(self.schema).validate(manifest)
        if manifest["business_id"] != data["business_id"]:
            raise ValueError("Wrong manifest business_id")
        report = ManifestValidator.validate(manifest, data["source_inventory"], data["_evidence_ids"])
        for table in manifest["tables"]:
            for column in table["columns"]:
                location = f"{table['source_file']}/{table['source_object']}/{column['source_name']}"
                report["errors"].extend(f"{location}: {message}" for message in ManifestValidator.validate_column_semantics(column))
        if complete and len(manifest["tables"]) != len(data["source_inventory"]["objects"]):
            report["errors"].append("Manifest must cover every source object")
        if report["errors"]:
            raise ValueError("; ".join(report["errors"]))
        return report
