from collections import Counter
from pathlib import Path
from typing import Any, TYPE_CHECKING
import json
import re
import unicodedata

from jsonschema import Draft202012Validator, ValidationError

from datamate.ingestion.sources.inventory import SourceInventory
from datamate.manifests.validation import ManifestValidator
from datamate.rules.rulebook import Rulebook

if TYPE_CHECKING:
    from datamate.llm.gemini import GeminiLLM

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "schemas" / "business_manifest.schema.json"


class BusinessManifestService:
    """Build evidence bundles and generate validated business manifests."""

    @staticmethod
    def build_draft(business_dir: Path, rulebook_text: str = "") -> dict[str, Any]:
        """Create a local, conservative v2 manifest without inferring undocumented semantics."""
        from datamate.manifests.migration import ManifestMigrator

        schema_path = SCHEMA_FILE.with_name("business_manifest.v2.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        inventory = SourceInventory.inspect(business_dir)
        if not inventory["objects"]:
            raise ValueError(f"No supported source objects found in {business_dir}")

        readme_path = business_dir / "README.md"
        readme = readme_path.read_text(encoding="utf-8-sig") if readme_path.exists() else ""
        readme_sections = BusinessManifestService._make_readme_sections(readme)
        rulebook_sections = BusinessManifestService._make_rulebook_sections(rulebook_text)
        survey_results = SourceInventory.evidence(inventory)
        evidence_ids = {
            section["evidence_id"]
            for section in readme_sections + rulebook_sections + survey_results
        }
        business_input = {
            "business_id": business_dir.name,
            "source_inventory": inventory,
            "readme_sections": readme_sections,
            "rulebook_sections": rulebook_sections,
            "_evidence_ids": sorted(evidence_ids),
        }
        legacy = {
            "schema_version": "1.0",
            "business_id": business_dir.name,
            "tables": [],
            "facts": [],
            "unresolved": [],
        }
        manifest = ManifestMigrator.upgrade(legacy, business_input)
        ManifestMigrator.enrich_explicit_readme_definitions(manifest, readme_sections)
        manifest["unresolved"].extend(
            ManifestValidator.issue(
                f"table_role:{table['source_file']}:{table['source_object']}",
                "Table role was not explicitly documented; this draft does not infer a business role from its name.",
                "Document and review the table role and grain in the BusinessManifest.",
            )
            for table in manifest["tables"] if table["table_role"] == "unknown"
        )
        manifest["unresolved"].extend(
            ManifestValidator.issue(
                f"grain:{table['source_file']}:{table['source_object']}",
                "Table grain was not explicitly documented.",
                "Document what one row represents in the BusinessManifest.",
            )
            for table in manifest["tables"] if table["grain"] is None
        )
        report = ManifestValidator.validate(manifest, inventory, evidence_ids)
        if report["errors"]:
            raise ValueError("Generated manifest draft failed validation: " + "; ".join(report["errors"]))
        manifest["unresolved"] = ManifestValidator.merge_issues(
            manifest["unresolved"], report["unresolved"]
        )
        Draft202012Validator(schema).validate(manifest)
        return manifest

    @staticmethod
    def _read_markdown_sections(text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, list[str]]] = []
        current_heading = ""
        current_lines: list[str] = []

        for line in text.splitlines():
            if line.startswith("#"):
                sections.append((current_heading, current_lines))
                current_heading = line
                current_lines = [line]
            else:
                current_lines.append(line)
        sections.append((current_heading, current_lines))

        return [
            (heading, "\n".join(lines).strip())
            for heading, lines in sections
            if "\n".join(lines).strip()
        ]

    @staticmethod
    def _slug(value: str) -> str:
        value = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
        value = "".join(character for character in value if not unicodedata.combining(character))
        value = re.sub(r"[^a-z0-9]+", "_", value)
        return value.strip("_")

    @staticmethod
    def _readme_evidence_id(heading: str, seen_ids: Counter[str]) -> str:
        if heading == "" or heading.startswith("# Dữ liệu"):
            base_id = "readme.overview"
        elif re.match(r"^##\s+File\s*$", heading, flags=re.IGNORECASE):
            base_id = "readme.overview"
        elif re.match(r"^##\s+Quy\s+ước\s*$", heading, flags=re.IGNORECASE):
            base_id = "readme.conventions"
        else:
            filename_match = re.search(r"`([^`]+)`", heading)
            if filename_match:
                stem = Path(filename_match.group(1)).stem
                normalized = BusinessManifestService._slug(stem)
                if "sobanhang" in normalized or "ban_hang" in normalized:
                    match = re.search(r"(?:sobanhang|ban_hang)_?(.+)?", normalized)
                    suffix = (match.group(1) or "").strip("_") if match else ""
                    base_id = f"readme.sales_{suffix}" if suffix else "readme.sales"
                elif any(term in normalized for term in ("danhmuc", "quydoi", "donvi")):
                    base_id = "readme.conversions"
                elif any(term in normalized for term in ("tonkho", "baocaoton")):
                    base_id = "readme.stock"
                elif any(term in normalized for term in ("nhaphang", "phieunhap")):
                    base_id = "readme.purchases"
                elif any(term in normalized for term in ("banggia", "lichsu_gia")):
                    base_id = "readme.prices"
                else:
                    base_id = f"readme.{normalized}"
                sheet_match = re.search(r"\bsheet\s+`([^`]+)`", heading, flags=re.IGNORECASE)
                if sheet_match:
                    base_id = f"{base_id}.sheet_{BusinessManifestService._slug(sheet_match.group(1))}"
            else:
                base_id = f"readme.{BusinessManifestService._slug(heading.lstrip('# ')) or 'section'}"

        seen_ids[base_id] += 1
        if seen_ids[base_id] == 1:
            return base_id
        return f"{base_id}_{seen_ids[base_id]}"

    @staticmethod
    def _make_readme_sections(readme_text: str) -> list[dict[str, str]]:
        sections = BusinessManifestService._read_markdown_sections(readme_text) # Trích xuất các đoạn từ README, mỗi đoạn là một tuple chứa tiêu đề và nội dung. Các đoạn này sẽ được chuyển đổi thành danh sách các dict với evidence_id và text.
        evidence: list[dict[str, str]] = []
        seen_ids: Counter[str] = Counter()
        overview_fragments: list[str] = []

        for heading, text in sections:
            
            # Nếu tiêu đề là trống, bắt đầu bằng "# Dữ liệu" hoặc khớp với mẫu "## File", thêm nội dung vào danh sách overview_fragments và tiếp tục. Ngược lại, tạo một dict với evidence_id được tạo từ tiêu đề và text, rồi thêm vào danh sách evidence.
            if heading == "" or heading.startswith("# Dữ liệu") or re.match(
                r"^##\s+File\s*$", heading, flags=re.IGNORECASE
            ):
                overview_fragments.append(text)
                continue
            evidence.append(
                {
                    "evidence_id": BusinessManifestService._readme_evidence_id(heading, seen_ids),
                    "text": text,
                }
            )

        if overview_fragments:
            evidence.insert(
                0,
                {
                    "evidence_id": "readme.overview",
                    "text": "\n\n".join(overview_fragments),
                },
            )
        return evidence

    @staticmethod
    def _make_rulebook_sections(rulebook_text: str) -> list[dict[str, str]]:
        return Rulebook.parse(rulebook_text)



    # Hàm này tạo đầu vào cho việc tạo manifest dựa trên thư mục của doanh nghiệp, nội dung rulebook và schema JSON. Nó tìm kiếm README trong thư mục doanh nghiệp, kiểm tra số lượng README, xây dựng danh sách file_inventory, đọc các đoạn README, kiểm
    @staticmethod
    def build_input(
        business_dir: Path,
        rulebook_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        
        # sắp xếp các file README trong thư mục doanh nghiệp, chỉ giữ lại các file có tên bắt đầu bằng "readme" (không phân biệt chữ hoa chữ thường). Nếu không tìm thấy đúng một README, ném ra lỗi ValueError.
        readme_paths = sorted(
            path
            for path in business_dir.iterdir()
            if path.is_file() and path.name.casefold().startswith("readme")
        )
        if len(readme_paths) != 1:
            raise ValueError(
                f"Expected exactly one README in {business_dir}, found {len(readme_paths)}."
            )


        # xây dựng danh sách file_inventory bằng cách tìm tất cả các file trong thư mục doanh nghiệp, bỏ qua các file ẩn, file ".DS_Store" và các file README. Danh sách này được sắp xếp theo đường dẫn tương đối từ thư mục doanh nghiệp và chuyển sang định dạng POSIX.
        file_inventory = sorted(
            path.relative_to(business_dir).as_posix()
            for path in business_dir.rglob("*")
            if path.is_file()
            and path.name != ".DS_Store"
            and not path.name.casefold().startswith("readme")
        )
        
        
        # đọc các đoạn README bằng cách gọi phương thức _make_readme_sections, truyền vào nội dung của README được đọc từ file. Đồng thời, sử dụng SourceInventory để kiểm tra các bảng nguồn trong thư mục doanh nghiệp và thu thập các quan sát dữ liệu từ inventory.
        readme_sections = BusinessManifestService._make_readme_sections(
            readme_paths[0].read_text(encoding="utf-8")
        )
        inventory = SourceInventory.inspect(business_dir) # Sử dụng SourceInventory để kiểm tra các bảng nguồn trong thư mục doanh nghiệp, trả về một danh sách các dict chứa thông tin về các bảng nguồn, cột, quan hệ và các quy tắc nghiệp vụ.
        survey_results = SourceInventory.evidence(inventory) # Thu thập các quan sát dữ liệu từ inventory, trả về một danh sách các dict chứa evidence_id và text cho từng quan sát.

        evidence_ids = {
            section["evidence_id"]
            for section in readme_sections + survey_results
        }
        rulebook_sections = BusinessManifestService._make_rulebook_sections(rulebook_text)
        evidence_ids.update(section["evidence_id"] for section in rulebook_sections)
    
        print(f"Built input for {business_dir.name}: {len(file_inventory)} files, {len(readme_sections)} README sections, {len(survey_results)} survey results, {len(rulebook_sections)} rulebook sections.")
        return {
            "business_id": business_dir.name,
            "source_inventory": inventory,
            "file_inventory": file_inventory,
            "readme_sections": readme_sections,
            "survey_results": survey_results,
            "rulebook_sections": rulebook_sections,
            "output_schema": output_schema,
            "_evidence_ids": sorted(evidence_ids),
        }

    @staticmethod
    def build_evidence(business_input: dict[str, Any]) -> dict[str, Any]:
        return {
            **({"source_inventory": business_input["source_inventory"]} if "source_inventory" in business_input else {}),
            "business_id": business_input["business_id"],
            "file_inventory": business_input["file_inventory"],
            "readme_sections": business_input["readme_sections"],
            "survey_results": business_input["survey_results"],
            "rulebook_sections": business_input["rulebook_sections"],
        }

    @staticmethod
    def generate(
        llm: "GeminiLLM",
        business_input: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        from datamate.llm.prompts import BUSINESS_MANIFEST_SYSTEM_PROMPT

        evidence_ids = set(business_input["_evidence_ids"])
        prompt_input = {
            key: value for key, value in business_input.items() if not key.startswith("_")
        }
        prompt = json.dumps(prompt_input, ensure_ascii=False, indent=2)
        validator = Draft202012Validator(output_schema)
        last_error = "No response was generated."

        for attempt in range(3):
            repair_instruction = (
                ""
                if attempt == 0
                else (
                    "\n\nYour previous response failed local validation. "
                    "Regenerate the complete JSON object from the supplied evidence "
                    f"and schema. Validation error: {last_error}"
                )
            )
            response_text = llm.generate(
                prompt + repair_instruction,
                system_instruction=BUSINESS_MANIFEST_SYSTEM_PROMPT,
                response_json_schema=output_schema,
            )
            try:
                manifest = json.loads(response_text)
                if "oneOf" in output_schema and manifest.get("schema_version") in {"1.0", "2.0"}:
                    branch = 0 if manifest["schema_version"] == "1.0" else 1
                    selected = {**output_schema["oneOf"][branch], "$defs": output_schema.get("$defs", {})}
                    Draft202012Validator(selected).validate(manifest)
                else:
                    validator.validate(manifest)
                if manifest["business_id"] != business_input["business_id"]:
                    raise ValueError(
                        "Gemini returned a manifest for a different business_id."
                    )

                inventory = set(business_input["file_inventory"])
                for table in manifest["tables"]:
                    unknown_files = set(table.get("source_files", [table.get("source_file")])) - inventory
                    if unknown_files:
                        raise ValueError(
                            "Manifest references files absent from inventory: "
                            f"{sorted(unknown_files)}"
                        )
                references = {
                    reference
                    for table in manifest["tables"]
                    for reference in table["source_refs"]
                }
                references.update(
                    reference
                    for fact in manifest["facts"]
                    for reference in fact["source_refs"]
                )
                unknown_references = references - evidence_ids
                if unknown_references:
                    raise ValueError(
                        "Manifest references unknown evidence IDs: "
                        f"{sorted(unknown_references)}"
                    )
                if "source_inventory" in business_input:
                    report = ManifestValidator.validate(manifest, business_input["source_inventory"], evidence_ids)
                    if report["errors"]:
                        raise ValueError("; ".join(report["errors"]))
                    manifest["unresolved"] = ManifestValidator.merge_issues(manifest["unresolved"], report["unresolved"])
                return manifest
            except json.JSONDecodeError as error:
                last_error = (
                    f"Response was invalid JSON at character {error.pos}: {error.msg}"
                )
            except ValidationError as error:
                path = "/".join(str(part) for part in error.absolute_path) or "(root)"
                last_error = f"Schema validation failed at {path}: {error.message}"
            except ValueError as error:
                last_error = str(error)

        raise RuntimeError(
            f"Gemini failed to generate a valid manifest for "
            f"{business_input['business_id']} after 3 attempts: {last_error}"
        )
