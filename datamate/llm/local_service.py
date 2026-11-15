"""Validated JSON and question rewriting using one local model instance."""
from collections import Counter
from copy import deepcopy
import json
import re

from jsonschema import Draft202012Validator, ValidationError

from datamate.llm.base import GenerationError
from datamate.llm.prompts import QUESTION_REWRITE_SYSTEM_PROMPT
from datamate.text import norm


class LocalLLMService:
    supports_batch = False
    VERSION = "local-json-v1"

    def __init__(self, backend, cfg, *, manifest_tokens=3072, rewrite_tokens=512):
        self.backend, self.cfg = backend, cfg
        self.manifest_tokens, self.rewrite_tokens = manifest_tokens, rewrite_tokens

    def tag(self):
        return f"{self.backend.tag()}|{self.VERSION}|sql={self.cfg.max_new_tokens}|manifest={self.manifest_tokens}"

    def count_tokens(self, text):
        return self.backend.count_tokens(text)

    @staticmethod
    def json_system(system, schema):
        return system + "\nReturn ONLY one JSON object, without Markdown. JSON schema:\n" + json.dumps(schema, ensure_ascii=False)

    def fits(self, system, prompt, output_tokens):
        return self.backend.prompt_tokens(system, prompt) + output_tokens <= self.cfg.max_model_len

    def json(self, system, prompt, schema, *, task="sql"):
        budget = {"manifest": self.manifest_tokens, "rewrite": self.rewrite_tokens,
                  "sql": self.cfg.max_new_tokens}[task]
        temperature = self.cfg.temperature if task == "sql" else 0.0
        system = self.json_system(system, schema)
        validator = Draft202012Validator(schema)
        repair = ""
        raw_outputs = []
        failure = ""
        for attempt in range(self.cfg.max_retry + 1):
            request = prompt + repair
            if not self.fits(system, request, budget):
                raise GenerationError(f"{task}: prompt exceeds local context budget ({self.cfg.max_model_len}); "
                                      "increase context or reduce the evidence batch")
            step_temp = 0.3 if (task == "manifest" and attempt > 0) else temperature
            raw = self.backend.generate_with_options(system, request, max_new_tokens=budget,
                                                     temperature=step_temp)[0]
            raw_outputs.append(raw)
            try:
                text = raw.strip()
                if text.startswith("```") and text.endswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)[:-3].strip()
                result = json.loads(text)
                validator.validate(result)
                return result
            except (ValueError, TypeError, IndexError) as error:
                failure = str(error)[:600]
            except ValidationError:
                err_msgs = []
                for err in list(validator.iter_errors(result))[:3]:
                    path = "/".join(str(p) for p in err.path)
                    if err.validator in ("maxItems", "minItems", "maxLength", "minLength"):
                        msg = f"violates {err.validator}={err.validator_value}"
                    else:
                        msg = err.message
                    err_msgs.append(f"{path}: {msg}" if path else msg)
                failure = ("; ".join(err_msgs) if err_msgs else "schema validation failed")[:600]
            repair = "\nPrevious response failed validation: " + failure + ". Return the complete corrected JSON."
        exc = GenerationError(f"{task}: invalid JSON after {self.cfg.max_retry + 1} attempts: {failure}")
        exc.raw_outputs = raw_outputs
        exc.last_raw = raw_outputs[-1] if raw_outputs else None
        raise exc

    def generate_json(self, system, prompt, schema):
        return json.dumps(self.json(system, prompt, schema), ensure_ascii=False)

    def rewrite_question(self, question, *, protected_terms=()):
        schema = {"type": "object", "properties": {"question": {"type": "string", "minLength": 1}},
                  "required": ["question"], "additionalProperties": False}
        # Explicit caller terms, quoted names and alphanumeric identifiers must survive rewriting.
        terms = set(protected_terms) | set(re.findall(r"\b(?=\w*[A-Za-z])(?=\w*\d)[\w-]+\b", question))
        terms.update(re.findall(r'["“]([^"”]+)["”]', question))
        prompt = json.dumps({"question": question, "protected_terms": sorted(terms)}, ensure_ascii=False)
        result = self.json(QUESTION_REWRITE_SYSTEM_PROMPT, prompt, schema, task="rewrite")["question"].strip()
        pattern = r"\d+(?:[.,:/-]\d+)*%?"
        if not result or Counter(re.findall(pattern, question)) != Counter(re.findall(pattern, result)):
            raise GenerationError("Rewrite changed numbers/dates or returned empty text")
        if any(question.count(term) != result.count(term) for term in terms):
            raise GenerationError("Rewrite changed a protected identifier/name")
        return result

    def fit_sql_context(self, context, system, schema):
        """Keep rules and complete selected table schemas; explicitly disclose omitted tables."""
        result = deepcopy(context)
        system = self.json_system(system, schema)
        # Reserve room for the SQL error feedback used by the solver on retries.
        def fits():
            return self.fits(system, json.dumps(result, ensure_ascii=False, default=str),
                             self.cfg.max_new_tokens + 768)
        if fits():
            return result
        for table in result["tables"]:
            table["sample_rows"] = []
            for column in table["columns"]:
                column.pop("profile", None)  # completeness/parse counts remain available
        if fits():
            return result
        words = set(norm(result["original_question"]).split())
        result["tables"].sort(key=lambda t: len(words & set(norm(json.dumps(t, ensure_ascii=False)).split())), reverse=True)
        result["omitted_tables"] = []
        result["context_notice"] = "Some tables were omitted for context size. Omission is not evidence that data does not exist."
        while len(result["tables"]) > 1 and not fits():
            removed = result["tables"].pop()
            result["omitted_tables"].append(removed["name"])
        if not fits():
            raise GenerationError("SQL rulebook/README and one table exceed context budget; increase max_model_len")
        return result
