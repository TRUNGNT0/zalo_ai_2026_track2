import json
from unittest.mock import Mock

import pytest

from datamate.llm import gemini
from datamate.llm.base import GenerationError, QuotaExhausted
from datamate.llm.prompts import QUESTION_REWRITE_SYSTEM_PROMPT


class TestQuestionRewrite:
    @staticmethod
    def backend(response):
        backend = gemini.GeminiBackend.__new__(gemini.GeminiBackend)
        backend.generate = Mock(return_value=[response])
        return backend

    def test_rewrite_sends_original_question_and_preserves_protected_terms(self):
        original = "sl SKU-001 cua ABC ngay 04/05/2026 la bn, ko tinh don huy?"
        expected = "Số lượng SKU-001 của ABC ngày 04/05/2026 là bao nhiêu, không tính đơn hủy?"
        backend = self.backend(json.dumps({"question": expected}, ensure_ascii=False))

        assert backend.rewrite_question(original, protected_terms=["SKU-001", "ABC"]) == expected
        backend.generate.assert_called_once_with(
            QUESTION_REWRITE_SYSTEM_PROMPT,
            json.dumps({"question": original, "protected_terms": ["ABC", "SKU-001"]}, ensure_ascii=False),
            n=1,
        )

    def test_plain_vietnamese_and_relative_time_can_remain_unchanged(self):
        original = "Doanh thu tháng này so với cùng kỳ năm ngoái là bao nhiêu?"
        backend = self.backend(json.dumps({"question": original}))
        assert backend.rewrite_question(original) == original

    @pytest.mark.parametrize("response", [
        "", "không phải JSON", "```json\n{}\n```", "null", "[]",
        '{}', '{"question": ""}', '{"question": "   "}', '{"question": 12}',
        '{"question": "Câu hỏi", "answer": 100}',
    ])
    def test_rejects_invalid_response(self, response):
        backend = self.backend(response)
        with pytest.raises(GenerationError):
            backend.rewrite_question("doanh thu hnay bn?")

    @pytest.mark.parametrize(("original", "rewritten"), [
        ("ngay 04/05/2026", "Ngày 05/04/2026"),
        ("gia 2tr", "Giá 2000000"),
        ("giam 10%", "Giảm 10"),
        ("tu 8:30 den 9:30", "Từ 8:00 đến 9:30"),
        ("ban 2 va nhap 2", "Bán 2"),
        ("doanh thu hnay", "Doanh thu ngày 06/11/2026"),
    ])
    def test_rejects_changed_numeric_literals(self, original, rewritten):
        backend = self.backend(json.dumps({"question": rewritten}))
        with pytest.raises(GenerationError, match="số liệu"):
            backend.rewrite_question(original)

    def test_accepts_abbreviation_expansion_without_changing_numbers(self):
        backend = self.backend('{"question": "Giá 2 triệu có giảm 10% không?"}')
        assert backend.rewrite_question("gia 2tr co giam 10% ko?") == "Giá 2 triệu có giảm 10% không?"

    @pytest.mark.parametrize("rewritten", ["Doanh thu ÁBC", "Doanh thu ABC và ABC"])
    def test_rejects_changed_protected_names(self, rewritten):
        backend = self.backend(json.dumps({"question": rewritten}))
        with pytest.raises(GenerationError, match="tên/mã"):
            backend.rewrite_question("doanh thu ABC", protected_terms=["ABC"])

    @pytest.mark.parametrize("question", ["", " \n\t"])
    def test_rejects_empty_input_without_api_call(self, question):
        backend = self.backend("unused")
        with pytest.raises(ValueError):
            backend.rewrite_question(question)
        backend.generate.assert_not_called()

    def test_rejects_non_string_input_without_api_call(self):
        backend = self.backend("unused")
        with pytest.raises(TypeError):
            backend.rewrite_question(None)
        backend.generate.assert_not_called()

    @pytest.mark.parametrize("terms", ["ABC", [""], [None]])
    def test_rejects_invalid_protected_terms(self, terms):
        backend = self.backend("unused")
        with pytest.raises(ValueError):
            backend.rewrite_question("doanh thu ABC", protected_terms=terms)
        backend.generate.assert_not_called()

    @pytest.mark.parametrize("error", [GenerationError("API failed"), QuotaExhausted("daily quota")])
    def test_propagates_api_failures_without_returning_a_fake_rewrite(self, error):
        backend = self.backend("unused")
        backend.generate.side_effect = error
        with pytest.raises(type(error)):
            backend.rewrite_question("hnay doanh thu bn?")

    def test_empty_response_list_is_a_generation_error(self):
        backend = self.backend("unused")
        backend.generate.return_value = []
        with pytest.raises(GenerationError):
            backend.rewrite_question("hnay doanh thu bn?")


class TestRewriteApiKey:
    @pytest.mark.parametrize("variable", ["GEMINI_API_KEY", "GEMINI_APIKEY"])
    def test_accepts_both_environment_names_without_reading_dotenv(self, monkeypatch, variable):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_APIKEY", raising=False)
        monkeypatch.setenv(variable, "test-key")
        loader = Mock()
        monkeypatch.setattr(gemini, "load_dotenv", loader)
        assert gemini._api_key() == "test-key"
        loader.assert_not_called()

    def test_loads_project_dotenv_when_environment_key_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_APIKEY", raising=False)
        # Track this variable so monkeypatch also removes the value loaded by dotenv.
        monkeypatch.setenv("GEMINI_APIKEY", "")
        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_API_KEY=test-dotenv-key\n")
        monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
        monkeypatch.delenv("GEMINI_API_KEY")
        monkeypatch.setattr(gemini, "ENV_FILE", env_file)
        assert gemini._api_key() == "test-dotenv-key"

    def test_standard_environment_name_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "standard-key")
        monkeypatch.setenv("GEMINI_APIKEY", "legacy-key")
        assert gemini._api_key() == "standard-key"
