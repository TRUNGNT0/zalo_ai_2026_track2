"""build_example của bước fine-tune: khuôn chat Qwen3.5 và che loss phần prompt."""
import re

from training.finetune_qlora import build_example

MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}, {"role": "assistant", "content": "CODE"}]
PIECE = re.compile(r"<\|[^|]*\|>|</?think>|\n\n|\n|[^\n<]+|<")


class FakeTokenizer:
    """Giả khuôn chat Qwen3.5. Lời nhắc ở chế độ suy nghĩ kết thúc bằng "<think>\\n"; chế độ không suy nghĩ kết thúc bằng
    khối rỗng "<think>\\n\\n</think>\\n\\n". Lượt assistant cuối của bản đầy đủ luôn có khối rỗng. Token hóa gộp "\\n\\n" thành
    MỘT token như BPE thật, nên "<think>\\n" không còn là tiền tố của "<think>\\n\\n</think>"."""

    def __init__(self):
        self.vocab = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n" + ("<think>\n\n</think>\n\n" if m["role"] == "assistant" else "")
            text += m["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if kwargs.get("enable_thinking") is False else "<think>\n")
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(piece, len(self.vocab)) for piece in PIECE.findall(text)]}


def test_prompt_is_masked_and_answer_is_trained_when_thinking_is_off():
    tok = FakeTokenizer()
    example = build_example(tok, MESSAGES, 10_000, {"enable_thinking": False})
    assert example is not None
    names = {v: k for k, v in tok.vocab.items()}
    trained = "".join(names[t] for t, label in zip(example["input_ids"], example["labels"]) if label != -100)
    assert trained == "CODE<|im_end|>\n"                       # chỉ phần trả lời (kèm dấu kết thúc lượt) có loss


def test_every_example_is_dropped_if_thinking_kwarg_differs_from_inference():
    # Không truyền enable_thinking=False thì lời nhắc kết thúc bằng "<think>\n" khác tiền tố của bản đầy đủ: mẫu bị loại.
    assert build_example(FakeTokenizer(), MESSAGES, 10_000) is None


def test_example_longer_than_max_len_is_dropped():
    assert build_example(FakeTokenizer(), MESSAGES, 5, {"enable_thinking": False}) is None
