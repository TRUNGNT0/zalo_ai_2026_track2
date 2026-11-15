"""Lõi giải câu hỏi: sinh code ×N, chạy, kiểm kiểu, bỏ phiếu đồng thuận, abstain khi không chắc."""
from __future__ import annotations

import time
import traceback
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from datamate.agent.cache import ResultCache
from datamate.agent.context import ContextBuilder
from datamate.agent.exemplars import ExemplarBank
from datamate.agent.prompts import build_system, prompt_hash
from datamate.config import Config
from datamate.ingestion.datasets.loader import Split
from datamate.llm.base import LLMBackend, QuotaExhausted
from datamate.submission import make_sub
from datamate.tools.results import (VALID_TYPES, canon, clean, coerce_result, extract_code, final_fmt,
                               is_empty_result)
from datamate.tools.rules import rule_solve
from datamate.tools.sandbox import run_code

MAX_FAIL_STREAK = 5      # chừng này câu liên tiếp sinh lỗi thì dừng hẳn, đừng im lặng abstain cả tập


class GenerationBroken(Exception):
    """Model sinh lỗi liên tiếp: dừng để khỏi biến cả tập thành abstain."""


@dataclass
class Stats:
    llm_failures: int = 0


class Solver:
    def __init__(self, cfg: Config, backend: LLMBackend | None):
        self.cfg, self.backend = cfg, backend
        self.stats = Stats()
        self._fail_streak = 0
        self.system = build_system(cfg)
        self.last_errors: list[str] = []
        bank = ExemplarBank.load(cfg.exemplar_path) if cfg.exemplar_path else None
        self.builder = ContextBuilder(cfg, backend.count_tokens if backend else None, bank)

    def _generate(self, prompt: str, n: int) -> list[str] | None:
        """Gọi backend; lỗi sinh được đếm và trả None (KHÔNG cache). Hết quota thì ném lên để dừng hẳn."""
        try:
            return self.backend.generate(self.system, prompt, n)
        except QuotaExhausted:
            raise
        except Exception as e:
            self.stats.llm_failures += 1
            if self.stats.llm_failures <= 3:
                print(f"[llm] lỗi sinh ({self.stats.llm_failures}): {e!r}"[:300])
            return None

    def one_sample(self, ctx: str, tables: dict, as_of, first: str | None = None) -> dict | None:
        """Một lời giải: `first` là văn bản đã sinh sẵn (batch) cho lần thử đầu; lỗi thì tự sửa tối đa max_retry lần."""
        err = prev = None
        for attempt in range(self.cfg.max_retry + 1):
            if attempt == 0 and first is not None:
                text = first
            else:
                prompt = ctx if not err else (ctx + f"\n\nLần trước bạn viết:\n{prev}\n\nGặp vấn đề, hãy sửa:\n{err}")
                out = self._generate(prompt, 1)
                if out is None:
                    return None
                text = out[0] or ""
            code = extract_code(text)
            if code is None:
                err, prev = "Không có khối ```python chứa result_type/result", text[:1500]
                continue
            try:
                typ, val = run_code(code, tables, as_of, self.cfg.code_timeout_s)
                if typ not in VALID_TYPES:
                    err, prev = f"result_type không hợp lệ: {typ}", code
                    continue
                val = coerce_result(typ, clean(val))
                if is_empty_result(typ, val):
                    err, prev = ("Kết quả rỗng/NaN: kiểm tra lại bộ lọc, tên cột, khoảng thời gian; "
                                 "nếu dữ liệu thật sự không có thì result_type='unanswerable'"), code
                    continue
                return dict(type=typ, val=val, code=code, attempt=attempt)
            except Exception as e:
                err, prev = f"{type(e).__name__}: {e}", code
        self.last_errors.append(str(err)[:300])           # để biết VÌ SAO câu bị llm_failed
        return None

    def llm_solve(self, ctx: str, tables: dict, as_of) -> dict:
        """Sinh n_samples lời giải độc lập, chỉ trả đáp án khi >= agree_min lần cho cùng kết quả."""
        n, base = self.cfg.n_samples, self.cfg.agree_min
        by_type = self.cfg.min_agree_by_type
        self.last_errors = []
        collect = self.cfg.collect_all

        def need(typ: str) -> int:
            return max(base, by_type.get(typ, 0))
        pre: list[str | None] = [None] * n
        if self.backend.supports_batch:
            texts = self._generate(ctx, n)
            if texts:
                pre = (list(texts) + [None] * n)[:n]
        samples: list = []
        votes: Counter = Counter()
        types: dict = {}

        def can_still_decide() -> bool:                 # còn khả năng đạt đồng thuận với số mẫu còn lại không
            left = n - len(samples)
            return left >= base or any(votes[k] + left >= need(types[k]) for k in votes)
        for i in range(n):
            s = self.one_sample(ctx, tables, as_of, first=pre[i])
            if s is None:
                samples.append(None)
                if not collect and not can_still_decide():
                    break
                continue
            s["canon"] = canon(s["type"], s["val"])
            types[s["canon"]] = s["type"]
            samples.append(s)
            votes[s["canon"]] += 1
            if not collect and (votes[s["canon"]] >= need(s["type"]) or not can_still_decide()):
                break
        decision, agree = None, 0
        if votes:
            key, agree = votes.most_common(1)[0]
            if agree >= need(types[key]):
                s = next(x for x in samples if x and x["canon"] == key)
                decision = (s["type"], s["val"])
        return dict(decision=decision, agree=agree, errors=list(self.last_errors),
                    samples=[None if s is None else {"type": s["type"], "canon": s["canon"], "code": s["code"],
                                                     "val": s["val"], "attempt": s["attempt"]}
                             for s in samples])

    def _solve_row(self, r, split: Split, cache: ResultCache, use_llm: bool):
        t0 = time.time()
        intent = note = ""
        src, codes, sample_results, agree, samples, errors = "rule", [], [], 0, [], []
        biz = split.business(r.business_id)
        if self.cfg.use_rules:
            typ, ans, intent, note = rule_solve(r.question, r.as_of_date, biz, self.cfg.rule_on, split.calendar)
        else:
            typ, ans, intent, note = "abstain", None, "rules_off", ""
        if typ != "abstain":
            codes = [f"# rule:{intent} fact_table={note}"]
        elif use_llm:
            ctx, tables = self.builder.build(r.question, r.as_of_date, biz, split, r.business_id)
            h = prompt_hash(self.cfg, self.backend.tag(), ctx)
            c = cache.get(r.id)
            if not c or c.get("hash") != h:                    # prompt/model đổi -> không dùng cache cũ
                before = self.stats.llm_failures
                res = self.llm_solve(ctx, tables, r.as_of_date)
                c = {"id": r.id, "hash": h, **res}
                if self.stats.llm_failures == before:          # có lỗi sinh thì KHÔNG cache, để lần sau gọi lại
                    cache.put(c)
                    self._fail_streak = 0
                else:
                    self._fail_streak += 1
                    if self._fail_streak >= MAX_FAIL_STREAK:
                        raise GenerationBroken(
                            f"Model sinh lỗi liên tiếp {self._fail_streak} câu (xem thông báo [llm] lỗi sinh ở trên). "
                            "Dừng để khỏi biến cả tập thành abstain.")
            agree, samples, errors = c["agree"], c["samples"], c.get("errors", [])
            codes = [s["code"] for s in c["samples"] if s]
            sample_results = [None if s is None else s["canon"] for s in c["samples"]]
            if c["decision"]:
                typ, ans = c["decision"]
                ans, src, intent = final_fmt(typ, ans), "llm", "llm"
            else:
                src, intent = "abstain", ("llm_no_consensus" if codes else "llm_failed")
        sub = make_sub(r.id, typ, ans)
        log = {"id": r.id, "type": typ, "answer": ans, "source": src, "intent": intent, "note": note,
               "elapsed_s": round(time.time() - t0, 3), "agree": agree, "codes": codes,
               "sample_results": sample_results, "samples": samples, "errors": errors}
        return sub, log

    def solve_split(self, split: Split, tag: str, *, limit_biz: int | None = None, per_biz: int | None = None,
                    limit_q: int | None = None, use_llm: bool = True, time_budget: float | None = None,
                    business_ids: list | None = None):
        """Giải một tập câu hỏi. Trả (danh sách bài nộp, danh sách log)."""
        q = split.questions
        if business_ids is not None:
            q = q[q.business_id.isin(business_ids)]
        if limit_biz:
            q = q[q.business_id.isin(q.business_id.unique()[:limit_biz])]
        if per_biz:
            q = q.groupby("business_id", sort=False).head(per_biz)
        if limit_q:
            q = q.head(limit_q)
        print(f"[{tag}] sẽ chạy {len(q)} câu / {q.business_id.nunique()} business "
              f"(tổng split: {len(split.questions)} câu / {split.questions.business_id.nunique()} business)")
        if self.cfg.run_name:
            tag = f"{tag}_{self.cfg.run_name}"              # mỗi model/cấu hình một cache riêng
        cache = ResultCache(f"{self.cfg.out_dir}/logs/llm_cache_{tag}.jsonl").load()
        print(f"[{tag}] cache hợp lệ đã nạp: {len(cache)} câu")
        subs, logs = [], []
        t_start, n_err, self._fail_streak = time.time(), 0, 0
        for r in q.itertuples():
            if time_budget and time.time() - t_start > time_budget:
                print(f"Hết ngân sách {time_budget}s sau {len(subs)}/{len(q)} câu: các câu còn lại sẽ ghi abstain.")
                break
            try:
                sub, log = self._solve_row(r, split, cache, use_llm)
            except (QuotaExhausted, GenerationBroken):
                print(f"Dừng ở câu {len(subs) + 1}/{len(q)}. Kết quả đã cache được giữ, chạy lại sẽ tiếp tục.")
                raise
            except Exception as e:
                n_err += 1
                if n_err <= 3:                      # đừng nuốt lỗi im lặng: in 3 lỗi đầu kèm traceback
                    print(f"⚠ lỗi ở câu {r.id} (business {r.business_id}): {e!r}")
                    traceback.print_exc(limit=3)
                sub = make_sub(r.id, "abstain", None)
                log = {"id": r.id, "type": "abstain", "answer": None, "source": "abstain", "intent": "error",
                       "note": repr(e)[:200], "elapsed_s": 0.0, "agree": 0, "codes": [], "sample_results": [], "samples": [], "errors": []}
            subs.append(sub)
            logs.append(log)
            if len(subs) % 20 == 0:
                by_source = pd.Series([lg["source"] for lg in logs]).value_counts().to_dict()
                print(f"  [{tag}] {len(subs)}/{len(q)} câu | {time.time() - t_start:.0f}s | nguồn {by_source} "
                      f"| lỗi code: {n_err} | lỗi sinh: {self.stats.llm_failures}")
        return subs, logs
