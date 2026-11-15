"""Äiá»ƒm vÃ o lÃºc cháº¥m bÃ i trong Docker: `bash /code/predict.sh` -> `python -m app.container_predict`.

Äá»c /data (chá»‰-Ä‘á»c), ghi /result/submission.jsonl vÃ  /result/logs/. Cháº¡y hoÃ n toÃ n ngoáº¡i tuyáº¿n.

Thá»© tá»± an toÃ n: file ná»™p Ä‘áº§y Ä‘á»§ abstain Ä‘Æ°á»£c ghi NGAY; sau Ä‘Ã³ pipeline Qwen cá»¥c bá»™ (manifest -> DuckDB -> SQL) ghi tá»«ng
cÃ¢u khi cÃ³ Ä‘Ã¡p Ã¡n. Háº¿t ngÃ¢n sÃ¡ch thá»i gian, lá»—i hoáº·c bá»‹ SIGTERM thÃ¬ nhá»¯ng cÃ¢u Ä‘Ã£ ghi váº«n Ä‘Æ°á»£c cháº¥m, pháº§n cÃ²n láº¡i lÃ  abstain.
Má»i tham sá»‘ cÃ³ giÃ¡ trá»‹ máº·c Ä‘á»‹nh cá»‘ Ä‘á»‹nh á»Ÿ Ä‘Ã¢y vÃ  trong predict.sh (ban tá»• chá»©c khÃ´ng Ä‘á»•i Ä‘Æ°á»£c).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

from datamate.submission_io import IncrementalSubmission


@dataclass
class ContainerSettings:
    data_dir: str = "/data"
    result_dir: str = "/result"
    work_dir: str = "/tmp/zaic_work"                 # nÆ¡i ghi táº¡m (DuckDB, manifest); /data lÃ  chá»‰-Ä‘á»c
    model_dir: str = "/code/models/qwen3.5-9b"
    lora_dir: str = "/code/models/lora"              # náº¿u cÃ³ adapter_config.json á»Ÿ Ä‘Ã¢y thÃ¬ gáº¯n LoRA
    total_budget_s: float = 19800.0                  # 5 giá» 30: chá»«a 30 phÃºt cá»§a giá»›i háº¡n 6 giá» (Ä‘Ã£ gá»“m náº¡p model)
    prepare_budget_s: float = 7200.0                 # tá»‘i Ä‘a cho bÆ°á»›c Qwen sinh manifest; quÃ¡ háº¡n thÃ¬ dÃ¹ng manifest nhÃ¡p táº¥t Ä‘á»‹nh
    margin_s: float = 300.0                          # chá»«a Ä‘á»ƒ ghi ná»‘t
    n_samples: int = 3
    agree_min: int = 2
    temperature: float = 0.7
    seed: int = 0
    limit_questions: int | None = None               # chá»‰ Ä‘á»ƒ cháº¡y thá»­ nhanh (smoke); ban tá»• chá»©c khÃ´ng Ä‘áº·t

    @classmethod
    def from_env(cls) -> "ContainerSettings":
        base = cls()
        env = os.environ
        limit = env.get("LIMIT_QUESTIONS")
        return cls(data_dir=env.get("DATA_DIR", base.data_dir), result_dir=env.get("RESULT_DIR", base.result_dir),
                   work_dir=env.get("WORK_DIR", base.work_dir), model_dir=env.get("MODEL_DIR", base.model_dir),
                   lora_dir=env.get("LORA_DIR", base.lora_dir),
                   total_budget_s=float(env.get("TOTAL_BUDGET_S", base.total_budget_s)),
                   prepare_budget_s=float(env.get("PREPARE_BUDGET_S", base.prepare_budget_s)),
                   margin_s=float(env.get("MARGIN_S", base.margin_s)),
                   n_samples=int(env.get("N_SAMPLES", base.n_samples)), agree_min=int(env.get("AGREE_MIN", base.agree_min)),
                   temperature=float(env.get("TEMPERATURE", base.temperature)), seed=int(env.get("SEED", base.seed)),
                   limit_questions=int(limit) if limit else None)


class ContainerPredictor:
    def __init__(self, settings: ContainerSettings):
        self.s = settings
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def say(self, message: str) -> None:
        print(f"[container {self.elapsed():7.0f}s] {message}", flush=True)

    def read_questions(self) -> list[dict]:
        path = Path(self.s.data_dir) / "questions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return rows[: self.s.limit_questions] if self.s.limit_questions else rows

    def stage_dataset(self, questions: list[dict]) -> Path:
        """ChÃ©p dá»¯ liá»‡u sang thÆ° má»¥c ghi Ä‘Æ°á»£c (pipeline táº¡o manifest/cache cáº¡nh dá»¯ liá»‡u vÃ  kiá»ƒm tra Ä‘Æ°á»ng dáº«n business
        nÃªn liÃªn káº¿t tÆ°á»£ng trÆ°ng tá»›i /data khÃ´ng dÃ¹ng Ä‘Æ°á»£c). Chá»‰ chÃ©p business cÃ³ cÃ¢u há»i."""
        src, root = Path(self.s.data_dir), Path(self.s.work_dir) / "dataset" / "public_test"
        if root.exists():
            shutil.rmtree(root)
        (root / "data").mkdir(parents=True)
        for name in ("rulebook.md", "calendar.csv"):
            shutil.copy2(src / name, root / name)
        for business in sorted({q["business_id"] for q in questions}):
            shutil.copytree(src / "data" / business, root / "data" / business)
        with (root / "questions.jsonl").open("w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in questions)
        return root

    def solve(self, writer: IncrementalSubmission, questions: list[dict]) -> None:
        # Import náº·ng Ä‘áº·t á»Ÿ Ä‘Ã¢y Ä‘á»ƒ file ná»™p abstain Ä‘Æ°á»£c ghi trÆ°á»›c khi náº¡p thÆ° viá»‡n/model.
        import datamate.agent.solver as solver_module
        from datamate.agent.sql_solver import SQLSolver
        from datamate.ingestion.datasets.loader import DatasetLoader
        from datamate.local_pipeline import DatasetPreparationPipeline, LocalPipelineConfig, LocalSQLPipeline

        solver_module.MAX_FAIL_STREAK = 10 ** 9       # má»™t chuá»—i cÃ¢u lá»—i khÃ´ng Ä‘Æ°á»£c dá»«ng cáº£ lÆ°á»£t cháº¡y: cÃ¢u lá»—i chá»‰ lÃ  abstain
        self.say(f"{len(questions)} cÃ¢u; dá»±ng dá»¯ liá»‡u táº¡m á»Ÿ {self.s.work_dir}")
        self.stage_dataset(questions)
        work = Path(self.s.work_dir)
        config = LocalPipelineConfig(dataset_dir=str(work / "dataset"), out_dir=str(work / "out"), model=self.s.model_dir,
                                     n_samples=self.s.n_samples, agree_min=self.s.agree_min)
        pipeline = LocalSQLPipeline(config)
        lora = Path(self.s.lora_dir)
        pipeline.base = replace(pipeline.base, temperature=self.s.temperature, seed=self.s.seed,
                                local_lora=str(lora) if (lora / "adapter_config.json").is_file() else None)
        self.say(f"náº¡p model {self.s.model_dir} (LoRA: {pipeline.base.local_lora})")
        llm = pipeline.llm                              # náº¡p model má»™t láº§n, dÃ¹ng cho manifest, viáº¿t láº¡i cÃ¢u há»i vÃ  SQL
        self.say("Ä‘Ã£ náº¡p model; sinh manifest báº±ng Qwen")

        prepared = False
        try:
            deadline = self.start + self.s.prepare_budget_s
            reports = DatasetPreparationPipeline(config, llm).prepare("public_test", deadline=deadline, keep_going=True)
            prepared = all(report["status"] == "ready" for report in reports)
            self.say("manifest: " + ", ".join(f"{r['business_id']}={r['status']}" for r in reports))
        except Exception:
            self.say("bÆ°á»›c manifest lá»—i, dÃ¹ng manifest nhÃ¡p táº¥t Ä‘á»‹nh cho business chÆ°a xong:\n" + traceback.format_exc())
        # Business nÃ o chÆ°a cÃ³ manifest/DuckDB há»£p lá»‡ sáº½ Ä‘Æ°á»£c dá»±ng á»Ÿ bÆ°á»›c SQL báº±ng manifest nhÃ¡p (khÃ´ng cáº§n model).
        cfg = replace(pipeline.split_config("public_test"), require_prepared_database=prepared)

        split = DatasetLoader.load_split(cfg.test_dir)
        solver = SQLSolver(cfg, llm)
        remaining = self.s.total_budget_s - self.s.margin_s - self.elapsed()
        self.say(f"giáº£i cÃ¢u há»i, cÃ²n {remaining:.0f}s trong ngÃ¢n sÃ¡ch")
        solver.solve_split(split, "test", time_budget=max(1.0, remaining), on_result=writer.add)

    def run(self) -> int:
        random.seed(self.s.seed)
        Path(self.s.result_dir).mkdir(parents=True, exist_ok=True)
        questions = self.read_questions()
        writer = IncrementalSubmission(self.s.result_dir, [q["id"] for q in questions])
        self.say(f"Ä‘Ã£ ghi file ná»™p abstain cho {len(writer.ids)} cÃ¢u; báº¯t Ä‘áº§u")

        def on_term(signum, frame):                      # `docker stop` / háº¿t giá»: ghi ná»‘t rá»“i thoÃ¡t
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, on_term)
        status = 0
        try:
            self.solve(writer, questions)
        except SystemExit:
            self.say("bá»‹ dá»«ng bá»Ÿi tÃ­n hiá»‡u, ghi ná»‘t káº¿t quáº£ Ä‘Ã£ cÃ³")
            status = 143
        except BaseException:
            self.say("Lá»–I khÃ´ng xá»­ lÃ½ Ä‘Æ°á»£c, giá»¯ káº¿t quáº£ Ä‘Ã£ ghi:\n" + traceback.format_exc())
            status = 1
        finally:
            missing = writer.finalize("not processed (time budget, error or stop)")
            self.say(f"xong: {len(writer.ids) - missing} cÃ¢u Ä‘Ã£ cÃ³ káº¿t quáº£, {missing} cÃ¢u abstain")
        return status

    @staticmethod
    def main() -> int:
        return ContainerPredictor(ContainerSettings.from_env()).run()


if __name__ == "__main__":
    sys.exit(ContainerPredictor.main())

