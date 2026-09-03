"""RAGAS 评估适配层（M3）：忠实度 / 答案相关性。

适配 ragas 库（0.2/0.3 系列），LLM 与 Embedding 均走 OpenAI 兼容端点
（对接硅基流动 DeepSeek + bge-m3）。防御性实现：版本导入路径差异、
异步 evaluate 返回 coroutine、网络异常均不向上抛——评估失败时返回 None，
由调用方在报告中记录 ragas_error，不阻塞其他指标落库。
"""

import asyncio
import inspect
import math

from finrag.config import Settings
from finrag.logging import get_logger

logger = get_logger("finrag.ragas_evaluator")


class RagasEvaluator:
    """RAGAS 指标计算器：faithfulness（忠实度）+ answer_relevancy（答案相关性）。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._evaluator_llm = None
        self._evaluator_embeddings = None

    def _build_clients(self):
        """延迟构建 LLM/Embedding wrapper（首次 evaluate 时）。"""
        if self._evaluator_llm is not None:
            return

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        s = self._settings
        chat = ChatOpenAI(
            base_url=s.llm_cloud_base_url,
            api_key=s.llm_cloud_api_key,
            model=s.llm_cloud_model,
            temperature=0,
        )
        emb_model = s.embedding_api_model or s.embedding_model
        emb = OpenAIEmbeddings(
            base_url=s.embedding_api_base_url,
            api_key=s.embedding_api_key,
            model=emb_model,
            check_embedding_ctx_length=False,  # 硅基流动不支持 tiktoken 预估
        )
        self._evaluator_llm = LangchainLLMWrapper(chat)
        self._evaluator_embeddings = LangchainEmbeddingsWrapper(emb)

    def evaluate(self, samples: list[dict]) -> dict | None:
        """计算 RAGAS 指标。

        Args:
            samples: [{"question", "answer", "contexts", "reference"}]

        Returns:
            {"faithfulness": float, "answer_relevancy": float}；不可用/失败返回 None
        """
        if not samples:
            return None
        try:
            return self._evaluate_inner(samples)
        except Exception as exc:
            logger.warning("ragas_evaluate_failed", error=str(exc)[:300])
            return None

    def _evaluate_inner(self, samples: list[dict]) -> dict | None:
        self._build_clients()

        # ragas 0.2/0.3 导入路径兼容：SingleTurnSample 在 dataset_schema 或 dataset
        try:
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        except ImportError:
            from ragas.dataset import EvaluationDataset, SingleTurnSample  # type: ignore[no-redef]

        from ragas import RunConfig, evaluate
        from ragas.metrics import AnswerRelevancy, Faithfulness

        ragas_samples = [
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s.get("reference") or "",
            )
            for s in samples
        ]

        # strictness=1：AnswerRelevancy 生成反向问题的数量；>1 时会传 n>1，
        # 硅基流动等 OpenAI 兼容端点不支持 n>1（BadRequest: n must be less than 1）
        metrics = [Faithfulness(), AnswerRelevancy(strictness=1)]
        run_config = RunConfig(max_workers=2, timeout=120)
        result = evaluate(
            EvaluationDataset(ragas_samples),  # 0.3.x 要求 EvaluationDataset（0.2.x 亦兼容）
            metrics=metrics,
            llm=self._evaluator_llm,
            embeddings=self._evaluator_embeddings,
            run_config=run_config,
            show_progress=False,
        )

        if inspect.iscoroutine(result):  # 新版 ragas evaluate 可能返回 coroutine
            result = asyncio.run(result)

        # EvaluationResult[metric] 返回逐样本分列表（可能含 None/NaN，judge 失败时），
        # 过滤无效值后取均值；全部无效返回 None（MySQL Float 列拒绝 NaN，必须拦下）
        def _avg(key: str) -> float | None:
            try:
                vals = result[key]
            except (KeyError, TypeError):
                return None
            xs = [
                v
                for v in vals
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            ]
            return round(sum(xs) / len(xs), 4) if xs else None

        return {
            "faithfulness": _avg("faithfulness"),
            "answer_relevancy": _avg("answer_relevancy"),
        }
