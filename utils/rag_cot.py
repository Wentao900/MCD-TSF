import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class RAGCoTConfig:
    use_retrieval: bool = True
    top_k: int = 3
    max_new_tokens: int = 96
    temperature: float = 0.7
    cot_model: Optional[str] = None
    cache_size: int = 1024
    local_files_only: bool = True
    device: Optional[str] = None
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False


class RAGCoTPipeline:
    """
    Lightweight retrieval + chain-of-thought text synthesizer.
    Builds a TF-IDF retriever over the domain search corpus and optionally
    calls a local causal LM to turn retrieved evidence + numeric stats into
    a short reasoning snippet that can be encoded by the text encoder.
    """

    def __init__(
        self,
        domain: str,
        search_df: Optional[pd.DataFrame],
        desc: str,
        lookback_len: int,
        pred_len: int,
        config: Optional[RAGCoTConfig] = None,
    ) -> None:
        self.domain = domain
        self.desc = desc
        self.lookback_len = lookback_len
        self.pred_len = pred_len
        self.config = config or RAGCoTConfig()
        self.search_df = self._prep_search_df(search_df)
        self.retriever = self._fit_retriever(self.search_df)
        self.generator = self._init_generator(self.config)
        self.cache: OrderedDict[str, Dict[str, str]] = OrderedDict()

    def _prep_search_df(self, search_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if search_df is None:
            return pd.DataFrame(columns=["fact", "start_date", "end_date"])
        df = search_df.copy()
        df["fact"] = df["fact"].fillna("").astype(str)
        if "start_date" in df.columns and not np.issubdtype(df["start_date"].dtype, np.datetime64):
            df["start_date"] = pd.to_datetime(df["start_date"])
        if "end_date" in df.columns and not np.issubdtype(df["end_date"].dtype, np.datetime64):
            df["end_date"] = pd.to_datetime(df["end_date"])
        return df

    def _fit_retriever(self, search_df: pd.DataFrame) -> Optional[Dict[str, object]]:
        if search_df.empty or not self.config.use_retrieval:
            return None
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4096, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(search_df["fact"].tolist())
        return {"vectorizer": vectorizer, "matrix": matrix}

    def _init_generator(self, config: RAGCoTConfig):
        if config.cot_model is None:
            return None
        device_index = self._resolve_device_index(config.device)
        trust_remote = config.trust_remote_code or (config.cot_model and "qwen" in config.cot_model.lower())
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.cot_model,
                local_files_only=config.local_files_only,
                trust_remote_code=trust_remote,
            )
            model_kwargs = {
                "local_files_only": config.local_files_only,
                "trust_remote_code": trust_remote,
            }
            if config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            if config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            model = AutoModelForCausalLM.from_pretrained(
                config.cot_model,
                **model_kwargs,
            )
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_index,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Falling back to template CoT because model '{config.cot_model}' "
                f"could not be loaded locally ({exc})."
            )
            return None

    def _resolve_device_index(self, device: Optional[str]) -> int:
        if device is None:
            return 0 if torch.cuda.is_available() else -1
        if device.startswith("cuda") and torch.cuda.is_available():
            parts = device.split(":")
            return int(parts[1]) if len(parts) > 1 else 0
        return -1

    def _retrieve(self, query_text: str) -> List[str]:
        if not self.retriever or not self.config.use_retrieval or self.config.top_k <= 0:
            return []
        query_vec = self.retriever["vectorizer"].transform([query_text])
        sims = cosine_similarity(query_vec, self.retriever["matrix"]).ravel()
        if sims.size == 0:
            return []
        top_idx = sims.argsort()[::-1][: self.config.top_k]
        return [
            self.search_df.iloc[i].fact
            for i in top_idx
            if sims[i] > 0 and len(self.search_df.iloc[i].fact.strip()) > 0
        ]

    def _summarize_numeric(self, numeric_history: Sequence[float]) -> str:
        arr = np.asarray(numeric_history, dtype=float).flatten()
        if arr.size == 0:
            return "No numeric history available."
        last = arr[-1]
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        slope = float(arr[-1] - arr[0]) / max(arr.size - 1, 1)
        direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
        return (
            f"last={last:.4f}, mean={mean:.4f}, std={std:.4f}, "
            f"trend is {direction} with slope {slope:.4f}"
        )

    def _format_prompt(
        self,
        numeric_summary: str,
        retrieved: List[str],
    ) -> str:
        evidence = "\n".join([f"- {item}" for item in retrieved]) if retrieved else "No extra evidence."
        return (
            f"{self.desc}\n"
            f"Historical summary (lookback {self.lookback_len}): {numeric_summary}\n"
            f"Retrieved evidence:\n{evidence}\n"
            f"Reason step by step to sketch an intermediate trend for the next {self.pred_len} steps "
            f"before a final forecast."
        )

    def _fallback_cot(self, numeric_summary: str, retrieved: List[str]) -> str:
        steps = [
            f"1) Summarize numeric window: {numeric_summary}.",
        ]
        if retrieved:
            steps.append(f"2) Align with retrieved signals: {' '.join(retrieved[:2])}.")
        steps.append(
            "3) Extrapolate a smooth intermediate trend that respects the direction and volatility, "
            "without giving exact predictions."
        )
        return " ".join(steps)

    def _generate_cot(self, prompt: str, numeric_summary: str, retrieved: List[str]) -> str:
        if self.generator is None:
            return self._fallback_cot(numeric_summary, retrieved)
        try:
            output = self.generator(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                num_return_sequences=1,
                do_sample=True,
            )
            text = output[0]["generated_text"]
            return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Falling back to template CoT because generation failed ({exc}).")
            return self._fallback_cot(numeric_summary, retrieved)

    def _compose_text(self, base_text: str, retrieved: List[str], cot_text: str) -> str:
        blocks = []
        if base_text and base_text != "NA":
            blocks.append(base_text)
        if retrieved:
            blocks.append("Retrieved evidence: " + " ".join(retrieved))
        if cot_text:
            blocks.append("Intermediate trend reasoning: " + cot_text)
        return "\n".join(blocks) if blocks else ""

    def build_guidance_text(
        self,
        numeric_history: Sequence[float],
        start_date,
        end_date,
        base_text: str,
    ) -> Dict[str, str]:
        cache_key = f"{start_date}-{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        numeric_summary = self._summarize_numeric(numeric_history)
        query = f"{self.domain} {numeric_summary} {base_text}"
        retrieved = self._retrieve(query)
        prompt = self._format_prompt(numeric_summary, retrieved)
        cot_text = self._generate_cot(prompt, numeric_summary, retrieved)
        composed_text = self._compose_text(base_text, retrieved, cot_text)

        packaged = {
            "cot_text": cot_text,
            "retrieved_text": " ".join(retrieved),
            "composed_text": composed_text,
        }
        self.cache[cache_key] = packaged
        if len(self.cache) > self.config.cache_size:
            self.cache.popitem(last=False)
        return packaged
