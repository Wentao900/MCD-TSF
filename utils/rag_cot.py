import math
import os
import re

try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


def resolve_local_model_path(model_name, aliases=()):
    candidates = [model_name, model_name.split("/")[-1], model_name.replace("/", "--")]
    candidates.extend(aliases)
    roots = [os.getcwd(), os.path.join(os.getcwd(), "llms"), os.path.join(os.getcwd(), "models")]
    for root in roots:
        for candidate in candidates:
            path = os.path.join(root, candidate)
            if os.path.isdir(path):
                return path
    raise FileNotFoundError(
        f"Local model files for {model_name} were not found under current directory, ./llms, or ./models. "
        f"Checked names: {', '.join(candidates)}"
    )


class ScaleAwareRAGCoT:
    def __init__(
        self,
        search_frame=None,
        rag_stage1_topk=12,
        rag_stage2_topk=3,
        cot_model_name="gpt2-medium",
        cot_local_files_only=True,
        cot_max_new_tokens=64,
        use_longformer_rerank=False,
        longformer_model_name="allenai/longformer-base-4096",
        longformer_local_files_only=True,
        longformer_max_length=2048,
        rag_long_topn=24,
        raw_text_max_chars=800,
        cot_generator=None,
    ):
        self.default_corpus = self._extract_corpus(search_frame)
        self.rag_stage1_topk = max(1, int(rag_stage1_topk))
        self.rag_stage2_topk = max(1, int(rag_stage2_topk))
        self.rag_long_topn = max(self.rag_stage1_topk, int(rag_long_topn))
        self.raw_text_max_chars = max(100, int(raw_text_max_chars))

        self.cot_model_name = cot_model_name
        self.cot_local_files_only = bool(cot_local_files_only)
        self.cot_max_new_tokens = max(1, int(cot_max_new_tokens))
        self.cot_generator = cot_generator
        self._cot_model = None
        self._cot_tokenizer = None
        self._cot_load_failed = False

        self.use_longformer_rerank = bool(use_longformer_rerank)
        self.longformer_model_name = longformer_model_name
        self.longformer_local_files_only = bool(longformer_local_files_only)
        self.longformer_max_length = max(128, int(longformer_max_length))
        self._longformer_model = None
        self._longformer_tokenizer = None
        self._longformer_load_failed = False

    def _extract_corpus(self, search_frame):
        if search_frame is None:
            return []
        if hasattr(search_frame, "__getitem__") and "fact" in getattr(search_frame, "columns", []):
            values = search_frame["fact"].dropna().astype(str).tolist()
        elif isinstance(search_frame, dict):
            values = search_frame.get("fact", [])
        else:
            values = search_frame
        corpus = []
        for value in values:
            text = str(value).strip()
            if text and text.lower() != "nan":
                corpus.append(text)
        return corpus

    def _tokenize(self, text):
        return set(re.findall(r"[a-zA-Z0-9_]+", str(text).lower()))

    def _rank_by_overlap(self, query, corpus, topk):
        query_terms = self._tokenize(query)
        scored = []
        for idx, text in enumerate(corpus):
            terms = self._tokenize(text)
            if not terms:
                continue
            score = len(query_terms & terms) / math.sqrt(len(terms))
            if score > 0:
                scored.append((score, idx))
        scored.sort(reverse=True)
        return [corpus[idx] for _, idx in scored[:topk]]

    def _retrieve(self, query, topk, corpus=None):
        corpus = self.default_corpus if corpus is None else self._extract_corpus(corpus)
        if not corpus or not str(query).strip():
            return []
        topk = min(max(1, int(topk)), len(corpus))
        if TfidfVectorizer is not None and cosine_similarity is not None and np is not None:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, matrix).ravel()
            indices = np.argsort(scores)[::-1][:topk]
            return [corpus[i] for i in indices if scores[i] > 0]
        return self._rank_by_overlap(query, corpus, topk)

    def _load_longformer(self):
        if not self.use_longformer_rerank or self._longformer_load_failed:
            return False
        if self._longformer_model is not None and self._longformer_tokenizer is not None:
            return True
        try:
            from transformers import AutoModel, AutoTokenizer

            model_path = resolve_local_model_path(
                self.longformer_model_name,
                aliases=("longformer-base-4096",),
            )
            self._longformer_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._longformer_model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._longformer_model.eval()
            return True
        except Exception:
            self._longformer_load_failed = True
            self._longformer_model = None
            self._longformer_tokenizer = None
            return False

    def _longformer_rerank(self, query, evidence, topk):
        if not evidence or not self._load_longformer():
            return evidence[:topk]
        try:
            import torch

            texts = [query] + evidence
            token_input = self._longformer_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.longformer_max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = self._longformer_model(**token_input).last_hidden_state
            mask = token_input["attention_mask"].unsqueeze(-1).clamp_min(1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            query_vec = pooled[:1]
            ev_vec = pooled[1:]
            scores = torch.nn.functional.cosine_similarity(query_vec, ev_vec, dim=1)
            indices = torch.argsort(scores, descending=True)[:topk].tolist()
            return [evidence[idx] for idx in indices]
        except Exception:
            return evidence[:topk]

    def _summarize_num(self, history_values):
        if np is None:
            return self._summarize_num_without_numpy(history_values)

        values = np.asarray(history_values, dtype=np.float32)
        if values.size == 0:
            stats = {"first": 0.0, "last": 0.0, "slope": 0.0, "mean": 0.0, "std": 0.0, "volatility": "low"}
            return "No numerical history is available.", stats
        values = values.reshape(values.shape[0], -1)
        series = np.nanmean(values, axis=1)
        series = np.nan_to_num(series, nan=0.0)
        first = float(series[0])
        last = float(series[-1])
        slope = float((last - first) / max(1, len(series) - 1))
        mean = float(np.mean(series))
        std = float(np.std(series))
        abs_changes = np.abs(np.diff(series)) if len(series) > 1 else np.asarray([0.0])
        change_mean = float(np.mean(abs_changes))
        volatility = "high" if std > 1.0 or change_mean > 0.5 else "medium" if std > 0.25 or change_mean > 0.1 else "low"
        stats = {
            "first": first,
            "last": last,
            "slope": slope,
            "mean": mean,
            "std": std,
            "change_mean": change_mean,
            "volatility": volatility,
        }
        return (
            f"first={first:.4f}; last={last:.4f}; slope={slope:.4f}; "
            f"mean={mean:.4f}; std={std:.4f}; volatility={volatility}",
            stats,
        )

    def _summarize_num_without_numpy(self, history_values):
        series = []
        for row in history_values or []:
            if isinstance(row, (list, tuple)):
                vals = [float(v) for v in row if v is not None]
                series.append(sum(vals) / len(vals) if vals else 0.0)
            else:
                series.append(float(row))
        if not series:
            stats = {"first": 0.0, "last": 0.0, "slope": 0.0, "mean": 0.0, "std": 0.0, "volatility": "low"}
            return "No numerical history is available.", stats
        first = series[0]
        last = series[-1]
        slope = (last - first) / max(1, len(series) - 1)
        mean = sum(series) / len(series)
        std = math.sqrt(sum((value - mean) ** 2 for value in series) / len(series))
        changes = [abs(series[idx] - series[idx - 1]) for idx in range(1, len(series))]
        change_mean = sum(changes) / len(changes) if changes else 0.0
        volatility = "high" if std > 1.0 or change_mean > 0.5 else "medium" if std > 0.25 or change_mean > 0.1 else "low"
        stats = {
            "first": first,
            "last": last,
            "slope": slope,
            "mean": mean,
            "std": std,
            "change_mean": change_mean,
            "volatility": volatility,
        }
        return (
            f"first={first:.4f}; last={last:.4f}; slope={slope:.4f}; "
            f"mean={mean:.4f}; std={std:.4f}; volatility={volatility}",
            stats,
        )

    def _fallback_cot(self, stats):
        slope = stats.get("slope", 0.0)
        std = stats.get("std", 0.0)
        direction = "upward" if slope > 1e-3 else "downward" if slope < -1e-3 else "flat"
        strength = "strong" if abs(slope) > 0.05 else "moderate" if abs(slope) > 0.01 else "weak"
        volatility = stats.get("volatility", "low")
        return (
            f"direction={direction}; strength={strength}; volatility={volatility}; "
            f"rationale=history slope is {slope:.4f} and standard deviation is {std:.4f}."
        )

    def _load_cot_model(self):
        if self._cot_load_failed:
            return False
        if self._cot_model is not None and self._cot_tokenizer is not None:
            return True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = resolve_local_model_path(
                self.cot_model_name,
                aliases=("gpt2-medium",),
            )
            self._cot_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._cot_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
            )
            if self._cot_tokenizer.pad_token is None:
                self._cot_tokenizer.pad_token = self._cot_tokenizer.eos_token
            self._cot_model.eval()
            return True
        except Exception:
            self._cot_load_failed = True
            self._cot_model = None
            self._cot_tokenizer = None
            return False

    def _generate_cot(self, prompt, stats):
        if self.cot_generator is not None:
            try:
                generated = self.cot_generator(prompt)
                if generated:
                    return str(generated).strip()[:500]
            except Exception:
                pass
        if not self._load_cot_model():
            return self._fallback_cot(stats)
        try:
            import torch

            inputs = self._cot_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
            with torch.no_grad():
                output_ids = self._cot_model.generate(
                    **inputs,
                    max_new_tokens=self.cot_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._cot_tokenizer.pad_token_id,
                )
            generated = self._cot_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated = generated.strip()
            return generated[:500] if generated else self._fallback_cot(stats)
        except Exception:
            return self._fallback_cot(stats)

    def build_guidance_text(self, raw_text, history_values, desc="", search_frame=None):
        raw_text = str(raw_text or "").strip()
        raw_text_or_na = raw_text if raw_text and raw_text != "NA" else "NA"
        raw_text_short = raw_text_or_na[: self.raw_text_max_chars]
        num_summary, stats = self._summarize_num(history_values)
        corpus = self._extract_corpus(search_frame) if search_frame is not None else self.default_corpus

        query1 = "\n".join(part for part in [desc, raw_text_short, "[NUMERICAL SUMMARY]", num_summary] if part)
        evidence0 = self._retrieve(query1, self.rag_long_topn, corpus)
        evidence0 = self._longformer_rerank(query1, evidence0, self.rag_stage1_topk)

        cot_prompt = (
            "Generate a short forecasting trend hypothesis.\n"
            f"Numerical summary: {num_summary}\n"
            f"Raw text: {raw_text_short}\n"
            f"Retrieved evidence: {' '.join(evidence0)[:1200]}\n"
            "Return direction, strength, volatility, and rationale."
        )
        trend_hypothesis = self._generate_cot(cot_prompt, stats)

        query2 = (
            f"{query1}\n[TREND HYPOTHESIS]\n{trend_hypothesis}\n"
            "Retrieve evidence that best supports or explains this trend hypothesis."
        )
        evidence1 = self._retrieve(query2, self.rag_long_topn, corpus)
        evidence1 = self._longformer_rerank(query2, evidence1, self.rag_stage2_topk) or evidence0[: self.rag_stage2_topk]
        evidence_text = "\n".join(f"{idx + 1}) {text[:350]}" for idx, text in enumerate(evidence1)) or "NA"

        return (
            f"[TREND HYPOTHESIS]\n{trend_hypothesis}\n\n"
            f"[NUMERICAL SUMMARY]\n{num_summary}\n\n"
            f"[RETRIEVED EVIDENCE - REFINED]\n{evidence_text}\n\n"
            f"[RAW TEXT]\n{raw_text_short}"
        )
