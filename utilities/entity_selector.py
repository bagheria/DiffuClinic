
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from typing import Optional

from utilities.extractor import (ClinicalEntities, _DOSAGE_PATTERN, extract_all, extract_diseases)

logger = logging.getLogger(__name__)

DEFAULT_IDF_PATH = "utilities/idf_weights.json"

_AGE_FORM = re.compile(r"^\d{1,3}-year-old$")

_VITAL_MARKERS = ("mmhg", "bpm", "°c", "°f", "%")


class EntitySelector:

    def __init__(self, max_medications: int = 4, max_diseases: int = 8, budget_fraction: float = 0.3, recency_threshold: float = 0.75, recency_multiplier: float = 2.0, idf_smoothing: float = 1.0, max_entity_tokens: int = 10) -> None:
        self.max_medications = max_medications
        self.max_diseases = max_diseases
        self.budget_fraction = budget_fraction
        self.recency_threshold = recency_threshold
        self.recency_multiplier = recency_multiplier
        self.idf_smoothing = idf_smoothing
        self.max_entity_tokens = max_entity_tokens

        self.n_docs: int = 0
        self.doc_freq: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._idf_unseen: float = 1.0


    def fit(self, corpus: list[str], precomputed_diseases: Optional[list[list[str]]] = None, show_progress: bool = True) -> "EntitySelector":
        n_docs = len(corpus) if precomputed_diseases is None else len(precomputed_diseases)
        doc_freq: Counter[str] = Counter()

        if precomputed_diseases is not None:
            for diseases in precomputed_diseases:
                for disease in set(diseases):
                    doc_freq[disease] += 1
        else:
            iterator = corpus
            if show_progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(corpus, desc="Fitting disease IDF")
                except ImportError:
                    logger.debug("tqdm unavailable, fitting without a progress bar")
            for doc in iterator:
                for disease in set(extract_diseases(doc)):
                    doc_freq[disease] += 1

        self.n_docs = n_docs
        self.doc_freq = dict(doc_freq)
        self._compute_idf()
        logger.debug("Fitted IDF over %d docs, %d distinct diseases", n_docs, len(doc_freq))
        return self

    def _compute_idf(self) -> None:
        if self.n_docs <= 0:
            self._idf = {}
            self._idf_unseen = 1.0
            return

        s = self.idf_smoothing
        max_raw = math.log((self.n_docs + s) / s)
        self._idf = {entity: math.log((self.n_docs + s) / (df + s)) / max_raw for entity, df in self.doc_freq.items()}
        self._idf_unseen = 1.0

    def _idf_for(self, disease: str) -> float:
        return self._idf.get(disease, self._idf_unseen)

    @property
    def normalized_idf(self) -> dict[str, float]:
        return dict(self._idf)

    def save(self, path: str = DEFAULT_IDF_PATH) -> None:
        payload = {
            "n_docs": self.n_docs,
            "idf_smoothing": self.idf_smoothing,
            "doc_freq": self.doc_freq,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        logger.debug("Saved IDF table to %s", path)

    @classmethod
    def load(cls, path: str = DEFAULT_IDF_PATH, **kwargs) -> "EntitySelector":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        kwargs.setdefault("idf_smoothing", payload.get("idf_smoothing", 1.0))
        selector = cls(**kwargs)
        selector.n_docs = payload["n_docs"]
        selector.doc_freq = {k: int(v) for k, v in payload["doc_freq"].items()}
        selector._compute_idf()
        logger.debug("Loaded IDF table from %s (%d diseases)", path, len(selector.doc_freq))
        return selector


    @staticmethod
    def _med_core(med: str) -> str:
        match = _DOSAGE_PATTERN.search(med)
        core = med[: match.start()] if match else med
        return core.strip()

    @staticmethod
    def _tf(core: str, source_lower: str, n_words: int) -> float:
        if not core:
            return 0.0
        count = source_lower.count(core.lower())
        if count == 0:
            count = 1
        return count / max(1, n_words)

    def _recency_weight(self, core: str, source_lower: str) -> float:
        if not core or not source_lower:
            return 1.0
        idx = source_lower.rfind(core.lower())
        if idx < 0:
            return 1.0
        position = idx / len(source_lower)
        return self.recency_multiplier if position >= self.recency_threshold else 1.0


    @staticmethod
    def _is_age(number: str) -> bool:
        return bool(_AGE_FORM.match(number))

    @staticmethod
    def _is_vital(number: str) -> bool:
        return any(marker in number for marker in _VITAL_MARKERS)


    @staticmethod
    def _count_tokens(text: str, tokenizer) -> int:
        if tokenizer is None:
            return max(1, len(text.split()))
        return len(tokenizer.encode(text, add_special_tokens=False))


    def select(self, entities: ClinicalEntities, source_text: str, gen_length: int = 128, tokenizer=None) -> list[str]:
        source_lower = source_text.lower()
        n_words = max(1, len(source_text.split()))
        budget = int(self.budget_fraction * gen_length)

        selected: list[str] = []
        used = 0

        def charge(item: str) -> None:
            nonlocal used
            used += self._count_tokens(item, tokenizer)
            selected.append(item)

        def fits(item: str) -> bool:
            return used + self._count_tokens(item, tokenizer) <= budget

        numbers = entities["numbers"]

        meds = [m for m in entities["medications"] if self._count_tokens(m, tokenizer) <= self.max_entity_tokens]
        diseases = [d for d in entities["diseases"] if self._count_tokens(d, tokenizer) <= self.max_entity_tokens]
        n_filtered = (len(entities["medications"]) - len(meds)) + (len(entities["diseases"]) - len(diseases))
        if n_filtered:
            logger.debug("Filtered %d entities exceeding max_entity_tokens=%d", n_filtered, self.max_entity_tokens)

        ages = [n for n in numbers if self._is_age(n)]
        if ages:
            best_age = max(ages, key=lambda a: self._tf(a, source_lower, n_words))
            charge(best_age)

        genders = entities["gender_terms"]
        if genders:
            best_gender = max(genders, key=lambda g: self._tf(g, source_lower, n_words))
            charge(best_gender)

        med_scored = []
        for med in meds:
            core = self._med_core(med)
            score = self._tf(core, source_lower, n_words) * self._recency_weight(core, source_lower)
            med_scored.append((score, med))
        med_scored.sort(key=lambda pair: pair[0], reverse=True)
        for _score, med in med_scored[: self.max_medications]:
            if fits(med):
                charge(med)

        disease_scored = []
        for disease in diseases:
            score = (self._tf(disease, source_lower, n_words) * self._recency_weight(disease, source_lower) * self._idf_for(disease))
            disease_scored.append((score, disease))
        disease_scored.sort(key=lambda pair: pair[0], reverse=True)
        for _score, disease in disease_scored[: self.max_diseases]:
            if fits(disease):
                charge(disease)

        non_age = [n for n in numbers if not self._is_age(n)]
        vitals = [n for n in non_age if self._is_vital(n)]
        dosages = [n for n in non_age if not self._is_vital(n)]
        for number in vitals + dosages:
            if any(number in item for item in selected):
                continue
            if fits(number):
                charge(number)

        logger.debug("Selected %d entities, %d/%d budget tokens used", len(selected), used, budget)
        return selected


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    _SAMPLE = (
        "A 49-year-old man presented with chest pain and a blood pressure of "
        "150/95 mmHg. He has a history of type 2 diabetes mellitus and "
        "hypertension. He was started on metformin 500 mg and aspirin 100 mg. "
        "On discharge the diagnosis was acute myocardial infarction and he was "
        "prescribed atorvastatin 40 mg for hyperlipidemia."
    )

    _CORPUS = [
        "patient with diabetes mellitus and hypertension",
        "patient with diabetes mellitus and asthma",
        "patient with diabetes mellitus and chest pain",
        "patient with acute myocardial infarction",
    ]

    selector = EntitySelector()
    selector.fit(_CORPUS, show_progress=False)

    result = selector.select(entities=extract_all(_SAMPLE), source_text=_SAMPLE, gen_length=128, tokenizer=None)
    logger.info("Selected entities (priority order): %s", result)
