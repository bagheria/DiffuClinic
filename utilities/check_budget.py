
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utilities.entity_selector import DEFAULT_IDF_PATH, EntitySelector
from utilities.extractor import _load_pipeline, extract_all

_GS_DIR = os.path.join(_REPO_ROOT, "Data", "multiclinsum_gs_train_en")
_CATEGORIES = ("numbers", "gender_terms", "medications", "diseases")

_GEN_LENGTH = 128

_BASELINE_FULL_COVER = 0.12
_BASELINE_MEAN_COVER = 0.52


def load_gs_fulltexts() -> list[str]:
    fulltext_dir = os.path.join(_GS_DIR, "fulltext")
    texts: list[str] = []
    for name in sorted(n for n in os.listdir(fulltext_dir) if n.endswith(".txt")):
        with open(os.path.join(fulltext_dir, name), "r", encoding="utf-8") as handle:
            texts.append(handle.read())
    return texts


def main() -> None:
    texts = load_gs_fulltexts()
    print(f"Loaded {len(texts)} GS fulltext documents")

    selector = EntitySelector.load(DEFAULT_IDF_PATH)
    _load_pipeline()

    full_cover = 0
    coverages: list[float] = []
    overflow_coverages: list[float] = []

    for text in texts:
        entities = extract_all(text)
        total = sum(len(entities[c]) for c in _CATEGORIES)
        if total == 0:
            continue
        selected = selector.select(entities=entities, source_text=text, gen_length=_GEN_LENGTH, tokenizer=None)
        coverage = len(selected) / total
        coverages.append(coverage)
        if len(selected) == total:
            full_cover += 1
        else:
            overflow_coverages.append(coverage)

    n = len(coverages)
    full_frac = full_cover / n
    mean_cover = sum(coverages) / n
    mean_overflow = (sum(overflow_coverages) / len(overflow_coverages) if overflow_coverages else 0.0)

    print("\n=== Section 5.4 budget analysis (post-fix) ===")
    print(f"Budget base gen_length:    {_GEN_LENGTH} tokens")
    print(f"Entity-token budget (30%): {int(0.3 * _GEN_LENGTH)} tokens")
    print(f"Documents scored:          {n}")
    print(f"Fully covered:             {full_cover}/{n} ({full_frac:.0%})")
    print(f"Mean coverage (all):       {mean_cover:.0%}")
    print(f"Mean coverage (overflow):  {mean_overflow:.0%}")

    print("\n=== Delta vs pre-fix baseline ===")
    print(f"Fully covered: {_BASELINE_FULL_COVER:.0%} -> {full_frac:.0%} " f"({full_frac - _BASELINE_FULL_COVER:+.0%})")
    print(f"Mean coverage: {_BASELINE_MEAN_COVER:.0%} -> {mean_cover:.0%} " f"({mean_cover - _BASELINE_MEAN_COVER:+.0%})")


if __name__ == "__main__":
    main()
