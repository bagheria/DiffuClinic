
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utilities.entity_selector import DEFAULT_IDF_PATH, EntitySelector

_TRAIN_FULLTEXT_DIRS = [
    os.path.join(_REPO_ROOT, "Data", "multiclinsum_gs_train_en", "fulltext"),
    os.path.join(_REPO_ROOT, "Data", "multiclinsum_large-scale_train_en", "fulltext"),
]


def load_corpus(directories: list[str]) -> list[str]:
    corpus: list[str] = []
    for directory in directories:
        names = sorted(name for name in os.listdir(directory) if name.endswith(".txt"))
        for name in names:
            with open(os.path.join(directory, name), "r", encoding="utf-8") as handle:
                corpus.append(handle.read())
        print(f"Loaded {len(names)} docs from {directory}")
    return corpus


def main() -> None:
    corpus = load_corpus(_TRAIN_FULLTEXT_DIRS)
    print(f"Total training documents: {len(corpus)}")

    selector = EntitySelector()
    selector.fit(corpus, show_progress=True)
    selector.save(DEFAULT_IDF_PATH)

    distinct = len(selector.doc_freq)
    singletons = sum(1 for freq in selector.doc_freq.values() if freq == 1)
    top10 = sorted(selector.doc_freq.items(), key=lambda kv: kv[1], reverse=True)[:10]

    print("\n=== IDF refit summary ===")
    print(f"Saved to:           {DEFAULT_IDF_PATH}")
    print(f"Documents (n_docs): {selector.n_docs}")
    print(f"Distinct diseases:  {distinct}")
    print(f"Singletons (df==1): {singletons} ({singletons / distinct:.1%} of distinct)")
    print("Top 10 most common diseases (document frequency):")
    for name, freq in top10:
        print(f"  df={freq:>5}  {name}")


if __name__ == "__main__":
    main()
