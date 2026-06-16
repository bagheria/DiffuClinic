
import re
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utilities.extractor import (extract_gender_terms, extract_numbers, extract_medications_and_diseases)


DATA_DIR     = Path("Data/multiclinsum_gs_train_en")
FULLTEXT_DIR = DATA_DIR / "fulltext"
SUMMARY_DIR  = DATA_DIR / "summaries"

N_EXAMPLES   = 5
WRAP_WIDTH   = 74
AGE_TOLERANCE = 2


_DOSAGE_CAPTURE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(mg|mcg|µg|ug|g|ml|l|mmol|µmol|umol|iu|meq|ng|pg|units?|kg)\b", re.IGNORECASE)


_SEP  = "=" * 80
_THIN = "─" * 80


def _load_pairs() -> list[tuple[str, str, str]]:
    pairs = []
    for ft_path in sorted(FULLTEXT_DIR.glob("*.txt")):
        stem = ft_path.stem
        sum_path = SUMMARY_DIR / f"{stem}_sum.txt"
        if not sum_path.exists():
            continue
        fulltext = ft_path.read_text(encoding="utf-8", errors="replace")
        summary  = sum_path.read_text(encoding="utf-8", errors="replace")
        pairs.append((stem, fulltext, summary))
    return pairs


def _find_sentence(text: str, term: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    term_lower = term.lower()
    for sent in sentences:
        if term_lower in sent.lower():
            return sent.strip()
    return "(sentence not isolated)"


def _gender_sentence(text: str, canonical: str) -> str:
    surface_map = {
        "male":   ["male", "man", "boy", "gentleman", " he ", " his ", " him "],
        "female": ["female", "woman", "girl", "lady", " she ", " her ", " herself"],
    }
    for surface in surface_map.get(canonical, [canonical]):
        sent = _find_sentence(text, surface)
        if sent != "(sentence not isolated)":
            return sent
    return "(sentence not isolated)"


def _wrap(label: str, text: str) -> str:
    lines = textwrap.wrap(text, width=WRAP_WIDTH - len(label) - 4)
    if not lines:
        return f"{label}: (empty)"
    indent = " " * (len(label) + 4)
    first = f"{label}: \"{lines[0]}"
    rest  = [f"{indent}{l}" for l in lines[1:]]
    closing = (rest[-1] if rest else first) + "\""
    if rest:
        rest[-1] = closing
        return "\n".join([first] + rest)
    return first + "\""


def _check_gender(stem, fulltext, summary) -> dict | None:
    ft_gender  = set(extract_gender_terms(fulltext))
    sum_gender = set(extract_gender_terms(summary))
    sum_only_male   = "male"   in sum_gender and "male"   not in ft_gender and "female" in ft_gender
    sum_only_female = "female" in sum_gender and "female" not in ft_gender and "male"   in ft_gender
    if not (sum_only_male or sum_only_female):
        return None
    ft_canon  = "female" if sum_only_male else "male"
    sum_canon = "male"   if sum_only_male else "female"
    return {
        "stem":       stem,
        "ft_gender":  ft_gender,
        "sum_gender": sum_gender,
        "ft_sent":    _gender_sentence(fulltext, ft_canon),
        "sum_sent":   _gender_sentence(summary,  sum_canon),
    }


def _extract_ages(text: str) -> list[int]:
    return [int(m.group(1)) for v in extract_numbers(text) if (m := re.match(r"^(\d+)-year-old$", v))]


def _age_sentence(text: str, age: int) -> str:
    for pattern in [f"{age}-year-old", f"{age} year", f"aged {age}", f"age {age}"]:
        sent = _find_sentence(text, pattern)
        if sent != "(sentence not isolated)":
            return sent
    return "(sentence not isolated)"


def _check_age(stem, fulltext, summary) -> dict | None:
    ft_ages  = _extract_ages(fulltext)
    sum_ages = _extract_ages(summary)
    if not ft_ages or not sum_ages:
        return None
    mismatched = [sa for sa in sum_ages if all(abs(sa - fa) > AGE_TOLERANCE for fa in ft_ages)]
    if not mismatched:
        return None
    sa = mismatched[0]
    return {
        "stem":     stem,
        "ft_ages":  ft_ages,
        "sum_ages": sum_ages,
        "example":  sa,
        "ft_sent":  _age_sentence(fulltext, ft_ages[0]),
        "sum_sent": _age_sentence(summary, sa),
    }


def _dosage_by_unit(text: str) -> dict[str, set[float]]:
    result: dict[str, set[float]] = {}
    for m in _DOSAGE_CAPTURE.finditer(text):
        val  = float(m.group(1).replace(",", "."))
        unit = m.group(2).lower().rstrip("s")
        result.setdefault(unit, set()).add(val)
    return result


def _check_numeric_change(stem, fulltext, summary) -> dict | None:
    ft_by_unit  = _dosage_by_unit(fulltext)
    sum_by_unit = _dosage_by_unit(summary)
    changes = []
    for unit, sum_vals in sum_by_unit.items():
        if unit in ft_by_unit:
            for v in sum_vals:
                if v not in ft_by_unit[unit]:
                    changes.append({"sum_val":  v, "unit":     unit, "ft_vals":  sorted(ft_by_unit[unit])})
    if not changes:
        return None
    ex = changes[0]
    val_str = f"{ex['sum_val']:g} {ex['unit']}"
    ft_val_str = f"{ex['ft_vals'][0]:g} {ex['unit']}"
    return {
        "stem":       stem,
        "changes":    changes,
        "example":    ex,
        "sum_sent":   _find_sentence(summary,  val_str),
        "ft_sent":    _find_sentence(fulltext, ft_val_str),
    }


def _drug_name(entry: str) -> str:
    return re.sub(r"\s+\d.*$", "", entry).strip()


def _check_medications(stem, fulltext, summary) -> dict | None:
    ft_meds, _ = extract_medications_and_diseases(fulltext)
    sum_meds, _ = extract_medications_and_diseases(summary)
    if not sum_meds:
        return None
    ft_drug_names = {_drug_name(m) for m in ft_meds}
    ft_lower = fulltext.lower()
    absent = []
    for entry in sum_meds:
        drug = _drug_name(entry)
        if drug not in ft_drug_names and drug.lower() not in ft_lower:
            absent.append(drug)
    if not absent:
        return None
    ex = absent[0]
    return {
        "stem":      stem,
        "absent":    absent,
        "example":   ex,
        "sum_sent":  _find_sentence(summary, ex),
        "ft_meds":   sorted(ft_drug_names),
    }


def _analyse(pairs):
    gender_flags  = []
    age_flags     = []
    numeric_flags = []
    med_flags     = []
    any_flag_ids  = set()

    print(f"Running checks on {len(pairs)} pairs …", flush=True)
    print("(Medication check loads scispaCy — first pair will be slow ~40s)", flush=True)

    for i, (stem, fulltext, summary) in enumerate(pairs, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i:>3}/{len(pairs)}] processing …", flush=True)

        g = _check_gender(stem, fulltext, summary)
        if g:
            gender_flags.append(g)
            any_flag_ids.add(stem)

        a = _check_age(stem, fulltext, summary)
        if a:
            age_flags.append(a)
            any_flag_ids.add(stem)

        nc = _check_numeric_change(stem, fulltext, summary)
        if nc:
            numeric_flags.append(nc)
            any_flag_ids.add(stem)

        m = _check_medications(stem, fulltext, summary)
        if m:
            med_flags.append(m)
            any_flag_ids.add(stem)

    return gender_flags, age_flags, numeric_flags, med_flags, any_flag_ids


def _print_gender_section(flags):
    n = min(N_EXAMPLES, len(flags))
    print(f"\n{_SEP}")
    print(f"GENDER CONTRADICTIONS — {n} EXAMPLE{'S' if n != 1 else ''}")
    print(_SEP)
    for i, f in enumerate(flags[:N_EXAMPLES], 1):
        ft_lbl  = " | ".join(sorted(f["ft_gender"]))
        sum_lbl = " | ".join(sorted(f["sum_gender"]))
        print(f"\n[{i}]  {f['stem']}")
        print(f"     Fulltext gender : {ft_lbl}   |   Summary gender : {sum_lbl}")
        print()
        print("     " + _wrap("SOURCE ", f["ft_sent"]))
        print("     " + _wrap("SUMMARY", f["sum_sent"]))


def _print_age_section(flags):
    n = min(N_EXAMPLES, len(flags))
    print(f"\n{_SEP}")
    print(f"AGE MISMATCHES (> {AGE_TOLERANCE} yr tolerance) — {n} EXAMPLE{'S' if n != 1 else ''}")
    print(_SEP)
    for i, f in enumerate(flags[:N_EXAMPLES], 1):
        print(f"\n[{i}]  {f['stem']}")
        print(f"     Source ages : {f['ft_ages']}   |   Summary ages : {f['sum_ages']}")
        print(f"     Mismatched summary age : {f['example']}")
        print()
        print("     " + _wrap("SOURCE ", f["ft_sent"]))
        print("     " + _wrap("SUMMARY", f["sum_sent"]))


def _print_numeric_section(flags):
    n = min(N_EXAMPLES, len(flags))
    print(f"\n{_SEP}")
    print(f"NUMERIC VALUE CHANGES — {n} EXAMPLE{'S' if n != 1 else ''}")
    print(_SEP)
    for i, f in enumerate(flags[:N_EXAMPLES], 1):
        ex = f["example"]
        ft_vals_str = ", ".join(f"{v:g}" for v in ex["ft_vals"])
        print(f"\n[{i}]  {f['stem']}")
        print(f"     Unit : {ex['unit']}   |   " f"Summary value : {ex['sum_val']:g}   |   Source value(s) : {ft_vals_str}")
        print()
        print("     " + _wrap("SOURCE ", f["ft_sent"]))
        print("     " + _wrap("SUMMARY", f["sum_sent"]))


def _print_med_section(flags):
    n = min(N_EXAMPLES, len(flags))
    print(f"\n{_SEP}")
    print(f"MEDICATION HALLUCINATIONS — {n} EXAMPLE{'S' if n != 1 else ''}")
    print(_SEP)
    for i, f in enumerate(flags[:N_EXAMPLES], 1):
        absent_str = ", ".join(f'"{d}"' for d in f["absent"])
        ft_str = ", ".join(f'"{d}"' for d in f["ft_meds"][:6])
        if len(f["ft_meds"]) > 6:
            ft_str += ", …"
        print(f"\n[{i}]  {f['stem']}")
        print(f"     Absent drug(s)  : {absent_str}")
        print(f"     Source drug(s)  : {ft_str if ft_str else '(none detected)'}")
        print()
        print("     " + _wrap("SUMMARY", f["sum_sent"]))
        print("     SOURCE : (drug absent from source text)")


def main():
    pairs = _load_pairs()
    n = len(pairs)
    if n == 0:
        print(f"ERROR: no pairs found under {DATA_DIR}")
        sys.exit(1)

    gender_flags, age_flags, numeric_flags, med_flags, any_ids = _analyse(pairs)

    ng  = len(gender_flags)
    na  = len(age_flags)
    nnc = len(numeric_flags)
    nm  = len(med_flags)
    nany = len(any_ids)

    def pct(k): return k / n * 100

    print(f"\n{_SEP}")
    print("MULTICLINSUM GS TRAIN — FULL ENTITY CONSISTENCY AUDIT")
    print(f"Dataset : {DATA_DIR.resolve()}")
    print(_SEP)
    print()
    print("STATISTICS")
    print(_THIN)
    print(f"Total pairs checked              : {n:>5}")
    print(f"Gender contradictions            : {ng:>5}  ({pct(ng):5.2f} %)")
    print(f"Age mismatches (> {AGE_TOLERANCE} yr)          : {na:>5}  ({pct(na):5.2f} %)")
    print(f"Numeric value changes            : {nnc:>5}  ({pct(nnc):5.2f} %)")
    print(f"Medication hallucinations        : {nm:>5}  ({pct(nm):5.2f} %)")
    print(_THIN)
    print(f"Pairs with ANY inconsistency     : {nany:>5}  ({pct(nany):5.2f} %)")
    print()

    if gender_flags:
        _print_gender_section(gender_flags)
    if age_flags:
        _print_age_section(age_flags)
    if numeric_flags:
        _print_numeric_section(numeric_flags)
    if med_flags:
        _print_med_section(med_flags)

    print(f"\n{_SEP}")
    print("END OF REPORT")
    print(_SEP)


if __name__ == "__main__":
    main()
