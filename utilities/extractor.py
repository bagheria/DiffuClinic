
import importlib.util
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import spacy
from scispacy.linking import EntityLinker


class ClinicalEntities(TypedDict):
    numbers: list[str]
    gender_terms: list[str]
    medications: list[str]
    diseases: list[str]


_BC5CDR_MODEL = "en_ner_bc5cdr_md"

_MEDICATION_TUIS = frozenset({"T121", "T200", "T195", "T125", "T129", "T127"})


def _ensure_bc5cdr_config_bools() -> None:
    # spaCy 3.8 rejects the quoted booleans the BC5CDR config ships, rewrite them to real bools
    try:
        spec = importlib.util.find_spec(_BC5CDR_MODEL)
        if spec is None or not spec.submodule_search_locations:
            return
        for cfg in Path(spec.submodule_search_locations[0]).rglob("config.cfg"):
            original = cfg.read_text(encoding="utf-8")
            patched = original.replace('= "True"', "= true").replace('= "False"', "= false")
            if patched != original:
                cfg.write_text(patched, encoding="utf-8")
    except Exception:
        return


@lru_cache(maxsize=1)
def _load_pipeline():
    _ensure_bc5cdr_config_bools()
    nlp = spacy.load(_BC5CDR_MODEL)
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp


def _linker():
    return _load_pipeline().get_pipe("scispacy_linker")


def _canonical_name(ent) -> str:
    if ent._.kb_ents:
        cui, _score = ent._.kb_ents[0]
        return _linker().kb.cui_to_entity[cui].canonical_name.lower()
    return ent.text.lower().strip()


def _is_medication(ent) -> bool:
    if not ent._.kb_ents:
        return True
    cui, _score = ent._.kb_ents[0]
    tuis = set(_linker().kb.cui_to_entity[cui].types)
    return bool(tuis & _MEDICATION_TUIS)


_DOSAGE_UNITS = r"(?:mg|mcg|µg|ug|g|ml|l|mmol|µmol|umol|iu|meq|ng|pg|units?)"
_PER_UNITS    = r"(?:kg|day|dose|hr|hour|min|ml|l|dl|week)"

_DOSAGE_PATTERN = re.compile(rf"\b\d+(?:[.,]\d+)?\s*{_DOSAGE_UNITS}(?:/{_PER_UNITS})?", re.IGNORECASE)

_AGE_PATTERN = re.compile(r"\b(\d{1,3})[- ]year[- ]old\b" r"|\baged?\s+(\d{1,3})\b" r"|\b(\d{1,3})\s+years?\s+old\b", re.IGNORECASE)

_VITAL_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?" r"\s*(?:mmHg|bpm|°C|°F|%)(?!\w)", re.IGNORECASE)

_GENDER_PATTERN = re.compile(r"\b(male|female|man|woman|boy|girl|gentleman|lady|" r"he|she|his|her|him|himself|herself|" r"transgender|non-binary|intersex)\b", re.IGNORECASE)

_GENDER_CANONICAL = {
    "he": "male", "his": "male", "him": "male", "himself": "male",
    "man": "male", "boy": "male", "gentleman": "male", "male": "male",
    "she": "female", "her": "female", "herself": "female",
    "woman": "female", "girl": "female", "lady": "female", "female": "female",
    "transgender": "transgender",
    "non-binary": "non-binary",
    "intersex": "intersex",
}


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_NUMBER_WORDS_RE = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

_SPELLED_AGE_PATTERN = re.compile(rf"\b((?:{_NUMBER_WORDS_RE})(?:[- ](?:{_NUMBER_WORDS_RE}))?)" r"[- ]years?[- ]old\b", re.IGNORECASE)


def _spelled_age_to_digit(match: re.Match) -> str:
    parts = re.split(r"[- ]", match.group(1).lower())
    return f"{sum(_NUMBER_WORDS[p] for p in parts)}-year-old"


def _canonical_age(match: re.Match) -> str:
    digit = match.group(1) or match.group(2) or match.group(3)
    return f"{int(digit)}-year-old"


def _canonical_number(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r",(\d)", r".\1", text)
    text = re.sub(r"(\d)([a-zµ°%])", r"\1 \2", text, flags=re.IGNORECASE)
    return text.lower()


def extract_numbers(text: str) -> list[str]:
    matches = {_canonical_number(match.group(0)) for pattern in (_DOSAGE_PATTERN, _VITAL_PATTERN) for match in pattern.finditer(text)}
    matches.update(_canonical_age(m) for m in _AGE_PATTERN.finditer(text))
    matches.update(_spelled_age_to_digit(m) for m in _SPELLED_AGE_PATTERN.finditer(text))
    return sorted(matches)


def extract_gender_terms(text: str) -> list[str]:
    matches = {_GENDER_CANONICAL[match.group(0).lower()] for match in _GENDER_PATTERN.finditer(text)}

    return sorted(matches)


def _attach_dosage(text_lower: str, start: int, end: int, drug: str) -> str:
    window_start = max(0, start - 60)
    window = text_lower[window_start:min(len(text_lower), end + 60)]

    closest_dosage = None
    closest_distance = float("inf")
    for match in _DOSAGE_PATTERN.finditer(window):
        absolute_start = window_start + match.start()
        absolute_end = window_start + match.end()
        distance = max(start - absolute_end, absolute_start - end, 0)
        if distance < closest_distance:
            closest_distance = distance
            closest_dosage = match.group(0).strip()

    return f"{drug} {closest_dosage}" if closest_dosage else drug


def extract_medications_and_diseases(text: str) -> tuple[list[str], list[str]]:
    doc = _load_pipeline()(text)
    text_lower = text.lower()

    medications = set()
    diseases = set()

    for ent in doc.ents:
        if not ent.text.strip():
            continue
        canonical = _canonical_name(ent)
        if ent.label_ == "CHEMICAL" and _is_medication(ent):
            medications.add(_attach_dosage(text_lower, ent.start_char, ent.end_char, canonical))
        elif ent.label_ == "DISEASE":
            diseases.add(canonical)

    return sorted(medications), sorted(diseases)


def extract_medications(text: str) -> list[str]:
    medications, _ = extract_medications_and_diseases(text)
    return medications


def extract_diseases(text: str) -> list[str]:
    _, diseases = extract_medications_and_diseases(text)
    return diseases


def extract_all(text: str) -> ClinicalEntities:
    medications, diseases = extract_medications_and_diseases(text)
    return ClinicalEntities(numbers=extract_numbers(text), gender_terms=extract_gender_terms(text), medications=medications, diseases=diseases)