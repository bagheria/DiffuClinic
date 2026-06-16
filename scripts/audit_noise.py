#!/usr/bin/env python3
"""
Entity consistency audit v2: stricter filtering to approximate the paper's 9.80% finding.
Focus on patient-level entity contradictions, not family members / literature refs.
"""
import zipfile, re
from collections import defaultdict

def get_patient_gender(text):
    """Get the PRIMARY patient gender from first few sentences.
    Clinical case reports typically state patient gender in the first 1-2 sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text[:800])
    for sent in sentences[:3]:
        s = sent.lower()
        # Direct patient descriptors
        if re.search(r'\b(female|woman|girl|lady)\b.*\bpatient\b', s) or \
           re.search(r'\bpatient\b.*\b(female|woman|girl)\b', s):
            return 'female'
        if re.search(r'\b(male|man|boy|gentleman)\b.*\bpatient\b', s) or \
           re.search(r'\bpatient\b.*\b(male|man|boy)\b', s):
            return 'male'
        # "X-year-old woman/man" pattern (almost always the patient)
        m = re.search(r'(\d{1,3})[- ]year[- ]old\s+(woman|man|female|male|boy|girl|lady|gentleman)', s)
        if m:
            g = m.group(2)
            if g in ('woman', 'female', 'girl', 'lady'): return 'female'
            if g in ('man', 'male', 'boy', 'gentleman'): return 'male'
        # "A XX-year-old woman/man presented/admitted" (typical case report opening)
        m = re.search(r'A\s+\d{1,3}[- ]year[- ]old\s+(woman|man|female|male)', s)
        if m:
            return 'female' if m.group(1) in ('woman', 'female') else 'male'
    return None

def get_patient_age(text):
    """Get the PRIMARY patient age from early text."""
    sentences = re.split(r'(?<=[.!?])\s+', text[:600])
    for sent in sentences[:3]:
        # "X-year-old" near "patient/presented/admitted/man/woman"
        m = re.search(r'(\d{1,3})[- ]year[- ]old\s+(?:woman|man|female|male|patient|who|presented|admitted|was)', sent.lower())
        if m:
            return int(m.group(1))
        # "aged X" near patient descriptors
        m = re.search(r'aged\s+(\d{1,3})\s*(?:year|yr)', sent.lower())
        if m:
            return int(m.group(1))
    return None

def audit_pair_v2(fulltext, summary, pair_id):
    """Stricter audit: patient-level only."""
    issues = []

    # ── Gender ──
    src_g = get_patient_gender(fulltext)
    ref_g = get_patient_gender(summary)
    if src_g and ref_g and src_g != ref_g:
        issues.append({'type': 'gender', 'source': src_g, 'reference': ref_g})

    # ── Age ──
    src_age = get_patient_age(fulltext)
    ref_age = get_patient_age(summary)
    if src_age and ref_age and abs(src_age - ref_age) > 2:
        issues.append({'type': 'age', 'source_age': src_age, 'ref_age': ref_age})

    # ── Medication hallucination ──
    # Check if summary mentions a medication NEVER mentioned in source
    common_meds = {
        'aspirin', 'ibuprofen', 'paracetamol', 'acetaminophen', 'morphine',
        'metformin', 'insulin', 'warfarin', 'heparin', 'prednisone',
        'dexamethasone', 'omeprazole', 'lisinopril', 'amlodipine',
        'metoprolol', 'atorvastatin', 'furosemide', 'levothyroxine',
        'gabapentin', 'sertraline', 'fluoxetine', 'ciprofloxacin',
        'amoxicillin', 'azithromycin', 'doxycycline', 'vancomycin',
        'ceftriaxone', 'propofol', 'midazolam', 'lidocaine', 'fentanyl',
        'ketamine', 'epinephrine', 'digoxin', 'amiodarone', 'clopidogrel',
        'apixaban', 'rivaroxaban', 'methotrexate', 'cyclophosphamide',
        'rituximab', 'bevacizumab', 'pembrolizumab', 'nivolumab',
        'cisplatin', 'carboplatin', 'paclitaxel', 'docetaxel',
        'doxorubicin', 'tamoxifen', 'imatinib', 'lenalidomide',
    }
    src_words = set(re.findall(r'\b\w+\b', fulltext.lower()))
    # Check summary for meds + dosages not in source
    ref_meds = re.findall(r'\b(\d+\.?\d*\s*(?:mg|g|mcg|µg|ml|mmol)(?:\s*(?:PO|IV|IM|SC|QD|BID|TID|QID))?)\s+(\w+)', summary.lower())
    for dosage, word in ref_meds:
        if word.lower() in common_meds and word.lower() not in src_words:
            issues.append({'type': 'medication', 'med': word, 'dosage': dosage})

    return issues


def audit_zip_v2(zip_path, name):
    print(f"\n{'='*60}")
    print(f"Auditing: {name}")
    print(f"{'='*60}")

    z = zipfile.ZipFile(zip_path)
    all_files = z.namelist()
    ft_files = sorted([f for f in all_files if '/fulltext/' in f and f.endswith('.txt')])

    results = {}
    type_counts = defaultdict(int)
    total_pairs = 0
    noisy_pairs = 0

    for ft_file in ft_files:
        total_pairs += 1
        sum_file = ft_file.replace('/fulltext/', '/summaries/').replace('.txt', '_sum.txt')
        if sum_file not in all_files:
            continue

        pair_id = ft_file.split('/')[-1].replace('.txt', '')
        fulltext = z.read(ft_file).decode('utf-8', errors='replace')
        summary = z.read(sum_file).decode('utf-8', errors='replace')

        issues = audit_pair_v2(fulltext, summary, pair_id)
        if issues:
            noisy_pairs += 1
            results[pair_id] = issues
            for iss in issues:
                type_counts[iss['type']] += 1

    print(f"\nTotal pairs: {total_pairs}")
    print(f"Pairs with ≥1 inconsistency: {noisy_pairs} ({100*noisy_pairs/total_pairs:.2f}%)")
    print(f"\nInconsistency breakdown:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    # Show all examples
    print(f"\n{'─'*40}")
    for check_type in ['gender', 'age', 'medication']:
        examples = [(pid, issues) for pid, issues in results.items()
                    if any(i['type'] == check_type for i in issues)]
        if examples:
            print(f"\n  [{check_type.upper()}] ({len(examples)} pairs)")
            for pid, issues in examples[:8]:
                for iss in issues:
                    if iss['type'] != check_type: continue
                    if check_type == 'gender':
                        print(f"    {pid}: src={iss['source']} → ref={iss['reference']}")
                    elif check_type == 'age':
                        print(f"    {pid}: src_age={iss['source_age']} → ref_age={iss['ref_age']}")
                    elif check_type == 'medication':
                        print(f"    {pid}: '{iss['med']}' ({iss['dosage']}) NOT in source")

    return results, type_counts, total_pairs, noisy_pairs


base = r'c:\Users\leeze\Documents\GitHub\DiffuClinic\data'

# GS
gs_results, gs_types, gs_total, gs_noisy = audit_zip_v2(
    f'{base}/multiclinsum_gs_train_en.zip', 'GS (Gold Standard)')

# LS (full — faster with v2)
ls_results, ls_types, ls_total, ls_noisy = audit_zip_v2(
    f'{base}/multiclinsum_large-scale_train_en.zip', 'LS (Large Scale)')
