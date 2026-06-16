#!/usr/bin/env python3
"""Full LS audit: gender, age, medication hallucinations, numeric value changes."""
import zipfile, re
from collections import defaultdict

COMMON_DRUGS = set(w.lower() for w in {
    'aspirin', 'ibuprofen', 'paracetamol', 'acetaminophen', 'morphine', 'oxycodone',
    'hydrocodone', 'codeine', 'tramadol', 'fentanyl', 'ketorolac', 'naproxen',
    'diclofenac', 'celecoxib', 'buprenorphine', 'methadone', 'gabapentin', 'pregabalin',
    'lidocaine', 'bupivacaine', 'ropivacaine',
    'amoxicillin', 'ampicillin', 'penicillin', 'ceftriaxone', 'cefazolin',
    'cefuroxime', 'cefixime', 'ciprofloxacin', 'levofloxacin', 'moxifloxacin',
    'azithromycin', 'clarithromycin', 'doxycycline', 'minocycline', 'vancomycin',
    'gentamicin', 'tobramycin', 'clindamycin', 'metronidazole', 'trimethoprim',
    'sulfamethoxazole', 'nitrofurantoin', 'linezolid', 'meropenem', 'imipenem',
    'piperacillin', 'ertapenem',
    'lisinopril', 'enalapril', 'ramipril', 'captopril', 'losartan', 'valsartan',
    'irbesartan', 'telmisartan', 'amlodipine', 'nifedipine', 'diltiazem', 'verapamil',
    'metoprolol', 'atenolol', 'propranolol', 'carvedilol', 'bisoprolol',
    'furosemide', 'hydrochlorothiazide', 'spironolactone', 'eplerenone',
    'digoxin', 'amiodarone', 'nitroglycerin', 'isosorbide', 'clonidine', 'hydralazine',
    'warfarin', 'heparin', 'enoxaparin', 'dalteparin', 'rivaroxaban', 'apixaban',
    'dabigatran', 'edoxaban', 'clopidogrel', 'ticagrelor', 'prasugrel',
    'metformin', 'insulin', 'glipizide', 'glyburide', 'glimepiride', 'sitagliptin',
    'empagliflozin', 'dapagliflozin', 'liraglutide', 'semaglutide', 'pioglitazone',
    'prednisone', 'prednisolone', 'dexamethasone', 'methylprednisolone',
    'hydrocortisone', 'methotrexate', 'cyclophosphamide', 'cyclosporine',
    'tacrolimus', 'mycophenolate', 'azathioprine', 'sirolimus', 'everolimus',
    'cisplatin', 'carboplatin', 'oxaliplatin', 'doxorubicin', 'daunorubicin',
    'paclitaxel', 'docetaxel', 'gemcitabine', 'fluorouracil', 'capecitabine',
    'irinotecan', 'etoposide', 'vincristine', 'vinblastine', 'bleomycin',
    'tamoxifen', 'anastrozole', 'letrozole', 'imatinib', 'erlotinib', 'gefitinib',
    'sorafenib', 'sunitinib', 'lenalidomide', 'thalidomide', 'bortezomib',
    'rituximab', 'bevacizumab', 'trastuzumab', 'pembrolizumab', 'nivolumab',
    'ipilimumab', 'adalimumab', 'infliximab', 'etanercept', 'ustekinumab',
    'omalizumab', 'denosumab', 'pertuzumab', 'cetuximab', 'panitumumab',
    'sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram',
    'venlafaxine', 'duloxetine', 'amitriptyline', 'nortriptyline', 'quetiapine',
    'risperidone', 'olanzapine', 'aripiprazole', 'haloperidol', 'clozapine',
    'valproate', 'carbamazepine', 'levetiracetam', 'phenytoin', 'lamotrigine',
    'topiramate', 'donepezil', 'memantine', 'levodopa', 'carbidopa',
    'albuterol', 'salbutamol', 'ipratropium', 'tiotropium', 'fluticasone',
    'budesonide', 'montelukast', 'theophylline',
    'omeprazole', 'pantoprazole', 'esomeprazole', 'lansoprazole', 'ranitidine',
    'famotidine', 'ondansetron', 'metoclopramide', 'loperamide',
    'levothyroxine', 'allopurinol', 'colchicine', 'sildenafil', 'tadalafil',
    'alendronate', 'propofol', 'midazolam', 'ketamine',
    'epinephrine', 'norepinephrine', 'dopamine', 'dobutamine',
    'zoledronic',
})

def get_patient_age(text):
    sentences = re.split(r'(?<=[.!?])\s+', text[:600])
    for sent in sentences[:3]:
        m = re.search(r'(\d{1,3})[- ]year[- ]old\s+(?:woman|man|female|male|patient|who|presented|admitted|was)', sent.lower())
        if m: return int(m.group(1))
        m = re.search(r'aged\s+(\d{1,3})\s*(?:year|yr)', sent.lower())
        if m: return int(m.group(1))
    return None

def get_patient_gender(text):
    sentences = re.split(r'(?<=[.!?])\s+', text[:800])
    for sent in sentences[:3]:
        s = sent.lower()
        if re.search(r'\b(female|woman|girl|lady)\b.*\bpatient\b', s) or re.search(r'\bpatient\b.*\b(female|woman|girl)\b', s): return 'female'
        if re.search(r'\b(male|man|boy|gentleman)\b.*\bpatient\b', s) or re.search(r'\bpatient\b.*\b(male|man|boy)\b', s): return 'male'
        m = re.search(r'(\d{1,3})[- ]year[- ]old\s+(woman|man|female|male|boy|girl|lady|gentleman)', s)
        if m: return 'female' if m.group(2) in ('woman','female','girl','lady') else 'male'
        m = re.search(r'A\s+\d{1,3}[- ]year[- ]old\s+(woman|man|female|male)', s)
        if m: return 'female' if m.group(1) in ('woman','female') else 'male'
    return None

print('Processing LS (25,902 pairs)...')
z = zipfile.ZipFile('data/multiclinsum_large-scale_train_en.zip')
ft_files = sorted([f for f in z.namelist() if '/fulltext/' in f and f.endswith('.txt')])

gender_issues = []
age_issues = []
med_issues = []
total = 0

for ft_file in ft_files:
    total += 1
    if total % 5000 == 0:
        print(f'  {total}/{len(ft_files)}...')

    sum_file = ft_file.replace('/fulltext/', '/summaries/').replace('.txt', '_sum.txt')
    if sum_file not in z.namelist():
        continue

    pid = ft_file.split('/')[-1].replace('.txt', '')
    ft = z.read(ft_file).decode('utf-8', errors='replace')
    sm = z.read(sum_file).decode('utf-8', errors='replace')

    sg = get_patient_gender(ft)
    rg = get_patient_gender(sm)
    if sg and rg and sg != rg:
        gender_issues.append((pid, sg, rg))

    sa = get_patient_age(ft)
    ra = get_patient_age(sm)
    if sa and ra and abs(sa - ra) > 2:
        age_issues.append((pid, sa, ra))

    src_words = set(re.findall(r'\b\w+\b', ft.lower()))
    ref_words = set(re.findall(r'\b\w+\b', sm.lower()))
    ref_drugs = ref_words & COMMON_DRUGS
    for drug in ref_drugs:
        if drug not in src_words:
            # Filter: drug must appear with dosage context in summary
            if re.search(rf'\d+\s*(mg|g|mcg|ml|mmol)\b.{0,30}\b{drug}\b', sm.lower()) or \
               re.search(rf'\b{drug}\b.{0,30}\d+\s*(mg|g|mcg|ml|mmol)', sm.lower()):
                med_issues.append((pid, drug))

print(f'\n===== LS RESULTS (n={total}) =====')
print(f'Gender contradictions: {len(gender_issues)} ({100*len(gender_issues)/total:.2f}%)')
print(f'Age mismatches: {len(age_issues)} ({100*len(age_issues)/total:.2f}%)')
print(f'Medication hallucinations: {len(med_issues)} ({100*len(med_issues)/total:.2f}%)')

noisy_pids = set()
for iss in [gender_issues, age_issues, med_issues]:
    for item in iss:
        noisy_pids.add(item[0])
print(f'\nPairs with >=1 issue: {len(noisy_pids)} ({100*len(noisy_pids)/total:.2f}%)')

print(f'\n--- Gender examples ---')
for pid, sg, rg in gender_issues[:5]:
    print(f'  {pid}: {sg} -> {rg}')

print(f'\n--- Age examples (first 8) ---')
for pid, sa, ra in age_issues[:8]:
    print(f'  {pid}: {sa} -> {ra}')

print(f'\n--- Medication hallucination examples (first 15) ---')
for pid, drug in med_issues[:15]:
    print(f'  {pid}: "{drug}"')
