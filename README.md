# DiffuClinic

DiffuClinic fine-tunes and compares three 8B models on clinical case-report
summarisation: an autoregressive baseline (Llama 3.1 8B) and two diffusion models
(LLaDA 8B, and LAD, a LoRA-adapted diffusion model). All three share the same LoRA
recipe, and the two diffusion models support **entity-pinned decoding**, a logit-level
bias that steers clinically important source entities (diseases, medications, ages,
vitals, gender) into the summary.

## Project structure

```
DiffuClinic/
├── configs/                 # one YAML per model + entity_pinning.yaml
├── Data/                    # MultiClinSum splits (gs / large-scale / test)
├── notebooks/
│   └── data_exploration.ipynb     # EDA
├── training/                # fine-tuning package (python -m training.train)
├── inference/               # entity-pinned inference + shared data loader
│   ├── llada_entity_pinning_inference.py
│   ├── lad_entity_pinning_inference.py
│   └── shared_utils.py
├── utilities/               # entity extractor, selector, fitted IDF table
├── analysis/                # entity-consistency audit of the GS split
├── models/                  # entity_pinned_generate.py (reference decode loop)
├── results/                 # generated summaries
├── third_party/lad-code/    # LAD reference model (git submodule)
└── requirements-{general,llama,llada,lad}.txt
```

Run all commands from the repo root.

## Setup

One virtualenv per environment. They pin different, incompatible `transformers`, so do not
mix them. All four are built and tested on **Python 3.12**:

| Environment | Used for                                                       | `transformers` |
|---|----------------------------------------------------------------|---|
| `requirements-general.txt` | EDA notebook + audit + entity stack  | 5.x |
| `requirements-llama.txt` | Llama 3.1 8B AR **fine-tuning** + **inference**                | 5.x |
| `requirements-llada.txt` | LLaDA 8B **fine-tuning + inference**                           | 4.38.2 (legacy, numpy-1.x) |
| `requirements-lad.txt` | LAD **fine-tuning + inference**                                | 5.12.0 (matches the pickle) |

`requirements-general.txt` is the local base/eval env (notebook + audit), the entity stack
(scispaCy) is bundled into general, llada, and lad, since EDA, the audit, and the two
diffusion inferences all use it.

The entity stack needs scispaCy and the BC5CDR model installed **with `--no-deps`**, their
resolver pulls the original `nmslib` (no Python 3.12 wheel, `nmslib-metabrainz` is the fork
that works) and an older spaCy/numpy:

```
pip install --no-deps scispacy==0.5.5
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

The BC5CDR model ships `config.cfg` booleans as quoted strings that spaCy 3.8 rejects,
`utilities/extractor.py` rewrites them on load, so no manual patch is
needed. The first extraction downloads ~1.5 GB of UMLS data into `~/.scispacy/`.

LAD also needs the submodule:

```
git submodule update --init third_party/lad-code
```

## Quickstart

Fine-tuning reads the Hugging Face token from `HF_TOKEN`. Run from the repo root.

### Exploratory data analysis & audit

```
pip install -r requirements-general.txt        # + scispaCy entity stack (see Setup)
python analysis/audit_entities.py              # GS-split entity-consistency audit
open notebooks/data_exploration.ipynb
```

### Llama 3.1 8B — autoregressive baseline

```
pip install -r requirements-llama.txt
HF_TOKEN=... python -m training.train --config configs/llama_ar.yaml
```

### LLaDA 8B — diffusion

```
pip install -r requirements-llada.txt          # + scispaCy entity stack (see Setup) for inference
HF_TOKEN=... python -m training.train --config configs/llada_diffusion.yaml
python inference/llada_entity_pinning_inference.py --data_path multiclinsum_test_en.zip --no_dllm_cache
```

### LAD — LoRA-adapted diffusion

```
pip install -r requirements-lad.txt            # + scispaCy entity stack (see Setup) for inference
git submodule update --init third_party/lad-code
HF_TOKEN=... python -m training.train --config configs/lad_finetune.yaml
python inference/lad_entity_pinning_inference.py --data_path multiclinsum_test_en.zip
```

`--self_test` runs the load/marker gates and exits, `--lambda_max 0.0` is the no-pin
control.

## Dataset

MultiClinSum clinical case reports (English). Training concatenates the gold-standard
(592) and large-scale (25,902) splits (26,494 pairs), evaluation uses the test split
(3,396). Each split is a zip of `*/fulltext/*.txt` paired with `*/summaries/*_sum.txt`,
loaded into `Full_Text` / `Summary` columns. The AR and LLaDA models are trained on the
prompt `Summarize this clinical note: {full_text}\nSummary:`, LAD uses a
`User:/Assistant:` variant required by its training corruption.

## Configurability

Models are swapped through the YAML config:

- **`configs/llama_ar.yaml`**: set `model_name` to any Hugging Face causal LM.
- **`configs/llada_diffusion.yaml`**: set `model_name` to any masked-diffusion model.
- **`configs/lad_finetune.yaml`**: LAD requires a separate config to support its custom pickle format, 
fp16 training, and `User:/Assistant:` prompt structure.

Entity pinning can be configured using the YAML file and some CLI parameters.

## Entity-pinned decoding

Clinical entities are extracted from the source note, diseases and medications via
scispaCy's BC5CDR model resolved against UMLS, and ages, vitals, dosages, and gender via
regex (`utilities/extractor.py`). The `EntitySelector` ranks them (medications by
term-frequency × recency, diseases also weighted by inverse document frequency) and keeps
a subset within a token budget (30% of the generation length, with per-category caps).
During decoding, the first token of each selected entity receives a logit bonus that is
largest when the sequence is fully masked and decays to zero by the final step, the
diffusion process still decides where each entity lands, the bias only raises its odds of
appearing. The mechanism is identical for LLaDA and LAD.
