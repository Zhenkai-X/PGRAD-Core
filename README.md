# PGRAD

**Prior-Guided Evidence-Chain Reasoning and Diagnosis for Quantitative CT-Based Pulmonary Function Evaluation**

![PGRAD framework](figure/pgrad_framework.png)

*Overview of the PGRAD framework.*

PGRAD is a prior-guided evidence-chain reasoning framework for interpretable quantitative CT-based differentiation of NC, PRISm, and COPD.

It integrates QCT-based prior modeling, subject-level evidence construction, retrieval of supportive and confusable cases, and constrained LLM reasoning to generate structured diagnostic reports.

## Workflow

```text
QCT features
  -> prior probabilities
  -> reference-normalized evidence
  -> supportive/confusable retrieval
  -> gating
  -> KEEP / SWITCH / ABSTAIN
  -> structured report
```

The main components are:

- `core_method/prior_model.py`: preprocessing, fused feature scoring, Top-20
  predictor selection, L1-regularized multinomial logistic regression, and
  out-of-fold probability estimation.
- `core_method/evidence_construction.py`: reference normalization and
  subject-level evidence construction.
- `core_method/pgrad_reasoning.py`: evidence retrieval, similarity scoring,
  gating, and constrained report generation.
- `pgrad_app/`: optional Streamlit application and integration pipeline.

## Repository Structure

```text
app.py
pgrad_pipeline.py
core_method/
  config.yaml
  config_loader.py
  prior_model.py
  evidence_construction.py
  pgrad_reasoning.py
pgrad_app/
data/
  reference/
  evidence/correct/
  evidence/error/
  examples/single_inspiratory/
  examples/paired_inspiratory_expiratory/
models/
outputs/
tests/
  fixtures/single_inspiratory/
  fixtures/paired_inspiratory_expiratory/
.streamlit/
```

Empty directories are retained with `.gitkeep` files. Runtime resources are
intentionally excluded from the public repository.

## Configuration

Core parameters are defined in `core_method/config.yaml`, including:

- random seed and class definitions
- Top-20 prior predictors
- Top-30 evidence items
- Top-3 supportive and Top-3 confusable case retrieval
- similarity weights
- gating thresholds
- LLM temperature

The YAML file is the source of truth for the public method configuration.

## Installation

```bash
git clone https://github.com/Zhenkai-X/PGRAD-Core.git
cd PGRAD-Core
python -m pip install -r requirements.txt
```

The repository does not download or install data, model artifacts, or
credentials.

## External Resources

The application requires externally supplied resources. Provide their paths
through function arguments or environment variables:

```text
PGRAD_MODEL_PATH
PGRAD_FEATURE_SCORES_PATH
PGRAD_TRAIN_STATS_PATH
PGRAD_CORRECT_EVIDENCE_PATH
PGRAD_ERROR_EVIDENCE_PATH
```

The application raises a clear error when a required resource is not
provided. No private runtime resources are bundled in this repository.

Optional local demo inputs can be configured with:

```text
PGRAD_SINGLE_INSPIRATORY_DEMO_PATH
PGRAD_PAIRED_INSPIRATORY_EXPIRATORY_DEMO_PATH
```

## Running the Application

```bash
streamlit run app.py
```

Upload a compatible table through the application, or configure an external
input path. The application does not assume that private resources are
present in the repository.

Provider credentials must be supplied through the local environment or
Streamlit secrets. Do not commit credential files.

## Tests

The public test checks:

- module availability
- configuration loading
- required directory structure

Run:

```bash
python -m unittest discover -s tests -v
```

The public test suite does not require private data, model artifacts, or
case-level evidence.

## Data Availability

Subject-level imaging, pulmonary function, clinical data, trained model artifacts, and case-level evidence are not distributed in this repository because of privacy, institutional, and data-use restrictions.

## Citation

Citation information will be added upon publication.
