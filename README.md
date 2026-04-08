# Legal Document Review OpenEnv

A submission for the Legal Document Review OpenEnv benchmark, implementing an LLM-based agent for contract and legal document analysis using the OpenAI API client.

## Overview

This repository contains:
- **`inference.py`** — Main agent script that runs 3 legal document review tasks
- **`requirements.txt`** — Python dependencies
- **`Dockerfile`** — Container configuration for deployment
- **`validate-submission.sh`** — Submission validation script

## Setup

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ayush1Deshmukh/my-openenv.git
   cd my-openenv
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   export HF_TOKEN=your_huggingface_token
   export API_BASE_URL=https://router.huggingface.co/v1  # optional
   export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct            # optional
   export ENV_BASE_URL=http://localhost:7860              # optional
   ```

4. **Run inference:**
   ```bash
   python inference.py
   ```

### Docker

1. **Build the image:**
   ```bash
   docker build -t legal-review-env .
   ```

2. **Run the container:**
   ```bash
   docker run \
     -e HF_TOKEN=your_token \
     -e API_BASE_URL=https://router.huggingface.co/v1 \
     -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
     -p 7860:7860 \
     legal-review-env
   ```

## Submission Checklist

### ✅ Pre-Submission Verification

- [x] **Followed sample `inference.py` strictly**
  - `[START]` emitted once at episode begin
  - `[STEP]` emitted once per step immediately after `env.step()`
  - `[END]` always emitted (even on exception)
  - All formatting rules followed

- [x] **Environment variables present**
  - `API_BASE_URL` — LLM endpoint (default: `https://router.huggingface.co/v1`)
  - `MODEL_NAME` — Model identifier (default: `Qwen/Qwen2.5-72B-Instruct`)
  - `HF_TOKEN` — API key (NO default, must be set in environment)
  - `LOCAL_IMAGE_NAME` — Docker image name (optional)

- [x] **Defaults set only for API_BASE_URL and MODEL_NAME**
  - `HF_TOKEN` explicitly has NO default

- [x] **All LLM calls use OpenAI client**
  - Configured via `OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`
  - All completions via `client.chat.completions.create()`

- [x] **Stdout logs follow required structured format**
  - `[START] task=<task_name> env=<benchmark> model=<model_name>`
  - `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
  - `[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>`

## Tasks

The agent handles 3 legal document review tasks:

1. **NDA Review** (`nda_review`)
   - Max steps: 14
   - Success threshold: 0.50
   - Temperature: 0.3

2. **Service Agreement Review** (`service_agreement_review`)
   - Max steps: 18
   - Success threshold: 0.45
   - Temperature: 0.3

3. **Merger Agreement Review** (`merger_agreement_review`)
   - Max steps: 22
   - Success threshold: 0.40
   - Temperature: 0.4

## Agent Actions

The legal review agent supports 11 action types:

1. `READ_SECTION` — Read a specific section of the document
2. `SEARCH_DOCUMENT` — Search for text in the document
3. `IDENTIFY_CLAUSE` — Identify a legal clause with confidence
4. `FLAG_ISSUE` — Flag a problematic provision with risk level
5. `ASSESS_RISK` — Set overall risk assessment
6. `EXTRACT_PARTY` — Extract a contracting party name
7. `EXTRACT_DATE` — Extract a key date
8. `EXTRACT_OBLIGATION` — Extract an obligation
9. `SUMMARIZE_DOCUMENT` — Write a comprehensive summary
10. `RECOMMEND_REVISION` — Recommend changes
11. `SUBMIT_REVIEW` — Submit final review with risk assessment

## Validation

Run the validation script to test the submission:

```bash
./validate-submission.sh https://your-hf-space-url .
```

## Dependencies

- `openai` — OpenAI API client
- `requests` — HTTP library for environment API calls
- `python >= 3.11`

## Submission Status

| Item | Status |
|------|--------|
| `inference.py` created | ✅ |
| Environment variables configured | ✅ |
| Defaults set correctly | ✅ |
| Logging format verified | ✅ |
| Dockerfile created | ✅ |
| Git repository initialized | ✅ |

## Repository

**GitHub**: `https://github.com/Ayush1Deshmukh/my-openenv.git`

**Hugging Face Space**: Deploy via HF and provide the URL

## Author

Ayush Deshmukh

## License

This submission is for the OpenEnv benchmark evaluation.
