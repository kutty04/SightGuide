# SightGuide: Diversity-Aware Caption-Pair Ranking & LLM-Based Fusion

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![TypeScript 6](https://img.shields.io/badge/TypeScript-6-3178C6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS v4](https://img.shields.io/badge/TailwindCSS-v4-38B2AC.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/tests-51%20passing-brightgreen.svg)]()

> **Final-Year B.E. Computer Science Research Project**  
> **Academic Foundation:** Celona et al., *"Improving image captioning descriptiveness by ranking and LLM-based fusion"*, **Neural Computing and Applications**, Springer Nature, 2025.

---

## 1. Project Overview

**SightGuide** is an accessible vision-assistance application that generates rich, grounded image descriptions by combining vision-language candidate generation, semantic ranking, and Large Language Model (LLM) fusion.

### The Research Question
> **Does diversity-aware caption-pair selection improve LLM-fused image caption quality compared with conventional CLIP Top-2 selection when all LLM generation settings are held constant?**

In the base paper (Celona et al., 2025), multiple candidate captions are generated with BLIP, ranked by CLIP image-text cosine similarity, and the top two candidates (Top-2) are fused using an LLM. However, vision-language models frequently output syntactic duplicates that achieve near-identical top CLIP scores, wasting the LLM's fusion context on redundant tokens.

SightGuide introduces **Diversity-Aware Caption-Pair Selection** based on **Maximal Marginal Relevance (MMR)**:
1. **Primary Visual Anchor ($c_1^*$):** Fixed to the highest CLIP visual-alignment candidate ($\arg\max_{c_i} S_{\text{CLIP}}(c_i, \text{Image})$).
2. **Secondary Complementary Candidate ($c_2^*$):** Selected to optimize visual relevance while explicitly penalizing lexical/semantic redundancy with $c_1^*$:
   $$\text{Score}(c_1^*, c_j; \lambda) = \lambda \cdot S_{\text{norm}}(c_j) - (1 - \lambda) \cdot \text{Sim}(c_1^*, c_j) - \text{Penalty}(\text{grounding})$$
   where $\lambda \in [0.0, 1.0]$ balances visual relevance against novelty, and candidates falling below normalized visual confidence threshold $\tau = 0.20$ receive a penalty of $10.0$ to prevent hallucinations.

---

## 2. System Architecture

```
D:\SightGuide\
├── backend/
│   └── app/
│       ├── api/
│       │   └── endpoints.py          # REST endpoints (health, generate, compare, benchmark)
│       ├── services/
│       │   ├── caption_pipeline.py   # BLIP + CLIP + MMR orchestration service
│       │   ├── diversity_ranker.py   # Historical Phase 1 ranker (protected artifact)
│       │   └── fusion_service.py     # Groq LLM Conservative V2 fusion client
│       ├── schemas.py                # Pydantic v2 data models & request/response contracts
│       └── main.py                   # FastAPI application entrypoint with CORS & lifecycle
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── BenchmarkViewer.tsx   # 12-image benchmark dataset inspector
│   │   │   ├── CandidateList.tsx     # Candidate pool visualizer with CLIP score meters
│   │   │   ├── CaptionPlayer.tsx     # Accessible Web Speech API Text-to-Speech audio bar
│   │   │   ├── ComparisonView.tsx    # Side-by-side Control vs Proposed comparison cards
│   │   │   ├── ImageUploader.tsx     # Drag-and-drop image upload & lambda slider
│   │   │   └── ResearchInfo.tsx      # Base paper citation, MMR math, and prompt rules
│   │   ├── services/
│   │   │   └── api.ts                # TypeScript client for backend API
│   │   ├── types/
│   │   │   └── api.ts                # TypeScript interfaces matching backend schemas
│   │   ├── App.tsx                   # Main tabbed UI (Interactive, Benchmark, Research)
│   │   └── index.css                 # Tailwind CSS v4 styling & dark theme
│   ├── package.json
│   └── vite.config.ts
├── dataset/                          # 12-image evaluation benchmark
├── data/                             # Historical candidate pools & human evaluations
├── experiments/                      # Fusion runners, quality evaluators, human eval tooling
├── results/                          # Controlled experiment CSVs and quality summaries
├── tests/
│   ├── test_backend_api.py           # Backend API integration tests (8 tests)
│   ├── test_diversity_ranking.py     # Diversity ranker unit tests (16 tests)
│   ├── test_fusion_comparison.py     # Controlled fusion tests (15 tests)
│   └── test_human_eval_analysis.py   # Blinded human evaluation tests (12 tests)
├── .env.example                      # Environment variables template
├── artifact_hashes.json              # SHA-256 integrity manifest for historical artifacts
└── README.md
```

---

## 3. Installation & Getting Started

### Prerequisites
- **Python**: 3.11 or 3.12 (with PyTorch, Transformers, Pillow, FastAPI, Groq)
- **Node.js**: v18+ or v20+ (with npm)
- **Groq API Key**: Optional for live LLM fusion; if omitted, the system operates in graceful dry-run / mock mode.

### Backend Setup
1. Open a terminal in the project root:
   ```bash
   cd D:\SightGuide
   ```
2. Configure your environment variables:
   ```bash
   copy .env.example .env
   ```
   Add your `GROQ_API_KEY` to `.env` if you have one.
3. Start the FastAPI backend:
   ```bash
   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
4. Verify the backend:
   - Interactive Swagger API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Frontend Setup
1. Open a second terminal:
   ```bash
   cd D:\SightGuide\frontend
   ```
2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Access the web application at [http://localhost:5173](http://localhost:5173).

---

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, model readiness, and Groq API key configuration status |
| `POST` | `/api/caption/generate` | Generate candidates, rank with MMR, and fuse using Conservative V2 |
| `POST` | `/api/caption/compare` | Controlled Control (Top-2) vs. Proposed (MMR) comparison for an image |
| `GET` | `/api/benchmark/summary` | Retrieve the 12-image benchmark metrics and comparison data |
| `GET` | `/api/benchmark/image/{image_name}` | Serve benchmark images directly for UI preview |

---

## 5. Conservative V2 Fusion Policy

To prevent visual hallucinations, SightGuide adheres to the **Conservative V2 prompt policy**:
- **Consensus Preference:** Prioritize details supported by both candidate captions.
- **Strict Anti-Hallucination:** Never invent objects, colors, locations, or identities not present in the candidate inputs.
- **Conflict Resolution:** If candidates conflict, choose the more conservative, generic description.
- **Formatting Constraint:** Output exactly one concise sentence without quotation marks or conversational framing.

---

## 6. Running Automated Tests & Integrity Verification

### Running the 51 Pytest Tests
```bash
python -m pytest D:\SightGuide\tests -v
```
All 51 tests verify:
- Diversity-aware ranking formulas, edge cases, lambda bounds, and grounding safeguards (16 tests).
- Controlled fusion comparison, prompts, determinism, and CSV record integrity (15 tests).
- Blinded human evaluation randomization, unblinding, and Cohen's Kappa inter-rater agreement (12 tests).
- FastAPI REST endpoints, multipart image uploads, CORS, and schema serialization (8 tests).

### Verifying Historical Artifact Integrity
All 14 historical baseline files from Phase 1 are protected and tracked via cryptographic hashes in `artifact_hashes.json`. To verify that not a single byte has drifted:
```bash
python -c "
import json, hashlib, os
with open('artifact_hashes.json') as f:
    hashes = json.load(f)
all_ok = True
for rel, data in hashes.items():
    p = os.path.join('.', rel)
    if not os.path.exists(p):
        print(f'MISSING: {rel}'); all_ok = False; continue
    with open(p, 'rb') as fp:
        actual = hashlib.sha256(fp.read()).hexdigest()
    if actual != data['sha256']:
        print(f'MISMATCH: {rel}'); all_ok = False
if all_ok:
    print('ALL 14 HISTORICAL ARTIFACTS VERIFIED 100% BIT-FOR-BIT')
"
```

---

## 7. Key Empirical Findings (Live Execution Completed)

1. **Candidate Pair Divergence & Empirical Resolution:**
   - On 10 out of 12 benchmark images (83.3%), Diversity-Aware MMR ($\lambda=0.5$) selected candidate pairs identical to naive CLIP Top-2, verifying visual stability.
   - On 2 out of 12 benchmark images (`edgar` and `sample-bumblebee`), Proposed selection diverged by promoting Candidate 3 to increase semantic complementarity (pair diversity deltas of $+0.314$ and $+0.077$).
   - Live fusion with `openai/gpt-oss-20b` under identical Conservative V2 prompts yielded:
     - **`edgar-nKC772R_qog-unsplash.jpg`**:
       - *Control*: `"small kitten sitting on the ground"`
       - *Proposed*: `"small kitten sits on the floor near a bench"` (incorporates grounded spatial context from Candidate 3).
     - **`sample-bumblebee-400x300.png`**:
       - *Control*: `"a bum on a yellow flower surrounded by green leaves"` (contains token corruption artifact `"bum"`).
       - *Proposed*: `"a bee sitting on a yellow flower surrounded by green leaves in a field"` (resolves token corruption into `"bee"` while integrating descriptive surrounding context).
2. **Quality Metric Summary (N = 11 valid images, 1 historical failure excluded):**
   - **Control ($\lambda=1.0$)**: Mean Input Pair Diversity = 0.6397, Mean Characters = 48.45, Mean Words = 10.0, Artifacts = 2.
   - **Proposed ($\lambda=0.5$)**: Mean Input Pair Diversity = 0.6723, Mean Characters = 51.00, Mean Words = 10.64, Artifacts = 1.
3. **Lexical Diversity vs. Factual Grounding:**
   - Higher diversity does not inherently imply higher quality unless constrained by conservative fusion rules.
   - The visual safeguard threshold $\tau=0.20$ prevented hallucinated captions, and Conservative V2 prompt policy ensured that descriptive details remained verifiably grounded.

---

## 8. Human Evaluation Protocol Status

- **Blinded Template:** [`results/sightguide_human_eval_blinded_template.csv`](file:///D:/SightGuide/results/sightguide_human_eval_blinded_template.csv) contains all 12 benchmark rows with randomized Method A / Method B presentations. All score fields remain blank until evaluated by human reviewers.
- **Blinding Key:** [`results/sightguide_human_eval_blinding_key.csv`](file:///D:/SightGuide/results/sightguide_human_eval_blinding_key.csv) is maintained separately.
- **Statistical Pipeline:** [`experiments/analyze_human_evaluation.py`](file:///D:/SightGuide/experiments/analyze_human_evaluation.py) aggregates ratings at the image level to prevent pseudoreplication and computes Cohen's Kappa and Wilcoxon signed-rank statistics once real ratings are collected.
